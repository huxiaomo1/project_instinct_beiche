"""Checkpoint-compatible AMP with a gradually introduced stopping reward."""

from __future__ import annotations

import torch
import torch.distributed as dist

from instinct_rl.algorithms.wasabi import WasabiPPO

from instinctlab.tasks.parkour.stand_control import blend_style_reward


class ParkourStandPPO(WasabiPPO):
    def __init__(
        self, *args, stand_warmup_updates=100, stand_blend_updates=2000, reset_optimizer_on_warm_start=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.discriminator_reward_type != "quad":
            raise ValueError("Standing reward blending requires bounded quad AMP rewards")
        if stand_warmup_updates < 0 or stand_blend_updates <= 0:
            raise ValueError("Invalid standing reward schedule")
        self.stand_warmup_updates = stand_warmup_updates
        self.stand_blend_updates = stand_blend_updates
        self.reset_optimizer_on_warm_start = reset_optimizer_on_warm_start
        self.stand_updates = 0
        self._stand_transition = None

    @property
    def stand_progress(self):
        return min(max((self.stand_updates - self.stand_warmup_updates) / self.stand_blend_updates, 0.0), 1.0)

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        if "parkour_stand" not in infos:
            raise RuntimeError("ParkourStandPPO must run with ParkourStandEnv")
        self._stand_transition = infos["parkour_stand"]
        try:
            super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        finally:
            self._stand_transition = None
        infos.setdefault("step", {}).update(self._stand_log)

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack):
        auxiliary = super().compute_auxiliary_reward(obs_pack)
        if self._stand_transition is None:
            raise RuntimeError("Missing pre-reset stopping reward data")
        state = self._stand_transition
        amp = auxiliary["discriminator_reward"]
        replacement = state["replacement"].reshape_as(amp)
        stop_mix = state["stop_mix"].reshape_as(amp)
        blended = blend_style_reward(amp, replacement, stop_mix, self.stand_progress)
        # PPO applies discriminator_reward_coef exactly once to this value.
        auxiliary["discriminator_reward"] = blended
        self._stand_log = {"stand/" + key: value.detach().clone() for key, value in state.items()}
        self._stand_log.update(
            {
                "stand/amp_raw": amp.detach().clone(),
                "stand/style_blended": blended.detach().clone(),
                "stand/style_contribution": blended.detach().clone() * self.discriminator_reward_coef,
                "stand/progress": amp.new_tensor(self.stand_progress),
            }
        )
        stopped = stop_mix >= 0.99
        count = stopped.sum().clamp_min(1)
        self._stand_log["stand/stop_fraction"] = stopped.float().mean()
        for name, value in (
            ("stop_speed", state.get("linear_speed", torch.zeros_like(stop_mix))),
            ("stop_amp_mean", amp),
            ("stop_replacement_mean", replacement),
        ):
            self._stand_log["stand/" + name] = (value.reshape_as(amp) * stopped).sum() / count
        return auxiliary

    def update(self, current_learning_iteration):
        result = super().update(current_learning_iteration)
        self.stand_updates += 1
        return result

    def state_dict(self):
        state = super().state_dict()
        state["parkour_stand_updates"] = self.stand_updates
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.stand_updates = int(state.get("parkour_stand_updates", 0))
        if "parkour_stand_updates" not in state and self.reset_optimizer_on_warm_start:
            self.optimizer.state.clear()
            self.discriminator_optimizer.state.clear()
        # Loading optimizer state restores the OLD lr; explicitly use this run's
        # fine-tuning rate instead. Preserve learned policy action noise.
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        for group in self.discriminator_optimizer.param_groups:
            group["lr"] = self.discriminator_optimizer_kwargs.get("lr", self.learning_rate)

    def distributed_data_parallel(self):
        super().distributed_data_parallel()
        # The runner loads only on rank 0; the local fine-tuning schedule must
        # agree across ranks, including when resuming a fine-tuned checkpoint.
        if dist.is_initialized():
            counter = torch.tensor(self.stand_updates, dtype=torch.long, device=self.device)
            dist.broadcast(counter, src=0)
            self.stand_updates = int(counter.item())
            for value in self.discriminator.state_dict().values():
                dist.broadcast(value, src=0)

            # The upstream runner loads optimizer states only on rank 0 too.
            def to_cpu(value):
                if isinstance(value, torch.Tensor):
                    return value.cpu()
                if isinstance(value, dict):
                    return {key: to_cpu(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [to_cpu(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(to_cpu(item) for item in value)
                return value

            states = [
                (
                    to_cpu((self.optimizer.state_dict(), self.discriminator_optimizer.state_dict()))
                    if dist.get_rank() == 0
                    else None
                )
            ]
            dist.broadcast_object_list(states, src=0)
            if dist.get_rank() != 0:
                self.optimizer.load_state_dict(states[0][0])
                self.discriminator_optimizer.load_state_dict(states[0][1])
