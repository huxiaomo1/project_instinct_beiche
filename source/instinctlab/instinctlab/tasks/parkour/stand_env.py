"""Capture auxiliary standing rewards before command resampling and auto-reset."""

from __future__ import annotations

import torch

from instinctlab.envs import InstinctRlEnv
from instinctlab.managers import MultiRewardManager
from instinctlab.tasks.parkour.stand_control import stopping_rewards


class StandRewardManager(MultiRewardManager):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.contacts = env.scene.sensors["contact_forces"]
        self.foot_ids, foot_names = self.robot.find_bodies(".*_ankle_roll_link")
        self.contact_ids, _ = self.contacts.find_bodies(foot_names, preserve_order=True)
        if len(self.foot_ids) != 2:
            raise ValueError("Standing fine-tuning requires exactly two ankle-roll bodies")
        self.leg_mask = torch.tensor(
            [any(part in name for part in ("hip_", "knee_", "ankle_")) for name in self.robot.joint_names],
            dtype=torch.bool,
            device=env.device,
        )

    def compute(self, dt):
        rewards = super().compute(dt)
        data = self.robot.data
        command = self._env.command_manager.get_command("base_velocity")
        forces = self.contacts.data.net_forces_w[:, self.contact_ids]
        components = stopping_rewards(
            command,
            data.root_lin_vel_b,
            data.root_ang_vel_b,
            data.projected_gravity_b,
            data.joint_pos - data.default_joint_pos,
            data.joint_vel,
            forces[..., 2].clamp_min(0.0),
            data.body_pos_w[:, self.foot_ids, 2],
            self.leg_mask,
            self._env.cfg.stand_control,
        )
        # Separate from environment rewards: no dt scaling and no extra critic.
        # Store transition data here; next observations may already have a new
        # command, or belong to the next episode after an automatic reset.
        self._env.extras["parkour_stand"] = {k: v.detach().clone() for k, v in components.items()}
        return rewards


class ParkourStandEnv(InstinctRlEnv):
    def load_managers(self):
        super().load_managers()
        self.reward_manager = StandRewardManager(self.cfg.rewards, self)
