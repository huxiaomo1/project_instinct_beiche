"""Parkour command generator with a bounded velocity-command slew rate."""

from __future__ import annotations

from isaaclab.utils import configclass

from instinctlab.tasks.parkour.stand_control import limit_velocity_command

from .commands_cfg import PoseVelocityCommandCfg
from .pose_velocity_command import PoseVelocityCommand


class StoppingVelocityCommand(PoseVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.raw_command = self.vel_command_b.clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[ids] = 0.0
        self.raw_command[ids] = 0.0
        return super().reset(env_ids)

    def _update_command(self):
        previous = self.vel_command_b.clone()
        super()._update_command()
        # The parent applies random terrain commands after standing masks.
        # In this task, sampled stops must take priority on every terrain.
        self.vel_command_b[self.is_standing_env] = 0.0
        self.raw_command.copy_(self.vel_command_b)
        self.vel_command_b.copy_(
            limit_velocity_command(previous, self.raw_command, self._env.step_dt, self.cfg.acceleration_limits)
        )


@configclass
class StoppingVelocityCommandCfg(PoseVelocityCommandCfg):
    class_type: type = StoppingVelocityCommand
    acceleration_limits: tuple[float, float, float] = (1.5, 1.5, 3.0)
