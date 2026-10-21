"""Warm-start ordinary Parkour with command-conditioned stopping rewards."""

from __future__ import annotations

import copy

from isaaclab.utils import configclass

from instinctlab.tasks.parkour.mdp.commands.stopping_velocity_command import StoppingVelocityCommandCfg
from instinctlab.tasks.parkour.stand_control import StandControlCfg

from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY


class StandFineTuneMixin:
    def configure_standing_finetune(self):
        self.stand_control = StandControlCfg()
        command = self.commands.base_velocity
        self.commands.base_velocity = StoppingVelocityCommandCfg(
            **{name: copy.deepcopy(value) for name, value in vars(command).items() if name != "class_type"}
        )
        command = self.commands.base_velocity
        command.acceleration_limits = self.stand_control.acceleration_limits
        command.resampling_time_range = (3.0, 5.0)
        command.rel_standing_envs = 0.35
        # Random commands on flat terrain reproduce joystick speed selection;
        # stairs retain their terrain-target direction for traversal practice.
        command.random_velocity_terrain = ["perlin_rough", "perlin_rough_stand"]
        command.ranges.lin_vel_x = (0.10, 0.80)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.6, 0.6)
        command.lin_vel_threshold = 0.0
        command.ang_vel_threshold = 0.0
        for terrain_name, ranges in command.velocity_ranges.items():
            if "stairs" in terrain_name:
                ranges["lin_vel_x"] = (0.20, 0.80)

        terrain = copy.deepcopy(self.scene.terrain.terrain_generator)
        selected = {
            "perlin_rough": 0.20,
            "perlin_rough_stand": 0.15,
            "pyramid_stairs": 0.25,
            "pyramid_stairs_inv": 0.25,
        }
        other_total = sum(cfg.proportion for name, cfg in terrain.sub_terrains.items() if name not in selected)
        for name, cfg in terrain.sub_terrains.items():
            cfg.proportion = selected[name] if name in selected else cfg.proportion * 0.15 / other_total
        self.scene.terrain.terrain_generator = terrain
        # Superseded by the bounded auxiliary posture reward. The old term also
        # affects low-speed commands and would tighten joints too early.
        self.rewards.rewards.stand_still.weight = 0.0


@configclass
class G1ParkourStandFineTuneEnvCfg(StandFineTuneMixin, G1ParkourEnvCfg):
    stand_control: StandControlCfg = StandControlCfg()
    stand_finetune: bool = True
    auto_depth_from_checkpoint: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.configure_standing_finetune()


@configclass
class G1ParkourStandFineTuneEnvCfg_PLAY(StandFineTuneMixin, G1ParkourEnvCfg_PLAY):
    stand_control: StandControlCfg = StandControlCfg()
    stand_finetune: bool = True
    auto_depth_from_checkpoint: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.configure_standing_finetune()
