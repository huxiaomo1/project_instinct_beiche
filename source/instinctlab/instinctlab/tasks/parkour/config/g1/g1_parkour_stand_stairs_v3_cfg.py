"""V3 standing/stairs configuration; shares the ordinary Parkour environment and PPO."""

import copy
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp

from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY

V3_DEPTH_CROP_REGION = (18, 0, 16, 16)


def _configure_v3(env_cfg, randomize_camera: bool):
    # The ordinary G1 configuration assigns a shared terrain generator in its
    # post-init. Copy before filtering or changing any terrain parameters.
    terrain = copy.deepcopy(env_cfg.scene.terrain.terrain_generator)
    proportions = {
        "perlin_rough": 0.10,
        "perlin_rough_stand": 0.30,
        "pyramid_stairs": 0.30,
        "pyramid_stairs_inv": 0.30,
    }
    terrain.sub_terrains = {name: terrain.sub_terrains[name] for name in proportions}
    for name, proportion in proportions.items():
        terrain.sub_terrains[name].proportion = proportion
    terrain.sub_terrains["perlin_rough_stand"].noise_scale = 0.0
    env_cfg.scene.terrain.terrain_generator = terrain

    # Match the source checkpoint explicitly without changing the ordinary task.
    env_cfg.scene.camera = copy.deepcopy(env_cfg.scene.camera)
    crop = env_cfg.scene.camera.noise_pipeline["crop_and_resize"]
    crop.crop_region = V3_DEPTH_CROP_REGION
    crop.resize_shape = None

    env_cfg.commands = copy.deepcopy(env_cfg.commands)
    command = env_cfg.commands.base_velocity
    command.velocity_ranges = {name: command.velocity_ranges[name] for name in proportions}
    command.random_velocity_terrain = []
    command.use_terrain_ang_vel_range = True

    # Preserve G1-specific reward adjustments, including the shoe height offset.
    env_cfg.rewards = copy.deepcopy(env_cfg.rewards)
    rewards = env_cfg.rewards.rewards
    rewards.stand_joint_vel_l2 = RewTerm(
        func=mdp.stand_joint_vel_l2,
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    rewards.stand_lin_vel_xy_l2 = RewTerm(
        func=mdp.stand_lin_vel_xy_l2, weight=-50.0, params={"command_name": "base_velocity"}
    )
    rewards.stand_ang_vel_l2 = RewTerm(func=mdp.stand_ang_vel_l2, weight=-5.0, params={"command_name": "base_velocity"})

    env_cfg.events = copy.deepcopy(env_cfg.events)
    env_cfg.events.randomize_camera_offset = (
        EventTerm(
            func=mdp.randomize_camera_offsets,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("camera"),
                "offset_pose_ranges": {
                    "roll": (-math.radians(2.0), math.radians(2.0)),
                    "pitch": (-math.radians(5.0), math.radians(5.0)),
                    "yaw": (-math.radians(2.0), math.radians(2.0)),
                },
                "distribution": "uniform",
            },
        )
        if randomize_camera
        else None
    )


@configclass
class G1ParkourStandStairsV3EnvCfg(G1ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_v3(self, randomize_camera=True)


@configclass
class G1ParkourStandStairsV3EnvCfg_PLAY(G1ParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        _configure_v3(self, randomize_camera=False)
