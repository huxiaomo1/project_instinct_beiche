import copy
import os

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    beyondmimic_g1_29dof_actuators,
    beyondmimic_g1_29dof_delayed_actuators,#这两个有什么区别
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.parkour.config.parkour_env_cfg_onlystairs import (
    PLAY_DUAL_PYRAMID_SPAWN_X,
    PLAY_DUAL_PYRAMID_SPEED,
    PLAY_DUAL_PYRAMID_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG_ONLYSTAIRS,
    ParkourEnvCfg,
)

__file_dir__ = os.path.dirname(os.path.realpath(__file__))
MOTION_DATA_DIR = os.path.abspath(
    os.path.join(__file_dir__, "../../../../../../../tool/dataset/division_npz")
)
MOTION_SELECTION_FILE = os.path.join(__file_dir__, "parkour_motion_terrain_conditioned_onlystairs.yaml")
G1_CFG = copy.deepcopy(G1_29DOF_TORSOBASE_POPSICLE_CFG)
G1_CFG.spawn.merge_fixed_joints = True
G1_CFG.init_state.pos = (0.0, 0.0, 0.9)
G1_with_shoe_CFG = copy.deepcopy(G1_CFG)
G1_with_shoe_CFG.spawn.asset_path = os.path.abspath(
    f"{__file_dir__}/../../urdf/g1_29dof_torsoBase_popsicle_without_shoe.urdf"
)


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    path = MOTION_DATA_DIR
    retargetting_func = None
    filtered_motion_selection_filepath = MOTION_SELECTION_FILE
    motion_start_from_middle_range = [0.0, 0.9]
    terrain_motion_file_patterns = {
        "perlin_rough_stand": ["stand/*.npz"],
        "perlin_rough": ["walk/*.npz"],
        "pyramid_stairs": ["parkour_motion_without_run_retargetted.npz"],
        "pyramid_stairs_high": ["parkour_motion_without_run_retargetted.npz"],
        "pyramid_stairs_inv": ["parkour_motion_without_run_retargetted.npz"],
        "pyramid_stairs_inv_high": ["parkour_motion_without_run_retargetted.npz"],
        "dual_pyramid_course": ["parkour_motion_without_run_retargetted.npz"],
    }
    terrain_motion_start_from_middle_range = {
        # Short clips start at frame zero so they provide the longest possible
        # reference window before dataset_exhausted resets the environment.
        "perlin_rough_stand": (0.0, 0.0),
        "perlin_rough": (0.0, 0.0),
        # The combined parkour sequence is long enough to sample throughout.
        "pyramid_stairs": (0.0, 0.9),
        "pyramid_stairs_high": (0.0, 0.9),
        "pyramid_stairs_inv": (0.0, 0.9),
        "pyramid_stairs_inv_high": (0.0, 0.9),
        "dual_pyramid_course": (0.0, 0.0),
    }
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    symmetric_augmentation_link_mapping=[0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12],
    symmetric_augmentation_joint_mapping=G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    symmetric_augmentation_joint_reverse_buf=G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={
        "run_walk": AmassMotionCfg(),
    },
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    mp_split_method="None",
)


ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG_ONLYSTAIRS)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class G1ParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_ONLYSTAIRS
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS))
        self.scene.motion_reference = motion_reference_cfg


class ShoeConfigMixin:
    def apply_shoe_config(self):
        self.scene.robot = G1_with_shoe_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.leg_volume_points.points_generator.z_min = -0.063
        self.scene.leg_volume_points.points_generator.z_max = -0.023
        self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.058
        self.rewards.rewards.feet_support_deficit.params["height_offset"] = 0.058


@configclass
class G1ParkourRoughEnvCfg_PLAY(G1ParkourRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = PLAY_DUAL_PYRAMID_TERRAINS_CFG
        # The play course is a single deterministic environment.
        self.scene.num_envs = 1
        self.viewer = ViewerCfg(
            eye=[4.0, 0.75, 1.0],
            lookat=[0.0, 0.75, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )

        self.scene.env_spacing = 2.5
        self.episode_length_s = 100
        self.terminations.root_height = None
        # Keep a single course so env 0 cannot be assigned to a flat terrain column.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1

        self.scene.leg_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (100.0, 100.0)
        self.commands.base_velocity.random_velocity_terrain = []
        self.commands.base_velocity.velocity_ranges = {
            "dual_pyramid_course": {
                "lin_vel_x": (PLAY_DUAL_PYRAMID_SPEED, PLAY_DUAL_PYRAMID_SPEED),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
        }

        self.events.physics_material = None
        self.events.reset_base.params["pose_range"] = {
            "x": (PLAY_DUAL_PYRAMID_SPAWN_X, PLAY_DUAL_PYRAMID_SPAWN_X),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class G1ParkourEnvCfg(G1ParkourRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()


@configclass
class G1ParkourEnvCfg_PLAY(G1ParkourRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
