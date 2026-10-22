"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Always pair this repository's play script with this repository's InstinctLab
# package. An older PYTHONPATH/editable install can otherwise mix code versions.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTINCTLAB_SOURCE_DIR = str(PROJECT_ROOT / "source" / "instinctlab")
if INSTINCTLAB_SOURCE_DIR in sys.path:
    sys.path.remove(INSTINCTLAB_SOURCE_DIR)
sys.path.insert(0, INSTINCTLAB_SOURCE_DIR)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--show_depth", action="store_true",
    help="Overlay the policy's latest depth frame in --video (headless supported).",
)
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--stairs_demo",
    action="store_true",
    help="Run the ordinary Parkour policy on the wall-free up/down-stairs demo course.",
)
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument(
    "--no_terminate", action="store_true", default=False, help="Do not remove termination conditions in simulation."
)
parser.add_argument(
    "--aux_reward",
    action="store_true",
    default=False,
    help="Whether to assign auxiliary rewards to each of the env's reward term.",
)
# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.show_depth and not args_cli.video:
    parser.error("--show_depth requires --video; depth is displayed in the saved video.")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from instinct_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab
import instinctlab.tasks  # noqa: F401
# Import Parkour registrations explicitly. This keeps repository-local demo tasks
# available even when the generic task auto-discovery has already been cached.
import instinctlab.tasks.parkour.config.g1  # noqa: F401
import instinctlab.terrains as terrain_gen
from instinctlab.managers.reward_manager import MultiRewardManager
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

STAIRS_DEMO_TASK_ID = "Instinct-Parkour-Target-Amp-G1-Stairs-Play-v0"

print(f"[INFO] InstinctLab source: {Path(instinctlab.__file__).resolve()}")
print(f"[INFO] InstinctLab terrains source: {Path(terrain_gen.__file__).resolve()}")

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


class PolicyDepthVideoWrapper(gym.Wrapper):
    """Overlay the depth used for the current action, without changing observations."""

    def __init__(self, env):
        super().__init__(env)
        import cv2

        self.cv2 = cv2
        self.depth_frame = None

    def render(self):
        frame = self.env.render()
        if frame is None or self.depth_frame is None:
            return frame
        frame = frame.copy()
        height, width = frame.shape[:2]
        depth = self.depth_frame
        panel_width = max(1, min(384, width // 3, int(max(1, height - 48) * depth.shape[1] / depth.shape[0])))
        panel_height = max(1, round(panel_width * depth.shape[0] / depth.shape[1]))
        panel = self.cv2.resize(depth, (panel_width, panel_height), interpolation=self.cv2.INTER_NEAREST)
        left, top = width - panel_width - 12, 36
        frame[top:top + panel_height, left:left + panel_width, :3] = panel[..., None]
        self.cv2.putText(
            frame, "Policy depth (action input)", (left, top - 10),
            self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        return frame


def main():
    """Play with Instinct-RL agent."""
    stairs_demo = args_cli.stairs_demo or args_cli.task == STAIRS_DEMO_TASK_ID
    runtime_task = "Instinct-Parkour-Target-Amp-G1-Play-v0" if stairs_demo else args_cli.task

    # parse configuration
    env_cfg = parse_env_cfg(
        runtime_task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(runtime_task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        env_cfg = load_yaml(os.path.join(log_dir, "params", "env.yaml"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    if stairs_demo:
        # Mutate the already parsed ordinary Parkour PLAY config. This preserves
        # its cropped 32x18 depth input, AMP layout, robot, actions and policy interface.
        stairs_spawn_x = -10.5
        stairs_target_x = 10.5
        stairs_speed = 0.65
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=0,
            size=(24.0, 8.0),
            border_width=0.5,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            slope_threshold=1.0,
            use_cache=False,
            curriculum=False,
            sub_terrains={
                "dual_pyramid_course": terrain_gen.DualPyramidStairsTerrainCfg(
                    proportion=1.0,
                    step_height=0.10,
                    step_width=0.40,
                    platform_width=1.60,
                    feature_width=6.80,
                    inverted_center_x=-5.0,
                    pyramid_center_x=5.0,
                    border_width=0.5,
                    wall_prob=[0.0, 0.0, 0.0, 0.0],
                    wall_height=5.0,
                    wall_thickness=0.05,
                    flat_patch_sampling={
                        "target": FlatPatchSamplingCfg(
                            num_patches=50,
                            patch_radius=[0.05, 0.10, 0.15, 0.20],
                            max_height_diff=0.05,
                            x_range=(stairs_target_x, stairs_target_x),
                            y_range=(0.0, 0.0),
                        ),
                    },
                ),
            },
        )
        env_cfg.scene.num_envs = 1
        env_cfg.scene.terrain.max_init_terrain_level = 0
        env_cfg.episode_length_s = 100.0
        env_cfg.terminations.terrain_out_bound.params["distance_buffer"] = 1.0
        env_cfg.scene.leg_volume_points.debug_vis = False

        command = env_cfg.commands.base_velocity
        command.debug_vis = False
        command.rel_standing_envs = 0.0
        command.resampling_time_range = (100.0, 100.0)
        command.random_velocity_terrain = []
        command.velocity_ranges = {
            "dual_pyramid_course": {
                "lin_vel_x": (stairs_speed, stairs_speed),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
        }
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (stairs_spawn_x, stairs_spawn_x),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.0, 3.0, 0.5)
        env_cfg.viewer.lookat = (0.0, 0.0, -0.3)
        env_cfg.viewer.resolution = (1280, 720)
        print("[INFO] Ordinary Parkour policy: wall-free up/down-stairs demonstration mode.")

    # Keep inference on the same device as the environment, including --device cuda:N.
    agent_cfg.device = env_cfg.sim.device

    # create isaac environment
    env = gym.make(runtime_task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    depth_video = None
    if args_cli.show_depth:
        depth_video = PolicyDepthVideoWrapper(env)
        env = depth_video
    # wrap for video recording
    if args_cli.video:
        video_prefix = f"model_{resume_path.split('_')[-1].split('.')[0]}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": video_prefix,
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # react to custom play arguments
    if args_cli.no_terminate:
        # NOTE: This is only applicable with shadowing task
        env.unwrapped.termination_manager._term_cfgs = [
            env.unwrapped.termination_manager._term_cfgs[
                env.unwrapped.termination_manager._term_names.index("dataset_exhausted")
            ]
        ]
        env.unwrapped.termination_manager._term_names = ["dataset_exhausted"]

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)
    if depth_video is not None:
        from instinct_rl.utils.utils import get_obs_slice

        if "depth_image" not in env.get_obs_segments():
            raise ValueError("--show_depth requires a depth_image policy observation.")
        depth_slice, depth_shape = get_obs_slice(env.get_obs_segments(), "depth_image")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.export_as_onnx(obs, export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            # env stepping
            if depth_video is not None:
                depth = obs[0, depth_slice].reshape(depth_shape)[-1]
                # Fixed normalized scale: near=black, far=white. No per-frame rescaling.
                depth_video.depth_frame = (
                    torch.nan_to_num(depth.detach(), nan=1.0, posinf=1.0, neginf=0.0)
                    .clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
                )
            obs, rewards, dones, infos = env.step(actions)
        timestep += 1

        # override reward terms if auxiliary reward is enabled
        if args_cli.aux_reward:
            # NOTE: This is only applicable when reward_term has `.reward` to be overridden
            aux_rewards = ppo_runner.alg.compute_auxiliary_reward(infos["observations"])
            for aux_reward_name, aux_reward in aux_rewards.items():
                aux_term_cfg = env.unwrapped.reward_manager.get_term_cfg(aux_reward_name)  # type: ignore
                aux_term_cfg.func.reward[:] = aux_reward * getattr(ppo_runner.alg, aux_reward_name + "_coef", 1.0)

        # exit the loop if video_length is meet
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_start_step + args_cli.video_length:
                break

    # close the simulator
    env.close()

    if args_cli.video:
        video_path = os.path.join(log_dir, "videos", "play", f"{video_prefix}-step-{args_cli.video_start_step}.mp4")
        print(f"[INFO] Video saved to: {video_path}")
        if not args_cli.headless:
            subprocess.run(["code", "-r", video_path])


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
