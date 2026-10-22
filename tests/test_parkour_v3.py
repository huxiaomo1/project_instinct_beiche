"""CPU regressions for V3; run with ``python tests/test_parkour_v3.py``.

Load real functions/configuration expressions without importing Isaac Sim.
These checks do not replace a simulator rollout or a trained-policy evaluation.
"""

from __future__ import annotations

import ast
import copy
import math
import numpy as np
import torch
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PARKOUR = ROOT / "source/instinctlab/instinctlab/tasks/parkour"


def entity(name="robot", **kwargs):
    return SimpleNamespace(name=name, joint_ids=kwargs.pop("joint_ids", slice(None)), **kwargs)


def execute(nodes, namespace, filename):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(filename), "exec"), namespace)
    return namespace


REWARDS_PATH = PARKOUR / "mdp/rewards.py"
REWARD_NAMES = {"_standing_command_mask", "stand_joint_vel_l2", "stand_lin_vel_xy_l2", "stand_ang_vel_l2"}
REWARDS = execute(
    [
        node
        for node in ast.parse(REWARDS_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in REWARD_NAMES
    ],
    {"torch": torch, "SceneEntityCfg": entity},
    REWARDS_PATH,
)


class ConfigFactory:
    """Retain configuration keyword arguments without loading simulator types."""

    def __call__(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self


CONFIG_PATH = PARKOUR / "config/parkour_env_cfg.py"
CONFIG_TREE = ast.parse(CONFIG_PATH.read_text())
FACTORY = ConfigFactory()
CONFIG = execute(
    [
        node
        for node in CONFIG_TREE.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ROUGH_TERRAINS_CFG" for t in node.targets)
        )
        or (isinstance(node, ast.ClassDef) and node.name in {"CommandsCfg", "EventCfg", "G1Rewards"})
    ],
    {
        "math": math,
        "terrain_gen": FACTORY,
        "TerrainGeneratorCfg": FACTORY,
        "FlatPatchSamplingCfg": FACTORY,
        "mdp": FACTORY,
        "instinct_mdp": FACTORY,
        "EventTerm": FACTORY,
        "RewTerm": FACTORY,
        "SceneEntityCfg": entity,
        "configclass": lambda cls: cls,
    },
    CONFIG_PATH,
)


def config_attributes(cls):
    return {name: value for name, value in vars(cls).items() if not name.startswith("_")}


PLAY_TERRAIN = copy.deepcopy(CONFIG["ROUGH_TERRAINS_CFG"])
PLAY_TERRAIN.num_rows = 4
PLAY_TERRAIN.num_cols = 10
for subterrain in PLAY_TERRAIN.sub_terrains.values():
    subterrain.wall_prob = [0.0, 0.0, 0.0, 0.0]
BASE_REWARDS = SimpleNamespace(**config_attributes(CONFIG["G1Rewards"]))
BASE_REWARDS.feet_at_plane.params["height_offset"] = 0.058
BASE_CAMERA = SimpleNamespace(
    noise_pipeline={"crop_and_resize": SimpleNamespace(crop_region=(18, 0, 16, 16), resize_shape=None)}
)


class OrdinaryEnvFixture:
    """Shared templates deliberately expose mutations that could leak into ordinary tasks."""

    def __init__(self):
        self.scene = SimpleNamespace(
            terrain=SimpleNamespace(terrain_generator=CONFIG["ROUGH_TERRAINS_CFG"]), camera=BASE_CAMERA
        )
        self.commands = SimpleNamespace(base_velocity=CONFIG["CommandsCfg"].base_velocity)
        self.rewards = SimpleNamespace(rewards=BASE_REWARDS)
        self.events = SimpleNamespace(**config_attributes(CONFIG["EventCfg"]))
        self.__post_init__()

    def __post_init__(self):
        pass


class OrdinaryPlayFixture(OrdinaryEnvFixture):
    def __post_init__(self):
        self.scene.terrain.terrain_generator = PLAY_TERRAIN


V3_PATH = PARKOUR / "config/g1/g1_parkour_stand_stairs_v3_cfg.py"
V3_CONFIG = execute(
    [
        node
        for node in ast.parse(V3_PATH.read_text()).body
        if isinstance(node, (ast.Assign, ast.FunctionDef, ast.ClassDef))
    ],
    {
        "copy": copy,
        "math": math,
        "mdp": FACTORY,
        "SceneEntityCfg": entity,
        "RewTerm": FACTORY,
        "EventTerm": FACTORY,
        "configclass": lambda cls: cls,
        "G1ParkourEnvCfg": OrdinaryEnvFixture,
        "G1ParkourEnvCfg_PLAY": OrdinaryPlayFixture,
    },
    V3_PATH,
)


class StandingRewardTests(unittest.TestCase):
    def make_env(self, commands):
        data = SimpleNamespace(
            joint_vel=torch.tensor([[1.0, -2.0, 0.5]]).repeat(len(commands), 1),
            root_lin_vel_b=torch.tensor([[0.3, -0.4, 10.0]]).repeat(len(commands), 1),
            root_ang_vel_b=torch.tensor([[0.1, -0.2, 0.3]]).repeat(len(commands), 1),
        )
        return SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(get_command=lambda name: torch.tensor(commands)),
        )

    def test_zero_command_penalizes_motion_and_does_not_gate_on_actual_drift(self):
        env = self.make_env([[0.0, 0.0, 0.0]])
        for name, expected in (("stand_joint_vel_l2", 5.25), ("stand_lin_vel_xy_l2", 0.25), ("stand_ang_vel_l2", 0.14)):
            self.assertAlmostEqual(REWARDS[name](env, "base_velocity").item(), expected, places=6)

    def test_forward_reverse_sideways_and_pure_turn_are_exempt(self):
        env = self.make_env([[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
        for name in REWARD_NAMES - {"_standing_command_mask"}:
            self.assertTrue(torch.equal(REWARDS[name](env, "base_velocity"), torch.zeros(4)))

    def test_command_deadzone_uses_xy_norm_and_yaw(self):
        env = self.make_env([[0.03, 0.03, 0.04], [0.04, 0.04, 0.0], [0.0, 0.0, 0.05]])
        mask = REWARDS["_standing_command_mask"](env, "base_velocity", 0.05)
        self.assertEqual(mask.tolist(), [True, False, False])

    def test_joint_selection_and_quiet_state(self):
        env = self.make_env([[0.0, 0.0, 0.0]])
        reward = REWARDS["stand_joint_vel_l2"](env, "base_velocity", entity(joint_ids=[0, 2]))
        self.assertAlmostEqual(reward.item(), 1.25)
        for value in vars(env.scene["robot"].data).values():
            value.zero_()
        for name in REWARD_NAMES - {"_standing_command_mask"}:
            self.assertEqual(REWARDS[name](env, "base_velocity").item(), 0.0)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.env = V3_CONFIG["G1ParkourStandStairsV3EnvCfg"]()

    def test_real_column_resolver_assigns_six_six_six_two(self):
        path = ROOT / "source/instinctlab/instinctlab/terrains/utils.py"
        namespace = {"np": np}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        terrain = self.env.scene.terrain.terrain_generator
        counts = Counter(namespace["resolve_subterrain_names_by_column"](terrain))
        self.assertEqual(
            counts, {"perlin_rough_stand": 6, "pyramid_stairs": 6, "pyramid_stairs_inv": 6, "perlin_rough": 2}
        )
        self.assertEqual(terrain.sub_terrains["perlin_rough_stand"].noise_scale, 0.0)
        self.assertEqual(set(terrain.sub_terrains), set(self.env.commands.base_velocity.velocity_ranges))

    def test_camera_ranges_are_radians_and_startup(self):
        event = self.env.events.randomize_camera_offset
        self.assertEqual(event.mode, "startup")
        self.assertEqual(event.params["asset_cfg"].name, "camera")
        for axis, degrees in (("roll", 2.0), ("pitch", 5.0), ("yaw", 2.0)):
            self.assertEqual(event.params["offset_pose_ranges"][axis], (-math.radians(degrees), math.radians(degrees)))

    def test_actual_command_update_keeps_stand_zero_despite_off_axis_target(self):
        path = PARKOUR / "mdp/commands/pose_velocity_command.py"
        cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_update_command")
        namespace = execute(
            [method],
            {
                "torch": torch,
                "yaw_quat": lambda quat: quat,
                # All test roots have identity orientation.
                "quat_apply_inverse": lambda quat, vector: vector,
                "wrap_to_pi": lambda angle: (angle + math.pi) % (2 * math.pi) - math.pi,
            },
            path,
        )
        cfg = self.env.commands.base_velocity
        names = list(cfg.velocity_ranges)
        n = len(names)
        command = SimpleNamespace(
            cfg=cfg,
            robot=SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=torch.zeros(n, 3),
                    root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
                    heading_w=torch.zeros(n),
                )
            ),
            pos_command_w=torch.tensor([[2.0, 1.0, 0.0]]).repeat(n, 1),
            pos_command_b=torch.zeros(n, 3),
            vel_command_b=torch.zeros(n, 3),
            max_command_b=torch.tensor(
                [
                    [cfg.velocity_ranges[name][axis][1] for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
                    for name in names
                ]
            ),
            ang_vel_z_range=torch.tensor([cfg.velocity_ranges[name]["ang_vel_z"] for name in names]),
            is_standing_env=torch.zeros(n, dtype=torch.bool),
            random_velocity_indices=torch.tensor([name in cfg.random_velocity_terrain for name in names]),
            random_lin_vel_x=torch.ones(n),
            random_lin_vel_y=torch.ones(n),
            random_ang_vel_z=torch.ones(n),
        )
        namespace["_update_command"](command)
        for index, name in enumerate(names):
            if name == "perlin_rough_stand":
                self.assertTrue(torch.equal(command.vel_command_b[index], torch.zeros(3)))
            else:
                self.assertGreater(command.vel_command_b[index, 0].item(), 0.0)
                self.assertGreater(command.vel_command_b[index, 2].item(), 0.0)
        command.is_standing_env[:] = True
        namespace["_update_command"](command)
        self.assertTrue(torch.equal(command.vel_command_b, torch.zeros(n, 3)))


class IsolationTests(unittest.TestCase):
    def test_constructing_v3_train_and_play_preserves_original_task(self):
        before = OrdinaryEnvFixture()
        V3_CONFIG["G1ParkourStandStairsV3EnvCfg"]()
        V3_CONFIG["G1ParkourStandStairsV3EnvCfg_PLAY"]()
        after = OrdinaryEnvFixture()
        for env in (before, after):
            terrain = env.scene.terrain.terrain_generator
            self.assertEqual(len(terrain.sub_terrains), 10)
            self.assertIn("square_gaps", terrain.sub_terrains)
            self.assertEqual(terrain.sub_terrains["perlin_rough_stand"].proportion, 0.05)
            self.assertEqual(terrain.sub_terrains["perlin_rough_stand"].noise_scale, [0.0, 0.1])
            self.assertEqual(env.commands.base_velocity.random_velocity_terrain, ["perlin_rough_stand"])
            self.assertFalse(hasattr(env.events, "randomize_camera_offset"))
            self.assertFalse(hasattr(env.rewards.rewards, "stand_joint_vel_l2"))

    def test_v3_instances_have_independent_nested_configuration(self):
        first = V3_CONFIG["G1ParkourStandStairsV3EnvCfg"]()
        second = V3_CONFIG["G1ParkourStandStairsV3EnvCfg"]()
        first.commands.base_velocity.velocity_ranges["perlin_rough_stand"]["lin_vel_x"] = (1.0, 1.0)
        first.rewards.rewards.stand_joint_vel_l2.weight = -999.0
        first.events.randomize_camera_offset.params["offset_pose_ranges"]["pitch"] = (-1.0, 1.0)
        first.scene.terrain.terrain_generator.sub_terrains["perlin_rough_stand"].proportion = 1.0
        first.scene.camera.noise_pipeline["crop_and_resize"].crop_region = (0, 0, 0, 0)
        self.assertEqual(second.commands.base_velocity.velocity_ranges["perlin_rough_stand"]["lin_vel_x"], (0.0, 0.0))
        self.assertEqual(second.rewards.rewards.stand_joint_vel_l2.weight, -0.5)
        self.assertEqual(
            second.events.randomize_camera_offset.params["offset_pose_ranges"]["pitch"][1], math.radians(5)
        )
        self.assertEqual(second.scene.terrain.terrain_generator.sub_terrains["perlin_rough_stand"].proportion, 0.3)
        self.assertEqual(second.scene.camera.noise_pipeline["crop_and_resize"].crop_region, (18, 0, 16, 16))
        self.assertEqual(BASE_CAMERA.noise_pipeline["crop_and_resize"].crop_region, (18, 0, 16, 16))

    def test_v3_play_disables_camera_randomization_and_preserves_shoe_adjustments(self):
        play = V3_CONFIG["G1ParkourStandStairsV3EnvCfg_PLAY"]()
        self.assertIsNone(play.events.randomize_camera_offset)
        self.assertEqual(play.scene.terrain.terrain_generator.num_cols, 10)
        self.assertEqual(play.scene.terrain.terrain_generator.num_rows, 4)
        self.assertTrue(
            all(
                terrain.wall_prob == [0.0] * 4 for terrain in play.scene.terrain.terrain_generator.sub_terrains.values()
            )
        )
        self.assertEqual(play.rewards.rewards.feet_at_plane.params["height_offset"], 0.058)
        self.assertEqual(play.rewards.rewards.stand_still.weight, -0.3)

    def test_ordinary_and_v3_registry_entries_are_separate(self):
        path = PARKOUR / "config/g1/__init__.py"
        registry = {}
        execute(
            [node for node in ast.parse(path.read_text()).body if not isinstance(node, (ast.Import, ast.ImportFrom))],
            {
                "gym": SimpleNamespace(register=lambda id, **kwargs: registry.update({id: kwargs})),
                "agents": SimpleNamespace(__name__="test_agents"),
            },
            path,
        )
        ordinary = registry["Instinct-Parkour-Target-Amp-G1-v0"]
        v3 = registry["Instinct-Parkour-Stand-Stairs-Finetune-G1-v3"]
        play = registry["Instinct-Parkour-Stand-Stairs-Finetune-G1-Play-v3"]
        self.assertIn("g1_parkour_target_amp_cfg:G1ParkourEnvCfg", ordinary["kwargs"]["env_cfg_entry_point"])
        self.assertIn(
            "g1_parkour_stand_stairs_v3_cfg:G1ParkourStandStairsV3EnvCfg", v3["kwargs"]["env_cfg_entry_point"]
        )
        self.assertTrue(play["kwargs"]["env_cfg_entry_point"].endswith("_PLAY"))
        self.assertEqual(ordinary["entry_point"], v3["entry_point"])
        self.assertNotEqual(
            ordinary["kwargs"]["instinct_rl_cfg_entry_point"], v3["kwargs"]["instinct_rl_cfg_entry_point"]
        )

    def test_v3_runner_uses_original_algorithm_with_independent_defaults(self):
        class Defaults:
            def __init__(self, **kwargs):
                for base in reversed(type(self).__mro__):
                    for name, value in vars(base).items():
                        if not name.startswith("_") and not callable(value):
                            setattr(self, name, copy.deepcopy(value))
                self.__dict__.update(kwargs)

        path = PARKOUR / "config/g1/agents/instinct_rl_amp_cfg.py"
        namespace = execute(
            [
                node
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef)
                and node.name in {"AmpAlgoCfg", "G1ParkourPPORunnerCfg", "G1ParkourStandStairsV3RunnerCfg"}
            ],
            {
                "configclass": lambda cls: cls,
                "InstinctRlPpoAlgorithmCfg": Defaults,
                "InstinctRlOnPolicyRunnerCfg": Defaults,
                "MoEPolicyCfg": FACTORY,
            },
            path,
        )
        ordinary = namespace["G1ParkourPPORunnerCfg"]()
        v3 = namespace["G1ParkourStandStairsV3RunnerCfg"]()
        self.assertEqual(ordinary.algorithm.schedule, "adaptive")
        self.assertEqual(ordinary.algorithm.learning_rate, 1e-3)
        self.assertEqual(v3.algorithm.schedule, "fixed")
        self.assertEqual(v3.algorithm.learning_rate, 5e-5)
        self.assertEqual(v3.experiment_name, "g1_parkour_finetune_v3")
        self.assertEqual(v3.save_interval, 1000)
        self.assertEqual(v3.max_iterations, 50000)
        self.assertIsNot(ordinary.algorithm, v3.algorithm)
        self.assertEqual(v3.algorithm.class_name, "WasabiPPO")
        self.assertEqual(v3.algorithm.discriminator_reward_coef, 0.25)
        self.assertEqual(v3.algorithm.clip_param, ordinary.algorithm.clip_param)
        self.assertEqual(v3.algorithm.entropy_coef, ordinary.algorithm.entropy_coef)


class ResumeTests(unittest.TestCase):
    def test_fixed_rate_overrides_loaded_optimizer_rate_and_preserves_momentum(self):
        path = ROOT / "scripts/instinct_rl/train.py"
        source = path.read_text()
        statement = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.If)
            and ast.get_source_segment(source, node.test) == 'agent_cfg.algorithm.schedule == "fixed"'
        )
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        parameter.square().sum().backward()
        optimizer.step()
        state = optimizer.state_dict()
        optimizer.param_groups[0]["lr"] = 5e-5
        optimizer.load_state_dict(state)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        momentum = optimizer.state[parameter]["exp_avg"].clone()
        runner = SimpleNamespace(alg=SimpleNamespace(optimizer=optimizer, learning_rate=1e-3))
        for schedule, expected in (("fixed", 5e-5), ("adaptive", 1e-3)):
            optimizer.param_groups[0]["lr"] = 1e-3
            execute(
                [statement],
                {
                    "agent_cfg": SimpleNamespace(algorithm=SimpleNamespace(schedule=schedule, learning_rate=5e-5)),
                    "runner": runner,
                },
                path,
            )
            self.assertEqual(optimizer.param_groups[0]["lr"], expected)
            self.assertTrue(torch.equal(optimizer.state[parameter]["exp_avg"], momentum))


if __name__ == "__main__":
    unittest.main()
