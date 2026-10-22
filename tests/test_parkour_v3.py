"""CPU regressions for V3; run with ``python tests/test_parkour_v3.py``.

Load real functions/configuration expressions without importing Isaac Sim.
These checks do not replace a simulator rollout or a trained-policy evaluation.
"""

from __future__ import annotations

import ast
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
        or (isinstance(node, ast.ClassDef) and node.name in {"CommandsCfg", "EventCfg"})
    ],
    {
        "math": math,
        "terrain_gen": FACTORY,
        "TerrainGeneratorCfg": FACTORY,
        "FlatPatchSamplingCfg": FACTORY,
        "mdp": FACTORY,
        "instinct_mdp": FACTORY,
        "EventTerm": FACTORY,
        "SceneEntityCfg": entity,
        "configclass": lambda cls: cls,
    },
    CONFIG_PATH,
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
    def test_real_column_resolver_assigns_six_six_six_two(self):
        path = ROOT / "source/instinctlab/instinctlab/terrains/utils.py"
        namespace = {"np": np}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        terrain = CONFIG["ROUGH_TERRAINS_CFG"]
        counts = Counter(namespace["resolve_subterrain_names_by_column"](terrain))
        self.assertEqual(
            counts, {"perlin_rough_stand": 6, "pyramid_stairs": 6, "pyramid_stairs_inv": 6, "perlin_rough": 2}
        )
        self.assertEqual(terrain.sub_terrains["perlin_rough_stand"].noise_scale, 0.0)
        self.assertEqual(set(terrain.sub_terrains), set(CONFIG["CommandsCfg"].base_velocity.velocity_ranges))

    def test_camera_ranges_are_radians_and_startup(self):
        event = CONFIG["EventCfg"].randomize_camera_offset
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
        cfg = CONFIG["CommandsCfg"].base_velocity
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
