"""Tensor-only stopping controls, shared by training, tests and deployment.

Reward functions return dimensionless values in [0, 1]. They do not multiply
by the simulation/control period. Only the command limiter uses ``dt``.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class StandControlCfg:
    acceleration_limits: tuple[float, float, float] = (1.5, 1.5, 3.0)
    stop_linear_speed: float = 0.03
    move_linear_speed: float = 0.30
    stop_yaw_speed: float = 0.05
    move_yaw_speed: float = 0.50
    settle_linear_speed: float = 0.08
    settle_yaw_speed: float = 0.15
    brake_velocity_std: float = 0.5
    upright_std: float = 0.3
    joint_position_std: float = 0.25
    joint_velocity_std: float = 0.8
    stationary_velocity_std: float = 0.08
    stationary_angular_std: float = 0.25
    contact_force: float = 5.0
    stair_height_std: float = 0.08
    minimum_leg_pose_weight: float = 0.15
    brake_weights: tuple[float, float] = (0.7, 0.3)
    stand_weights: tuple[float, float, float, float] = (0.25, 0.20, 0.40, 0.15)

    def __post_init__(self):
        for weights, length in ((self.brake_weights, 2), (self.stand_weights, 4)):
            if len(weights) != length or min(weights) < 0 or abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Standing/braking reward weights must be nonnegative and sum to one")
        for name in (
            "settle_linear_speed",
            "settle_yaw_speed",
            "brake_velocity_std",
            "upright_std",
            "joint_position_std",
            "joint_velocity_std",
            "stationary_velocity_std",
            "stationary_angular_std",
            "contact_force",
            "stair_height_std",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.minimum_leg_pose_weight <= 1:
            raise ValueError("minimum_leg_pose_weight must be in [0, 1]")
        if self.move_linear_speed <= self.stop_linear_speed or self.move_yaw_speed <= self.stop_yaw_speed:
            raise ValueError("Motion command thresholds must exceed stopping thresholds")


def smoothstep(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if upper <= lower:
        raise ValueError("smoothstep requires upper > lower")
    t = ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
    return t.square() * (3.0 - 2.0 * t)


def limit_velocity_command(previous, target, dt: float, acceleration_limits):
    """Limit each command axis; use this same function before deployment inference."""
    if dt <= 0 or len(acceleration_limits) != 3 or min(acceleration_limits) <= 0:
        raise ValueError("dt and all three acceleration limits must be positive")
    step = previous.new_tensor(acceleration_limits) * dt
    return previous + torch.maximum(torch.minimum(target - previous, step), -step)


def stopping_rewards(
    command,
    linear_velocity,
    angular_velocity,
    projected_gravity,
    joint_position_error,
    joint_velocity,
    foot_forces,
    foot_heights,
    leg_mask,
    cfg: StandControlCfg,
):
    """Compute braking/standing rewards from one transition, without hidden timers.

    The AMP mix depends ONLY on the filtered command. Contact and body speed can
    advance posture settling, but cannot turn AMP back on when the robot drifts.
    Once the command reaches zero, settling is fully active even without double
    support: the deceleration interval is the bounded landing grace period.
    """
    linear_command = torch.linalg.vector_norm(command[:, :2], dim=-1)
    yaw_command = command[:, 2].abs()
    motion = torch.maximum(
        smoothstep(linear_command, cfg.stop_linear_speed, cfg.move_linear_speed),
        smoothstep(yaw_command, cfg.stop_yaw_speed, cfg.move_yaw_speed),
    )
    stop_mix = 1.0 - motion
    settle = 1.0 - torch.maximum(
        smoothstep(linear_command, 0.0, cfg.settle_linear_speed),
        smoothstep(yaw_command, 0.0, cfg.settle_yaw_speed),
    )
    support = (foot_forces / cfg.contact_force).clamp(0.0, 1.0).prod(dim=-1)
    quiet_body = torch.exp(-linear_velocity[:, :2].square().sum(-1) / 0.15**2)
    settle = settle + (1.0 - settle) * stop_mix * support * quiet_body * 0.5

    # Feet on different steps need different leg poses. Arms/waist keep their
    # nominal target; relax the legs instead of demanding a flat-floor pose.
    height_difference = (foot_heights[:, 0] - foot_heights[:, 1]).abs()
    leg_weight = cfg.minimum_leg_pose_weight + (1.0 - cfg.minimum_leg_pose_weight) * torch.exp(
        -(height_difference / cfg.stair_height_std).square()
    )
    pose_weights = torch.where(leg_mask.unsqueeze(0), leg_weight.unsqueeze(-1), torch.ones_like(joint_position_error))
    pose_error = (pose_weights * (joint_position_error / cfg.joint_position_std).square()).mean(-1)
    pose = torch.exp(-pose_error)
    joints_quiet = torch.exp(-(joint_velocity / cfg.joint_velocity_std).square().mean(-1))
    body_quiet = torch.exp(-linear_velocity.square().sum(-1) / cfg.stationary_velocity_std**2)
    angular_quiet = torch.exp(-angular_velocity.square().sum(-1) / cfg.stationary_angular_std**2)
    upright = torch.exp(-projected_gravity[:, :2].square().sum(-1) / cfg.upright_std**2)
    # Exclude upside-down poses, which can also have zero projected x/y gravity.
    upright = upright * (-projected_gravity[:, 2]).clamp(0.0, 1.0)
    tracking_error = (linear_velocity[:, :2] - command[:, :2]).square().sum(-1)
    tracking_error = tracking_error + (angular_velocity[:, 2] - command[:, 2]).square()
    tracking = torch.exp(-tracking_error / cfg.brake_velocity_std**2)
    brake = cfg.brake_weights[0] * tracking + cfg.brake_weights[1] * upright
    stand = upright * (
        cfg.stand_weights[0] * pose
        + cfg.stand_weights[1] * joints_quiet
        + cfg.stand_weights[2] * body_quiet
        + cfg.stand_weights[3] * angular_quiet
    )
    replacement = ((1.0 - settle) * brake + settle * stand).clamp(0.0, 1.0)
    return {
        "stop_mix": stop_mix,
        "settle_mix": settle,
        "brake": brake,
        "stand": stand,
        "replacement": replacement,
        "pose": pose,
        "joints_quiet": joints_quiet,
        "body_quiet": body_quiet,
        "angular_quiet": angular_quiet,
        "upright": upright,
        "double_support": support,
        "linear_speed": torch.linalg.vector_norm(linear_velocity[:, :2], dim=-1),
    }


def blend_style_reward(amp, replacement, stop_mix, progress: float):
    """Convex blend before the ONE shared AMP coefficient; never apply dt here."""
    mix = stop_mix * min(max(progress, 0.0), 1.0)
    return (1.0 - mix) * amp + mix * replacement
