#!/usr/bin/env python3
"""Split a retargeted G1 motion NPZ into labeled continuous clips.

The detector uses four signals:

1. root-position discontinuities (hard clip boundaries),
2. a smoothed root-height trend (upstairs/downstairs),
3. horizontal root speed (standing/locomotion), and
4. hip/knee joint motion (standing/locomotion and hard jumps).

Automatic labels are deliberately conservative. Review ``segments.csv`` and
the rendered clips before using the generated YAML for AMP training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path("/home/dhy/Datasets/parkour_motion_without_run_retargetted.npz")
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset" / "division_npz"
LABELS = ("stand", "walk", "stairs_up", "stairs_down", "unknown")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
DEFAULT_CLASS_WEIGHTS = "stand=0.25,walk=0.25,stairs_up=0.20,stairs_down=0.30,unknown=0.0"


@dataclass(frozen=True)
class DetectorConfig:
    jump_xy_m: float
    jump_z_m: float
    joint_jump_rad: float
    smooth_window_s: float
    height_trend_window_s: float
    label_filter_s: float
    min_state_s: float
    min_output_s: float
    min_stairs_output_s: float
    max_output_s: float
    stand_speed_mps: float
    locomotion_speed_mps: float
    stand_joint_speed_radps: float
    stairs_height_rate_mps: float
    stairs_min_net_height_m: float


@dataclass
class MotionData:
    framerate: float
    joint_names: np.ndarray
    joint_pos: np.ndarray
    base_pos_w: np.ndarray
    base_quat_w: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


@dataclass(frozen=True)
class HardBoundary:
    frame: int
    root_jump_xy_m: float
    root_jump_z_m: float
    max_hip_knee_jump_rad: float
    reasons: str


@dataclass
class Segment:
    index: int
    label: str
    start_frame: int
    end_frame: int
    boundary_reason: str
    duration_s: float
    confidence: float
    mean_speed_xy_mps: float
    mean_abs_height_rate_mps: float
    mean_height_rate_mps: float
    net_height_change_m: float
    mean_hip_knee_speed_radps: float
    exported: bool = False
    output_file: str = ""


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value}")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a retargeted G1 NPZ using root jumps, smoothed height trend, "
            "horizontal speed, and hip/knee motion."
        )
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT, help="Input retargeted NPZ file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for clips and reports.")
    parser.add_argument("--jump-xy", type=positive_float, default=0.25, help="Hard XY jump threshold in m/frame.")
    parser.add_argument("--jump-z", type=positive_float, default=0.15, help="Hard Z jump threshold in m/frame.")
    parser.add_argument(
        "--joint-jump",
        type=positive_float,
        default=0.75,
        help="Hard maximum hip/knee position jump threshold in rad/frame.",
    )
    parser.add_argument("--smooth-window-s", type=positive_float, default=0.20, help="Feature smoothing window.")
    parser.add_argument(
        "--height-trend-window-s", type=positive_float, default=1.00, help="Window used to estimate root-height slope."
    )
    parser.add_argument("--label-filter-s", type=nonnegative_float, default=0.30, help="Majority filter for frame labels.")
    parser.add_argument("--min-state-s", type=nonnegative_float, default=0.40, help="Merge shorter state runs.")
    parser.add_argument("--min-output-s", type=positive_float, default=0.30, help="Do not export shorter clips.")
    parser.add_argument(
        "--min-stairs-output-s",
        type=positive_float,
        default=1.00,
        help="Do not export stair clips shorter than this duration; they remain listed in segments.csv.",
    )
    parser.add_argument(
        "--max-output-s",
        type=nonnegative_float,
        default=10.0,
        help="Split longer same-label clips; 0 disables length splitting.",
    )
    parser.add_argument("--stand-speed", type=positive_float, default=0.08, help="Maximum standing XY speed in m/s.")
    parser.add_argument(
        "--locomotion-speed", type=positive_float, default=0.12, help="Minimum clear locomotion XY speed in m/s."
    )
    parser.add_argument(
        "--stand-joint-speed",
        type=positive_float,
        default=0.35,
        help="Maximum standing mean absolute hip/knee speed in rad/s.",
    )
    parser.add_argument(
        "--stairs-height-rate",
        type=positive_float,
        default=0.06,
        help="Minimum smoothed signed root-height rate for stair labels in m/s.",
    )
    parser.add_argument(
        "--stairs-min-net-height",
        type=positive_float,
        default=0.12,
        help="Minimum robust root-height change used to label a whole continuous block as stairs.",
    )
    parser.add_argument(
        "--class-weights",
        default=DEFAULT_CLASS_WEIGHTS,
        help="Comma-separated YAML class weights, for example stand=0.25,walk=0.25.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include unknown clips in the generated training YAML when their class weight is positive.",
    )
    parser.add_argument(
        "--keep-absolute-xy",
        action="store_true",
        help="Keep source root XY instead of moving each clip start to the local origin.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze and print a summary without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated output directory.")
    return parser.parse_args()


def load_motion(path: Path) -> MotionData:
    if not path.is_file():
        raise FileNotFoundError(f"input NPZ does not exist: {path}")

    with np.load(path, allow_pickle=False) as raw:
        required = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
        missing = required.difference(raw.files)
        if missing:
            raise ValueError(f"missing NPZ fields: {sorted(missing)}")
        motion = MotionData(
            framerate=float(np.asarray(raw["framerate"]).item()),
            joint_names=np.array(raw["joint_names"], copy=True),
            joint_pos=np.array(raw["joint_pos"], copy=True),
            base_pos_w=np.array(raw["base_pos_w"], copy=True),
            base_quat_w=np.array(raw["base_quat_w"], copy=True),
        )

    if not math.isfinite(motion.framerate) or motion.framerate <= 0.0:
        raise ValueError(f"invalid framerate: {motion.framerate}")
    if motion.joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be two-dimensional, got {motion.joint_pos.shape}")
    if motion.base_pos_w.shape != (motion.num_frames, 3):
        raise ValueError(f"base_pos_w must have shape ({motion.num_frames}, 3), got {motion.base_pos_w.shape}")
    if motion.base_quat_w.shape != (motion.num_frames, 4):
        raise ValueError(f"base_quat_w must have shape ({motion.num_frames}, 4), got {motion.base_quat_w.shape}")
    if motion.joint_names.shape != (motion.joint_pos.shape[1],):
        raise ValueError("joint_names count does not match joint_pos columns")
    if motion.num_frames < 2:
        raise ValueError("at least two frames are required")
    for name, values in (
        ("joint_pos", motion.joint_pos),
        ("base_pos_w", motion.base_pos_w),
        ("base_quat_w", motion.base_quat_w),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinity")

    quat_norms = np.linalg.norm(motion.base_quat_w, axis=1)
    if np.any(quat_norms < 1e-8):
        raise ValueError("base_quat_w contains a zero-length quaternion")
    return motion


def find_hip_knee_indices(joint_names: Iterable[object]) -> tuple[np.ndarray, list[str]]:
    names = [str(name) for name in joint_names]
    indices = [index for index, name in enumerate(names) if "_hip_" in name or "_knee_joint" in name]
    if not indices:
        raise ValueError("no hip/knee joints were found in joint_names")
    return np.asarray(indices, dtype=np.int64), [names[index] for index in indices]


def odd_window(seconds: float, fps: float, available_frames: int) -> int:
    if available_frames <= 1 or seconds <= 0.0:
        return 1
    window = max(1, int(round(seconds * fps)))
    if window % 2 == 0:
        window += 1
    largest_odd = available_frames if available_frames % 2 == 1 else available_frames - 1
    return max(1, min(window, largest_odd))


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 2:
        return np.asarray(values, dtype=np.float64).copy()
    pad = window // 2
    padded = np.pad(np.asarray(values, dtype=np.float64), (pad, pad), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def rolling_slope(values: np.ndarray, fps: float, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) <= 2:
        return np.gradient(values, 1.0 / fps) if len(values) > 1 else np.zeros_like(values)
    pad = window // 2
    x_s = (np.arange(window, dtype=np.float64) - pad) / fps
    kernel = x_s / np.sum(np.square(x_s))
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.correlate(padded, kernel, mode="valid")


def robust_height_change(values: np.ndarray, fps: float) -> float:
    """Estimate block height change without relying on noisy endpoint frames."""
    edge_frames = max(1, int(round(0.10 * fps)))
    edge_frames = min(edge_frames, max(1, len(values) // 3))
    return float(np.median(values[-edge_frames:]) - np.median(values[:edge_frames]))


def run_length_encode(values: np.ndarray) -> list[tuple[int, int, int]]:
    if len(values) == 0:
        return []
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(values)]))
    return [(int(start), int(end), int(values[start])) for start, end in zip(bounds[:-1], bounds[1:])]


def majority_filter(labels: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return labels.copy()
    pad = window // 2
    padded = np.pad(labels, (pad, pad), mode="edge")
    filtered = labels.copy()
    for frame in range(len(labels)):
        counts = np.bincount(padded[frame : frame + window], minlength=len(LABELS))
        winners = np.flatnonzero(counts == counts.max())
        filtered[frame] = labels[frame] if labels[frame] in winners else winners[0]
    return filtered


def merge_short_runs(labels: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1:
        return labels.copy()
    merged = labels.copy()
    for _ in range(20):
        runs = run_length_encode(merged)
        candidate = next(((i, run) for i, run in enumerate(runs) if run[1] - run[0] < min_frames), None)
        if candidate is None or len(runs) == 1:
            break
        run_index, (start, end, _) = candidate
        if 0 < run_index < len(runs) - 1 and runs[run_index - 1][2] == runs[run_index + 1][2]:
            replacement = runs[run_index - 1][2]
        elif run_index == 0:
            replacement = runs[1][2]
        elif run_index == len(runs) - 1:
            replacement = runs[-2][2]
        else:
            left_length = runs[run_index - 1][1] - runs[run_index - 1][0]
            right_length = runs[run_index + 1][1] - runs[run_index + 1][0]
            replacement = runs[run_index - 1][2] if left_length >= right_length else runs[run_index + 1][2]
        merged[start:end] = replacement
    return merged


def detect_hard_boundaries(
    motion: MotionData, hip_knee_indices: np.ndarray, cfg: DetectorConfig
) -> tuple[np.ndarray, list[HardBoundary]]:
    root_delta = np.diff(motion.base_pos_w.astype(np.float64), axis=0)
    root_jump_xy = np.linalg.norm(root_delta[:, :2], axis=1)
    root_jump_z = np.abs(root_delta[:, 2])
    joint_delta = np.abs(np.diff(motion.joint_pos[:, hip_knee_indices].astype(np.float64), axis=0))
    max_joint_jump = np.max(joint_delta, axis=1)

    is_xy_jump = root_jump_xy > cfg.jump_xy_m
    is_z_jump = root_jump_z > cfg.jump_z_m
    is_joint_jump = max_joint_jump > cfg.joint_jump_rad
    candidate_indices = np.flatnonzero(is_xy_jump | is_z_jump | is_joint_jump)

    candidates: list[HardBoundary] = []
    for index in candidate_indices:
        reasons = []
        if is_xy_jump[index]:
            reasons.append("root_xy_jump")
        if is_z_jump[index]:
            reasons.append("root_z_jump")
        if is_joint_jump[index]:
            reasons.append("hip_knee_jump")
        candidates.append(
            HardBoundary(
                frame=int(index + 1),
                root_jump_xy_m=float(root_jump_xy[index]),
                root_jump_z_m=float(root_jump_z[index]),
                max_hip_knee_jump_rad=float(max_joint_jump[index]),
                reasons="+".join(reasons),
            )
        )

    bounds = np.concatenate(([0], candidate_indices + 1, [motion.num_frames])).astype(np.int64)
    return np.unique(bounds), candidates


def compute_features_and_labels(
    motion: MotionData, hip_knee_indices: np.ndarray, hard_bounds: np.ndarray, cfg: DetectorConfig
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    frame_count = motion.num_frames
    speed_xy = np.zeros(frame_count, dtype=np.float64)
    height_rate = np.zeros(frame_count, dtype=np.float64)
    hip_knee_speed = np.zeros(frame_count, dtype=np.float64)
    labels = np.full(frame_count, LABEL_TO_ID["unknown"], dtype=np.int8)
    dt = 1.0 / motion.framerate

    for block_start, block_end in zip(hard_bounds[:-1], hard_bounds[1:]):
        start, end = int(block_start), int(block_end)
        length = end - start
        if length <= 0:
            continue
        pos = motion.base_pos_w[start:end].astype(np.float64)
        joints = motion.joint_pos[start:end, hip_knee_indices].astype(np.float64)

        if length == 1:
            velocity = np.zeros((1, 3), dtype=np.float64)
            joint_velocity = np.zeros_like(joints)
        else:
            velocity = np.gradient(pos, dt, axis=0)
            joint_velocity = np.gradient(joints, dt, axis=0)

        smooth_window = odd_window(cfg.smooth_window_s, motion.framerate, length)
        trend_window = odd_window(cfg.height_trend_window_s, motion.framerate, length)
        block_speed_xy = smooth_1d(np.linalg.norm(velocity[:, :2], axis=1), smooth_window)
        smooth_height = smooth_1d(pos[:, 2], smooth_window)
        block_height_rate = rolling_slope(smooth_height, motion.framerate, trend_window)
        block_joint_speed = smooth_1d(np.mean(np.abs(joint_velocity), axis=1), smooth_window)

        speed_xy[start:end] = block_speed_xy
        height_rate[start:end] = block_height_rate
        hip_knee_speed[start:end] = block_joint_speed

        block_height_change = robust_height_change(pos[:, 2], motion.framerate)
        if abs(block_height_change) >= cfg.stairs_min_net_height_m:
            stair_label = "stairs_up" if block_height_change > 0.0 else "stairs_down"
            block_labels = np.full(length, LABEL_TO_ID[stair_label], dtype=np.int8)
        else:
            abs_height_rate = np.abs(block_height_rate)
            is_stand = (
                (block_speed_xy <= cfg.stand_speed_mps)
                & (block_joint_speed <= cfg.stand_joint_speed_radps)
                & (abs_height_rate <= cfg.stairs_height_rate_mps)
            )
            is_walk = (
                (~is_stand)
                & (
                    (block_speed_xy >= cfg.locomotion_speed_mps)
                    | (block_joint_speed >= cfg.stand_joint_speed_radps)
                )
            )

            block_labels = np.full(length, LABEL_TO_ID["unknown"], dtype=np.int8)
            block_labels[is_stand] = LABEL_TO_ID["stand"]
            block_labels[is_walk] = LABEL_TO_ID["walk"]

            filter_window = odd_window(cfg.label_filter_s, motion.framerate, length)
            block_labels = majority_filter(block_labels, filter_window)
            min_state_frames = max(1, int(round(cfg.min_state_s * motion.framerate)))
            block_labels = merge_short_runs(block_labels, min_state_frames)
        labels[start:end] = block_labels

    return {
        "speed_xy": speed_xy,
        "height_rate": height_rate,
        "hip_knee_speed": hip_knee_speed,
    }, labels


def segment_confidence(label: str, segment_slice: slice, features: dict[str, np.ndarray], cfg: DetectorConfig) -> float:
    speed = float(np.mean(features["speed_xy"][segment_slice]))
    height_rate = float(np.mean(features["height_rate"][segment_slice]))
    joint_speed = float(np.mean(features["hip_knee_speed"][segment_slice]))
    if label == "stand":
        speed_score = np.clip(1.0 - speed / cfg.stand_speed_mps, 0.0, 1.0)
        joint_score = np.clip(1.0 - joint_speed / cfg.stand_joint_speed_radps, 0.0, 1.0)
        height_score = np.clip(1.0 - abs(height_rate) / cfg.stairs_height_rate_mps, 0.0, 1.0)
        return float(np.mean([speed_score, joint_score, height_score]))
    if label == "stairs_up":
        return float(np.clip(height_rate / (2.0 * cfg.stairs_height_rate_mps), 0.0, 1.0))
    if label == "stairs_down":
        return float(np.clip(-height_rate / (2.0 * cfg.stairs_height_rate_mps), 0.0, 1.0))
    if label == "walk":
        motion_score = max(
            np.clip(speed / (2.0 * cfg.locomotion_speed_mps), 0.0, 1.0),
            np.clip(joint_speed / (2.0 * cfg.stand_joint_speed_radps), 0.0, 1.0),
        )
        flat_score = np.clip(1.0 - abs(height_rate) / cfg.stairs_height_rate_mps, 0.0, 1.0)
        return float(0.5 * (motion_score + flat_score))
    return 0.0


def split_run_by_max_length(start: int, end: int, max_frames: int) -> list[tuple[int, int]]:
    length = end - start
    if max_frames <= 0 or length <= max_frames:
        return [(start, end)]
    chunk_count = int(math.ceil(length / max_frames))
    edges = np.rint(np.linspace(start, end, chunk_count + 1)).astype(np.int64)
    return [(int(left), int(right)) for left, right in zip(edges[:-1], edges[1:]) if right > left]


def build_segments(
    motion: MotionData,
    hard_bounds: np.ndarray,
    labels: np.ndarray,
    features: dict[str, np.ndarray],
    cfg: DetectorConfig,
) -> list[Segment]:
    segments: list[Segment] = []
    max_frames = int(round(cfg.max_output_s * motion.framerate)) if cfg.max_output_s > 0.0 else 0
    hard_bound_set = set(int(frame) for frame in hard_bounds[1:-1])

    for block_start, block_end in zip(hard_bounds[:-1], hard_bounds[1:]):
        start, end = int(block_start), int(block_end)
        for local_start, local_end, label_id in run_length_encode(labels[start:end]):
            run_start, run_end = start + local_start, start + local_end
            chunks = split_run_by_max_length(run_start, run_end, max_frames)
            for chunk_index, (chunk_start, chunk_end) in enumerate(chunks):
                label = ID_TO_LABEL[label_id]
                segment_slice = slice(chunk_start, chunk_end)
                boundary_reason = "hard_jump" if chunk_start in hard_bound_set else "state_change"
                if chunk_index > 0:
                    boundary_reason = "max_duration"
                height_rate_values = features["height_rate"][segment_slice]
                segment = Segment(
                    index=len(segments),
                    label=label,
                    start_frame=chunk_start,
                    end_frame=chunk_end,
                    boundary_reason=boundary_reason,
                    duration_s=(chunk_end - chunk_start) / motion.framerate,
                    confidence=segment_confidence(label, segment_slice, features, cfg),
                    mean_speed_xy_mps=float(np.mean(features["speed_xy"][segment_slice])),
                    mean_abs_height_rate_mps=float(np.mean(np.abs(height_rate_values))),
                    mean_height_rate_mps=float(np.mean(height_rate_values)),
                    net_height_change_m=float(motion.base_pos_w[chunk_end - 1, 2] - motion.base_pos_w[chunk_start, 2]),
                    mean_hip_knee_speed_radps=float(np.mean(features["hip_knee_speed"][segment_slice])),
                )
                segments.append(segment)
    return segments


def parse_class_weights(specification: str) -> dict[str, float]:
    weights = {label: 0.0 for label in LABELS}
    for item in specification.split(","):
        if not item.strip():
            continue
        try:
            label, raw_weight = item.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"invalid class weight item: {item!r}") from exc
        label = label.strip()
        if label not in weights:
            raise ValueError(f"unknown class in --class-weights: {label!r}")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"invalid weight for {label}: {raw_weight}")
        weights[label] = weight
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one class weight must be positive")
    return weights


def continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions).copy()
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(norms, 1e-12)
    for frame in range(1, len(result)):
        if np.dot(result[frame - 1], result[frame]) < 0.0:
            result[frame] *= -1.0
    return result.astype(quaternions.dtype, copy=False)


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}; use --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def segment_is_exportable(segment: Segment, motion: MotionData, cfg: DetectorConfig) -> bool:
    minimum_s = cfg.min_output_s
    if segment.label in {"stairs_up", "stairs_down"}:
        minimum_s = max(minimum_s, cfg.min_stairs_output_s)
    min_frames = max(2, int(math.ceil(minimum_s * motion.framerate)))
    return segment.end_frame - segment.start_frame >= min_frames


def export_segments(
    motion: MotionData,
    segments: list[Segment],
    output_dir: Path,
    cfg: DetectorConfig,
    keep_absolute_xy: bool,
) -> list[Path]:
    label_counts: Counter[str] = Counter()
    exported_paths: list[Path] = []

    for segment in segments:
        if not segment_is_exportable(segment, motion, cfg):
            continue
        label_counts[segment.label] += 1
        output_name = (
            f"{segment.label}_{label_counts[segment.label]:04d}_"
            f"f{segment.start_frame:06d}-{segment.end_frame:06d}_retargetted.npz"
        )
        output_path = output_dir / segment.label / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        section = slice(segment.start_frame, segment.end_frame)
        base_pos_w = motion.base_pos_w[section].copy()
        if not keep_absolute_xy:
            base_pos_w[:, :2] -= base_pos_w[0, :2]
        base_quat_w = continuous_quaternions(motion.base_quat_w[section])

        np.savez_compressed(
            output_path,
            framerate=np.asarray(motion.framerate, dtype=np.float64),
            joint_names=motion.joint_names,
            joint_pos=motion.joint_pos[section],
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
        )
        segment.exported = True
        segment.output_file = output_path.relative_to(output_dir).as_posix()
        exported_paths.append(output_path)
    return exported_paths


def write_segments_csv(output_dir: Path, segments: list[Segment]) -> None:
    fieldnames = list(asdict(segments[0]).keys()) if segments else list(Segment.__dataclass_fields__.keys())
    with (output_dir / "segments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            row = asdict(segment)
            for key, value in row.items():
                if isinstance(value, float):
                    row[key] = f"{value:.8f}"
            writer.writerow(row)


def write_boundaries_csv(output_dir: Path, candidates: list[HardBoundary]) -> None:
    fieldnames = list(HardBoundary.__dataclass_fields__.keys())
    with (output_dir / "hard_boundaries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def write_training_yaml(
    output_dir: Path,
    segments: list[Segment],
    class_weights: dict[str, float],
    include_unknown: bool,
) -> None:
    selected = [
        segment
        for segment in segments
        if segment.exported
        and class_weights.get(segment.label, 0.0) > 0.0
        and (segment.label != "unknown" or include_unknown)
    ]
    counts = Counter(segment.label for segment in selected)
    present_weight_sum = sum(class_weights[label] for label in counts)
    if not selected or present_weight_sum <= 0.0:
        raise ValueError("no exported clips have a positive YAML class weight")

    per_file_weights = [
        class_weights[segment.label] / present_weight_sum / counts[segment.label] for segment in selected
    ]
    with (output_dir / "selected_motions.yaml").open("w", encoding="utf-8") as stream:
        stream.write("selected_files:\n")
        for segment in selected:
            stream.write(f"  - {segment.output_file}\n")
        stream.write("motion_weights:\n")
        for weight in per_file_weights:
            stream.write(f"  - {weight:.10f}\n")


def print_summary(
    motion: MotionData,
    candidates: list[HardBoundary],
    segments: list[Segment],
    cfg: DetectorConfig,
    dry_run: bool,
) -> None:
    exportable = [segment for segment in segments if segment_is_exportable(segment, motion, cfg)]
    counts = Counter(segment.label for segment in exportable)
    durations = Counter()
    for segment in exportable:
        durations[segment.label] += segment.duration_s

    print("=" * 72)
    print("Motion division summary")
    print("=" * 72)
    print(f"frames                 : {motion.num_frames}")
    print(f"framerate              : {motion.framerate:.3f} Hz")
    print(f"duration               : {motion.num_frames / motion.framerate:.3f} s")
    print(f"hard boundary candidates: {len(candidates)}")
    print(f"detected segments      : {len(segments)}")
    print(f"exportable segments    : {len(exportable)}")
    for label in LABELS:
        print(f"  {label:12s}: {counts[label]:4d} clips, {durations[label]:9.3f} s")
    if dry_run:
        print("dry-run: no files were written")


def main() -> None:
    args = parse_args()
    if args.stand_speed >= args.locomotion_speed:
        raise ValueError("--stand-speed must be smaller than --locomotion-speed")
    cfg = DetectorConfig(
        jump_xy_m=args.jump_xy,
        jump_z_m=args.jump_z,
        joint_jump_rad=args.joint_jump,
        smooth_window_s=args.smooth_window_s,
        height_trend_window_s=args.height_trend_window_s,
        label_filter_s=args.label_filter_s,
        min_state_s=args.min_state_s,
        min_output_s=args.min_output_s,
        min_stairs_output_s=args.min_stairs_output_s,
        max_output_s=args.max_output_s,
        stand_speed_mps=args.stand_speed,
        locomotion_speed_mps=args.locomotion_speed,
        stand_joint_speed_radps=args.stand_joint_speed,
        stairs_height_rate_mps=args.stairs_height_rate,
        stairs_min_net_height_m=args.stairs_min_net_height,
    )
    class_weights = parse_class_weights(args.class_weights)
    motion = load_motion(args.input.expanduser().resolve())
    hip_knee_indices, hip_knee_names = find_hip_knee_indices(motion.joint_names)
    hard_bounds, candidates = detect_hard_boundaries(motion, hip_knee_indices, cfg)
    features, labels = compute_features_and_labels(motion, hip_knee_indices, hard_bounds, cfg)
    segments = build_segments(motion, hard_bounds, labels, features, cfg)

    if args.dry_run:
        print_summary(motion, candidates, segments, cfg, dry_run=True)
        print(f"hip/knee joints        : {', '.join(hip_knee_names)}")
        return

    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_directory(output_dir, args.overwrite)
    export_segments(motion, segments, output_dir, cfg, args.keep_absolute_xy)
    write_segments_csv(output_dir, segments)
    write_boundaries_csv(output_dir, candidates)
    write_training_yaml(output_dir, segments, class_weights, args.include_unknown)
    report = {
        "source_file": str(args.input.expanduser().resolve()),
        "output_directory": str(output_dir),
        "num_frames": motion.num_frames,
        "framerate": motion.framerate,
        "detector_config": asdict(cfg),
        "hip_knee_joints": hip_knee_names,
        "class_weights": class_weights,
        "hard_boundary_count": len(candidates),
        "segment_count": len(segments),
        "exported_segment_count": sum(segment.exported for segment in segments),
    }
    with (output_dir / "division_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2)
        stream.write("\n")

    print_summary(motion, candidates, segments, cfg, dry_run=False)
    print(f"output directory       : {output_dir}")
    print("Review segments.csv and rendered clips before training.")


if __name__ == "__main__":
    main()
