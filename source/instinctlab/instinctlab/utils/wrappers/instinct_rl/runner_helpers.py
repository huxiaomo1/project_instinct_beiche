"""Checkpoint depth matching for Parkour fine-tuning."""

from __future__ import annotations

import torch


def match_finetune_depth_to_checkpoint(env_cfg, runner_config, checkpoint_path):
    """Keep 32x18 / 64x36 encoder dimensions when warm-starting ordinary Parkour.

    No weights are resized or discarded. Unexpected architectures fail early;
    the model's ordinary strict load still validates all remaining dimensions.
    """
    if not getattr(env_cfg, "stand_finetune", False):
        return
    if not getattr(env_cfg, "auto_depth_from_checkpoint", True):
        return
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = state["model_state_dict"]
    suffix = "_parallel_blocks.depth_encoder.head.model.0.weight"
    sizes = {int(value.shape[1]) for name, value in weights.items() if name.endswith(suffix)}
    if len(sizes) != 1:
        raise ValueError("Cannot identify matching actor/critic depth encoders in the checkpoint")
    encoder = runner_config["policy"]["encoder_configs"]["depth_encoder"]
    expected = {"channels": [4], "kernel_sizes": [3], "strides": [1], "paddings": [1]}
    if any(list(encoder.get(key, [])) != value for key, value in expected.items()):
        raise ValueError("Automatic depth matching supports the original Parkour encoder only")
    size = sizes.pop()
    crops = {4 * 18 * 32: (18, 0, 16, 16), 4 * 36 * 64: (0, 0, 0, 0)}
    if size not in crops:
        raise ValueError(f"Unsupported depth encoder input size: {size}; match the original model configuration")
    camera = env_cfg.scene.camera
    if (camera.pattern_cfg.width, camera.pattern_cfg.height) != (64, 36):
        raise ValueError("Checkpoint depth matching expects a 64x36 raw camera")
    crop = camera.noise_pipeline["crop_and_resize"]
    crop.crop_region = crops[size]
    crop.resize_shape = None
    width = 64 - crop.crop_region[2] - crop.crop_region[3]
    height = 36 - crop.crop_region[0] - crop.crop_region[1]
    print(f"[INFO] Fine-tuning checkpoint depth input: {width}x{height}; crop={crop.crop_region}")
