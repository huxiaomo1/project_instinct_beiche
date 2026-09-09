"""Parkour-specific observation terms."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from instinctlab.terrains import resolve_subterrain_names_by_column


def terrain_type_one_hot(env, terrain_name_groups: list[list[str]]) -> torch.Tensor:
    """Encode each environment's terrain group as a one-hot observation.

    The returned tensor is cached because terrain columns do not change when the
    curriculum changes terrain difficulty levels.
    """
    terrain = env.scene["terrain"]
    cache_key = tuple(tuple(group) for group in terrain_name_groups)
    cache = getattr(terrain, "_instinctlab_terrain_type_one_hot_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    terrain_generator_cfg = terrain.cfg.terrain_generator
    column_names = resolve_subterrain_names_by_column(terrain_generator_cfg)
    name_to_group = {
        terrain_name: group_idx
        for group_idx, terrain_names in enumerate(terrain_name_groups)
        for terrain_name in terrain_names
    }
    missing_names = sorted(set(column_names) - set(name_to_group))
    if missing_names:
        raise ValueError(
            f"Terrain context groups do not cover generated terrains: {missing_names}"
        )

    column_groups = torch.tensor(
        [name_to_group[name] for name in column_names],
        dtype=torch.long,
        device=terrain.terrain_types.device,
    )
    env_groups = column_groups[terrain.terrain_types]
    encoded = functional.one_hot(env_groups, num_classes=len(terrain_name_groups)).to(
        torch.float32
    )
    cache[cache_key] = encoded
    terrain._instinctlab_terrain_type_one_hot_cache = cache
    return encoded
