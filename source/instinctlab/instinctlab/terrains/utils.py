"""Utilities for resolving generated terrain columns back to sub-terrain names."""

from __future__ import annotations

import numpy as np


def resolve_subterrain_names_by_column(terrain_generator_cfg) -> list[str]:
    """Return the configured sub-terrain name for every generated terrain column.

    Isaac Lab assigns environments to terrain columns, while the generator accepts
    proportions per named sub-terrain.  Keep this conversion in one place so command,
    observation, and motion-reference sampling use exactly the same assignment.
    """
    sub_terrains = terrain_generator_cfg.sub_terrains
    if not sub_terrains:
        raise ValueError("The terrain generator must define at least one sub-terrain.")

    names = list(sub_terrains.keys())
    proportions = np.asarray(
        [cfg.proportion for cfg in sub_terrains.values()], dtype=np.float64
    )
    total = float(proportions.sum())
    if total <= 0.0:
        raise ValueError("The sum of sub-terrain proportions must be positive.")
    proportions /= total

    cumulative = np.cumsum(proportions)
    column_names: list[str] = []
    for column_idx in range(terrain_generator_cfg.num_cols):
        # Match TerrainGenerator's deterministic, proportion-based column layout.
        sample = column_idx / terrain_generator_cfg.num_cols + 0.001
        subterrain_idx = int(np.min(np.where(sample < cumulative)[0]))
        column_names.append(names[subterrain_idx])
    return column_names
