from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .objects import detect_background
from .types import Grid

Axis = Literal["vertical", "horizontal"]
ColorMode = Literal["exact", "agnostic"]


@dataclass(frozen=True)
class SymmetryReport:
    axis: Axis
    color_mode: ColorMode
    score: float
    checked_pairs: int
    matching_pairs: int
    asymmetric_cells: list[tuple[int, int]]
    mismatched_pairs: list[tuple[tuple[int, int], tuple[int, int]]]

    @property
    def is_symmetric(self) -> bool:
        return self.score == 1.0


def mirror_coord(r: int, c: int, height: int, width: int, axis: Axis) -> tuple[int, int]:
    if axis == "vertical":
        return r, width - 1 - c
    if axis == "horizontal":
        return height - 1 - r, c
    raise ValueError("axis must be 'vertical' or 'horizontal'")


def _pair_matches(a: int, b: int, background: int, color_mode: ColorMode) -> bool:
    if color_mode == "exact":
        return a == b
    if color_mode == "agnostic":
        return (a == background) == (b == background)
    raise ValueError("color_mode must be 'exact' or 'agnostic'")


def symmetry_report(
    grid: Grid,
    *,
    axis: Axis = "vertical",
    color_mode: ColorMode = "exact",
    background: int | None = None,
) -> SymmetryReport:
    """Analyze grid symmetry.

    `exact` mode requires mirrored cells to have the same color.
    `agnostic` mode only checks foreground/background occupancy.

    This is an analyzer, not a generator. It does not infer missing pixels or
    decide which color should be used to repair asymmetry.
    """
    if background is None:
        background = detect_background(grid)

    height, width = grid.shape
    visited: set[tuple[int, int]] = set()
    checked_pairs = 0
    matching_pairs = 0
    asymmetric_cells: set[tuple[int, int]] = set()
    mismatched_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if (r, c) in visited:
                continue
            mr, mc = mirror_coord(r, c, height, width, axis)
            visited.add((r, c))
            visited.add((mr, mc))

            # Center line cells mirror to themselves. They are trivially symmetric.
            if (r, c) == (mr, mc):
                continue

            checked_pairs += 1
            a = int(grid[r, c])
            b = int(grid[mr, mc])
            if _pair_matches(a, b, background, color_mode):
                matching_pairs += 1
            else:
                asymmetric_cells.add((r, c))
                asymmetric_cells.add((mr, mc))
                mismatched_pairs.append(((r, c), (mr, mc)))

    score = matching_pairs / checked_pairs if checked_pairs else 1.0
    return SymmetryReport(
        axis=axis,
        color_mode=color_mode,
        score=score,
        checked_pairs=checked_pairs,
        matching_pairs=matching_pairs,
        asymmetric_cells=sorted(asymmetric_cells),
        mismatched_pairs=mismatched_pairs,
    )


def symmetry_score(
    grid: Grid,
    *,
    axis: Axis = "vertical",
    color_mode: ColorMode = "exact",
    background: int | None = None,
) -> float:
    return symmetry_report(grid, axis=axis, color_mode=color_mode, background=background).score


def find_asymmetric_cells(
    grid: Grid,
    *,
    axis: Axis = "vertical",
    color_mode: ColorMode = "exact",
    background: int | None = None,
) -> list[tuple[int, int]]:
    return symmetry_report(grid, axis=axis, color_mode=color_mode, background=background).asymmetric_cells


def is_symmetric(
    grid: Grid,
    *,
    axis: Axis = "vertical",
    color_mode: ColorMode = "exact",
    background: int | None = None,
) -> bool:
    return symmetry_score(grid, axis=axis, color_mode=color_mode, background=background) == 1.0


def occupancy_grid(grid: Grid, *, background: int | None = None) -> Grid:
    """Return a binary foreground/background grid for color-agnostic inspection."""
    if background is None:
        background = detect_background(grid)
    return (grid != background).astype(int)
