from __future__ import annotations

from .objects import detect_background, extract_objects, select_objects
from .types import Grid, RecolorProgram, SegmentationProfile, SeqProgram, ShiftProgram


class InvalidProgram(Exception):
    pass


def apply_shift_strict(
    grid: Grid,
    program: ShiftProgram,
    profile: SegmentationProfile | None = None,
) -> Grid:
    if profile is None:
        profile = SegmentationProfile()
    background = profile.background if profile.background is not None else detect_background(grid)
    profile = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=background)
    objects = extract_objects(grid, background, profile=profile)
    selected = select_objects(objects, program.selector)

    if not selected:
        raise InvalidProgram("empty selector")

    h, w = grid.shape
    selected_pixels = {px for obj in selected for px in obj.pixels}
    non_selected_active = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] != background and (r, c) not in selected_pixels
    }

    moved: dict[tuple[int, int], int] = {}
    for r, c in selected_pixels:
        nr, nc = r + program.dy, c + program.dx
        if not (0 <= nr < h and 0 <= nc < w):
            raise InvalidProgram("SHIFT produces out-of-bounds pixel")
        if (nr, nc) in non_selected_active:
            raise InvalidProgram("SHIFT collides with non-selected object")
        moved[(nr, nc)] = int(grid[r, c])

    result = grid.copy()
    for r, c in selected_pixels:
        result[r, c] = background
    for (r, c), color in moved.items():
        result[r, c] = color
    return result


def apply_recolor(
    grid: Grid,
    program: RecolorProgram,
    profile: SegmentationProfile | None = None,
) -> Grid:
    if profile is None:
        profile = SegmentationProfile()
    background = profile.background if profile.background is not None else detect_background(grid)
    profile = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=background)
    objects = extract_objects(grid, background, profile=profile)
    selected = select_objects(objects, program.selector)

    if not selected:
        raise InvalidProgram("empty selector")

    result = grid.copy()
    for obj in selected:
        new_color = program.color_map.get(obj.color, obj.color)
        for r, c in obj.pixels:
            result[r, c] = new_color
    return result


def execute(program, grid: Grid, profile: SegmentationProfile | None = None) -> Grid:
    if isinstance(program, ShiftProgram):
        return apply_shift_strict(grid, program, profile)
    if isinstance(program, RecolorProgram):
        return apply_recolor(grid, program, profile)
    if isinstance(program, SeqProgram):
        intermediate = execute(program.first, grid, profile)
        return execute(program.second, intermediate, profile)
    raise InvalidProgram(f"unsupported program type: {type(program)}")
