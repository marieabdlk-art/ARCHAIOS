from __future__ import annotations

from .objects import detect_background, extract_objects, select_objects
from .types import DeleteProgram, FillBBoxProgram, Grid, RecolorProgram, SegmentationProfile, SeqProgram, ShiftProgram


class InvalidProgram(Exception):
    pass


def _objects_for_selector(grid: Grid, selector, profile: SegmentationProfile | None = None):
    if profile is None:
        profile = SegmentationProfile()
    background = profile.background if profile.background is not None else detect_background(grid)
    resolved = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=background)
    objects = extract_objects(grid, background, profile=resolved)
    selected = select_objects(objects, selector)
    if not selected:
        raise InvalidProgram("empty selector")
    return background, selected, resolved


def apply_shift_strict(
    grid: Grid,
    program: ShiftProgram,
    profile: SegmentationProfile | None = None,
) -> Grid:
    background, selected, _ = _objects_for_selector(grid, program.selector, profile)

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
    _, selected, _ = _objects_for_selector(grid, program.selector, profile)

    result = grid.copy()
    for obj in selected:
        new_color = program.color_map.get(obj.color, obj.color)
        for r, c in obj.pixels:
            result[r, c] = new_color
    return result


def apply_delete(
    grid: Grid,
    program: DeleteProgram,
    profile: SegmentationProfile | None = None,
) -> Grid:
    background, selected, _ = _objects_for_selector(grid, program.selector, profile)

    result = grid.copy()
    for obj in selected:
        for r, c in obj.pixels:
            result[r, c] = background
    return result


def apply_fill_bbox(
    grid: Grid,
    program: FillBBoxProgram,
    profile: SegmentationProfile | None = None,
) -> Grid:
    background, selected, _ = _objects_for_selector(grid, program.selector, profile)

    r_min = min(obj.bbox[0] for obj in selected)
    c_min = min(obj.bbox[1] for obj in selected)
    r_max = max(obj.bbox[2] for obj in selected)
    c_max = max(obj.bbox[3] for obj in selected)

    result = grid.copy()
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            if result[r, c] == background:
                result[r, c] = program.color
    return result


def execute(program, grid: Grid, profile: SegmentationProfile | None = None) -> Grid:
    if isinstance(program, ShiftProgram):
        return apply_shift_strict(grid, program, profile)
    if isinstance(program, RecolorProgram):
        return apply_recolor(grid, program, profile)
    if isinstance(program, DeleteProgram):
        return apply_delete(grid, program, profile)
    if isinstance(program, FillBBoxProgram):
        return apply_fill_bbox(grid, program, profile)
    if isinstance(program, SeqProgram):
        intermediate = execute(program.first, grid, profile)
        return execute(program.second, intermediate, profile)
    raise InvalidProgram(f"unsupported program type: {type(program)}")
