from __future__ import annotations

from collections import deque
from typing import Optional
import numpy as np

from .types import ArcObject, Grid, SegmentationMode, SegmentationProfile


def parse_grid(raw: list[list[int]]) -> Grid:
    return np.array(raw, dtype=int)


def detect_background(grid: Grid) -> int:
    values, counts = np.unique(grid, return_counts=True)

    if 0 in values:
        zero_idx = list(values).index(0)
        zero_ratio = counts[zero_idx] / grid.size
        if zero_ratio >= 0.10:
            return 0

    return int(values[np.argmax(counts)])


def neighbors(r: int, c: int, h: int, w: int, connectivity: int = 4) -> list[tuple[int, int]]:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    result = []
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc))
    return result


def extract_objects(
    grid: Grid,
    background: Optional[int] = None,
    *,
    profile: Optional[SegmentationProfile] = None,
) -> list[ArcObject]:
    if profile is None:
        profile = SegmentationProfile(background=background)
    if profile.background is not None:
        background = profile.background
    if background is None:
        background = detect_background(grid)

    mode = profile.mode
    connectivity = profile.connectivity

    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[ArcObject] = []
    obj_counter = 0

    for start_r in range(h):
        for start_c in range(w):
            start_color = int(grid[start_r, start_c])
            if start_color == background or visited[start_r, start_c]:
                continue

            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            pixels: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                pixels.append((r, c))

                for nr, nc in neighbors(r, c, h, w, connectivity):
                    if visited[nr, nc]:
                        continue
                    color = int(grid[nr, nc])
                    if color == background:
                        continue
                    if mode == SegmentationMode.COLOR and color != start_color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            r_min, r_max = min(rows), max(rows)
            c_min, c_max = min(cols), max(cols)
            rep_color = start_color
            shape_signature = tuple(sorted((r - r_min, c - c_min) for r, c in pixels))

            objects.append(
                ArcObject(
                    id=f"obj_{obj_counter}",
                    color=rep_color,
                    pixels=tuple(sorted(pixels)),
                    bbox=(r_min, c_min, r_max, c_max),
                    area=len(pixels),
                    centroid=(sum(rows) / len(rows), sum(cols) / len(cols)),
                    touches_border=(r_min == 0 or r_max == h - 1 or c_min == 0 or c_max == w - 1),
                    width=c_max - c_min + 1,
                    height=r_max - r_min + 1,
                    shape_signature=shape_signature,
                )
            )
            obj_counter += 1

    return objects


def _extreme(objects: list[ArcObject], key_fn, reverse: bool = False) -> list[ArcObject]:
    if not objects:
        return []
    value = key_fn(max(objects, key=key_fn) if reverse else min(objects, key=key_fn))
    return [obj for obj in objects if key_fn(obj) == value]


def select_objects(objects: list[ArcObject], selector) -> list[ArcObject]:
    if selector.kind == "ALL":
        return objects
    if selector.kind == "COLOR":
        return [obj for obj in objects if obj.color == selector.value]
    if selector.kind == "SIZE":
        return [obj for obj in objects if obj.area == selector.value]
    if selector.kind == "LARGEST":
        if not objects:
            return []
        max_area = max(obj.area for obj in objects)
        return [obj for obj in objects if obj.area == max_area]
    if selector.kind == "SMALLEST":
        if not objects:
            return []
        min_area = min(obj.area for obj in objects)
        return [obj for obj in objects if obj.area == min_area]
    if selector.kind == "TOUCHING_BORDER":
        return [obj for obj in objects if obj.touches_border]
    if selector.kind == "NOT_TOUCHING_BORDER":
        return [obj for obj in objects if not obj.touches_border]
    if selector.kind == "POSITION":
        if selector.value == "left":
            return _extreme(objects, lambda obj: obj.bbox[1])
        if selector.value == "right":
            return _extreme(objects, lambda obj: obj.bbox[3], reverse=True)
        if selector.value == "top":
            return _extreme(objects, lambda obj: obj.bbox[0])
        if selector.value == "bottom":
            return _extreme(objects, lambda obj: obj.bbox[2], reverse=True)
    return []
