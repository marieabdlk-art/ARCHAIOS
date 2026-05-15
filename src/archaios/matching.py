from __future__ import annotations

from dataclasses import dataclass

from .types import ArcObject


@dataclass(frozen=True)
class ObjectMatch:
    obj_in: ArcObject
    obj_out: ArcObject
    dx: int
    dy: int
    confidence: float
    strategy: str
    ambiguous: bool = False


def centroid_distance(a: ArcObject, b: ArcObject) -> float:
    return ((a.centroid[0] - b.centroid[0]) ** 2 + (a.centroid[1] - b.centroid[1]) ** 2) ** 0.5


def bbox_delta(a: ArcObject, b: ArcObject) -> tuple[int, int]:
    dy = b.bbox[0] - a.bbox[0]
    dx = b.bbox[1] - a.bbox[1]
    return dx, dy


def _match_greedy(
    objs_in: list[ArcObject],
    objs_out: list[ArcObject],
    strategies,
) -> tuple[list[ObjectMatch], float]:
    if not objs_in or not objs_out:
        return [], 0.0

    used_out: set[int] = set()
    matches: list[ObjectMatch] = []
    matched_in: set[str] = set()

    for name, confidence, candidate_fn in strategies:
        for o_in in objs_in:
            if o_in.id in matched_in:
                continue
            candidates = [
                (idx, o_out)
                for idx, o_out in enumerate(objs_out)
                if idx not in used_out and candidate_fn(o_in, o_out)
            ]
            if not candidates:
                continue
            ambiguous = len(candidates) > 1
            idx, o_out = min(candidates, key=lambda pair: centroid_distance(o_in, pair[1]))
            dx, dy = bbox_delta(o_in, o_out)
            used_out.add(idx)
            matched_in.add(o_in.id)
            matches.append(
                ObjectMatch(
                    obj_in=o_in,
                    obj_out=o_out,
                    dx=dx,
                    dy=dy,
                    confidence=confidence * (0.85 if ambiguous else 1.0),
                    strategy=name,
                    ambiguous=ambiguous,
                )
            )

    total_conf = sum(m.confidence for m in matches) / len(objs_in)
    return matches, total_conf


def match_objects_for_translation(objs_in: list[ArcObject], objs_out: list[ArcObject]) -> tuple[list[ObjectMatch], float]:
    strategies = [
        (
            "same_color+same_shape_signature",
            1.0,
            lambda a, b: a.color == b.color and a.shape_signature == b.shape_signature,
        ),
        (
            "same_color+same_area+same_bbox_size",
            0.85,
            lambda a, b: a.color == b.color and a.area == b.area and a.width == b.width and a.height == b.height,
        ),
        (
            "same_shape_signature+nearest",
            0.75,
            lambda a, b: a.shape_signature == b.shape_signature,
        ),
        (
            "same_color+nearest",
            0.60,
            lambda a, b: a.color == b.color,
        ),
        (
            "same_area+nearest",
            0.40,
            lambda a, b: a.area == b.area,
        ),
        (
            "nearest_only",
            0.20,
            lambda a, b: True,
        ),
    ]
    return _match_greedy(objs_in, objs_out, strategies)


def match_objects_for_recolor(objs_in: list[ArcObject], objs_out: list[ArcObject]) -> tuple[list[ObjectMatch], float]:
    strategies = [
        (
            "same_shape_signature+same_bbox_position",
            1.0,
            lambda a, b: a.shape_signature == b.shape_signature and a.bbox == b.bbox,
        ),
        (
            "same_area+same_bbox_position",
            0.80,
            lambda a, b: a.area == b.area and a.bbox == b.bbox,
        ),
        (
            "same_shape_signature+nearest",
            0.60,
            lambda a, b: a.shape_signature == b.shape_signature,
        ),
        (
            "nearest_only",
            0.30,
            lambda a, b: True,
        ),
    ]
    return _match_greedy(objs_in, objs_out, strategies)
