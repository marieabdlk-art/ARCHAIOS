from __future__ import annotations

from .objects import detect_background, extract_objects
from .types import ArcObject, Grid, Invariants, SegmentationProfile


def color_multiset(objects: list[ArcObject]) -> dict[int, int]:
    result: dict[int, int] = {}
    for obj in objects:
        result[obj.color] = result.get(obj.color, 0) + 1
    return result


def area_multiset(objects: list[ArcObject]) -> list[int]:
    return sorted(obj.area for obj in objects)


def bbox_size_multiset(objects: list[ArcObject]) -> list[tuple[int, int]]:
    return sorted((obj.width, obj.height) for obj in objects)


def shape_signature_multiset(objects: list[ArcObject]) -> list[tuple[tuple[int, int], ...]]:
    return sorted(obj.shape_signature for obj in objects)


def centroid_set(objects: list[ArcObject]) -> list[tuple[float, float]]:
    return sorted((round(obj.centroid[0], 3), round(obj.centroid[1], 3)) for obj in objects)


def relative_layout_bbox(objects: list[ArcObject]) -> list[tuple[int, int, int, int, int]]:
    if not objects:
        return []
    anchor_r = min(o.bbox[0] for o in objects)
    anchor_c = min(o.bbox[1] for o in objects)
    return sorted(
        (o.bbox[0] - anchor_r, o.bbox[1] - anchor_c, o.width, o.height, o.color)
        for o in objects
    )


def border_touching_multiset(objects: list[ArcObject]) -> list[bool]:
    return sorted(obj.touches_border for obj in objects)


def compare_train_pair(inp: Grid, out: Grid, profile: SegmentationProfile | None = None) -> dict:
    if profile is None:
        profile = SegmentationProfile()
    bg_in = profile.background if profile.background is not None else detect_background(inp)
    bg_out = profile.background if profile.background is not None else detect_background(out)
    profile_in = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=bg_in)
    profile_out = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=bg_out)

    objs_in = extract_objects(inp, bg_in, profile=profile_in)
    objs_out = extract_objects(out, bg_out, profile=profile_out)

    return {
        "grid_size_in": inp.shape,
        "grid_size_out": out.shape,
        "background_in": bg_in,
        "background_out": bg_out,
        "objects_in": objs_in,
        "objects_out": objs_out,
        "object_count_in": len(objs_in),
        "object_count_out": len(objs_out),
        "color_multiset_in": color_multiset(objs_in),
        "color_multiset_out": color_multiset(objs_out),
        "area_multiset_in": area_multiset(objs_in),
        "area_multiset_out": area_multiset(objs_out),
        "bbox_sizes_in": bbox_size_multiset(objs_in),
        "bbox_sizes_out": bbox_size_multiset(objs_out),
        "shape_signatures_in": shape_signature_multiset(objs_in),
        "shape_signatures_out": shape_signature_multiset(objs_out),
        "centroids_in": centroid_set(objs_in),
        "centroids_out": centroid_set(objs_out),
        "relative_layout_in": relative_layout_bbox(objs_in),
        "relative_layout_out": relative_layout_bbox(objs_out),
        "border_touching_in": border_touching_multiset(objs_in),
        "border_touching_out": border_touching_multiset(objs_out),
    }


def detect_invariants(
    train_pairs: list[tuple[Grid, Grid]],
    profile: SegmentationProfile | None = None,
) -> Invariants:
    if not train_pairs:
        raise ValueError("train_pairs must not be empty")
    if profile is None:
        profile = SegmentationProfile()

    comparisons = [compare_train_pair(inp, out, profile) for inp, out in train_pairs]
    inv = Invariants(profile_label=profile.label())

    checks = [
        ("grid_size", lambda c: c["grid_size_in"] == c["grid_size_out"]),
        ("background_color", lambda c: c["background_in"] == c["background_out"]),
        ("object_count", lambda c: c["object_count_in"] == c["object_count_out"]),
        ("color_multiset", lambda c: c["color_multiset_in"] == c["color_multiset_out"]),
        ("area_multiset", lambda c: c["area_multiset_in"] == c["area_multiset_out"]),
        ("bbox_size", lambda c: c["bbox_sizes_in"] == c["bbox_sizes_out"]),
        ("shape_signature", lambda c: c["shape_signatures_in"] == c["shape_signatures_out"]),
        ("absolute_position", lambda c: c["centroids_in"] == c["centroids_out"]),
        ("relative_layout", lambda c: c["relative_layout_in"] == c["relative_layout_out"]),
        ("border_touching", lambda c: c["border_touching_in"] == c["border_touching_out"]),
    ]

    for name, predicate in checks:
        if all(predicate(c) for c in comparisons):
            inv.preserved.append(name)
        else:
            inv.changed.append(name)

    preserved = set(inv.preserved)
    changed = set(inv.changed)

    if {"object_count", "color_multiset", "area_multiset", "shape_signature"} <= preserved and "absolute_position" in changed:
        if "relative_layout" in preserved:
            inv.candidate_transform_families.append("translation_uniform")
            inv.notes.append("Shape/color/count preserved while absolute positions changed; relative layout preserved.")
        else:
            inv.candidate_transform_families.append("translation_individual")
            inv.notes.append("Shape/color/count preserved while relative layout changed.")

    if {"object_count", "area_multiset", "shape_signature", "absolute_position"} <= preserved and "color_multiset" in changed:
        inv.candidate_transform_families.append("recoloring")
        inv.notes.append("Objects preserve shape and position while colors change.")

    if {"object_count", "color_multiset", "area_multiset"} <= preserved and "shape_signature" in changed:
        inv.candidate_transform_families.append("geometric_transform")
        inv.notes.append("Object count, colors and areas are preserved but shape signatures change.")

    if "object_count" in changed:
        counts_in = [c["object_count_in"] for c in comparisons]
        counts_out = [c["object_count_out"] for c in comparisons]
        if all(ci > co for ci, co in zip(counts_in, counts_out)):
            inv.candidate_transform_families.append("deletion")
            inv.notes.append("Object count decreases across train pairs.")
        elif all(ci < co for ci, co in zip(counts_in, counts_out)):
            inv.candidate_transform_families.append("copy_or_fill")
            inv.notes.append("Object count increases across train pairs.")

    if "grid_size" in changed:
        inv.candidate_transform_families.append("crop_or_resize")
        inv.notes.append("Grid size changes between input and output.")

    if not inv.candidate_transform_families:
        inv.candidate_transform_families.append("unknown")
        inv.notes.append("No transform family confidently detected; fallback or pixel-level analysis required.")

    return inv


def format_invariants(inv: Invariants) -> str:
    lines = ["=== Invariant Analysis ==="]
    lines.append(f"Profile   : {inv.profile_label}")
    lines.append(f"Preserved : {', '.join(inv.preserved) or '—'}")
    lines.append(f"Changed   : {', '.join(inv.changed) or '—'}")
    lines.append(f"Candidates: {', '.join(inv.candidate_transform_families)}")
    if inv.notes:
        lines.append("Notes:")
        for note in inv.notes:
            lines.append(f"  · {note}")
    return "\n".join(lines)
