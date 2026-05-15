from __future__ import annotations

import numpy as np

from .executor import InvalidProgram, execute
from .matching import match_objects_for_recolor, match_objects_for_translation
from .objects import detect_background, extract_objects
from .types import DeleteProgram, Grid, Hypothesis, Invariants, RecolorProgram, SegmentationProfile, Selector, SeqProgram, ShiftProgram


def verify_program(program, train_pairs: list[tuple[Grid, Grid]], profile: SegmentationProfile | None = None) -> tuple[int, int]:
    passed = 0
    total = len(train_pairs)
    for inp, expected in train_pairs:
        try:
            predicted = execute(program, inp, profile)
        except InvalidProgram:
            continue
        if np.array_equal(predicted, expected):
            passed += 1
    return passed, total


def _translation_allowed(inv: Invariants) -> bool:
    families = set(inv.candidate_transform_families)
    if families & {"translation_uniform", "translation_individual", "unknown"}:
        return True
    return "absolute_position" in inv.changed and "shape_signature" in inv.preserved


def _recolor_allowed(inv: Invariants) -> bool:
    families = set(inv.candidate_transform_families)
    if families & {"recoloring", "unknown"}:
        return True
    return "color_multiset" in inv.changed and "shape_signature" in inv.preserved


def _delete_allowed(inv: Invariants) -> bool:
    families = set(inv.candidate_transform_families)
    if families & {"deletion", "unknown"}:
        return True
    return "object_count" in inv.changed


def _candidate_selectors_for_objects(train_pairs: list[tuple[Grid, Grid]], profile: SegmentationProfile) -> list[Selector]:
    selectors: list[Selector] = [
        Selector.all(),
        Selector.largest(),
        Selector.smallest(),
        Selector.touching_border(),
        Selector.not_touching_border(),
        Selector.position("left"),
        Selector.position("right"),
        Selector.position("top"),
        Selector.position("bottom"),
    ]
    colors = set()
    sizes = set()
    for inp, _ in train_pairs:
        bg = profile.background if profile.background is not None else detect_background(inp)
        profile_in = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=bg)
        for obj in extract_objects(inp, bg, profile=profile_in):
            colors.add(obj.color)
            sizes.add(obj.area)
    selectors.extend(Selector.color(c) for c in sorted(colors))
    selectors.extend(Selector.size(s) for s in sorted(sizes))

    unique: dict[str, Selector] = {}
    for selector in selectors:
        unique[selector.to_dsl()] = selector
    return list(unique.values())


def generate_translation_hypotheses(
    train_pairs: list[tuple[Grid, Grid]],
    inv: Invariants,
    profile: SegmentationProfile | None = None,
) -> list[Hypothesis]:
    if not _translation_allowed(inv):
        return []
    if profile is None:
        profile = SegmentationProfile()

    all_matches = []
    matching_confidences = []

    for inp, out in train_pairs:
        bg_in = profile.background if profile.background is not None else detect_background(inp)
        bg_out = profile.background if profile.background is not None else detect_background(out)
        profile_in = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=bg_in)
        profile_out = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=bg_out)
        objs_in = extract_objects(inp, bg_in, profile=profile_in)
        objs_out = extract_objects(out, bg_out, profile=profile_out)
        matches, conf = match_objects_for_translation(objs_in, objs_out)
        all_matches.append(matches)
        matching_confidences.append(conf)

    all_dx = [m.dx for pair in all_matches for m in pair]
    all_dy = [m.dy for pair in all_matches for m in pair]
    if not all_dx:
        return []

    hypotheses: list[Hypothesis] = []
    dx_std = float(np.std(all_dx))
    dy_std = float(np.std(all_dy))
    dx_mean = int(round(float(np.mean(all_dx))))
    dy_mean = int(round(float(np.mean(all_dy))))
    matching_conf = float(np.mean(matching_confidences)) if matching_confidences else 0.0
    uniform_threshold = 0.0

    if dx_std <= uniform_threshold and dy_std <= uniform_threshold:
        candidates = [ShiftProgram(selector, dx_mean, dy_mean) for selector in _candidate_selectors_for_objects(train_pairs, profile)]

        for program in candidates:
            passed, total = verify_program(program, train_pairs, profile)
            selector_penalty = 0.0 if program.selector.kind == "ALL" else 0.10
            confidence = max(0.0, 0.95 * matching_conf - selector_penalty)
            hypotheses.append(
                Hypothesis(
                    program=program,
                    generator="TranslationGenerator",
                    confidence=confidence,
                    train_match=f"{passed}/{total}",
                    match_rate=passed / total if total else 0.0,
                    notes=[
                        f"Uniform bbox delta detected: dx={dx_mean}, dy={dy_mean}.",
                        f"std(dx)={dx_std:.2f}, std(dy)={dy_std:.2f}.",
                        f"matching_confidence={matching_conf:.2f}.",
                    ],
                )
            )
    else:
        by_color: dict[int, list[tuple[int, int]]] = {}
        for pair in all_matches:
            for match in pair:
                by_color.setdefault(match.obj_in.color, []).append((match.dx, match.dy))

        for color, deltas in by_color.items():
            dxs = [d[0] for d in deltas]
            dys = [d[1] for d in deltas]
            if np.std(dxs) == 0 and np.std(dys) == 0:
                dx = int(dxs[0])
                dy = int(dys[0])
                program = ShiftProgram(Selector.color(color), dx, dy)
                passed, total = verify_program(program, train_pairs, profile)
                hypotheses.append(
                    Hypothesis(
                        program=program,
                        generator="TranslationGenerator",
                        confidence=0.65 * matching_conf,
                        train_match=f"{passed}/{total}",
                        match_rate=passed / total if total else 0.0,
                        notes=[f"Color-conditioned translation: color={color}, dx={dx}, dy={dy}."],
                    )
                )

    return sorted(hypotheses, key=lambda h: (h.match_rate, h.confidence), reverse=True)


def _extract_color_map(matches) -> tuple[dict[int, int], float]:
    color_map: dict[int, int] = {}
    for m in matches:
        c_in = m.obj_in.color
        c_out = m.obj_out.color
        if c_in in color_map and color_map[c_in] != c_out:
            return color_map, 0.0
        color_map[c_in] = c_out
    return color_map, 1.0


def _merge_color_maps(maps: list[dict[int, int]]) -> tuple[dict[int, int], float]:
    merged: dict[int, int] = {}
    for mapping in maps:
        for c_in, c_out in mapping.items():
            if c_in in merged and merged[c_in] != c_out:
                return merged, 0.0
            merged[c_in] = c_out
    return merged, 1.0


def generate_recolor_hypotheses(
    train_pairs: list[tuple[Grid, Grid]],
    inv: Invariants,
    profile: SegmentationProfile | None = None,
) -> list[Hypothesis]:
    if not _recolor_allowed(inv):
        return []
    if profile is None:
        profile = SegmentationProfile()

    maps: list[dict[int, int]] = []
    match_confs = []
    for inp, out in train_pairs:
        bg_in = profile.background if profile.background is not None else detect_background(inp)
        bg_out = profile.background if profile.background is not None else detect_background(out)
        profile_in = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=bg_in)
        profile_out = SegmentationProfile(mode=profile.mode, connectivity=profile.connectivity, background=bg_out)
        objs_in = extract_objects(inp, bg_in, profile=profile_in)
        objs_out = extract_objects(out, bg_out, profile=profile_out)
        matches, match_conf = match_objects_for_recolor(objs_in, objs_out)
        color_map, map_conf = _extract_color_map(matches)
        if map_conf == 0.0:
            return []
        maps.append(color_map)
        match_confs.append(match_conf)

    merged, merge_conf = _merge_color_maps(maps)
    if merge_conf == 0.0:
        return []

    active_map = {k: v for k, v in merged.items() if k != v}
    if not active_map:
        return []

    hypotheses: list[Hypothesis] = []
    matching_conf = float(np.mean(match_confs)) if match_confs else 0.0

    candidates: list[RecolorProgram] = [RecolorProgram(selector, merged if selector.kind != "COLOR" else {selector.value: active_map[selector.value]}) for selector in _candidate_selectors_for_objects(train_pairs, profile) if selector.kind != "COLOR" or selector.value in active_map]

    for program in candidates:
        passed, total = verify_program(program, train_pairs, profile)
        selector_penalty = 0.0 if program.selector.kind == "ALL" else 0.08
        confidence = max(0.0, 0.90 * matching_conf - selector_penalty)
        hypotheses.append(
            Hypothesis(
                program=program,
                generator="RecolorGenerator",
                confidence=confidence,
                train_match=f"{passed}/{total}",
                match_rate=passed / total if total else 0.0,
                notes=[
                    f"Color map detected: {active_map}.",
                    f"matching_confidence={matching_conf:.2f}.",
                ],
            )
        )

    return sorted(hypotheses, key=lambda h: (h.match_rate, h.confidence), reverse=True)


def generate_delete_hypotheses(
    train_pairs: list[tuple[Grid, Grid]],
    inv: Invariants,
    profile: SegmentationProfile | None = None,
) -> list[Hypothesis]:
    if not _delete_allowed(inv):
        return []
    if profile is None:
        profile = SegmentationProfile()

    hypotheses: list[Hypothesis] = []
    for selector in _candidate_selectors_for_objects(train_pairs, profile):
        program = DeleteProgram(selector)
        passed, total = verify_program(program, train_pairs, profile)
        selector_bonus = 0.10 if selector.kind in {"LARGEST", "SMALLEST", "TOUCHING_BORDER", "NOT_TOUCHING_BORDER", "POSITION"} else 0.0
        selector_penalty = 0.15 if selector.kind == "ALL" else 0.0
        confidence = max(0.0, 0.75 + selector_bonus - selector_penalty)
        hypotheses.append(
            Hypothesis(
                program=program,
                generator="DeleteGenerator",
                confidence=confidence,
                train_match=f"{passed}/{total}",
                match_rate=passed / total if total else 0.0,
                notes=[f"Deletion selector candidate: {selector.to_dsl()}."],
            )
        )

    return sorted(hypotheses, key=lambda h: (h.match_rate, h.confidence), reverse=True)


def _residual_train_pairs(
    first_program,
    train_pairs: list[tuple[Grid, Grid]],
    profile: SegmentationProfile | None = None,
) -> list[tuple[Grid, Grid]] | None:
    residual: list[tuple[Grid, Grid]] = []
    for inp, expected in train_pairs:
        try:
            intermediate = execute(first_program, inp, profile)
        except InvalidProgram:
            return None
        residual.append((intermediate, expected))
    return residual


def _generate_single_step_hypotheses(
    train_pairs: list[tuple[Grid, Grid]],
    profile: SegmentationProfile | None = None,
) -> list[Hypothesis]:
    from .invariants import detect_invariants

    inv = detect_invariants(train_pairs, profile)
    hypotheses: list[Hypothesis] = []
    hypotheses.extend(generate_translation_hypotheses(train_pairs, inv, profile))
    hypotheses.extend(generate_recolor_hypotheses(train_pairs, inv, profile))
    hypotheses.extend(generate_delete_hypotheses(train_pairs, inv, profile))
    return hypotheses


def generate_sequence_hypotheses(
    train_pairs: list[tuple[Grid, Grid]],
    base_hypotheses: list[Hypothesis],
    profile: SegmentationProfile | None = None,
    *,
    top_k_first: int = 8,
) -> list[Hypothesis]:
    """Generate residual-guided two-step programs.

    This is intentionally not a full Cartesian-product search. It takes the
    strongest structural first-step hypotheses, applies each to train inputs,
    then runs one-step generators on the residual task:

        first(input) -> intermediate
        second(intermediate) -> expected_output

    In mixed tasks, a useful first step can have train_match=0/ N because the
    remaining transformation is still missing. Therefore selection is based on
    structural confidence, not only partial exact matches.
    """
    if profile is None:
        profile = SegmentationProfile()

    candidates = [h for h in base_hypotheses if h.match_rate < 1.0 and h.confidence > 0.0]
    candidates.sort(key=lambda h: (h.confidence, h.match_rate), reverse=True)
    sequence_hypotheses: list[Hypothesis] = []

    for first in candidates[:top_k_first]:
        residual = _residual_train_pairs(first.program, train_pairs, profile)
        if residual is None:
            continue

        second_candidates = _generate_single_step_hypotheses(residual, profile)
        for second in second_candidates:
            seq = SeqProgram(first=first.program, second=second.program)
            passed, total = verify_program(seq, train_pairs, profile)
            if passed == 0:
                continue
            confidence = max(0.0, 0.5 * first.confidence + 0.5 * second.confidence - 0.05)
            sequence_hypotheses.append(
                Hypothesis(
                    program=seq,
                    generator="SequenceGenerator",
                    confidence=confidence,
                    train_match=f"{passed}/{total}",
                    match_rate=passed / total if total else 0.0,
                    notes=[
                        "Residual-guided composition.",
                        f"First: {first.program_dsl}",
                        f"Second: {second.program_dsl}",
                    ],
                )
            )

    return sorted(sequence_hypotheses, key=lambda h: (h.match_rate, h.confidence), reverse=True)
