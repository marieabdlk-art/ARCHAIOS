from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .executor import InvalidProgram, execute
from .generators import (
    generate_recolor_hypotheses,
    generate_sequence_hypotheses,
    generate_translation_hypotheses,
)
from .invariants import detect_invariants, format_invariants
from .objects import parse_grid
from .types import Grid, Hypothesis, Invariants, SegmentationMode, SegmentationProfile


@dataclass
class RankedHypothesis:
    hypothesis: Hypothesis
    score: float
    rank: int


@dataclass
class PipelineResult:
    task_id: str
    prediction: Optional[Grid]
    program: str
    confidence: float
    train_match: str
    match_rate: float
    score: float
    generator: str
    invariants: Invariants
    all_hypotheses: list[RankedHypothesis]
    trace: list[str]
    status: str
    profile: SegmentationProfile


def program_length(program_dsl: str) -> int:
    return program_dsl.count("(") + program_dsl.count(",") + 1


def special_case_penalty(h: Hypothesis) -> float:
    dsl = h.program_dsl
    penalty = 0.0
    if "size=" in dsl:
        penalty += 0.05
    if "color=" in dsl and "ALL" not in dsl:
        penalty += 0.02
    if dsl.startswith("SEQ"):
        penalty += 0.03
    return penalty


def rank_exact(hypotheses: list[Hypothesis]) -> list[RankedHypothesis]:
    ranked = []
    for h in hypotheses:
        simplicity = 1.0 - min(program_length(h.program_dsl) / 14.0, 1.0)
        score = 0.65 * h.confidence + 0.25 * simplicity - 0.10 * special_case_penalty(h)
        ranked.append(RankedHypothesis(hypothesis=h, score=round(score, 4), rank=0))
    ranked.sort(key=lambda item: item.score, reverse=True)
    for idx, item in enumerate(ranked):
        item.rank = idx + 1
    return ranked


def rank_fallback(hypotheses: list[Hypothesis]) -> list[RankedHypothesis]:
    ranked = []
    for h in hypotheses:
        score = 1.00 * h.match_rate + 0.20 * h.confidence - 0.05 * program_length(h.program_dsl)
        ranked.append(RankedHypothesis(hypothesis=h, score=round(score, 4), rank=0))
    ranked.sort(key=lambda item: item.score, reverse=True)
    for idx, item in enumerate(ranked):
        item.rank = idx + 1
    return ranked


def build_trace(inv: Invariants, ranked: list[RankedHypothesis], selected: RankedHypothesis | None) -> list[str]:
    trace = []
    trace.extend(format_invariants(inv).split("\n"))
    trace.append("")
    trace.append("=== Hypotheses ===")
    trace.append(f"Generated: {len(ranked)}")
    for r in ranked:
        h = r.hypothesis
        marker = "SELECTED" if selected and r.rank == selected.rank else f"rank={r.rank}"
        trace.append(
            f"[{marker}] {h.program_dsl} | train={h.train_match} | score={r.score:.4f} | {h.generator}"
        )
        for note in h.notes:
            trace.append(f"  · {note}")
    return trace


def run_pipeline(
    task_id: str,
    train_pairs: list[tuple[list[list[int]], list[list[int]]]],
    test_input: list[list[int]],
    profile: SegmentationProfile | None = None,
    try_profiles: bool = True,
    allow_composition: bool = True,
) -> PipelineResult:
    parsed_train = [(parse_grid(inp), parse_grid(out)) for inp, out in train_pairs]
    test_grid = parse_grid(test_input)

    profiles = [profile] if profile is not None else [SegmentationProfile()]
    if try_profiles and profile is None:
        profiles += [
            SegmentationProfile(mode=SegmentationMode.COLOR, connectivity=8),
            SegmentationProfile(mode=SegmentationMode.SPATIAL, connectivity=4),
            SegmentationProfile(mode=SegmentationMode.SPATIAL, connectivity=8),
        ]

    best_result: PipelineResult | None = None

    for current_profile in profiles:
        inv = detect_invariants(parsed_train, current_profile)
        hypotheses: list[Hypothesis] = []
        hypotheses.extend(generate_translation_hypotheses(parsed_train, inv, current_profile))
        hypotheses.extend(generate_recolor_hypotheses(parsed_train, inv, current_profile))

        exact_single = [h for h in hypotheses if h.match_rate == 1.0]
        if allow_composition and not exact_single and hypotheses:
            hypotheses.extend(generate_sequence_hypotheses(parsed_train, hypotheses, current_profile))

        if not hypotheses:
            result = PipelineResult(
                task_id=task_id,
                prediction=None,
                program="—",
                confidence=0.0,
                train_match="0/0",
                match_rate=0.0,
                score=0.0,
                generator="—",
                invariants=inv,
                all_hypotheses=[],
                trace=build_trace(inv, [], None),
                status="failed",
                profile=current_profile,
            )
        else:
            exact = [h for h in hypotheses if h.match_rate == 1.0]
            if exact:
                ranked = rank_exact(exact)
                selected = ranked[0]
                status = "solved"
            else:
                ranked = rank_fallback(hypotheses)
                selected = ranked[0]
                status = "fallback_low_confidence"

            try:
                prediction = execute(selected.hypothesis.program, test_grid, current_profile)
            except InvalidProgram:
                prediction = None
                status = "failed"

            result = PipelineResult(
                task_id=task_id,
                prediction=prediction,
                program=selected.hypothesis.program_dsl,
                confidence=selected.hypothesis.confidence,
                train_match=selected.hypothesis.train_match,
                match_rate=selected.hypothesis.match_rate,
                score=selected.score,
                generator=selected.hypothesis.generator,
                invariants=inv,
                all_hypotheses=ranked,
                trace=build_trace(inv, ranked, selected),
                status=status,
                profile=current_profile,
            )

        if best_result is None:
            best_result = result
        elif result.status == "solved" and best_result.status != "solved":
            best_result = result
        elif result.status == best_result.status and result.score > best_result.score:
            best_result = result

        if result.status == "solved":
            break

    assert best_result is not None
    return best_result


def result_to_dict(result: PipelineResult) -> dict:
    return {
        "task_id": result.task_id,
        "status": result.status,
        "profile": result.profile.label(),
        "program": result.program,
        "train_match": result.train_match,
        "confidence": result.confidence,
        "score": result.score,
        "generator": result.generator,
        "prediction": result.prediction.tolist() if result.prediction is not None else None,
        "trace": result.trace,
    }
