from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .pipeline import result_to_dict, run_pipeline


def _load_task(
    path: Path,
) -> tuple[
    str,
    list[tuple[list[list[int]], list[list[int]]]],
    list[list[int]],
    list[list[int]] | None,
]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    task_id = data.get("task_id", path.stem)
    train_raw = data.get("train")
    test_raw = data.get("test")

    if not isinstance(train_raw, list) or not train_raw:
        raise ValueError("task JSON must contain non-empty 'train' list")
    if not isinstance(test_raw, list) or not test_raw:
        raise ValueError("task JSON must contain non-empty 'test' list")

    train_pairs: list[tuple[list[list[int]], list[list[int]]]] = []
    for idx, pair in enumerate(train_raw):
        if "input" not in pair or "output" not in pair:
            raise ValueError(f"train[{idx}] must contain 'input' and 'output'")
        train_pairs.append((pair["input"], pair["output"]))

    first_test = test_raw[0]
    if "input" not in first_test:
        raise ValueError("test[0] must contain 'input'")

    expected_test_output = first_test.get("output")
    return task_id, train_pairs, first_test["input"], expected_test_output


def _test_match(prediction, expected_test_output: list[list[int]] | None) -> str | None:
    if expected_test_output is None or prediction is None:
        return None
    expected = np.array(expected_test_output, dtype=int)
    return "1/1" if np.array_equal(prediction, expected) else "0/1"


def _solve_path(task_path: Path, *, no_profile_fallback: bool, no_composition: bool) -> dict:
    task_id, train_pairs, test_input, expected_test_output = _load_task(task_path)
    result = run_pipeline(
        task_id=task_id,
        train_pairs=train_pairs,
        test_input=test_input,
        try_profiles=not no_profile_fallback,
        allow_composition=not no_composition,
    )
    payload = result_to_dict(result)
    payload["path"] = str(task_path)
    payload["test_match"] = _test_match(result.prediction, expected_test_output)
    return payload


def _cmd_solve(args: argparse.Namespace) -> int:
    payload = _solve_path(
        Path(args.task),
        no_profile_fallback=args.no_profile_fallback,
        no_composition=args.no_composition,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Task       : {payload['task_id']}")
        print(f"Status     : {payload['status']}")
        print(f"Profile    : {payload['profile']}")
        print(f"Program    : {payload['program']}")
        print(f"Train match: {payload['train_match']}")
        if payload.get("test_match") is not None:
            print(f"Test match : {payload['test_match']}")
        print(f"Confidence : {payload['confidence']:.3f}")
        print(f"Score      : {payload['score']:.4f}")
        print("Prediction :")
        print(json.dumps(payload["prediction"], ensure_ascii=False))
        if args.trace:
            print("\nTrace:")
            for line in payload["trace"]:
                print(line)

    return 0 if payload["status"] == "solved" else 1


def _task_files(task_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(path for path in task_dir.glob(pattern) if path.is_file())


def _summary(rows: list[dict]) -> dict:
    total = len(rows)
    solved = sum(1 for row in rows if row["status"] == "solved")
    fallback = sum(1 for row in rows if row["status"] == "fallback_low_confidence")
    failed = total - solved - fallback
    test_evaluable = sum(1 for row in rows if row.get("test_match") is not None)
    test_solved = sum(1 for row in rows if row.get("test_match") == "1/1")
    return {
        "total": total,
        "solved": solved,
        "fallback_low_confidence": fallback,
        "failed": failed,
        "solve_rate": solved / total if total else 0.0,
        "test_evaluable": test_evaluable,
        "test_solved": test_solved,
        "test_accuracy": test_solved / test_evaluable if test_evaluable else None,
    }


def _print_eval_table(rows: list[dict], summary: dict) -> None:
    headers = ["task_id", "status", "generator", "train", "test", "score", "program"]
    widths = {
        "task_id": max([len("task_id")] + [len(str(r["task_id"])) for r in rows]),
        "status": max([len("status")] + [len(str(r["status"])) for r in rows]),
        "generator": max([len("generator")] + [len(str(r["generator"])) for r in rows]),
        "train": max([len("train")] + [len(str(r["train_match"])) for r in rows]),
        "test": max([len("test")] + [len(str(r.get("test_match") or "—")) for r in rows]),
        "score": len("score"),
        "program": max([len("program")] + [min(len(str(r["program"])), 60) for r in rows]),
    }

    print(
        f"{headers[0]:<{widths['task_id']}}  "
        f"{headers[1]:<{widths['status']}}  "
        f"{headers[2]:<{widths['generator']}}  "
        f"{headers[3]:<{widths['train']}}  "
        f"{headers[4]:<{widths['test']}}  "
        f"{headers[5]:>{widths['score']}}  "
        f"{headers[6]}"
    )
    print("-" * (sum(widths.values()) + 14))

    for row in rows:
        program = str(row["program"])
        if len(program) > 60:
            program = program[:57] + "..."
        print(
            f"{row['task_id']:<{widths['task_id']}}  "
            f"{row['status']:<{widths['status']}}  "
            f"{row['generator']:<{widths['generator']}}  "
            f"{row['train_match']:<{widths['train']}}  "
            f"{(row.get('test_match') or '—'):<{widths['test']}}  "
            f"{row['score']:>{widths['score']}.3f}  "
            f"{program}"
        )

    print("\nSummary:")
    print(f"  total   : {summary['total']}")
    print(f"  solved  : {summary['solved']}")
    print(f"  fallback: {summary['fallback_low_confidence']}")
    print(f"  failed  : {summary['failed']}")
    print(f"  solve_rate: {summary['solve_rate']:.2%}")
    if summary["test_accuracy"] is not None:
        print(f"  test_accuracy: {summary['test_accuracy']:.2%} ({summary['test_solved']}/{summary['test_evaluable']})")


def _cmd_eval(args: argparse.Namespace) -> int:
    task_dir = Path(args.task_dir)
    if not task_dir.exists() or not task_dir.is_dir():
        raise ValueError(f"task directory not found: {task_dir}")

    files = _task_files(task_dir, recursive=args.recursive)
    if not files:
        raise ValueError(f"no JSON task files found in {task_dir}")

    rows = []
    errors = []
    for path in files:
        try:
            rows.append(
                _solve_path(
                    path,
                    no_profile_fallback=args.no_profile_fallback,
                    no_composition=args.no_composition,
                )
            )
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})

    summary = _summary(rows)
    report = {
        "task_dir": str(task_dir),
        "summary": summary,
        "results": rows,
        "errors": errors,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_eval_table(rows, summary)
        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"  · {error['path']}: {error['error']}")

    return 0 if errors == [] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archaios",
        description="ARCHAIOS ARC-style symbolic reasoning prototype",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="solve a JSON task")
    solve.add_argument("task", help="path to task JSON")
    solve.add_argument("--json", action="store_true", help="print full JSON result")
    solve.add_argument("--trace", action="store_true", help="print reasoning trace in text mode")
    solve.add_argument("--output", "-o", help="write JSON result to path")
    solve.add_argument("--no-profile-fallback", action="store_true", help="disable alternative segmentation profiles")
    solve.add_argument("--no-composition", action="store_true", help="disable residual-guided SeqProgram composition")
    solve.set_defaults(func=_cmd_solve)

    eval_cmd = sub.add_parser("eval", help="evaluate all JSON tasks in a directory")
    eval_cmd.add_argument("task_dir", help="directory containing task JSON files")
    eval_cmd.add_argument("--recursive", "-r", action="store_true", help="search for JSON files recursively")
    eval_cmd.add_argument("--json", action="store_true", help="print full JSON report")
    eval_cmd.add_argument("--output", "-o", help="write JSON report to path")
    eval_cmd.add_argument("--no-profile-fallback", action="store_true", help="disable alternative segmentation profiles")
    eval_cmd.add_argument("--no-composition", action="store_true", help="disable residual-guided SeqProgram composition")
    eval_cmd.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"archaios: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
