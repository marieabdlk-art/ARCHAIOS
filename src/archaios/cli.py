from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import result_to_dict, run_pipeline


def _load_task(path: Path) -> tuple[str, list[tuple[list[list[int]], list[list[int]]]], list[list[int]]]:
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

    return task_id, train_pairs, first_test["input"]


def _cmd_solve(args: argparse.Namespace) -> int:
    task_path = Path(args.task)
    task_id, train_pairs, test_input = _load_task(task_path)

    result = run_pipeline(
        task_id=task_id,
        train_pairs=train_pairs,
        test_input=test_input,
        try_profiles=not args.no_profile_fallback,
        allow_composition=not args.no_composition,
    )

    payload = result_to_dict(result)

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
        print(f"Confidence : {payload['confidence']:.3f}")
        print(f"Score      : {payload['score']:.4f}")
        print("Prediction :")
        print(json.dumps(payload["prediction"], ensure_ascii=False))
        if args.trace:
            print("\nTrace:")
            for line in payload["trace"]:
                print(line)

    return 0 if result.status == "solved" else 1


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
