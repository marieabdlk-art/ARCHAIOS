from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_TASK_IDS = [
    # Intentionally small placeholder list. Replace or extend with ARC task ids
    # after local inspection. The script never downloads or vendors ARC data.
]


def _find_task_file(arc_root: Path, task_id: str) -> Path | None:
    candidates = [
        arc_root / "data" / "training" / f"{task_id}.json",
        arc_root / "data" / "evaluation" / f"{task_id}.json",
        arc_root / "training" / f"{task_id}.json",
        arc_root / "evaluation" / f"{task_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(arc_root.glob(f"**/{task_id}.json"))
    return matches[0] if matches else None


def _validate_arc_task(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "train" not in data or "test" not in data:
        raise ValueError(f"{path} is not ARC-like: missing train/test")
    if not isinstance(data["train"], list) or not isinstance(data["test"], list):
        raise ValueError(f"{path} is not ARC-like: train/test must be lists")


def prepare_subset(arc_root: Path, output_dir: Path, task_ids: list[str]) -> list[dict]:
    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    for task_id in task_ids:
        source = _find_task_file(arc_root, task_id)
        if source is None:
            missing.append(task_id)
            continue
        _validate_arc_task(source)
        destination = tasks_dir / f"{task_id}.json"
        shutil.copyfile(source, destination)
        copied.append({"task_id": task_id, "source": str(source), "destination": str(destination)})

    readme = output_dir / "README.md"
    readme.write_text(
        "# ARC Easy Subset\n\n"
        "This directory is generated locally by `scripts/prepare_arc_subset.py`.\n\n"
        "The ARC dataset is not vendored in this repository.\n"
        "Provide a local ARC checkout and copy only the selected JSON tasks.\n\n"
        f"Copied tasks: {len(copied)}\n\n"
        "## Run\n\n"
        "```bash\n"
        "archaios eval datasets/arc_easy_subset/tasks --json --output reports/arc_eval.json\n"
        "```\n",
        encoding="utf-8",
    )

    return [{"copied": copied, "missing": missing}]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a local ARC subset for ARCHAIOS evaluation")
    parser.add_argument("--arc-root", required=True, help="path to local ARC/ARC-AGI repository")
    parser.add_argument("--output", default="datasets/arc_easy_subset", help="output subset directory")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="ARC task id to copy; can be repeated")
    parser.add_argument("--task-list", help="text file with one ARC task id per line")
    args = parser.parse_args()

    task_ids = list(DEFAULT_TASK_IDS)
    if args.task_ids:
        task_ids.extend(args.task_ids)
    if args.task_list:
        task_list_path = Path(args.task_list)
        task_ids.extend(
            line.strip()
            for line in task_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    task_ids = sorted(set(task_ids))
    if not task_ids:
        raise SystemExit("No task ids provided. Use --task-id or --task-list.")

    report = prepare_subset(Path(args.arc_root), Path(args.output), task_ids)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
