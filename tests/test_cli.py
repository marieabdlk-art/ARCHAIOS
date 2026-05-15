import json
import subprocess
import sys
from pathlib import Path


def test_cli_solve_json(tmp_path: Path):
    task_path = Path("examples/tasks/shift_then_recolor.json")
    result_path = tmp_path / "result.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "archaios.cli",
            "solve",
            str(task_path),
            "--json",
            "--output",
            str(result_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "solved"
    assert payload["program"].startswith("SEQ")
    assert payload["prediction"] == [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]


def test_cli_eval_json(tmp_path: Path):
    report_path = tmp_path / "eval.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "archaios.cli",
            "eval",
            "examples/tasks",
            "--json",
            "--output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] >= 3
    assert payload["summary"]["solved"] >= 3
    assert payload["summary"]["solve_rate"] == 1.0
    task_ids = {row["task_id"] for row in payload["results"]}
    assert {"shift_right_2", "recolor_1_to_3", "shift_then_recolor"} <= task_ids
