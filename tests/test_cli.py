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
