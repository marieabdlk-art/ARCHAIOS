# ARCHAIOS

**ARCHAIOS** is an interpretable ARC-style reasoning prototype based on object-centric program synthesis.

Instead of directly predicting an output grid, ARCHAIOS:

1. parses grids into objects;
2. detects preserved and changed invariants;
3. activates targeted hypothesis generators;
4. produces executable DSL-like programs;
5. verifies candidates on train pairs;
6. composes residual-guided two-step programs when one operation is insufficient;
7. ranks exact programs separately from fallback candidates;
8. returns prediction + reasoning trace.

Current MVP supports:

- object extraction with configurable segmentation profiles;
- invariant analysis;
- translation hypotheses;
- recolor hypotheses;
- `SeqProgram` composition via residual-guided search;
- strict `SHIFT` semantics: no clipping, no collisions with non-selected objects;
- exact-match verification;
- traceable pipeline;
- CLI solver for JSON task files;
- evaluation harness for task directories.

## Install

```bash
pip install -e ".[dev]"
```

## CLI usage

Solve a task JSON:

```bash
archaios solve examples/tasks/shift_then_recolor.json --trace
```

Print full JSON result:

```bash
archaios solve examples/tasks/shift_then_recolor.json --json
```

Write result to a file:

```bash
archaios solve examples/tasks/shift_then_recolor.json --json --output result.json
```

Evaluate all JSON tasks in a directory:

```bash
archaios eval examples/tasks
```

Write an evaluation report:

```bash
archaios eval examples/tasks --json --output reports/eval.json
```

The mixed example task should produce a residual-guided composition similar to:

```text
SEQ(SHIFT(OBJECTS(color=2), dx=2, dy=0), RECOLOR(OBJECTS(color=1), {1->3}))
```

## Python example

```bash
python examples/run_basic.py
```

## Run tests

```bash
pytest
```

## Task JSON format

```json
{
  "task_id": "example",
  "train": [
    {"input": [[0, 1]], "output": [[0, 2]]}
  ],
  "test": [
    {"input": [[1, 0]]}
  ]
}
```

## Status

This is an MVP scaffold, not a full ARC solver.

Implemented:

- `TranslationGenerator`
- `RecolorGenerator`
- `SequenceGenerator`
- `InvariantAnalyzer`
- `Pipeline`
- `archaios solve` CLI
- `archaios eval` CLI

Planned:

- deeper selector language: largest/smallest/touching-border;
- topology layer: inside/frame/hole/border-touching;
- robustness evaluator via metamorphic consistency;
- evaluation on curated ARC-Easy subset.
