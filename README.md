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
- expanded selectors: `ALL`, `COLOR`, `SIZE`, `LARGEST`, `SMALLEST`, `TOUCHING_BORDER`, `NOT_TOUCHING_BORDER`, `POSITION`, `SHAPE`;
- shape selectors: `square`, `rectangle`, `line_horizontal`, `line_vertical`, `single_pixel`;
- translation hypotheses;
- recolor hypotheses;
- deletion hypotheses;
- bbox-fill hypotheses;
- `SeqProgram` composition via residual-guided search;
- strict `SHIFT` semantics: no clipping, no collisions with non-selected objects;
- exact-match verification;
- traceable pipeline;
- CLI solver for JSON task files;
- evaluation harness for task directories;
- ARC-style `test[0].output` scoring when available.

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

Prepare a local ARC subset:

```bash
python scripts/prepare_arc_subset.py \
  --arc-root /path/to/ARC-AGI \
  --output datasets/arc_easy_subset \
  --task-list docs/arc_easy_task_ids.txt
```

Evaluate the prepared ARC subset:

```bash
archaios eval datasets/arc_easy_subset/tasks \
  --json \
  --output reports/arc_eval.json
```

See [`docs/ARC_EVAL.md`](docs/ARC_EVAL.md) for details.

The mixed example task should produce a residual-guided composition similar to:

```text
SEQ(SHIFT(OBJECTS(color=2), dx=2, dy=0), RECOLOR(OBJECTS(color=1), {1->3}))
```

The deletion example should produce a selector-based program similar to:

```text
DELETE(SELECT_SMALLEST(OBJECTS()))
```

The shape-selector example should produce a program similar to:

```text
DELETE(OBJECTS(shape=single_pixel))
```

The bbox-fill example should produce a program similar to:

```text
FILL_BBOX(SELECT_LARGEST(OBJECTS()), color=3)
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
    {"input": [[1, 0]], "output": [[2, 0]]}
  ]
}
```

`test[0].output` is optional. If present, CLI reports `test_match` and eval reports aggregate `test_accuracy`.

## Status

This is an MVP scaffold, not a full ARC solver.

Implemented:

- `TranslationGenerator`
- `RecolorGenerator`
- `DeleteGenerator`
- `FillBBoxGenerator`
- `SequenceGenerator`
- shape-based selectors
- `InvariantAnalyzer`
- `Pipeline`
- `archaios solve` CLI
- `archaios eval` CLI
- ARC-style test scoring

Planned:

- deeper selector language: inside/outside selectors, relation selectors;
- topology layer: inside/frame/hole/border-touching;
- robustness evaluator via metamorphic consistency;
- evaluation on curated ARC-Easy subset.
