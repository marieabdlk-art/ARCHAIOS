# ARC Evaluation

ARCHAIOS can evaluate ARC-style JSON tasks.

A task is expected to use the standard structure:

```json
{
  "train": [
    {"input": [[0]], "output": [[1]]}
  ],
  "test": [
    {"input": [[0]], "output": [[1]]}
  ]
}
```

`test[0].output` is optional. If present, ARCHAIOS reports `test_match` and aggregate `test_accuracy`.

## Why ARC data is not committed

The repository does not vendor the ARC dataset. Keep the public ARC repository as a local checkout and prepare a small local subset for experiments.

## Prepare a local subset

```bash
python scripts/prepare_arc_subset.py \
  --arc-root /path/to/ARC-AGI \
  --output datasets/arc_easy_subset \
  --task-id <task_id_1> \
  --task-id <task_id_2>
```

Or provide a text file:

```bash
python scripts/prepare_arc_subset.py \
  --arc-root /path/to/ARC-AGI \
  --output datasets/arc_easy_subset \
  --task-list docs/arc_easy_task_ids.txt
```

## Run evaluation

```bash
archaios eval datasets/arc_easy_subset/tasks \
  --json \
  --output reports/arc_eval.json
```

The report contains:

- `solve_rate`: fraction of tasks where an exact train-fitting program was found;
- `test_accuracy`: fraction of evaluable tasks where prediction matched `test[0].output`;
- executable program DSL;
- generator name;
- train/test match fields;
- full reasoning trace.

## Important limitations

`FILL_BBOX` is not a topology primitive. It fills background cells inside the selected object's bounding box. It does not detect enclosed regions, holes, or true inside/outside topology.

`SHIFT` currently uses strict boundary semantics: no clipping, no collisions, no out-of-bounds pixels. Some real ARC tasks may require a future boundary mode such as `drop` or `clip`.
