# Evaluation Protocol

ARCHAIOS is currently a prototype. Evaluation should measure not only solved tasks, but also search quality and interpretability.

## Dataset plan

Start with a manually curated ARC-Easy subset:

- 5–10 translation tasks;
- 5–10 recolor tasks;
- 5–10 deletion/copy tasks;
- 5–10 simple topology tasks.

The current implementation only targets translation and recolor.

## Metrics

- `solved_tasks`: exact test predictions;
- `train_pass_rate`: candidate verification on train pairs;
- `hypotheses_generated`: number of candidates per task;
- `fallback_rate`: fraction of tasks without exact candidate;
- `program_length`: proxy for rule simplicity;
- `trace_completeness`: whether invariant, generator, verification and rank information are logged;
- `failure_type`: segmentation failure, primitive missing, wrong invariant, ambiguous rule, search timeout.

## Baselines

Planned baselines:

1. naive brute-force DSL;
2. ARCHAIOS without invariant-guided generator selection;
3. ARCHAIOS without segmentation profile fallback;
4. ARCHAIOS full symbolic core.

## Ablation plan

| Variant | Disabled component | Purpose |
|---|---|---|
| `base` | profile fallback | tests segmentation sensitivity |
| `no-inv` | invariant pruning | measures search expansion |
| `no-rank` | exact ranker | measures over-specific candidates |
| `full` | none | final prototype |
