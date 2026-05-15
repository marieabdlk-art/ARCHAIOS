# ARCHAIOS

**ARCHAIOS** is an interpretable ARC-style reasoning prototype based on object-centric program synthesis.

Instead of directly predicting an output grid, ARCHAIOS:

1. parses grids into objects;
2. detects preserved and changed invariants;
3. activates targeted hypothesis generators;
4. produces executable DSL-like programs;
5. verifies candidates on train pairs;
6. ranks exact programs separately from fallback candidates;
7. returns prediction + reasoning trace.

Current MVP supports:

- object extraction with configurable segmentation profiles;
- invariant analysis;
- translation hypotheses;
- recolor hypotheses;
- strict `SHIFT` semantics: no clipping, no collisions with non-selected objects;
- exact-match verification;
- traceable pipeline.

## Install

```bash
pip install -e .
```

## Run examples

```bash
python examples/run_basic.py
```

## Run tests

```bash
pytest
```

## Status

This is an MVP scaffold, not a full ARC solver.

Implemented:

- `TranslationGenerator`
- `RecolorGenerator`
- `InvariantAnalyzer`
- `Pipeline`

Planned:

- `SeqProgram` and residual-guided composition;
- topology layer: inside/frame/hole/border-touching;
- robustness evaluator via metamorphic consistency;
- evaluation harness on curated ARC-Easy subset.
