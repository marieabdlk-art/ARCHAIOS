# ARCHAIOS Architecture v0.1

## Core Pipeline

```text
Grid Parser
→ Object Extractor
→ Invariant Analyzer
→ Targeted Generators
→ Strict Executor
→ Exact Verifier
→ Ranker
→ Prediction + Trace
```

## Design decisions

### TrainMatch is a hard filter

Exact train match is required for the normal `solved` status. Partial candidates are ranked only in fallback mode.

### Strict SHIFT

`SHIFT` does not clip pixels and does not overwrite non-selected objects.

Invalid cases:

- selected pixels move out of bounds;
- target pixels collide with non-selected active pixels;
- selector is empty.

### Matching is generator-specific

Translation and recolor use different object-matching strategies:

- translation prioritizes same color + same shape;
- recolor prioritizes same position + same shape, because color is expected to change.

### Segmentation profile

Segmentation is explicit:

- `COLOR`: connected components by color;
- `SPATIAL`: connected components over all non-background cells;
- `connectivity`: 4 or 8.

The default pipeline tries `COLOR+4` first and can fallback to alternative profiles.

## Known limitations

- no `SEQ` composition yet;
- no topology layer yet;
- no real ARC dataset loader yet;
- no robustness evaluator yet;
- no learned ranker.
