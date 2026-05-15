# ARCHAIOS Roadmap

## v0.1 — Current MVP

Implemented:

- grid parsing;
- object extraction;
- segmentation profiles: `COLOR/SPATIAL`, `4/8 connectivity`;
- invariant analysis;
- translation generator;
- recolor generator;
- strict executor;
- exact verifier;
- basic pipeline trace;
- unit tests for translation and recolor.

## v0.2 — Residual-guided composition

Implemented:

- `SeqProgram`;
- residual-guided two-step composition;
- `SHIFT → RECOLOR` support;
- verification over composed programs;
- mixed-task unit test.

## v0.3 — Strict symbolic core improvements

Planned:

- structured selector expansion: largest/smallest/touching-border;
- more explicit failure taxonomy;
- better fallback reporting;
- JSON trace schema;
- CLI entrypoint.

## v0.4 — Topology layer

Planned:

- frame detection;
- inside/outside relationships;
- holes;
- border-touching selectors;
- `FILL_HOLES`;
- `COLOR_INSIDE`;
- `REMOVE_BORDER_TOUCHING_OBJECTS`.

## v0.5 — Evaluation harness

Planned:

- curated ARC-Easy subset loader;
- baseline comparison;
- ablation: no-invariants / no-profile-fallback / no-ranker;
- per-task failure reports.
