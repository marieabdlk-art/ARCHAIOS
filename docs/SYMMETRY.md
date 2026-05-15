# Symmetry Analyzer

ARCHAIOS includes a lightweight symmetry analysis layer.

This layer is intentionally analyzer-only. It does not generate or repair grids yet.

## Supported axes

- `vertical`
- `horizontal`

## Supported color modes

### `exact`

Mirrored cells must have exactly the same color.

```text
0 2 0 2 0
0 2 3 2 0
0 2 0 2 0
```

### `agnostic`

Mirrored cells only need to agree on foreground/background occupancy.
Color differences are ignored.

```text
0 2 0 3 0
0 4 0 5 0
0 2 0 3 0
```

This is useful for structural inspection where geometry matters more than color.

## Public API

```python
from archaios.symmetry import symmetry_report, symmetry_score, is_symmetric

report = symmetry_report(grid, axis="vertical", color_mode="agnostic")
print(report.score)
print(report.asymmetric_cells)
```

## Why analyzer before generator

Color-agnostic symmetry detection is not the same as symmetry generation.

Detection asks:

```text
Are the foreground/background structures mirrored?
```

Generation must also answer:

```text
If a mirrored cell is missing, which color should be drawn?
```

That second question is task-dependent. Therefore `CompleteSymmetryGenerator` is planned as a future layer, after the analyzer has stable tests and semantics.

## Current limitations

- No symmetry repair yet.
- No object-level symmetry yet.
- No rotational symmetry yet.
- No continuous or 3D symmetry.
- No connection to the main solver pipeline yet.

Planned next step:

```text
COMPLETE_SYMMETRY(selector, axis=vertical|horizontal, color_mode=exact)
```
