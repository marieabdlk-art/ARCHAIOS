import numpy as np

from archaios.symmetry import find_asymmetric_cells, is_symmetric, occupancy_grid, symmetry_report, symmetry_score


def test_vertical_exact_symmetry():
    grid = np.array([
        [0, 2, 0, 2, 0],
        [0, 2, 3, 2, 0],
        [0, 2, 0, 2, 0],
    ])

    report = symmetry_report(grid, axis="vertical", color_mode="exact")
    assert report.is_symmetric
    assert report.score == 1.0
    assert report.asymmetric_cells == []
    assert is_symmetric(grid, axis="vertical", color_mode="exact")


def test_horizontal_exact_symmetry():
    grid = np.array([
        [0, 0, 2, 0],
        [0, 3, 3, 0],
        [0, 0, 2, 0],
    ])

    assert symmetry_score(grid, axis="horizontal", color_mode="exact") == 1.0
    assert is_symmetric(grid, axis="horizontal", color_mode="exact")


def test_vertical_color_agnostic_symmetry():
    grid = np.array([
        [0, 2, 0, 3, 0],
        [0, 4, 0, 5, 0],
        [0, 2, 0, 3, 0],
    ])

    assert symmetry_score(grid, axis="vertical", color_mode="exact") < 1.0
    assert symmetry_score(grid, axis="vertical", color_mode="agnostic") == 1.0
    assert is_symmetric(grid, axis="vertical", color_mode="agnostic")


def test_broken_symmetry_reports_asymmetric_cells():
    grid = np.array([
        [0, 2, 0, 2, 0],
        [0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0],
    ])

    report = symmetry_report(grid, axis="vertical", color_mode="exact")
    assert report.score < 1.0
    assert (1, 1) in report.asymmetric_cells
    assert (1, 3) in report.asymmetric_cells
    assert find_asymmetric_cells(grid, axis="vertical", color_mode="exact") == report.asymmetric_cells


def test_occupancy_grid():
    grid = np.array([
        [0, 2, 0],
        [3, 0, 4],
    ])
    assert occupancy_grid(grid).tolist() == [
        [0, 1, 0],
        [1, 0, 1],
    ]
