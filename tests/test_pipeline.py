from archaios.pipeline import run_pipeline


def test_translation_basic():
    result = run_pipeline(
        task_id="shift_right_2",
        train_pairs=[
            (
                [[0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0], [0, 0, 0, 0, 0, 0]],
            ),
            (
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 2, 2, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0]],
            ),
        ],
        test_input=[[0, 0, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    )
    assert result.status == "solved"
    assert result.program.startswith("SHIFT")
    assert result.prediction.tolist() == [[0, 0, 0, 0, 0, 0], [0, 0, 2, 2, 0, 0], [0, 0, 0, 0, 0, 0]]


def test_recolor_basic():
    result = run_pipeline(
        task_id="recolor_1_to_3",
        train_pairs=[
            ([[0, 1, 0], [1, 1, 0], [0, 0, 0]], [[0, 3, 0], [3, 3, 0], [0, 0, 0]]),
            ([[1, 0, 0], [1, 0, 0], [0, 0, 0]], [[3, 0, 0], [3, 0, 0], [0, 0, 0]]),
        ],
        test_input=[[0, 0, 1], [0, 1, 1], [0, 0, 0]],
    )
    assert result.status == "solved"
    assert result.program.startswith("RECOLOR")
    assert result.prediction.tolist() == [[0, 0, 3], [0, 3, 3], [0, 0, 0]]


def test_delete_smallest():
    result = run_pipeline(
        task_id="delete_smallest",
        train_pairs=[
            (
                [[0, 0, 0, 0, 0],
                 [0, 2, 2, 0, 3],
                 [0, 2, 2, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0],
                 [0, 2, 2, 0, 0],
                 [0, 2, 2, 0, 0],
                 [0, 0, 0, 0, 0]],
            ),
            (
                [[0, 0, 0, 0, 0],
                 [4, 0, 0, 2, 2],
                 [0, 0, 0, 2, 2],
                 [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 2],
                 [0, 0, 0, 2, 2],
                 [0, 0, 0, 0, 0]],
            ),
        ],
        test_input=[[0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0],
                    [0, 2, 2, 0, 0],
                    [0, 0, 0, 5, 0]],
    )
    assert result.status == "solved"
    assert result.generator == "DeleteGenerator"
    assert result.program.startswith("DELETE")
    assert result.prediction.tolist() == [[0, 0, 0, 0, 0],
                                          [0, 2, 2, 0, 0],
                                          [0, 2, 2, 0, 0],
                                          [0, 0, 0, 0, 0]]


def test_delete_single_pixels_shape_selector():
    result = run_pipeline(
        task_id="delete_single_pixels",
        train_pairs=[
            (
                [[0, 0, 0, 0, 0, 0],
                 [0, 2, 2, 0, 3, 0],
                 [0, 2, 2, 0, 0, 0],
                 [0, 0, 0, 4, 0, 0]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 2, 2, 0, 0, 0],
                 [0, 2, 2, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
            ),
            (
                [[0, 5, 0, 0, 0, 0],
                 [0, 0, 2, 2, 2, 0],
                 [0, 0, 2, 2, 2, 0],
                 [0, 0, 0, 0, 0, 6]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 2, 2, 2, 0],
                 [0, 0, 2, 2, 2, 0],
                 [0, 0, 0, 0, 0, 0]],
            ),
        ],
        test_input=[[0, 0, 7, 0, 0, 0],
                    [0, 2, 2, 2, 0, 0],
                    [0, 2, 2, 2, 0, 8],
                    [0, 0, 0, 0, 0, 0]],
    )
    assert result.status == "solved"
    assert result.generator == "DeleteGenerator"
    assert "shape=single_pixel" in result.program
    assert result.prediction.tolist() == [[0, 0, 0, 0, 0, 0],
                                          [0, 2, 2, 2, 0, 0],
                                          [0, 2, 2, 2, 0, 0],
                                          [0, 0, 0, 0, 0, 0]]


def test_fill_bbox_largest():
    result = run_pipeline(
        task_id="fill_bbox_largest",
        train_pairs=[
            (
                [[0, 0, 0, 0, 0],
                 [0, 2, 2, 2, 0],
                 [0, 2, 0, 2, 0],
                 [0, 2, 2, 2, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0],
                 [0, 2, 2, 2, 0],
                 [0, 2, 3, 2, 0],
                 [0, 2, 2, 2, 0],
                 [0, 0, 0, 0, 0]],
            ),
            (
                [[0, 2, 2, 2, 0],
                 [0, 2, 0, 2, 0],
                 [0, 2, 2, 2, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 2, 2, 2, 0],
                 [0, 2, 3, 2, 0],
                 [0, 2, 2, 2, 0],
                 [0, 0, 0, 0, 0]],
            ),
        ],
        test_input=[[0, 0, 0, 0, 0],
                    [2, 2, 2, 0, 0],
                    [2, 0, 2, 0, 0],
                    [2, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0]],
    )
    assert result.status == "solved"
    assert result.generator == "FillBBoxGenerator"
    assert result.program.startswith("FILL_BBOX")
    assert result.prediction.tolist() == [[0, 0, 0, 0, 0],
                                          [2, 2, 2, 0, 0],
                                          [2, 3, 2, 0, 0],
                                          [2, 2, 2, 0, 0],
                                          [0, 0, 0, 0, 0]]


def test_sequence_shift_then_recolor():
    result = run_pipeline(
        task_id="shift_then_recolor",
        train_pairs=[
            (
                [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 2, 2, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 2, 0, 3, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            ),
            (
                [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 2, 2, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 2, 2, 3, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            ),
        ],
        test_input=[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    )
    assert result.status == "solved"
    assert result.generator == "SequenceGenerator"
    assert result.program.startswith("SEQ")
    assert result.prediction.tolist() == [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 2, 2, 0, 0, 3, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]
