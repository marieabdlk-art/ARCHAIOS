import json

from archaios.pipeline import result_to_dict, run_pipeline


def show(result):
    print("=" * 70)
    print(f"Task: {result.task_id} | Status: {result.status} | Profile: {result.profile.label()}")
    print(f"Program: {result.program}")
    print(f"Train match: {result.train_match} | Score: {result.score:.4f}")
    print("Prediction:")
    print(result.prediction)
    print("\nTrace:")
    for line in result.trace:
        print(line)


if __name__ == "__main__":
    shift = run_pipeline(
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
    show(shift)

    recolor = run_pipeline(
        task_id="recolor_1_to_3",
        train_pairs=[
            ([[0, 1, 0], [1, 1, 0], [0, 0, 0]], [[0, 3, 0], [3, 3, 0], [0, 0, 0]]),
            ([[1, 0, 0], [1, 0, 0], [0, 0, 0]], [[3, 0, 0], [3, 0, 0], [0, 0, 0]]),
        ],
        test_input=[[0, 0, 1], [0, 1, 1], [0, 0, 0]],
    )
    show(recolor)

    print("\nJSON export:")
    print(json.dumps(result_to_dict(recolor), indent=2, ensure_ascii=False))
