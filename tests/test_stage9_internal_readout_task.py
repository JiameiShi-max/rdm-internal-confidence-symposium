import os
import tempfile

import numpy as np

from generate_stage9_internal_teacher_dataset import save_internal_teacher_dataset
from sure_target_stage9_internal_readout import (
    RDM_SureTarget_InternalReadoutDatasetTask,
    build_arg_parser,
    build_post_go_output_records,
    categorical_soft_postgo_loss,
    categorical_soft_postgo_loss_components,
    configure_student_action_objective,
    load_stage9_dataset,
)


def make_test_dataset(path):
    x = np.zeros((3, 20, 4), dtype=np.float32)
    y = np.zeros((3, 20, 4), dtype=np.float32)
    mask = np.ones((3, 20, 4), dtype=np.float32)
    y_internal = np.zeros((3, 20, 4), dtype=np.float32)
    mask_internal = np.ones((3, 20, 4), dtype=np.float32)
    y_internal[:, :12, 0] = 1.0
    y_internal[0, 12:, 1] = 0.8
    y_internal[1, 12:, 2] = 0.4
    y_internal[2, 12:, 1] = 1.0
    y_internal[:, 12:, 3] = np.array([0.2, 0.6, 0.0], dtype=np.float32)[:, None]
    trial_info = [
        {
            "dir_choice": 0,
            "sure_available": True,
            "fixation_end": 2,
            "stimulus_end": 10,
            "ts_onset": 11,
            "delay_end": 12,
            "coh": 0.032,
        },
        {
            "dir_choice": 1,
            "sure_available": True,
            "fixation_end": 2,
            "stimulus_end": 10,
            "ts_onset": 11,
            "delay_end": 12,
            "coh": 0.128,
        },
        {
            "dir_choice": 0,
            "sure_available": False,
            "fixation_end": 2,
            "stimulus_end": 10,
            "ts_onset": 11,
            "delay_end": 12,
            "coh": 0.512,
        },
    ]
    teacher_records = {
        "teacher_internal_margin": np.array([0.5, 1.0, 2.0], dtype=float),
        "teacher_sure_strength": np.array([0.2, 0.6, 0.0], dtype=float),
        "teacher_expected_direction_value": np.array([0.4, 0.7, 0.9], dtype=float),
        "teacher_direction_success_proxy": np.array([0.4, 0.7, 0.9], dtype=float),
        "external_sensory_margin": np.array([1.0, 2.0, 3.0], dtype=float),
        "external_sure_strength": np.array([0.8, 0.3, 0.0], dtype=float),
    }
    save_internal_teacher_dataset(
        path,
        x,
        y,
        mask,
        trial_info,
        teacher_records,
        y_internal=y_internal,
        mask_internal=mask_internal,
    )
    return x, y_internal, mask_internal, trial_info


def test_load_stage9_dataset_reads_internal_targets_and_metadata():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        _, y_internal, _, trial_info = make_test_dataset(path)

        dataset = load_stage9_dataset(path)

        assert dataset["x"].shape == (3, 20, 4)
        assert np.allclose(dataset["y_internal"], y_internal)
        assert dataset["trial_info"][1]["coh"] == trial_info[1]["coh"]
        assert dataset["teacher_sure_strength"][1] == 0.6


def test_dataset_task_returns_internal_targets_in_batches():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        _, y_internal, mask_internal, _ = make_test_dataset(path)
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=path,
            dt=10,
            tau=100,
            N_batch=2,
            sample_mode="sequential",
        )

        batch_x, batch_y, batch_mask, params = task.get_trial_batch()

        assert batch_x.shape == (2, 20, 4)
        assert np.allclose(batch_y[0], y_internal[0])
        assert np.allclose(batch_y[1], y_internal[1])
        assert np.allclose(batch_mask[0], mask_internal[0])
        assert params[0]["dataset_index"] == 0
        assert task.trial_info[1]["dataset_index"] == 1
        assert task.trial_info[1]["teacher_sure_strength"] == 0.6


def test_dataset_task_wraps_sequential_indices():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        make_test_dataset(path)
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=path,
            dt=10,
            tau=100,
            N_batch=2,
            sample_mode="sequential",
        )

        task.get_trial_batch()
        _, _, _, params = task.get_trial_batch()

        assert [int(p["dataset_index"]) for p in params] == [2, 0]


def test_soft_action_targets_sum_to_one_and_incorrect_targets_are_zero():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        _, y_internal, _, _ = make_test_dataset(path)

        np.testing.assert_allclose(np.sum(y_internal[:, 12:, 1:4], axis=2), 1.0)
        assert np.all(y_internal[0, 12:, 2] == 0.0)
        assert np.all(y_internal[1, 12:, 1] == 0.0)


def test_legacy_objective_remains_default_and_unmodified():
    args = build_arg_parser().parse_args([])
    params = {"sentinel": 1}

    assert args.student_action_objective == "independent_mse"
    assert configure_student_action_objective(params, "independent_mse") == {"sentinel": 1}


def test_categorical_loss_weights_change_only_their_declared_terms():
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()
    predictions = tf.constant(
        [[[0.8, 0.1, -0.1, 0.0], [0.4, 0.3, -0.2, 0.1]]], dtype=tf.float32
    )
    targets = tf.constant(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.4, 0.0, 0.6]]], dtype=tf.float32
    )
    mask = tf.ones_like(targets)
    retained, post_fix, action = categorical_soft_postgo_loss_components(
        predictions, targets, mask
    )
    base = categorical_soft_postgo_loss(predictions, targets, mask)
    fix_doubled = categorical_soft_postgo_loss(
        predictions, targets, mask, lambda_fix_postgo=2.0, lambda_action=1.0
    )
    action_doubled = categorical_soft_postgo_loss(
        predictions, targets, mask, lambda_fix_postgo=1.0, lambda_action=2.0
    )
    with tf.compat.v1.Session() as session:
        retained_value, post_fix_value, action_value, base_value, fix_value, action2_value = session.run(
            [retained, post_fix, action, base, fix_doubled, action_doubled]
        )

    assert abs(base_value - (retained_value + action_value)) < 1e-7
    assert abs((fix_value - base_value) - post_fix_value) < 1e-7
    assert abs((action2_value - base_value) - action_value) < 1e-7


def test_categorical_objective_trains_without_nan():
    import tensorflow as tf
    from psychrnn.backend.models.basic import Basic

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        make_test_dataset(path)
        tf.compat.v1.reset_default_graph()
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=path,
            dt=10,
            tau=100,
            N_batch=2,
            sample_mode="sequential",
        )
        params = task.get_task_params()
        params.update({"name": "categorical_test", "N_rec": 5, "rec_noise": 0.0})
        configure_student_action_objective(params, "categorical_soft")
        model = Basic(params)
        losses, _, _ = model.train(
            task,
            train_params={"training_iters": 6, "loss_epoch": 1, "verbosity": False},
        )
        assert losses
        assert np.all(np.isfinite(losses))
        model.destruct()


def test_legacy_objective_trains_without_nan():
    import tensorflow as tf
    from psychrnn.backend.models.basic import Basic

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        make_test_dataset(path)
        tf.compat.v1.reset_default_graph()
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=path,
            dt=10,
            tau=100,
            N_batch=2,
            sample_mode="sequential",
        )
        params = task.get_task_params()
        params.update({"name": "legacy_test", "N_rec": 5, "rec_noise": 0.0})
        configure_student_action_objective(params, "independent_mse")
        model = Basic(params)
        losses, _, _ = model.train(
            task,
            train_params={"training_iters": 6, "loss_epoch": 1, "verbosity": False},
        )
        assert losses
        assert np.all(np.isfinite(losses))
        model.destruct()


def test_four_channel_output_persistence_excludes_fixation_from_categorical_argmax():
    outputs = np.zeros((1, 20, 4), dtype=float)
    outputs[0, 12:, :] = [9.0, 0.2, 0.1, 0.7]
    targets = np.zeros_like(outputs)
    targets[0, 12:, :] = [0.0, 0.3, 0.0, 0.7]
    info = [{
        "dataset_index": 4,
        "delay_end": 12,
        "dir_choice": 0,
        "true_direction": 0,
        "sure_available": True,
        "teacher_sure_strength": 0.7,
        "coh": 0.032,
        "stimulus_dur": 500,
        "teacher_internal_margin": 0.2,
        "teacher_direction_success_proxy": 0.4,
    }]

    record = build_post_go_output_records(
        outputs, targets, info, objective="categorical_soft", readout_window=5
    )[0]

    assert record["predicted_fixation"] == 9.0
    assert record["predicted_left"] == 0.2
    assert record["predicted_right"] == 0.1
    assert record["predicted_sure"] == 0.7
    assert record["final_argmax_choice"] == 3
    assert record["raw_post_go_fixation"] == 9.0
    assert record["p_left"] == 0.2
    assert record["p_right"] == 0.1
    assert record["p_sure"] == 0.7
    assert record["teacher_preferred_action"] == 3
    assert abs(record["target_action_margin"] - 0.4) < 1e-12
    assert abs(record["predicted_action_margin"] - 0.5) < 1e-12


def test_candidate_standard_evaluation_indices_are_identical():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        make_test_dataset(path)
        sequences = []
        for _ in range(2):
            task = RDM_SureTarget_InternalReadoutDatasetTask(
                dataset_path=path,
                dt=10,
                tau=100,
                N_batch=2,
                sample_mode="sequential",
                seed=7,
            )
            sequence = []
            for _ in range(3):
                _, _, _, params = task.get_trial_batch()
                sequence.extend(int(item["dataset_index"]) for item in params)
            sequences.append(sequence)
        assert sequences[0] == sequences[1] == [0, 1, 2, 0, 1, 2]


def main():
    test_load_stage9_dataset_reads_internal_targets_and_metadata()
    test_dataset_task_returns_internal_targets_in_batches()
    test_dataset_task_wraps_sequential_indices()
    test_soft_action_targets_sum_to_one_and_incorrect_targets_are_zero()
    test_legacy_objective_remains_default_and_unmodified()
    test_categorical_loss_weights_change_only_their_declared_terms()
    test_categorical_objective_trains_without_nan()
    test_legacy_objective_trains_without_nan()
    test_four_channel_output_persistence_excludes_fixation_from_categorical_argmax()
    test_candidate_standard_evaluation_indices_are_identical()
    print("stage9 internal readout task test passed")


if __name__ == "__main__":
    main()
