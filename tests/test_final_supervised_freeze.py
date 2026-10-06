import json

import numpy as np

from final_supervised_freeze import (
    BATCH_SIZE,
    LAMBDA_ACTION,
    LAMBDA_FIX_POSTGO,
    N_REC,
    OBJECTIVE,
    SPLIT_SEED,
    STUDENT_SEEDS,
    TRAINING_ITERS,
    create_or_load_split,
    validate_split,
)
from sure_target_stage9_internal_readout import RDM_SureTarget_InternalReadoutDatasetTask


def make_dataset(path, n_trials=20):
    x = np.zeros((n_trials, 20, 4), dtype=np.float32)
    y = np.zeros((n_trials, 20, 4), dtype=np.float32)
    mask = np.ones_like(y)
    y[:, :12, 0] = 1.0
    for index in range(n_trials):
        direction = index % 2
        sure = 0.3 if index % 3 == 0 else 0.0
        y[index, 12:, 1 + direction] = 1.0 - sure
        y[index, 12:, 3] = sure
    info = [
        {
            "dir_choice": index % 2,
            "true_direction": index % 2,
            "sure_available": index % 2 == 0,
            "fixation_end": 2,
            "stimulus_end": 10,
            "ts_onset": 11,
            "delay_end": 12,
            "coh": [0.0, 0.032, 0.064, 0.128][index % 4],
            "stimulus_dur": 500 + 10 * index,
        }
        for index in range(n_trials)
    ]
    np.savez(
        path,
        x=x,
        y_internal=y,
        mask_internal=mask,
        trial_info_json=np.asarray([json.dumps(item) for item in info], dtype=object),
        teacher_internal_margin=np.linspace(0.0, 1.0, n_trials),
        teacher_sure_strength=np.linspace(0.5, 0.0, n_trials),
        teacher_expected_direction_value=np.linspace(0.5, 1.0, n_trials),
        teacher_direction_success_proxy=np.linspace(0.5, 1.0, n_trials),
        teacher_sure_advantage=np.linspace(0.05, -0.45, n_trials),
    )


def sampled_indices(task, n_batches):
    observed = []
    for _ in range(n_batches):
        _, _, _, params = task.get_trial_batch()
        observed.extend(int(item["dataset_index"]) for item in params)
    return observed


def test_split_is_exact_disjoint_exhaustive_and_reused(tmp_path):
    output = tmp_path / "freeze"
    split, split_path, _ = create_or_load_split(output, 20, "dataset-digest")
    checked = validate_split(split["train"], split["validation"], split["test"], 20)

    assert {key: len(value) for key, value in checked.items()} == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }
    with np.load(split_path, allow_pickle=False) as stored:
        assert int(stored["split_seed"]) == SPLIT_SEED
    reused, _, _ = create_or_load_split(output, 20, "dataset-digest", resume=True)
    for name in split:
        assert np.array_equal(split[name], reused[name])


def test_split_generation_is_reproducible_across_directories(tmp_path):
    first, _, _ = create_or_load_split(tmp_path / "one", 20, "same")
    second, _, _ = create_or_load_split(tmp_path / "two", 20, "same")
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_train_validation_and_test_tasks_cannot_cross_splits(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    make_dataset(dataset_path)
    split, _, _ = create_or_load_split(tmp_path / "freeze", 20, "same")
    observed_sets = {}
    for name in ("train", "validation", "test"):
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=dataset_path,
            N_batch=2,
            sample_mode="random",
            seed=7,
            allowed_indices=split[name],
        )
        observed_sets[name] = set(sampled_indices(task, 100))
        assert observed_sets[name] <= set(split[name])

    assert observed_sets["train"].isdisjoint(observed_sets["validation"])
    assert observed_sets["train"].isdisjoint(observed_sets["test"])
    assert observed_sets["validation"].isdisjoint(observed_sets["test"])


def test_sequential_evaluators_visit_only_their_exact_split(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    make_dataset(dataset_path)
    split, _, _ = create_or_load_split(tmp_path / "freeze", 20, "same")
    for name in ("train", "validation", "test"):
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=dataset_path,
            N_batch=1,
            sample_mode="sequential",
            allowed_indices=split[name],
        )
        assert sampled_indices(task, len(split[name])) == split[name].tolist()


def test_random_sampler_seed_is_reproducible_within_allowed_indices(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    make_dataset(dataset_path)
    allowed = np.asarray([1, 5, 9, 13], dtype=np.int64)
    sequences = []
    for _ in range(2):
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=dataset_path,
            N_batch=2,
            sample_mode="random",
            seed=8,
            allowed_indices=allowed,
        )
        sequences.append(sampled_indices(task, 10))
    assert sequences[0] == sequences[1]
    assert set(sequences[0]) <= set(allowed)


def test_frozen_categorical_configuration_is_unchanged():
    assert OBJECTIVE == "categorical_soft"
    assert STUDENT_SEEDS == (7, 8, 9)
    assert TRAINING_ITERS == 50000
    assert BATCH_SIZE == 50
    assert N_REC == 50
    assert LAMBDA_FIX_POSTGO == 1.0
    assert LAMBDA_ACTION == 1.0


def test_checkpoint_reload_reproduces_outputs(tmp_path):
    import tensorflow as tf
    from psychrnn.backend.models.basic import Basic
    from psychrnn.backend.simulation import BasicSimulator

    from sure_target_stage9_internal_readout import configure_student_action_objective

    dataset_path = tmp_path / "dataset.npz"
    weights_path = tmp_path / "weights.npz"
    make_dataset(dataset_path, n_trials=4)
    tf.compat.v1.reset_default_graph()
    task = RDM_SureTarget_InternalReadoutDatasetTask(
        dataset_path=dataset_path,
        N_batch=2,
        sample_mode="sequential",
        allowed_indices=np.arange(4),
    )
    params = task.get_task_params()
    params.update({"name": "final_freeze_reload_test", "N_rec": 5, "rec_noise": 0.0})
    configure_student_action_objective(params, "categorical_soft")
    model = Basic(params)
    model.train(
        task,
        train_params={
            "training_iters": 6,
            "loss_epoch": 1,
            "verbosity": False,
            "save_weights_path": str(weights_path),
        },
    )
    model.destruct()
    with np.load(dataset_path, allow_pickle=True) as data:
        inputs = np.asarray(data["x"][:2], dtype=np.float32)
    first = BasicSimulator(
        params={"alpha": 0.1, "rec_noise": 0.0},
        weights=dict(np.load(weights_path, allow_pickle=True)),
    )
    second = BasicSimulator(
        params={"alpha": 0.1, "rec_noise": 0.0},
        weights=dict(np.load(weights_path, allow_pickle=True)),
    )
    expected_outputs, expected_states = first.run_trials(inputs)
    observed_outputs, observed_states = second.run_trials(inputs)

    np.testing.assert_allclose(observed_outputs, expected_outputs, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(observed_states, expected_states, rtol=1e-7, atol=1e-7)
