from pathlib import Path

import numpy as np

from dual_head_student import (
    dual_head_loss,
    dual_head_loss_numpy,
    dual_head_probabilities,
    split_output_head_weights,
)
from rdm_frozen_readout_diagnostic import sha256


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/model_freeze/teacher_seed7.npz"
CONTROL = (
    ROOT
    / "results/model_freeze/student_head_calibration_v2/stage_a/fix_1_action_1/weights.npz"
)
DUAL_HEAD = ROOT / "results/model_freeze/dual_head_seed7/weights.npz"
REFERENCE = ROOT / "results/model_freeze/dual_head_seed7/reconstruction_reference.npz"
EXPECTED_TEACHER_SHA256 = "8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d"


def example_batch():
    raw = np.array(
        [
            [
                [2.0, 0.2, -0.1, 0.5],
                [-2.0, 1.2, -0.3, 0.4],
            ]
        ],
        dtype=np.float32,
    )
    targets = np.zeros_like(raw)
    targets[0, 0, 0] = 1.0
    targets[0, 1, 1:4] = [0.7, 0.0, 0.3]
    mask = np.zeros_like(raw)
    mask[:, :, 0] = 1.0
    mask[0, 1, 1:4] = 1.0
    return raw, targets, mask


def test_independent_fixation_and_action_parameter_slices():
    weights = {"W_out": np.arange(20).reshape(4, 5), "b_out": np.arange(4.0)}
    fixation, action = split_output_head_weights(weights)

    fixation["W_fix"][:] = -1

    assert action["W_action"].shape == (3, 5)
    assert np.array_equal(action["W_action"], np.arange(20).reshape(4, 5)[1:4])
    assert np.array_equal(weights["W_out"][0], np.arange(5))


def test_sigmoid_fixation_and_three_way_softmax_are_separate():
    raw, _, _ = example_batch()
    fixation_before, action_before = dual_head_probabilities(raw)
    changed = raw.copy()
    changed[..., 0] += 100.0
    _, action_after = dual_head_probabilities(changed)

    assert np.all((fixation_before > 0) & (fixation_before < 1))
    assert np.array_equal(action_before, action_after)
    assert np.allclose(np.sum(action_before, axis=-1), 1.0)
    assert action_before.shape[-1] == 3


def test_action_soft_ce_is_inactive_before_go():
    raw, targets, mask = example_batch()
    baseline = dual_head_loss_numpy(raw, targets, mask)
    changed = raw.copy()
    changed[0, 0, 1:4] = [100.0, -100.0, 50.0]
    observed = dual_head_loss_numpy(changed, targets, mask)

    assert baseline["n_fixation_active"] == 2
    assert baseline["n_action_active"] == 1
    assert observed["action_soft_ce"] == baseline["action_soft_ce"]
    assert observed["total"] == baseline["total"]


def test_tensorflow_dual_head_loss_matches_reference():
    import tensorflow as tf

    raw, targets, mask = example_batch()
    expected = dual_head_loss_numpy(raw, targets, mask)
    tf.compat.v1.reset_default_graph()
    loss = dual_head_loss(
        tf.constant(raw), tf.constant(targets), tf.constant(mask)
    )
    if tf.executing_eagerly():
        observed = float(loss.numpy())
    else:
        with tf.compat.v1.Session() as session:
            observed = float(session.run(loss))

    assert np.isclose(observed, expected["total"], rtol=1e-6, atol=1e-7)


def test_teacher_dataset_and_targets_are_unchanged():
    before = sha256(DATASET)
    with np.load(DATASET, allow_pickle=True) as dataset:
        targets_before = dataset["y_internal"].copy()
    with np.load(DATASET, allow_pickle=True) as dataset:
        targets_after = dataset["y_internal"].copy()

    assert before == EXPECTED_TEACHER_SHA256
    assert sha256(DATASET) == before
    assert np.array_equal(targets_after, targets_before)


def test_saved_dual_head_checkpoint_reconstructs_predictions():
    from psychrnn.backend.simulation import BasicSimulator

    assert DUAL_HEAD.exists()
    assert REFERENCE.exists()
    with np.load(REFERENCE, allow_pickle=False) as reference:
        indices = reference["dataset_indices"]
        expected_outputs = reference["raw_outputs"]
        expected_states = reference["states"]
    with np.load(DATASET, allow_pickle=True) as dataset:
        inputs = np.asarray(dataset["x"][indices], dtype=np.float32)
    weights = dict(np.load(DUAL_HEAD, allow_pickle=True))
    simulator = BasicSimulator(params={"alpha": 0.1, "rec_noise": 0.0}, weights=weights)
    outputs, states = simulator.run_trials(inputs)

    assert np.array_equal(outputs, expected_outputs)
    assert np.array_equal(states, expected_states)


def test_legacy_categorical_checkpoint_remains_runnable():
    from psychrnn.backend.simulation import BasicSimulator

    before = sha256(CONTROL)
    weights = dict(np.load(CONTROL, allow_pickle=True))
    with np.load(DATASET, allow_pickle=True) as dataset:
        inputs = np.asarray(dataset["x"][:1], dtype=np.float32)
    simulator = BasicSimulator(params={"alpha": 0.1, "rec_noise": 0.0}, weights=weights)
    outputs, states = simulator.run_trials(inputs)

    assert np.all(np.isfinite(outputs))
    assert np.all(np.isfinite(states))
    assert sha256(CONTROL) == before
