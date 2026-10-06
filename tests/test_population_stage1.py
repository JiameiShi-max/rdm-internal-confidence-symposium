import inspect

import numpy as np

import population_stage1
from population_stage1 import (
    PRE_GO_STEPS,
    confidence_cross_validation,
    extract_primary_states,
    fit_pca,
    fit_scaler,
)


def test_primary_state_is_exactly_ten_steps_before_go():
    states = np.arange(2 * 30 * 3, dtype=np.float32).reshape(2, 30, 3)
    info = [{"delay_end": 20, "stimulus_end": 10}, {"delay_end": 25, "stimulus_end": 12}]
    pre_go, evidence_end, starts, stops = extract_primary_states(states, info)

    assert PRE_GO_STEPS == 10
    np.testing.assert_array_equal(starts, [10, 15])
    np.testing.assert_array_equal(stops, [20, 25])
    np.testing.assert_allclose(pre_go[0], states[0, 10:20].mean(axis=0))
    np.testing.assert_allclose(pre_go[1], states[1, 15:25].mean(axis=0))
    np.testing.assert_array_equal(evidence_end[0], states[0, 9])


def test_scaler_uses_train_only_and_handles_zero_variance():
    states = np.asarray([[0, 4], [2, 4], [100, 40], [200, 80]], dtype=float)
    standardized, mean, std, scale, zero = fit_scaler(states, np.asarray([0, 1]))

    np.testing.assert_allclose(mean, [1, 4])
    np.testing.assert_allclose(std, [1, 0])
    np.testing.assert_allclose(scale, [1, 1])
    np.testing.assert_array_equal(zero, [False, True])
    assert np.all(np.isfinite(standardized))


def test_pca_fit_mean_depends_only_on_train():
    states = np.asarray([[0, 0], [2, 2], [100, -100], [200, -200]], dtype=float)
    split = {"train": np.asarray([0, 1]), "validation": np.asarray([2]), "test": np.asarray([3])}
    pca, projected = fit_pca(states, split)

    np.testing.assert_allclose(pca.mean_, [1, 1])
    assert projected.shape == states.shape


def test_confidence_axis_accepts_continuous_target_without_sure_labels():
    rng = np.random.RandomState(3)
    x = rng.normal(size=(30, 4))
    target = 0.7 * x[:, 0] - 0.2 * x[:, 1]
    train = np.arange(25)
    result = confidence_cross_validation(x, target, train, seed=7, target_name="empirical_p_correct")

    assert result["target_name"] == "empirical_p_correct"
    assert np.array_equal(result["eligible_indices"], train)
    assert np.isclose(np.linalg.norm(result["vector"]), 1.0)
    assert np.corrcoef(x[train] @ result["vector"], target[train])[0, 1] > 0


def test_analysis_source_has_no_network_training_call_or_trainable_basic_model():
    source = inspect.getsource(population_stage1)
    assert "model" + ".train(" not in source
    assert "Basic(" not in source
