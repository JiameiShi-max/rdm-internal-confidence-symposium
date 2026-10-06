import inspect

import numpy as np

from population_stage3_variability import (
    center_within_rows,
    fit_fixed_effect_logistic,
    noise_seed,
    operational_negative_onset,
    simulate_frozen_repeats,
)


def small_weights():
    return {
        "init_state": np.asarray([[0.1, -0.1]], dtype=np.float32),
        "W_in": np.asarray([[0.2], [-0.3]], dtype=np.float32),
        "W_rec": np.asarray([[0.5, 0.1], [-0.2, 0.4]], dtype=np.float32),
        "W_out": np.asarray([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.2], [0.4, -0.2]], dtype=np.float32),
        "b_rec": np.asarray([0.01, -0.02], dtype=np.float32),
        "b_out": np.asarray([0.0, 0.1, -0.1, 0.2], dtype=np.float32),
    }


def test_noise_seed_schedule_is_unique_and_reproducible():
    seeds = [noise_seed(model, source, repeat) for model in (7, 8, 9) for source in (0, 999) for repeat in (0, 99)]
    assert len(seeds) == len(set(seeds))
    assert noise_seed(7, 19, 2) == noise_seed(7, 19, 2)


def test_zero_noise_replays_are_exactly_identical():
    source_input = np.linspace(-1, 1, 12, dtype=np.float32).reshape(12, 1)
    outputs, states = simulate_frozen_repeats(small_weights(), source_input, [1, 2, 3], 0.0)
    np.testing.assert_array_equal(outputs, np.repeat(outputs[0:1], 3, axis=0))
    np.testing.assert_array_equal(states, np.repeat(states[0:1], 3, axis=0))


def test_recurrent_noise_changes_state_but_not_input_contract():
    source_input = np.ones((12, 1), dtype=np.float32)
    _, states = simulate_frozen_repeats(small_weights(), source_input, [1, 2], 0.05)
    assert not np.array_equal(states[0], states[1])


def test_center_within_stimulus_has_zero_row_means():
    values = np.asarray([[1.0, 2.0, 6.0], [-4.0, 3.0, 8.0]])
    centered = center_within_rows(values)
    np.testing.assert_allclose(np.mean(centered, axis=1), 0.0, atol=1e-15)


def test_fixed_effect_logistic_recovers_negative_relation():
    rng = np.random.RandomState(4)
    x = rng.normal(size=(12, 80))
    x -= np.mean(x, axis=1, keepdims=True)
    intercept = rng.uniform(-0.8, 0.8, size=(12, 1))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(intercept - 1.2 * x))))
    variable = (np.sum(y, axis=1) > 0) & (np.sum(y, axis=1) < y.shape[1])
    result = fit_fixed_effect_logistic(x[variable], y[variable])
    assert result["converged"]
    assert result["beta"][0] < -0.8


def test_operational_onset_requires_five_consecutive_points():
    times = np.arange(-100, 1, 10)
    beta = np.asarray([0, -2, -2, -2, -2, 0, -2, -2, -2, -2, -2], dtype=float)
    assert operational_negative_onset(times, beta, 1.0) == -40


def test_source_contains_no_network_training_or_neural_decoder_fit():
    import population_stage3_variability as module

    source = inspect.getsource(module)
    assert "model" + ".train(" not in source
    assert "LogisticRegression(" not in source
    assert "Ridge(" not in source
