import inspect

import numpy as np

import population_stage2_time
from population_stage2_time import (
    DT_MS,
    ONSET_RUN_POINTS,
    TRAILING_STEPS,
    aligned_trailing_states,
    fit_time_scaler,
    operational_onset,
)


def synthetic_metadata():
    return [
        {
            "source_trial_index": index,
            "split": "train" if index < 2 else "test",
            "seed": 7,
            "motion_onset_step": 20,
            "motion_offset_step": 70,
            "sure_target_onset_step": 125,
            "go_cue_step": 150,
            "coherence": 0.1,
            "sure_offered": index % 2 == 0,
            "student_final_choice": 1 if index % 2 == 0 else 2,
            "empirical_p_correct": 0.5 + 0.1 * index,
        }
        for index in range(4)
    ]


def test_alignment_uses_exact_five_step_trailing_window():
    states = np.arange(4 * 300 * 2, dtype=np.float32).reshape(4, 300, 2)
    metadata = synthetic_metadata()
    source_ids, offsets, aligned = aligned_trailing_states(states, metadata, "motion_onset")
    position = int(np.flatnonzero(offsets == 0)[0])

    assert TRAILING_STEPS == 5
    assert DT_MS == 10
    np.testing.assert_array_equal(source_ids, np.arange(4))
    np.testing.assert_allclose(aligned[position, 0], states[0, 15:20].mean(axis=0))


def test_sure_alignment_includes_only_offered_trials():
    states = np.zeros((4, 300, 2), dtype=np.float32)
    source_ids, _, _ = aligned_trailing_states(states, synthetic_metadata(), "sure_target_onset")
    np.testing.assert_array_equal(source_ids, [0, 2])


def test_time_scaler_uses_supplied_train_only_and_handles_constant_unit():
    train = np.asarray([[0.0, 4.0], [2.0, 4.0]])
    test = np.asarray([[100.0, 40.0]])
    scaled_train, scaled_test, zero_count = fit_time_scaler(train, test)

    np.testing.assert_allclose(scaled_train[:, 0], [-1, 1])
    np.testing.assert_allclose(scaled_test[:, 0], [99])
    np.testing.assert_allclose(scaled_train[:, 1], [0, 0])
    assert zero_count == 1


def test_operational_onset_requires_five_consecutive_points():
    assert ONSET_RUN_POINTS == 5
    rows = [
        {"event": "motion_onset", "relative_time_ms": -20 + 10 * index, "metric": value}
        for index, value in enumerate([0.8, 0.9, 0.1, 0.7, 0.8, 0.9, 1.0, 0.95])
    ]
    assert operational_onset(rows, "motion_onset", "metric", 0.5) == 10


def test_stage2_source_has_no_rnn_training_or_replay_construction():
    source = inspect.getsource(population_stage2_time)
    assert "model" + ".train(" not in source
    assert "Basic" + "(" not in source
    assert "BasicSimulator" + "(" not in source
