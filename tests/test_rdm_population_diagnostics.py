import os
import tempfile

import numpy as np

from rdm_population_diagnostics import (
    compute_axis_timecourses,
    linear_r2,
    summarize_axis_r2_metrics,
    summarize_population_dynamics,
    summarize_pre_ts_leakage,
    summarize_repeated_stimulus_trials,
    write_json,
    write_repeated_trials_csv,
)


def test_linear_r2_matches_perfect_and_null_fit():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    assert abs(linear_r2(x[:, None], y) - 1.0) < 1e-12
    assert abs(linear_r2(np.ones((4, 1)), y) - 0.0) < 1e-12


def test_axis_r2_includes_residual_sure_axis_after_evidence():
    evidence = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    sure = np.array([0.0, 0.1, 1.2, 1.3, 2.4])
    p_sure = np.array([0.0, 0.1, 1.0, 1.1, 2.0])
    confidence = 2.0 * evidence + 1.0
    offered = np.array([True, True, True, True, True])

    metrics = summarize_axis_r2_metrics(
        evidence_axis=evidence,
        sure_axis=sure,
        p_sure=p_sure,
        confidence=confidence,
        offered=offered,
    )

    assert metrics["r2_evidence_axis_explains_confidence"] > 0.999
    assert metrics["r2_evidence_axis_explains_p_sure"] > 0.0
    assert metrics["r2_sure_axis_explains_p_sure"] > metrics["r2_evidence_axis_explains_p_sure"]
    assert metrics["r2_sure_axis_explains_residual_p_sure_after_evidence"] > 0.0


def test_repeated_stimulus_summary_detects_stable_repeats():
    records = [
        {"stimulus_id": 0, "repeat_index": 0, "choice": 1, "sure_output": 0.2, "confidence": 0.4},
        {"stimulus_id": 0, "repeat_index": 1, "choice": 1, "sure_output": 0.2, "confidence": 0.4},
        {"stimulus_id": 1, "repeat_index": 0, "choice": 3, "sure_output": 0.8, "confidence": 0.1},
        {"stimulus_id": 1, "repeat_index": 1, "choice": 2, "sure_output": 0.7, "confidence": 0.2},
    ]
    outputs = np.zeros((4, 3, 4))
    states = np.zeros((4, 3, 2))
    outputs[3, :, 3] = 0.1
    states[3, :, 1] = 0.2

    summary = summarize_repeated_stimulus_trials(records, outputs, states)

    assert summary["n_stimuli"] == 2
    assert summary["n_repeats_per_stimulus"] == 2
    assert summary["all_repeats_same_choice_fraction"] == 0.5
    assert summary["max_output_abs_diff"] == 0.1
    assert summary["max_state_abs_diff"] == 0.2


def test_pre_ts_leakage_summary_and_writers_create_outputs():
    n_trials, n_steps, n_rec = 4, 8, 2
    states = np.zeros((n_trials, n_steps, n_rec))
    states[0, 2:5, 0] = 1.0
    states[1, 2:5, 0] = 0.0
    states[2, 2:5, 0] = 0.5
    states[3, 2:5, 0] = 0.5
    trial_info = [
        {"sure_available": True, "ts_onset": 5, "delay_end": 7},
        {"sure_available": True, "ts_onset": 5, "delay_end": 7},
        {"sure_available": False, "ts_onset": 5, "delay_end": 7},
        {"sure_available": False, "ts_onset": 5, "delay_end": 7},
    ]
    choices = np.array([3, 1, 1, 2])
    axes_data = {"sure_axis": np.array([1.0, 0.0])}

    summary, plot_data = summarize_pre_ts_leakage(
        states,
        trial_info,
        choices,
        axes_data,
        pre_window=3,
        post_window=2,
    )

    assert summary["n_offered_sure_choice"] == 1
    assert summary["n_offered_direction_choice"] == 1
    assert summary["pre_ts_sure_axis_mean_difference"] == 1.0
    assert len(plot_data["relative_steps"]) == 6

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "repeated.csv")
        json_path = os.path.join(tmp_dir, "summary.json")
        write_repeated_trials_csv(csv_path, records=[{"a": 1, "b": 2}])
        write_json(json_path, summary)
        assert os.path.getsize(csv_path) > 0
        assert os.path.getsize(json_path) > 0


def test_axis_timecourses_project_states_onto_named_axes():
    states = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 3.0]],
        ]
    )
    axes_data = {
        "evidence_axis": np.array([1.0, 0.0]),
        "sure_axis": np.array([0.0, 2.0]),
        "time_axis": np.array([1.0, 1.0]),
    }

    timecourses = compute_axis_timecourses(states, axes_data)

    np.testing.assert_allclose(timecourses["evidence"], [[1.0, 2.0], [0.0, 0.0]])
    np.testing.assert_allclose(timecourses["sure"], [[0.0, 0.0], [1.0, 3.0]])
    np.testing.assert_allclose(
        timecourses["time"],
        [[1.0 / np.sqrt(2.0), 2.0 / np.sqrt(2.0)], [1.0 / np.sqrt(2.0), 3.0 / np.sqrt(2.0)]],
    )


def test_population_dynamics_summary_finds_ts_aligned_sure_separation():
    states = np.zeros((4, 8, 2))
    states[0, 5:, 0] = 2.0
    states[1, 5:, 0] = 2.0
    states[2, 5:, 0] = -1.0
    states[3, 5:, 0] = -1.0
    trial_info = [
        {"sure_available": True, "ts_onset": 5, "delay_end": 7},
        {"sure_available": True, "ts_onset": 5, "delay_end": 7},
        {"sure_available": True, "ts_onset": 5, "delay_end": 7},
        {"sure_available": True, "ts_onset": 5, "delay_end": 7},
    ]
    choices = np.array([3, 3, 1, 2])
    axes_data = {
        "evidence_axis": np.array([0.0, 1.0]),
        "sure_axis": np.array([1.0, 0.0]),
        "time_axis": np.array([1.0, 1.0]),
    }

    summary, plot_data = summarize_population_dynamics(
        states,
        trial_info,
        choices,
        axes_data,
        align_event="ts_onset",
        pre_window=3,
        post_window=2,
    )

    assert summary["align_event"] == "ts_onset"
    assert summary["n_offered_sure_choice"] == 2
    assert summary["n_offered_direction_choice"] == 2
    assert summary["sure_axis_pre_event_mean_difference"] == 0.0
    assert summary["sure_axis_post_event_mean_difference"] == 3.0
    assert summary["sure_axis_half_peak_onset_relative_step"] == 0
    assert len(plot_data["relative_steps"]) == 6


def main():
    test_linear_r2_matches_perfect_and_null_fit()
    test_axis_r2_includes_residual_sure_axis_after_evidence()
    test_repeated_stimulus_summary_detects_stable_repeats()
    test_pre_ts_leakage_summary_and_writers_create_outputs()
    test_axis_timecourses_project_states_onto_named_axes()
    test_population_dynamics_summary_finds_ts_aligned_sure_separation()
    print("rdm population diagnostics test passed")


if __name__ == "__main__":
    main()
