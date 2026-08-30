import csv
import json
import os
import tempfile

from rdm_matched_trial_controls import (
    build_matched_controls,
    compute_delta_inference,
    load_trial_records,
    summarize_matched_controls,
    write_matched_control_outputs,
)


def _write_trials(path):
    rows = [
        # coh 0.032: one sure, one waived, two no-sure matches
        {
            "trial_index": 0,
            "coh": 0.032,
            "sure_available": True,
            "chose_sure": True,
            "chose_direction": False,
            "correct": False,
            "waived_sure": False,
            "evidence_axis": 0.10,
            "sure_axis": 0.80,
            "confidence_near_ts": 0.40,
            "sensory_margin": 1.0,
        },
        {
            "trial_index": 1,
            "coh": 0.032,
            "sure_available": True,
            "chose_sure": False,
            "chose_direction": True,
            "correct": True,
            "waived_sure": True,
            "evidence_axis": 0.12,
            "sure_axis": 0.20,
            "confidence_near_ts": 0.75,
            "sensory_margin": 1.1,
        },
        {
            "trial_index": 2,
            "coh": 0.032,
            "sure_available": False,
            "chose_sure": False,
            "chose_direction": True,
            "correct": True,
            "waived_sure": False,
            "evidence_axis": 0.11,
            "sure_axis": 0.15,
            "confidence_near_ts": 0.70,
            "sensory_margin": 1.0,
        },
        {
            "trial_index": 3,
            "coh": 0.032,
            "sure_available": False,
            "chose_sure": False,
            "chose_direction": True,
            "correct": False,
            "waived_sure": False,
            "evidence_axis": 0.50,
            "sure_axis": 0.10,
            "confidence_near_ts": 0.65,
            "sensory_margin": 1.4,
        },
        # coh 0.512: one sure, one waived, two no-sure matches
        {
            "trial_index": 4,
            "coh": 0.512,
            "sure_available": True,
            "chose_sure": True,
            "chose_direction": False,
            "correct": False,
            "waived_sure": False,
            "evidence_axis": 0.90,
            "sure_axis": 0.70,
            "confidence_near_ts": 0.50,
            "sensory_margin": 9.0,
        },
        {
            "trial_index": 5,
            "coh": 0.512,
            "sure_available": True,
            "chose_sure": False,
            "chose_direction": True,
            "correct": True,
            "waived_sure": True,
            "evidence_axis": 0.95,
            "sure_axis": 0.05,
            "confidence_near_ts": 0.95,
            "sensory_margin": 9.5,
        },
        {
            "trial_index": 6,
            "coh": 0.512,
            "sure_available": False,
            "chose_sure": False,
            "chose_direction": True,
            "correct": True,
            "waived_sure": False,
            "evidence_axis": 0.96,
            "sure_axis": 0.04,
            "confidence_near_ts": 0.90,
            "sensory_margin": 9.4,
        },
        {
            "trial_index": 7,
            "coh": 0.512,
            "sure_available": False,
            "chose_sure": False,
            "chose_direction": True,
            "correct": True,
            "waived_sure": False,
            "evidence_axis": 0.88,
            "sure_axis": 0.30,
            "confidence_near_ts": 0.92,
            "sensory_margin": 9.1,
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_trial_records_casts_booleans_and_floats():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "trials.csv")
        _write_trials(path)

        records = load_trial_records(path)

        assert len(records) == 8
        assert records[0]["sure_available"] is True
        assert records[2]["sure_available"] is False
        assert records[0]["coh"] == 0.032
        assert records[0]["evidence_axis"] == 0.10


def test_build_matched_controls_matches_within_coherence_by_evidence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "trials.csv")
        _write_trials(path)
        records = load_trial_records(path)

        controls = build_matched_controls(records, match_key="evidence_axis")

        assert len(controls["offered_to_no_sure_matches"]) == 4
        first = controls["offered_to_no_sure_matches"][0]
        assert first["offered_trial_index"] == 0
        assert first["matched_no_sure_trial_index"] == 2
        assert abs(first["match_delta"] - 0.01) < 1e-12
        assert len(controls["sure_to_waived_matches"]) == 2
        assert controls["sure_to_waived_matches"][0]["sure_axis_delta_sure_minus_waived"] > 0


def test_summary_and_outputs_include_required_controls():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trial_path = os.path.join(tmp_dir, "trials.csv")
        output_dir = os.path.join(tmp_dir, "exports")
        figure_dir = os.path.join(tmp_dir, "figures")
        _write_trials(trial_path)
        records = load_trial_records(trial_path)
        controls = build_matched_controls(records, match_key="evidence_axis")

        summary = summarize_matched_controls(records, controls, match_key="evidence_axis")

        assert summary["n_trials"] == 8
        assert summary["n_offered_to_no_sure_matches"] == 4
        assert summary["n_sure_to_waived_matches"] == 2
        assert summary["waived_accuracy"] == 1.0
        assert summary["matched_no_sure_accuracy_for_waived"] == 1.0
        assert summary["mean_sure_axis_delta_sure_minus_waived"] > 0

        paths = write_matched_control_outputs(output_dir, figure_dir, summary, controls)
        assert os.path.getsize(paths["summary_json"]) > 0
        assert os.path.getsize(paths["matches_csv"]) > 0
        assert os.path.getsize(paths["figure"]) > 0
        with open(paths["summary_json"], encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["match_key"] == "evidence_axis"


def test_compute_delta_inference_returns_ci_and_permutation_p_value():
    deltas = [0.2, 0.3, 0.4, 0.5]

    inference = compute_delta_inference(deltas, n_bootstrap=50, n_permutations=50, seed=3)

    assert inference["n"] == 4
    assert inference["mean"] == 0.35
    assert inference["bootstrap_ci95_low"] <= inference["mean"]
    assert inference["bootstrap_ci95_high"] >= inference["mean"]
    assert 0.0 <= inference["permutation_p_two_sided"] <= 1.0


def main():
    test_load_trial_records_casts_booleans_and_floats()
    test_build_matched_controls_matches_within_coherence_by_evidence()
    test_summary_and_outputs_include_required_controls()
    test_compute_delta_inference_returns_ci_and_permutation_p_value()
    print("rdm matched trial controls test passed")


if __name__ == "__main__":
    main()
