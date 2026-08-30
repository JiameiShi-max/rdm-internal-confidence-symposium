import csv
import json
import os
import tempfile

from rdm_report_tables import build_report_payload, write_report_outputs


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_build_report_payload_includes_single_and_multiseed_rows():
    with tempfile.TemporaryDirectory() as tmp_dir:
        original = os.path.join(tmp_dir, "original_summary.json")
        timed = os.path.join(tmp_dir, "timed_summary.json")
        timed_pre = os.path.join(tmp_dir, "timed_summary_pre_ts_leakage_summary.json")
        timed_det = os.path.join(tmp_dir, "timed_summary_deterministic_repeated_stimulus_summary.json")
        timed_pop = os.path.join(tmp_dir, "timed_summary_population_dynamics_summary.json")
        multiseed = os.path.join(tmp_dir, "multiseed_summary.json")

        _write_json(
            original,
            {
                "n_trials": 400,
                "n_sure_offered": 321,
                "n_sure_choice": 184,
                "corr_sure_axis_with_sure_residual_after_evidence": 0.78,
                "r2_sure_axis_explains_residual_p_sure_after_evidence": 0.61,
            },
        )
        _write_json(
            timed,
            {
                "n_trials": 400,
                "n_sure_offered": 198,
                "n_sure_choice": 115,
                "corr_sure_axis_with_sure_residual_after_evidence": 0.60,
                "r2_sure_axis_explains_residual_p_sure_after_evidence": 0.36,
            },
        )
        _write_json(timed_pre, {"corr_pre_ts_sure_axis_with_sure_choice": 0.20})
        _write_json(timed_det, {"all_repeats_same_choice_fraction": 1.0})
        _write_json(timed_pop, {"sure_axis_peak_abs_difference": 0.09})
        _write_json(
            multiseed,
            {
                "n_seeds": 5,
                "pass": False,
                "metrics": {
                    "corr_sure_axis_with_sure_residual_after_evidence": {
                        "mean": 0.60,
                        "std": 0.05,
                        "min": 0.53,
                        "max": 0.66,
                    },
                    "pre_ts_leakage.corr_pre_ts_sure_axis_with_sure_choice": {
                        "mean": 0.30,
                        "std": 0.17,
                        "min": 0.15,
                        "max": 0.57,
                    },
                },
            },
        )

        payload = build_report_payload(
            original_summary=original,
            timed_summary=timed,
            timed_multiseed_summary=multiseed,
        )

        assert payload["single_model_rows"][0]["model"] == "original_single_seed"
        assert payload["single_model_rows"][1]["sure_offered_rate"] == 0.495
        assert payload["single_model_rows"][1]["pre_ts_corr"] == 0.20
        assert payload["multiseed_rows"][0]["n_seeds"] == 5
        assert payload["multiseed_rows"][0]["residual_corr_mean"] == 0.60


def test_write_report_outputs_creates_markdown_json_and_csv():
    payload = {
        "single_model_rows": [
            {
                "model": "timed_single_seed",
                "n_trials": 400,
                "sure_offered_rate": 0.495,
                "sure_choice_rate_offered": 0.58,
                "residual_corr": 0.60,
                "residual_r2": 0.36,
                "pre_ts_corr": 0.20,
                "deterministic_stability": 1.0,
                "population_peak_abs_diff": 0.09,
            }
        ],
        "multiseed_rows": [],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = write_report_outputs(tmp_dir, payload)
        assert os.path.getsize(paths["report_md"]) > 0
        assert os.path.getsize(paths["report_json"]) > 0
        assert os.path.getsize(paths["single_model_csv"]) > 0
        with open(paths["single_model_csv"], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["model"] == "timed_single_seed"


def main():
    test_build_report_payload_includes_single_and_multiseed_rows()
    test_write_report_outputs_creates_markdown_json_and_csv()
    print("rdm report tables test passed")


if __name__ == "__main__":
    main()
