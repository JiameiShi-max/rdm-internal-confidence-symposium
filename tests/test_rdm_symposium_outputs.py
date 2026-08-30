import csv
import json
import os
import tempfile

from rdm_symposium_outputs import (
    build_symposium_payload,
    write_symposium_outputs,
)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_model_bundle(summary_path, n_offered, n_sure, residual_r2, pre_ts_corr):
    _write_json(
        summary_path,
        {
            "n_trials": 400,
            "n_sure_offered": n_offered,
            "n_sure_choice": n_sure,
            "corr_sure_axis_with_sure_residual_after_evidence": residual_r2 ** 0.5,
            "r2_sure_axis_explains_residual_p_sure_after_evidence": residual_r2,
            "r2_evidence_axis_explains_p_sure": 0.25,
            "r2_evidence_axis_explains_confidence": 0.50,
            "dataset": "dataset.npz",
        },
    )
    root, _ = os.path.splitext(summary_path)
    _write_json(
        f"{root}_behavior.json",
        {
            "overall_p_sure": n_sure / float(n_offered),
            "no_sure_accuracy": 0.90,
            "waived_sure_accuracy": 1.0,
            "coherence": [
                {
                    "coh": 0.0,
                    "n_sure_offered": 20,
                    "p_sure": 0.90,
                    "no_sure_accuracy": 0.55,
                    "waived_sure_accuracy": 1.0,
                },
                {
                    "coh": 0.512,
                    "n_sure_offered": 20,
                    "p_sure": 0.10,
                    "no_sure_accuracy": 1.0,
                    "waived_sure_accuracy": 1.0,
                },
            ],
        },
    )
    _write_json(
        f"{root}_pre_ts_leakage_summary.json",
        {
            "corr_pre_ts_sure_axis_with_sure_choice": pre_ts_corr,
            "corr_post_ts_sure_axis_with_sure_choice": pre_ts_corr + 0.05,
        },
    )
    _write_json(
        f"{root}_deterministic_repeated_stimulus_summary.json",
        {"all_repeats_same_choice_fraction": 1.0},
    )
    _write_json(
        f"{root}_population_dynamics_summary.json",
        {"sure_axis_peak_abs_difference": 0.08},
    )


def test_build_symposium_payload_includes_metrics_variants_and_coherence_rows():
    with tempfile.TemporaryDirectory() as tmp_dir:
        original = os.path.join(tmp_dir, "original_summary.json")
        timed = os.path.join(tmp_dir, "timed_summary.json")
        variant = os.path.join(tmp_dir, "variant_summary.json")
        multiseed = os.path.join(tmp_dir, "multiseed_summary.json")
        _write_model_bundle(original, 320, 180, 0.49, 0.45)
        _write_model_bundle(timed, 200, 100, 0.30, 0.25)
        _write_model_bundle(variant, 200, 130, 0.42, 0.40)
        _write_json(
            multiseed,
            {
                "n_seeds": 5,
                "pass": False,
                "metrics": {
                    "n_sure_offered": {"mean": 198.0},
                    "n_sure_choice": {"mean": 116.0},
                    "r2_sure_axis_explains_residual_p_sure_after_evidence": {
                        "mean": 0.36,
                        "std": 0.05,
                        "min": 0.28,
                        "max": 0.43,
                    },
                    "r2_evidence_axis_explains_p_sure": {"mean": 0.45, "std": 0.04},
                    "pre_ts_leakage.corr_pre_ts_sure_axis_with_sure_choice": {
                        "mean": 0.30,
                        "max": 0.57,
                    },
                    "deterministic_repeated.all_repeats_same_choice_fraction": {"min": 1.0},
                    "population_dynamics.sure_axis_peak_abs_difference": {"mean": 0.12},
                },
            },
        )

        payload = build_symposium_payload(
            model_specs=[
                ("original", original, "single"),
                ("baseline_timed", timed, "main"),
            ],
            parameter_specs=[("reward_057", variant, "parameter")],
            multiseed_summary=multiseed,
        )

        names = [row["model"] for row in payload["metrics_rows"]]
        assert names == ["original", "baseline_timed", "timed_5seed", "reward_057"]
        assert payload["metrics_rows"][1]["sure_offered_rate"] == 0.5
        assert payload["metrics_rows"][2]["n_seeds"] == 5
        assert len(payload["coherence_rows"]) == 6
        assert payload["coherence_rows"][0]["p_sure"] == 0.9


def test_write_symposium_outputs_creates_tables_and_four_figures():
    payload = {
        "metrics_rows": [
            {
                "model": "original",
                "model_group": "single",
                "sure_offered_rate": 0.8,
                "sure_choice_rate_offered": 0.58,
                "residual_r2": 0.49,
                "evidence_r2_p_sure": 0.28,
                "evidence_r2_confidence": 0.55,
                "pre_ts_corr": 0.45,
                "deterministic_stability": 1.0,
                "population_peak_abs_diff": 0.19,
            },
            {
                "model": "baseline_timed",
                "model_group": "main",
                "sure_offered_rate": 0.5,
                "sure_choice_rate_offered": 0.51,
                "residual_r2": 0.30,
                "evidence_r2_p_sure": 0.47,
                "evidence_r2_confidence": 0.50,
                "pre_ts_corr": 0.30,
                "deterministic_stability": 1.0,
                "population_peak_abs_diff": 0.06,
            },
            {
                "model": "reward_057",
                "model_group": "parameter",
                "sure_offered_rate": 0.5,
                "sure_choice_rate_offered": 0.68,
                "residual_r2": 0.42,
                "evidence_r2_p_sure": 0.35,
                "evidence_r2_confidence": 0.66,
                "pre_ts_corr": 0.47,
                "deterministic_stability": 1.0,
                "population_peak_abs_diff": 0.12,
            },
        ],
        "coherence_rows": [
            {"model": "baseline_timed", "coh": 0.0, "p_sure": 1.0, "no_sure_accuracy": 0.5, "waived_sure_accuracy": None},
            {"model": "baseline_timed", "coh": 0.512, "p_sure": 0.0, "no_sure_accuracy": 1.0, "waived_sure_accuracy": 1.0},
            {"model": "reward_057", "coh": 0.0, "p_sure": 1.0, "no_sure_accuracy": 0.5, "waived_sure_accuracy": None},
            {"model": "reward_057", "coh": 0.512, "p_sure": 0.25, "no_sure_accuracy": 1.0, "waived_sure_accuracy": 1.0},
        ],
        "multiseed_rows": [],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = write_symposium_outputs(tmp_dir, os.path.join(tmp_dir, "figures"), payload)
        assert os.path.getsize(paths["payload_json"]) > 0
        with open(paths["metrics_csv"], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[1]["model"] == "baseline_timed"
        assert len(paths["figures"]) == 4
        for figure_path in paths["figures"].values():
            assert os.path.getsize(figure_path) > 0


def main():
    test_build_symposium_payload_includes_metrics_variants_and_coherence_rows()
    test_write_symposium_outputs_creates_tables_and_four_figures()
    print("rdm symposium outputs test passed")


if __name__ == "__main__":
    main()
