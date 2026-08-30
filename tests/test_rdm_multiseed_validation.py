import json
import os
import tempfile

from rdm_multiseed_validation import (
    aggregate_seed_summaries,
    default_validation_metrics,
    write_aggregate_outputs,
)


def test_aggregate_seed_summaries_computes_mean_std_and_thresholds():
    summaries = [
        {
            "seed": 7,
            "n_sure_offered": 198,
            "n_trials": 400,
            "corr_sure_axis_with_sure_residual_after_evidence": 0.60,
            "r2_sure_axis_explains_residual_p_sure_after_evidence": 0.36,
            "pre_ts_leakage": {"corr_pre_ts_sure_axis_with_sure_choice": 0.20},
            "deterministic_repeated": {"all_repeats_same_choice_fraction": 1.0},
        },
        {
            "seed": 8,
            "n_sure_offered": 202,
            "n_trials": 400,
            "corr_sure_axis_with_sure_residual_after_evidence": 0.50,
            "r2_sure_axis_explains_residual_p_sure_after_evidence": 0.25,
            "pre_ts_leakage": {"corr_pre_ts_sure_axis_with_sure_choice": 0.30},
            "deterministic_repeated": {"all_repeats_same_choice_fraction": 1.0},
        },
    ]

    aggregate = aggregate_seed_summaries(
        summaries,
        metrics=default_validation_metrics(),
        min_residual_corr=0.45,
        max_pre_ts_corr=0.35,
        min_deterministic_stability=0.999,
    )

    assert aggregate["n_seeds"] == 2
    assert aggregate["pass"] is True
    assert aggregate["metrics"]["n_sure_offered"]["mean"] == 200.0
    assert aggregate["metrics"]["corr_sure_axis_with_sure_residual_after_evidence"]["min"] == 0.50
    assert aggregate["criteria"]["min_residual_corr"]["pass"] is True
    assert aggregate["criteria"]["max_pre_ts_corr"]["pass"] is True
    assert aggregate["criteria"]["min_deterministic_stability"]["pass"] is True
    assert aggregate["criteria"]["all_required_metrics_finite"]["pass"] is True


def test_aggregate_seed_summaries_fails_when_required_metric_is_missing():
    summaries = [
        {
            "seed": 7,
            "corr_sure_axis_with_sure_residual_after_evidence": 0.60,
            "pre_ts_leakage": {"corr_pre_ts_sure_axis_with_sure_choice": 0.20},
            "deterministic_repeated": {"all_repeats_same_choice_fraction": 1.0},
        },
        {
            "seed": 8,
            "corr_sure_axis_with_sure_residual_after_evidence": float("nan"),
            "pre_ts_leakage": {"corr_pre_ts_sure_axis_with_sure_choice": 0.20},
            "deterministic_repeated": {"all_repeats_same_choice_fraction": 1.0},
        },
    ]

    aggregate = aggregate_seed_summaries(summaries)

    assert aggregate["pass"] is False
    assert aggregate["criteria"]["all_required_metrics_finite"]["pass"] is False


def test_write_aggregate_outputs_creates_json_and_csv():
    aggregate = {
        "n_seeds": 1,
        "pass": True,
        "per_seed": [
            {
                "seed": 7,
                "corr_sure_axis_with_sure_residual_after_evidence": 0.6,
                "r2_sure_axis_explains_residual_p_sure_after_evidence": 0.36,
            }
        ],
        "metrics": {
            "corr_sure_axis_with_sure_residual_after_evidence": {
                "mean": 0.6,
                "std": 0.0,
                "min": 0.6,
                "max": 0.6,
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = write_aggregate_outputs(tmp_dir, aggregate)
        assert os.path.getsize(paths["summary_json"]) > 0
        assert os.path.getsize(paths["per_seed_csv"]) > 0
        with open(paths["summary_json"], encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["pass"] is True


def main():
    test_aggregate_seed_summaries_computes_mean_std_and_thresholds()
    test_aggregate_seed_summaries_fails_when_required_metric_is_missing()
    test_write_aggregate_outputs_creates_json_and_csv()
    print("rdm multiseed validation test passed")


if __name__ == "__main__":
    main()
