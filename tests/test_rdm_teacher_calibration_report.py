import csv
import json
import os
import tempfile

import numpy as np

from rdm_teacher_calibration_report import (
    build_calibration_bins,
    load_teacher_dataset,
    summarize_teacher_calibration,
    write_teacher_calibration_outputs,
)


def _write_tiny_dataset(path):
    trial_info = []
    coherences = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
    sure_available = [True, False, True, False, True, False]
    for idx, coh in enumerate(coherences):
        trial_info.append(
            json.dumps(
                {
                    "coh": coh,
                    "sure_available": sure_available[idx],
                    "stimulus_dur": 600 + 10 * idx,
                }
            )
        )
    np.savez_compressed(
        path,
        trial_info_json=np.asarray(trial_info, dtype=object),
        teacher_internal_margin=np.asarray([0.1, 0.3, 0.8, 1.4, 2.0, 3.0], dtype=float),
        teacher_direction_success_proxy=np.asarray([0.52, 0.55, 0.65, 0.75, 0.86, 0.95], dtype=float),
        teacher_expected_direction_value=np.asarray([0.52, 0.55, 0.65, 0.75, 0.86, 0.95], dtype=float),
        teacher_sure_strength=np.asarray([0.8, 0.0, 0.7, 0.0, 0.2, 0.0], dtype=float),
        external_sensory_margin=np.asarray([0.2, 0.4, 0.7, 1.0, 1.9, 3.5], dtype=float),
        external_sure_strength=np.asarray([0.78, 0.0, 0.72, 0.0, 0.25, 0.0], dtype=float),
        internal_margin_center_used=np.asarray(1.1, dtype=float),
        internal_margin_temp_used=np.asarray(0.5, dtype=float),
        internal_margin_calibration_mode=np.asarray("auto_quantile"),
        internal_margin_n_calibration_trials=np.asarray(3, dtype=int),
    )


def test_load_teacher_dataset_extracts_required_vectors():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "tiny_teacher_dataset.npz")
        _write_tiny_dataset(path)

        dataset = load_teacher_dataset(path)

        assert dataset["n_trials"] == 6
        assert dataset["sure_available"].sum() == 3
        assert dataset["coherence"][0] == 0.0
        assert dataset["teacher_internal_margin"][-1] == 3.0


def test_build_calibration_bins_orders_margin_and_averages_targets():
    margin = np.asarray([0.1, 0.2, 1.0, 1.2, 2.5, 2.8], dtype=float)
    p_correct = np.asarray([0.51, 0.52, 0.65, 0.68, 0.90, 0.94], dtype=float)
    expected_value = p_correct.copy()
    sure_strength = np.asarray([0.9, 0.8, 0.5, 0.4, 0.1, 0.05], dtype=float)
    coherence = np.asarray([0.0, 0.0, 0.064, 0.064, 0.512, 0.512], dtype=float)
    sure_available = np.asarray([True, True, True, True, True, True])

    rows = build_calibration_bins(
        margin,
        p_correct,
        expected_value,
        sure_strength,
        coherence,
        sure_available,
        n_bins=3,
    )

    assert len(rows) == 3
    assert rows[0]["internal_margin_mean"] < rows[-1]["internal_margin_mean"]
    assert rows[0]["p_correct_proxy_mean"] < rows[-1]["p_correct_proxy_mean"]
    assert rows[0]["sure_target_mean"] > rows[-1]["sure_target_mean"]


def test_summary_and_outputs_include_proxy_boundary_and_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "tiny_teacher_dataset.npz")
        output_dir = os.path.join(tmp_dir, "exports")
        figure_dir = os.path.join(tmp_dir, "figures")
        _write_tiny_dataset(dataset_path)
        dataset = load_teacher_dataset(dataset_path)
        bins = build_calibration_bins(
            dataset["teacher_internal_margin"],
            dataset["teacher_direction_success_proxy"],
            dataset["teacher_expected_direction_value"],
            dataset["teacher_sure_strength"],
            dataset["coherence"],
            dataset["sure_available"],
            n_bins=3,
        )
        summary = summarize_teacher_calibration(dataset, bins)

        assert summary["calibration_target"] == "teacher_direction_success_proxy"
        assert summary["empirical_teacher_accuracy_available"] is False
        assert summary["n_trials"] == 6
        assert summary["corr_internal_margin_p_correct_proxy"] > 0.9

        paths = write_teacher_calibration_outputs(output_dir, figure_dir, summary, bins)
        assert os.path.getsize(paths["summary_json"]) > 0
        assert os.path.getsize(paths["bins_csv"]) > 0
        assert os.path.getsize(paths["figure"]) > 0
        with open(paths["bins_csv"], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["bin_index"] == "0"


def main():
    test_load_teacher_dataset_extracts_required_vectors()
    test_build_calibration_bins_orders_margin_and_averages_targets()
    test_summary_and_outputs_include_proxy_boundary_and_files()
    print("rdm teacher calibration report test passed")


if __name__ == "__main__":
    main()
