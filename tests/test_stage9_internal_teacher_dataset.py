import numpy as np
import os
import tempfile

from generate_stage9_internal_teacher_dataset import (
    build_internal_teacher_records,
    build_stage9_targets_and_masks,
    collect_internal_teacher_dataset,
    compute_output_internal_margin,
    get_internal_readout_step,
    estimate_internal_margin_calibration,
    save_internal_teacher_dataset,
    summarize_internal_teacher_dataset,
)


def make_outputs(left_value, right_value, n_steps=20):
    outputs = np.zeros((n_steps, 4), dtype=float)
    outputs[:, 1] = float(left_value)
    outputs[:, 2] = float(right_value)
    return outputs


def test_compute_output_internal_margin_uses_window_average():
    outputs = make_outputs(0.0, 0.0)
    outputs[8:12, 1] = 0.8
    outputs[8:12, 2] = 0.2

    margin = compute_output_internal_margin(outputs, center_step=10, half_window=2)

    assert abs(margin - 0.6) < 1e-12


def test_internal_readout_step_supports_pre_go_anchor():
    info = {
        "sure_available": True,
        "stimulus_end": 40,
        "ts_onset": 90,
        "delay_end": 130,
    }

    assert get_internal_readout_step(info, readout_anchor="ts_onset") == 90
    assert get_internal_readout_step(info, readout_anchor="pre_go", pre_go_offset_steps=7) == 123
    assert get_internal_readout_step(info, readout_anchor="go") == 130
    assert get_internal_readout_step(info, readout_anchor="stimulus_end") == 40


def test_internal_teacher_records_decrease_sure_strength_with_internal_margin():
    trial_info = [
        {
            "sure_available": True,
            "sensory_margin": 0.5,
            "sure_strength": 0.8,
            "ts_onset": 10,
            "delay_end": 15,
        },
        {
            "sure_available": True,
            "sensory_margin": 0.5,
            "sure_strength": 0.8,
            "ts_onset": 10,
            "delay_end": 15,
        },
    ]
    outputs = np.stack(
        [
            make_outputs(0.55, 0.45),
            make_outputs(0.95, 0.05),
        ],
        axis=0,
    )

    records = build_internal_teacher_records(outputs, trial_info, half_window=2)

    assert records["teacher_internal_margin"][0] < records["teacher_internal_margin"][1]
    assert records["teacher_sure_strength"][0] > records["teacher_sure_strength"][1]
    assert records["margin_source"][0] == "internal_output"


def test_estimate_internal_margin_calibration_uses_available_distribution():
    margins = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 5.0])
    sure_available = np.array([True, True, True, True, True, False])

    calibration = estimate_internal_margin_calibration(margins, sure_available)

    assert abs(calibration["margin_center"] - 0.5) < 1e-12
    assert abs(calibration["margin_temp"] - 0.2) < 1e-12
    assert calibration["calibration_mode"] == "auto_quantile"
    assert calibration["n_calibration_trials"] == 5


def test_internal_teacher_records_auto_calibrate_internal_margin_scale():
    trial_info = []
    outputs = []
    for margin in [0.1, 0.3, 0.5, 0.7, 0.9]:
        trial_info.append(
            {
                "sure_available": True,
                "sensory_margin": 0.5,
                "sure_strength": 0.8,
                "ts_onset": 10,
                "delay_end": 15,
            }
        )
        outputs.append(make_outputs(0.5 + margin / 2.0, 0.5 - margin / 2.0))

    records = build_internal_teacher_records(
        np.stack(outputs, axis=0),
        trial_info,
        half_window=2,
        internal_calibration="auto",
    )

    assert abs(float(records["internal_margin_center_used"]) - 0.5) < 1e-12
    assert abs(float(records["internal_margin_temp_used"]) - 0.2) < 1e-12
    assert records["internal_margin_calibration_mode"] == "auto_quantile"
    assert records["teacher_sure_strength"][0] > records["teacher_sure_strength"][-1]
    assert records["teacher_sure_strength"][0] < 0.95
    assert records["teacher_sure_strength"][-1] > 0.05


def test_internal_teacher_records_can_use_pre_go_readout_anchor():
    trial_info = [
        {
            "sure_available": True,
            "sensory_margin": 0.5,
            "sure_strength": 0.8,
            "ts_onset": 8,
            "delay_end": 16,
        }
    ]
    outputs = np.zeros((1, 20, 4), dtype=float)
    outputs[0, 6:10, 1] = 0.55
    outputs[0, 6:10, 2] = 0.45
    outputs[0, 13:17, 1] = 0.95
    outputs[0, 13:17, 2] = 0.05

    ts_records = build_internal_teacher_records(
        outputs,
        trial_info,
        half_window=2,
        internal_calibration="manual",
        margin_center=0.5,
        margin_temp=0.2,
        readout_anchor="ts_onset",
    )
    pre_go_records = build_internal_teacher_records(
        outputs,
        trial_info,
        half_window=2,
        internal_calibration="manual",
        margin_center=0.5,
        margin_temp=0.2,
        readout_anchor="pre_go",
        pre_go_offset_steps=1,
    )

    assert ts_records["teacher_internal_margin"][0] < pre_go_records["teacher_internal_margin"][0]
    assert pre_go_records["internal_readout_anchor"] == "pre_go"


def test_internal_teacher_records_keep_unavailable_sure_strength_zero():
    trial_info = [
        {
            "sure_available": False,
            "sensory_margin": 0.5,
            "sure_strength": 0.0,
            "ts_onset": 10,
            "delay_end": 15,
        }
    ]
    outputs = np.stack([make_outputs(0.55, 0.45)], axis=0)

    records = build_internal_teacher_records(outputs, trial_info, half_window=2)

    assert records["teacher_sure_strength"][0] == 0.0
    assert records["external_sensory_margin"][0] == 0.5
    assert records["external_sure_strength"][0] == 0.0


class FakeTask:
    def __init__(self):
        self.batch_index = 0
        self.trial_info = []

    def reset_trial_info(self):
        self.trial_info = []

    def get_trial_batch(self):
        x = np.full((2, 20, 4), float(self.batch_index), dtype=float)
        y = np.zeros((2, 20, 4), dtype=float)
        mask = np.ones((2, 20, 4), dtype=float)
        self.trial_info = [
            {
                "dir_choice": 0,
                "sure_available": True,
                "fixation_end": 2,
                "sensory_margin": 0.5,
                "sure_strength": 0.8,
                "ts_onset": 10,
                "delay_end": 15,
                "batch_index": self.batch_index,
                "trial_index": 0,
            },
            {
                "dir_choice": 1,
                "sure_available": False,
                "fixation_end": 2,
                "sensory_margin": 3.0,
                "sure_strength": 0.0,
                "ts_onset": 10,
                "delay_end": 15,
                "batch_index": self.batch_index,
                "trial_index": 1,
            },
        ]
        self.batch_index += 1
        return x, y, mask, None


class FakeModel:
    def test(self, batch_x):
        outputs = np.zeros_like(batch_x)
        outputs[:, :, 1] = 0.7
        outputs[:, :, 2] = 0.2
        states = np.zeros((batch_x.shape[0], batch_x.shape[1], 3), dtype=float)
        return outputs, states


def test_collect_internal_teacher_dataset_concatenates_batches_and_records():
    dataset = collect_internal_teacher_dataset(
        model=FakeModel(),
        task=FakeTask(),
        n_batches=2,
        half_window=2,
    )

    assert dataset["x"].shape == (4, 20, 4)
    assert dataset["y"].shape == (4, 20, 4)
    assert dataset["mask"].shape == (4, 20, 4)
    assert len(dataset["trial_info"]) == 4
    assert dataset["teacher_internal_margin"].shape == (4,)
    assert dataset["teacher_sure_strength"][1] == 0.0
    assert dataset["trial_info"][2]["batch_index"] == 1


def test_save_internal_teacher_dataset_writes_expected_npz_fields():
    dataset = collect_internal_teacher_dataset(
        model=FakeModel(),
        task=FakeTask(),
        n_batches=1,
        half_window=2,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "stage9_dataset.npz")
        save_internal_teacher_dataset(
            path,
            dataset["x"],
            dataset["y"],
            dataset["mask"],
            dataset["trial_info"],
            dataset["teacher_records"],
            y_internal=dataset["y_internal"],
            mask_internal=dataset["mask_internal"],
        )
        loaded = np.load(path, allow_pickle=True)

        assert loaded["x"].shape == (2, 20, 4)
        assert loaded["y_internal"].shape == (2, 20, 4)
        assert loaded["mask_internal"].shape == (2, 20, 4)
        assert loaded["teacher_internal_margin"].shape == (2,)
        assert loaded["trial_info_json"].shape == (2,)


def test_build_stage9_targets_uses_internal_sure_strength_after_go():
    trial_info = [
        {
            "dir_choice": 0,
            "sure_available": True,
            "fixation_end": 2,
            "ts_onset": 8,
            "delay_end": 12,
        }
    ]

    y, mask = build_stage9_targets_and_masks(
        trial_info,
        teacher_sure_strength=np.array([0.7]),
        n_steps=20,
        n_out=4,
    )

    assert y[0, 0, 0] == 1.0
    assert abs(float(y[0, 12, 1]) - 0.3) < 1e-7
    assert y[0, 12, 2] == 0.0
    assert abs(float(y[0, 12, 3]) - 0.7) < 1e-7
    assert mask[0, 11, 1] == 0.0
    assert mask[0, 12, 1] == 1.0


def test_build_stage9_targets_keeps_unavailable_trial_as_direction_choice():
    trial_info = [
        {
            "dir_choice": 1,
            "sure_available": False,
            "fixation_end": 2,
            "ts_onset": 8,
            "delay_end": 12,
        }
    ]

    y, mask = build_stage9_targets_and_masks(
        trial_info,
        teacher_sure_strength=np.array([0.7]),
        n_steps=20,
        n_out=4,
    )

    assert y[0, 12, 1] == 0.0
    assert y[0, 12, 2] == 1.0
    assert y[0, 12, 3] == 0.0
    assert mask[0, 1, 1] == 1.0
    assert mask[0, 2, 1] == 0.0


def test_summarize_internal_teacher_dataset_reports_key_fields():
    dataset = collect_internal_teacher_dataset(
        model=FakeModel(),
        task=FakeTask(),
        n_batches=1,
        half_window=2,
    )

    summary = summarize_internal_teacher_dataset(dataset)

    assert summary["n_trials"] == 2
    assert summary["n_sure_available"] == 1
    assert abs(summary["mean_teacher_internal_margin"] - 0.5) < 1e-12
    assert "mean_teacher_sure_strength" in summary
    assert "mean_external_sensory_margin" in summary
    assert "internal_margin_center_used" in summary
    assert "internal_margin_temp_used" in summary
    assert abs(summary["mean_stage9_post_go_sure_target"] - dataset["teacher_sure_strength"][0]) < 1e-7


def main():
    test_compute_output_internal_margin_uses_window_average()
    test_internal_readout_step_supports_pre_go_anchor()
    test_internal_teacher_records_decrease_sure_strength_with_internal_margin()
    test_estimate_internal_margin_calibration_uses_available_distribution()
    test_internal_teacher_records_auto_calibrate_internal_margin_scale()
    test_internal_teacher_records_can_use_pre_go_readout_anchor()
    test_internal_teacher_records_keep_unavailable_sure_strength_zero()
    test_collect_internal_teacher_dataset_concatenates_batches_and_records()
    test_save_internal_teacher_dataset_writes_expected_npz_fields()
    test_build_stage9_targets_uses_internal_sure_strength_after_go()
    test_build_stage9_targets_keeps_unavailable_trial_as_direction_choice()
    test_summarize_internal_teacher_dataset_reports_key_fields()
    print("stage9 internal teacher dataset test passed")


if __name__ == "__main__":
    main()
