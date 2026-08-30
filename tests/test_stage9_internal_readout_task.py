import os
import tempfile

import numpy as np

from generate_stage9_internal_teacher_dataset import save_internal_teacher_dataset
from sure_target_stage9_internal_readout import (
    RDM_SureTarget_InternalReadoutDatasetTask,
    load_stage9_dataset,
)


def make_test_dataset(path):
    x = np.zeros((3, 20, 4), dtype=np.float32)
    y = np.zeros((3, 20, 4), dtype=np.float32)
    mask = np.ones((3, 20, 4), dtype=np.float32)
    y_internal = np.zeros((3, 20, 4), dtype=np.float32)
    mask_internal = np.ones((3, 20, 4), dtype=np.float32)
    y_internal[:, 12:, 3] = np.array([0.2, 0.6, 0.0], dtype=np.float32)[:, None]
    trial_info = [
        {
            "dir_choice": 0,
            "sure_available": True,
            "fixation_end": 2,
            "stimulus_end": 10,
            "ts_onset": 11,
            "delay_end": 12,
            "coh": 0.032,
        },
        {
            "dir_choice": 1,
            "sure_available": True,
            "fixation_end": 2,
            "stimulus_end": 10,
            "ts_onset": 11,
            "delay_end": 12,
            "coh": 0.128,
        },
        {
            "dir_choice": 0,
            "sure_available": False,
            "fixation_end": 2,
            "stimulus_end": 10,
            "ts_onset": 11,
            "delay_end": 12,
            "coh": 0.512,
        },
    ]
    teacher_records = {
        "teacher_internal_margin": np.array([0.5, 1.0, 2.0], dtype=float),
        "teacher_sure_strength": np.array([0.2, 0.6, 0.0], dtype=float),
        "teacher_expected_direction_value": np.array([0.4, 0.7, 0.9], dtype=float),
        "teacher_direction_success_proxy": np.array([0.4, 0.7, 0.9], dtype=float),
        "external_sensory_margin": np.array([1.0, 2.0, 3.0], dtype=float),
        "external_sure_strength": np.array([0.8, 0.3, 0.0], dtype=float),
    }
    save_internal_teacher_dataset(
        path,
        x,
        y,
        mask,
        trial_info,
        teacher_records,
        y_internal=y_internal,
        mask_internal=mask_internal,
    )
    return x, y_internal, mask_internal, trial_info


def test_load_stage9_dataset_reads_internal_targets_and_metadata():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        _, y_internal, _, trial_info = make_test_dataset(path)

        dataset = load_stage9_dataset(path)

        assert dataset["x"].shape == (3, 20, 4)
        assert np.allclose(dataset["y_internal"], y_internal)
        assert dataset["trial_info"][1]["coh"] == trial_info[1]["coh"]
        assert dataset["teacher_sure_strength"][1] == 0.6


def test_dataset_task_returns_internal_targets_in_batches():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        _, y_internal, mask_internal, _ = make_test_dataset(path)
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=path,
            dt=10,
            tau=100,
            N_batch=2,
            sample_mode="sequential",
        )

        batch_x, batch_y, batch_mask, params = task.get_trial_batch()

        assert batch_x.shape == (2, 20, 4)
        assert np.allclose(batch_y[0], y_internal[0])
        assert np.allclose(batch_y[1], y_internal[1])
        assert np.allclose(batch_mask[0], mask_internal[0])
        assert params[0]["dataset_index"] == 0
        assert task.trial_info[1]["dataset_index"] == 1
        assert task.trial_info[1]["teacher_sure_strength"] == 0.6


def test_dataset_task_wraps_sequential_indices():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "dataset.npz")
        make_test_dataset(path)
        task = RDM_SureTarget_InternalReadoutDatasetTask(
            dataset_path=path,
            dt=10,
            tau=100,
            N_batch=2,
            sample_mode="sequential",
        )

        task.get_trial_batch()
        _, _, _, params = task.get_trial_batch()

        assert [int(p["dataset_index"]) for p in params] == [2, 0]


def main():
    test_load_stage9_dataset_reads_internal_targets_and_metadata()
    test_dataset_task_returns_internal_targets_in_batches()
    test_dataset_task_wraps_sequential_indices()
    print("stage9 internal readout task test passed")


if __name__ == "__main__":
    main()
