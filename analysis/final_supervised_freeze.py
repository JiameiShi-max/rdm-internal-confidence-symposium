"""Final train/validation/test freeze for the categorical supervised student.

This module deliberately contains no population-representation analysis.  It
only creates/reuses one trial-identity split, trains the three predetermined
student seeds, evaluates validation and held-out test trials, and freezes the
checkpoints and provenance needed by later analyses.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rdm_task_params import task_params_from_task
from sure_target_stage9_internal_readout import (
    RDM_SureTarget_InternalReadoutDatasetTask,
    build_post_go_output_records,
    configure_student_action_objective,
    load_stage9_dataset,
    readout_post_go,
    set_global_seed,
)


SPLIT_SEED = 2026
STUDENT_SEEDS = (7, 8, 9)
EXAMPLE_SEED = 7
OBJECTIVE = "categorical_soft"
TRAINING_ITERS = 50000
LOSS_EPOCH = 1000
BATCH_SIZE = 50
N_REC = 50
DT_MS = 10
TAU_MS = 100
LEARNING_RATE = 0.001
RECURRENT_NOISE = 0.05
ACTIVATION = "rectified_linear"
LAMBDA_FIX_POSTGO = 1.0
LAMBDA_ACTION = 1.0
READOUT_WINDOW = 10


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _jsonable(value.item())
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_split(train_indices, validation_indices, test_indices, n_trials):
    arrays = {
        "train": np.asarray(train_indices, dtype=np.int64),
        "validation": np.asarray(validation_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }
    for name, values in arrays.items():
        if values.ndim != 1:
            raise AssertionError(f"{name} indices must be one-dimensional")
        if np.unique(values).size != values.size:
            raise AssertionError(f"{name} indices contain duplicates")
        if np.any(values < 0) or np.any(values >= int(n_trials)):
            raise AssertionError(f"{name} indices are out of range")
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        overlap = np.intersect1d(arrays[left], arrays[right])
        if overlap.size:
            raise AssertionError(f"{left}/{right} overlap: {overlap[:10].tolist()}")
    union = np.concatenate(list(arrays.values()))
    if union.size != int(n_trials) or not np.array_equal(
        np.sort(union), np.arange(int(n_trials), dtype=np.int64)
    ):
        raise AssertionError("split union does not equal all dataset trial identities")
    return arrays


def create_or_load_split(output_dir, n_trials, dataset_sha256, resume=False):
    output_dir = Path(output_dir)
    split_path = output_dir / "split_indices.npz"
    summary_path = output_dir / "split_summary.json"
    if split_path.exists():
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing split: {split_path}. Use --resume to reuse it."
            )
        with np.load(split_path, allow_pickle=False) as stored:
            split_seed = int(stored["split_seed"])
            stored_n_trials = int(stored["n_trials"])
            stored_dataset_sha256 = str(stored["dataset_sha256"].item())
            arrays = validate_split(
                stored["train_indices"],
                stored["validation_indices"],
                stored["test_indices"],
                n_trials,
            )
        if split_seed != SPLIT_SEED or stored_n_trials != int(n_trials):
            raise ValueError("existing split metadata does not match the frozen split specification")
        if stored_dataset_sha256 != str(dataset_sha256):
            raise ValueError("existing split belongs to a different teacher dataset")
        return arrays, split_path, summary_path

    rng = np.random.RandomState(SPLIT_SEED)
    permutation = rng.permutation(int(n_trials)).astype(np.int64)
    n_train = int(np.floor(0.70 * n_trials))
    n_validation = int(np.floor(0.15 * n_trials))
    arrays = validate_split(
        permutation[:n_train],
        permutation[n_train : n_train + n_validation],
        permutation[n_train + n_validation :],
        n_trials,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        split_path,
        train_indices=arrays["train"],
        validation_indices=arrays["validation"],
        test_indices=arrays["test"],
        split_seed=np.asarray(SPLIT_SEED, dtype=np.int64),
        n_trials=np.asarray(n_trials, dtype=np.int64),
        dataset_sha256=np.asarray(str(dataset_sha256)),
    )
    write_json(
        summary_path,
        {
            "dataset_sha256": dataset_sha256,
            "split_seed": SPLIT_SEED,
            "n_dataset_trials": int(n_trials),
            "counts": {name: int(values.size) for name, values in arrays.items()},
            "fractions": {
                name: float(values.size / n_trials) for name, values in arrays.items()
            },
            "checks": {
                "train_validation_overlap": 0,
                "train_test_overlap": 0,
                "validation_test_overlap": 0,
                "union_equals_all_trials": True,
                "all_timepoints_follow_trial_identity": True,
            },
        },
    )
    return arrays, split_path, summary_path


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if np.sum(usable) < 2 or np.std(x[usable]) == 0 or np.std(y[usable]) == 0:
        return np.nan
    return float(np.corrcoef(x[usable], y[usable])[0, 1])


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if np.sum(usable) < 2:
        return np.nan
    return pearson(_rankdata(x[usable]), _rankdata(y[usable]))


def _direction_correct(choice, direction):
    return bool((int(choice) == 1 and int(direction) == 0) or (int(choice) == 2 and int(direction) == 1))


def evaluate_model(model, dataset_path, indices, seed, split_name):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size % BATCH_SIZE:
        raise ValueError(f"{split_name} size must be divisible by frozen batch size {BATCH_SIZE}")
    task = RDM_SureTarget_InternalReadoutDatasetTask(
        dataset_path=dataset_path,
        dt=DT_MS,
        tau=TAU_MS,
        N_batch=BATCH_SIZE,
        sample_mode="sequential",
        seed=seed,
        allowed_indices=indices,
    )
    raw_batches = []
    target_batches = []
    info = []
    for _ in range(indices.size // BATCH_SIZE):
        task.reset_trial_info()
        batch_x, batch_y, _, params = task.get_trial_batch()
        observed = np.asarray([int(item["dataset_index"]) for item in params], dtype=np.int64)
        expected = indices[len(info) : len(info) + BATCH_SIZE]
        if not np.array_equal(observed, expected):
            raise AssertionError(f"{split_name} evaluator escaped or reordered its allowed indices")
        raw, _ = model.test(batch_x)
        raw_batches.append(raw)
        target_batches.append(batch_y)
        info.extend(task.trial_info)
    raw_outputs = np.concatenate(raw_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    action_logits = raw_outputs[:, :, 1:4]
    shifted = action_logits - np.max(action_logits, axis=2, keepdims=True)
    action_probabilities = np.exp(shifted)
    action_probabilities /= np.sum(action_probabilities, axis=2, keepdims=True)
    outputs = raw_outputs.copy()
    outputs[:, :, 1:4] = action_probabilities
    records = build_post_go_output_records(
        outputs,
        targets,
        info,
        objective=OBJECTIVE,
        readout_window=READOUT_WINDOW,
        raw_outputs=raw_outputs,
    )
    for record in records:
        choice = int(record["final_argmax_choice"])
        record["split"] = split_name
        record["direction_correct"] = _direction_correct(choice, record["dir_choice"])
        record["chose_sure"] = choice == 3
    if not np.array_equal(
        np.asarray([record["dataset_index"] for record in records], dtype=np.int64), indices
    ):
        raise AssertionError(f"{split_name} records do not match their exact source trial identities")
    return records, raw_outputs


def summarize_records(records, raw_outputs):
    offered = np.asarray([bool(row["sure_available"]) for row in records], dtype=bool)
    choices = np.asarray([int(row["final_argmax_choice"]) for row in records], dtype=int)
    directions = np.asarray([int(row["dir_choice"]) for row in records], dtype=int)
    correct = np.asarray(
        [_direction_correct(choice, direction) for choice, direction in zip(choices, directions)],
        dtype=bool,
    )
    chose_direction = np.isin(choices, [1, 2])
    soft_sure = np.asarray([float(row["p_sure"]) for row in records], dtype=float)
    coherence = np.asarray([float(row["stimulus_coherence"]) for row in records], dtype=float)
    offered_direction = offered & chose_direction
    no_sure = ~offered

    coherence_rows = []
    for value in sorted(np.unique(coherence[np.isfinite(coherence)])):
        selected = coherence == value
        selected_offered = selected & offered
        selected_direction = selected & chose_direction
        coherence_rows.append(
            {
                "coherence": float(value),
                "n_trials": int(np.sum(selected)),
                "n_sure_offered": int(np.sum(selected_offered)),
                "p_sure_given_offered": (
                    float(np.mean(choices[selected_offered] == 3))
                    if np.any(selected_offered)
                    else np.nan
                ),
                "mean_predicted_p_sure_given_offered": (
                    float(np.mean(soft_sure[selected_offered]))
                    if np.any(selected_offered)
                    else np.nan
                ),
                "direction_accuracy": (
                    float(np.mean(correct[selected_direction]))
                    if np.any(selected_direction)
                    else np.nan
                ),
                "n_direction_choices": int(np.sum(selected_direction)),
            }
        )

    def values(key):
        return np.asarray([float(row[key]) for row in records], dtype=float)

    correlations = {
        "teacher_sure_strength": pearson(soft_sure[offered], values("teacher_sure_strength")[offered]),
        "empirical_p_correct": pearson(soft_sure[offered], values("empirical_p_correct")[offered]),
        "internal_margin": pearson(soft_sure[offered], values("internal_margin")[offered]),
        "coherence": pearson(soft_sure[offered], coherence[offered]),
        "stimulus_duration": pearson(soft_sure[offered], values("stimulus_duration_ms")[offered]),
    }
    coh_x = np.asarray([row["coherence"] for row in coherence_rows], dtype=float)
    coh_y = np.asarray([row["p_sure_given_offered"] for row in coherence_rows], dtype=float)
    coherence_trend = spearman(coh_x, coh_y)
    finite_outputs = bool(np.all(np.isfinite(raw_outputs)) and np.all(np.isfinite(soft_sure)))
    n_sure = int(np.sum(offered & (choices == 3)))
    n_offered_direction = int(np.sum(offered_direction))
    summary = {
        "n_trials": int(len(records)),
        "n_sure_offered": int(np.sum(offered)),
        "p_sure_given_offered": float(np.mean(choices[offered] == 3)) if np.any(offered) else np.nan,
        "no_sure_direction_accuracy": float(np.mean(correct[no_sure])) if np.any(no_sure) else np.nan,
        "offered_direction_choice_accuracy": (
            float(np.mean(correct[offered_direction])) if np.any(offered_direction) else np.nan
        ),
        "choice_proportions": {
            "left": float(np.mean(choices == 1)),
            "right": float(np.mean(choices == 2)),
            "sure": float(np.mean(choices == 3)),
        },
        "coherence": coherence_rows,
        "correlations_predicted_p_sure_offered": correlations,
        "qc": {
            "nondegenerate_direction_behavior": bool(np.any(choices == 1) and np.any(choices == 2)),
            "nondegenerate_sure_behavior": bool(n_sure > 0 and n_offered_direction > 0),
            "p_sure_generally_decreases_with_coherence": bool(
                np.isfinite(coherence_trend) and coherence_trend < 0
            ),
            "p_sure_coherence_spearman": coherence_trend,
            "p_sure_negatively_related_to_empirical_p_correct": bool(
                np.isfinite(correlations["empirical_p_correct"])
                and correlations["empirical_p_correct"] < 0
            ),
            "p_sure_negatively_related_to_internal_margin": bool(
                np.isfinite(correlations["internal_margin"])
                and correlations["internal_margin"] < 0
            ),
            "finite_stable_network_outputs": finite_outputs,
        },
    }
    return summary


def _network_params(task):
    params = task_params_from_task(task)
    params.update(
        {
            "name": "RDM_SureTarget_FinalCategoricalStudent",
            "N_rec": N_REC,
            "rec_noise": RECURRENT_NOISE,
            "activation": ACTIVATION,
            "alpha": DT_MS / TAU_MS,
        }
    )
    return configure_student_action_objective(
        params,
        OBJECTIVE,
        lambda_fix_postgo=LAMBDA_FIX_POSTGO,
        lambda_action=LAMBDA_ACTION,
    )


def _checkpoint_reload_reference(weights_path, dataset, test_indices, output_path):
    from psychrnn.backend.simulation import BasicSimulator

    reference_indices = np.asarray(test_indices[:8], dtype=np.int64)
    inputs = np.asarray(dataset["x"][reference_indices], dtype=np.float32)
    weights = dict(np.load(weights_path, allow_pickle=True))
    simulator = BasicSimulator(
        params={"alpha": DT_MS / TAU_MS, "rec_noise": 0.0}, weights=weights
    )
    expected_outputs, expected_states = simulator.run_trials(inputs)
    reloaded_weights = dict(np.load(weights_path, allow_pickle=True))
    reloaded = BasicSimulator(
        params={"alpha": DT_MS / TAU_MS, "rec_noise": 0.0}, weights=reloaded_weights
    )
    observed_outputs, observed_states = reloaded.run_trials(inputs)
    np.testing.assert_allclose(observed_outputs, expected_outputs, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(observed_states, expected_states, rtol=1e-7, atol=1e-7)
    np.savez(
        output_path,
        dataset_indices=reference_indices,
        raw_outputs=expected_outputs,
        states=expected_states,
        rec_noise=np.asarray(0.0),
    )
    return {
        "n_reference_trials": int(reference_indices.size),
        "rtol": 1e-7,
        "atol": 1e-7,
        "max_abs_output_difference": float(np.max(np.abs(observed_outputs - expected_outputs))),
        "max_abs_state_difference": float(np.max(np.abs(observed_states - expected_states))),
        "passed": True,
    }


def _dataset_frozen_config(dataset):
    info = dataset["trial_info"]
    scalar = {}
    for key in (
        "confidence_mapping",
        "empirical_confidence_coefficients",
        "internal_margin_center_used",
        "internal_margin_temp_used",
        "internal_margin_calibration_mode",
        "internal_readout_anchor",
        "internal_pre_go_offset_steps",
    ):
        if key in dataset:
            scalar[key] = _jsonable(np.asarray(dataset[key]))
    sure_rewards = []
    if "teacher_sure_advantage" in dataset and "teacher_expected_direction_value" in dataset:
        sure_rewards = np.asarray(dataset["teacher_sure_advantage"], dtype=float) + np.asarray(
            dataset["teacher_expected_direction_value"], dtype=float
        )
    return {
        **scalar,
        "sure_reward_inferred_from_teacher_values": (
            float(np.median(sure_rewards[np.isfinite(sure_rewards)])) if np.any(np.isfinite(sure_rewards)) else None
        ),
        "sure_offer_probabilities": sorted(
            {float(item.get("sure_offer_prob", np.nan)) for item in info if np.isfinite(item.get("sure_offer_prob", np.nan))}
        ),
        "n_steps": int(dataset["x"].shape[1]),
        "n_inputs": int(dataset["x"].shape[2]),
        "n_outputs": int(dataset["y_internal"].shape[2]),
        "stimulus_duration_ms_range": [
            int(min(item.get("stimulus_duration", item.get("stimulus_dur")) for item in info)),
            int(max(item.get("stimulus_duration", item.get("stimulus_dur")) for item in info)),
        ],
        "ts_latency_ms_range": [
            int(min(item.get("ts_latency_from_motion_offset", item.get("ts_delay")) for item in info)),
            int(max(item.get("ts_latency_from_motion_offset", item.get("ts_delay")) for item in info)),
        ],
    }


def frozen_config(dataset):
    return {
        "student_action_objective": OBJECTIVE,
        "categorical_loss": "retained legacy masked MSE plus post-go Left/Right/Sure soft-label categorical cross-entropy",
        "lambda_fix_postgo": LAMBDA_FIX_POSTGO,
        "lambda_action": LAMBDA_ACTION,
        "learning_rate": LEARNING_RATE,
        "training_iters": TRAINING_ITERS,
        "loss_epoch": LOSS_EPOCH,
        "batch_size": BATCH_SIZE,
        "n_recurrent_units": N_REC,
        "recurrent_noise": RECURRENT_NOISE,
        "activation": ACTIVATION,
        "dt_ms": DT_MS,
        "tau_ms": TAU_MS,
        "alpha": DT_MS / TAU_MS,
        "post_go_behavioral_readout": "mean over 10 steps; argmax of categorical Left/Right/Sure probabilities; fixation excluded",
        "teacher_dataset_configuration": _dataset_frozen_config(dataset),
    }


def run_seed(seed, dataset_path, dataset, split, output_dir, resume=False):
    import tensorflow as tf
    from psychrnn.backend.models.basic import Basic

    seed_dir = Path(output_dir) / f"seed{seed}"
    complete_path = seed_dir / "test_summary.json"
    if complete_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite completed seed: {seed_dir}")
        with complete_path.open(encoding="utf-8") as handle:
            prior = json.load(handle)
        if prior.get("technical_validity") is True:
            return prior
        raise ValueError(f"Existing seed directory is incomplete or invalid: {seed_dir}")
    partial_required = [
        seed_dir / "weights.npz",
        seed_dir / "validation_summary.json",
        seed_dir / "test_trials.csv",
        seed_dir / "reconstruction_reference.npz",
    ]
    if resume and all(path.exists() for path in partial_required):
        rows = read_csv(seed_dir / "test_trials.csv")
        bool_fields = ("sure_available", "direction_correct", "chose_sure")
        int_fields = ("final_argmax_choice", "dir_choice", "dataset_index")
        float_fields = (
            "p_sure",
            "teacher_sure_strength",
            "empirical_p_correct",
            "internal_margin",
            "stimulus_coherence",
            "stimulus_duration_ms",
        )
        for row in rows:
            for key in bool_fields:
                row[key] = str(row[key]).lower() in ("true", "1")
            for key in int_fields:
                row[key] = int(row[key])
            for key in float_fields:
                row[key] = float(row[key])
        test_summary = summarize_records(
            rows,
            np.asarray(
                [[row["predicted_left"], row["predicted_right"], row["predicted_sure"]] for row in rows],
                dtype=float,
            ),
        )
        with (seed_dir / "validation_summary.json").open(encoding="utf-8") as handle:
            validation_summary = json.load(handle)
        reload_check = _checkpoint_reload_reference(
            seed_dir / "weights.npz",
            dataset,
            split["test"],
            seed_dir / "reconstruction_reference.npz",
        )
        weights_finite = True
        with np.load(seed_dir / "weights.npz", allow_pickle=True) as weights:
            for key in weights.files:
                if np.issubdtype(weights[key].dtype, np.number) and not np.all(np.isfinite(weights[key])):
                    weights_finite = False
                    break
        technical_validity = bool(
            weights_finite
            and validation_summary["qc"]["finite_stable_network_outputs"]
            and test_summary["qc"]["finite_stable_network_outputs"]
            and reload_check["passed"]
        )
        config = frozen_config(dataset)
        config.update(
            {
                "seed": seed,
                "split_seed": SPLIT_SEED,
                "split_counts": {key: int(value.size) for key, value in split.items()},
                "checkpoint_reload_check": reload_check,
            }
        )
        write_json(seed_dir / "config.json", config)
        test_summary.update(
            {
                "seed": seed,
                "example_network": seed == EXAMPLE_SEED,
                "checkpoint_path": str((seed_dir / "weights.npz").resolve()),
                "checkpoint_sha256": sha256(seed_dir / "weights.npz"),
                "training_elapsed_seconds": None,
                "psychrnn_training_seconds": None,
                "psychrnn_initialization_seconds": None,
                "n_recorded_training_losses": 0,
                "training_history_note": "The frozen loss interval produced no entries; timing metadata was unavailable after artifact finalization was resumed from the intact seed-7 checkpoint.",
                "checkpoint_reload_check": reload_check,
                "weights_finite": weights_finite,
                "technical_validity": technical_validity,
                "resumed_from_intact_partial_artifacts": True,
            }
        )
        write_json(complete_path, test_summary)
        return test_summary
    seed_dir.mkdir(parents=True, exist_ok=True)
    tf.compat.v1.reset_default_graph()
    set_global_seed(seed)
    train_task = RDM_SureTarget_InternalReadoutDatasetTask(
        dataset_path=dataset_path,
        dt=DT_MS,
        tau=TAU_MS,
        N_batch=BATCH_SIZE,
        sample_mode="random",
        seed=seed,
        allowed_indices=split["train"],
    )
    if not np.array_equal(train_task.allowed_indices, split["train"]):
        raise AssertionError("training task was not restricted to the exact train split")
    weights_path = seed_dir / "weights.npz"
    model = Basic(_network_params(train_task))
    started = time.time()
    losses, training_seconds, initialization_seconds = model.train(
        train_task,
        train_params={
            "training_iters": TRAINING_ITERS,
            "loss_epoch": LOSS_EPOCH,
            "learning_rate": LEARNING_RATE,
            "save_weights_path": str(weights_path),
        },
    )
    elapsed_seconds = time.time() - started
    history_rows = [
        {
            "record_index": index,
            "optimizer_epoch": (index + 1) * LOSS_EPOCH,
            "approx_training_trials_seen": (index + 1) * LOSS_EPOCH * BATCH_SIZE,
            "regularized_loss": float(loss),
        }
        for index, loss in enumerate(losses)
    ]
    write_csv(
        seed_dir / "training_history.csv",
        history_rows,
        ["record_index", "optimizer_epoch", "approx_training_trials_seen", "regularized_loss"],
    )
    validation_records, validation_raw = evaluate_model(
        model, dataset_path, split["validation"], seed, "validation"
    )
    validation_summary = summarize_records(validation_records, validation_raw)
    validation_summary.update(
        {
            "role": "quality control only; not used for model or hyperparameter selection",
            "seed": seed,
        }
    )
    write_json(seed_dir / "validation_summary.json", validation_summary)

    test_records, test_raw = evaluate_model(model, dataset_path, split["test"], seed, "test")
    test_summary = summarize_records(test_records, test_raw)
    write_csv(seed_dir / "test_trials.csv", test_records)
    write_csv(seed_dir / "post_go_outputs.csv", test_records)
    model.destruct()

    reload_check = _checkpoint_reload_reference(
        weights_path, dataset, split["test"], seed_dir / "reconstruction_reference.npz"
    )
    weights_finite = True
    with np.load(weights_path, allow_pickle=True) as weights:
        for key in weights.files:
            if np.issubdtype(weights[key].dtype, np.number) and not np.all(np.isfinite(weights[key])):
                weights_finite = False
                break
    technical_validity = bool(
        weights_path.exists()
        and weights_finite
        and validation_summary["qc"]["finite_stable_network_outputs"]
        and test_summary["qc"]["finite_stable_network_outputs"]
        and reload_check["passed"]
    )
    config = frozen_config(dataset)
    config.update(
        {
            "seed": seed,
            "split_seed": SPLIT_SEED,
            "split_counts": {key: int(value.size) for key, value in split.items()},
            "checkpoint_reload_check": reload_check,
        }
    )
    write_json(seed_dir / "config.json", config)
    test_summary.update(
        {
            "seed": seed,
            "example_network": seed == EXAMPLE_SEED,
            "checkpoint_path": str(weights_path.resolve()),
            "checkpoint_sha256": sha256(weights_path),
            "training_elapsed_seconds": float(elapsed_seconds),
            "psychrnn_training_seconds": float(training_seconds),
            "psychrnn_initialization_seconds": float(initialization_seconds),
            "n_recorded_training_losses": int(len(losses)),
            "training_history_note": (
                "PsychRNN records only at exact loss_epoch optimizer steps; the frozen 50,000-trial/50-batch run completes before optimizer epoch 1000."
                if not losses
                else "Losses recorded at the frozen loss_epoch interval."
            ),
            "checkpoint_reload_check": reload_check,
            "weights_finite": weights_finite,
            "technical_validity": technical_validity,
        }
    )
    write_json(complete_path, test_summary)
    return test_summary


def _package_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "tensorflow", "psychrnn", "pandas", "matplotlib", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_value(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def write_multiseed(output_dir, summaries):
    output_dir = Path(output_dir)
    multiseed_dir = output_dir / "multiseed"
    rows = []
    coherence_by_value = {}
    for summary in summaries:
        corr = summary["correlations_predicted_p_sure_offered"]
        rows.append(
            {
                "seed": summary["seed"],
                "test_direction_accuracy": summary["no_sure_direction_accuracy"],
                "p_sure_given_offered": summary["p_sure_given_offered"],
                "corr_p_sure_teacher_sure_strength": corr["teacher_sure_strength"],
                "corr_p_sure_empirical_p_correct": corr["empirical_p_correct"],
                "corr_p_sure_internal_margin": corr["internal_margin"],
                "corr_p_sure_coherence": corr["coherence"],
                "corr_p_sure_stimulus_duration": corr["stimulus_duration"],
                "checkpoint_path": summary["checkpoint_path"],
                "technical_validity": summary["technical_validity"],
            }
        )
        for item in summary["coherence"]:
            coherence_by_value.setdefault(float(item["coherence"]), []).append(
                (int(summary["seed"]), float(item["p_sure_given_offered"]))
            )
    write_csv(multiseed_dir / "behavioral_metrics.csv", rows)
    coherence_summary = []
    for coherence, seed_values in sorted(coherence_by_value.items()):
        values = np.asarray([value for _, value in seed_values], dtype=float)
        coherence_summary.append(
            {
                "coherence": coherence,
                "individual_seeds": {
                    str(seed): value for seed, value in seed_values
                },
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            }
        )
    freeze_summary = {
        "student_seeds": list(STUDENT_SEEDS),
        "example_seed": EXAMPLE_SEED,
        "all_technical_runs_valid": bool(all(item["technical_validity"] for item in summaries)),
        "individual_seed_metrics": rows,
        "p_sure_given_offered_by_coherence": coherence_summary,
        "selection_statement": "Seeds 7, 8, and 9 were predetermined; no seed was selected or replaced based on scientific results.",
        "scope_statement": "No PCA, decoding, repeated-stimulus variability, TDR, dPCA, or fixed-point analysis was run.",
    }
    write_json(multiseed_dir / "freeze_summary.json", freeze_summary)
    return freeze_summary


def run(args):
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite non-empty final freeze directory: {output_dir}. Use --resume only for this exact run."
        )
    dataset_sha = sha256(dataset_path)
    dataset = load_stage9_dataset(dataset_path)
    split, split_path, split_summary_path = create_or_load_split(
        output_dir, dataset["x"].shape[0], dataset_sha, resume=args.resume
    )
    summaries = []
    for seed in STUDENT_SEEDS:
        print(f"\n=== Final categorical freeze: seed {seed} ===", flush=True)
        summaries.append(
            run_seed(seed, str(dataset_path), dataset, split, output_dir, resume=args.resume)
        )
    freeze_summary = write_multiseed(output_dir, summaries)
    status = _git_value("status", "--short")
    command = " ".join([sys.executable, *sys.argv])
    commands = [command]
    if args.resume:
        commands.insert(0, command.replace(" --resume", ""))
    manifest = {
        "created_unix_time": time.time(),
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(status),
        "git_status_at_manifest": status.splitlines(),
        "teacher_dataset_path": str(dataset_path),
        "teacher_dataset_sha256": dataset_sha,
        "split_file": str(split_path.resolve()),
        "split_file_sha256": sha256(split_path),
        "split_summary": str(split_summary_path.resolve()),
        "split_seed": SPLIT_SEED,
        "split_counts": {key: int(value.size) for key, value in split.items()},
        "student_seeds": list(STUDENT_SEEDS),
        "example_seed": EXAMPLE_SEED,
        "frozen_configuration": frozen_config(dataset),
        "package_versions": _package_versions(),
        "checkpoint_paths": [summary["checkpoint_path"] for summary in summaries],
        "checkpoint_sha256": {
            str(summary["seed"]): summary["checkpoint_sha256"] for summary in summaries
        },
        "exact_commands": commands,
        "all_technical_runs_valid": freeze_summary["all_technical_runs_valid"],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(_jsonable(freeze_summary), indent=2, sort_keys=True), flush=True)
    return manifest


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/model_freeze/teacher_seed7.npz")
    parser.add_argument("--output-dir", default="results/final_supervised_freeze")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the exact existing split and completed valid seeds after an interrupted run.",
    )
    return parser


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
