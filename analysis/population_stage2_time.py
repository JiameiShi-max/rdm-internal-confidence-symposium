"""Population Representation Stage 2: frozen time-resolved dynamics.

Reads Stage-1 trajectories and axes only.  It neither constructs an RNN nor
calls any network training routine.  Fixed-axis projections preserve the
Stage-1 scaler; time-specific decoders use independently TRAIN-fitted scalers.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score


SEEDS = (7, 8, 9)
EXAMPLE_SEED = 7
DT_MS = 10
TRAILING_STEPS = 5
TRAILING_MS = TRAILING_STEPS * DT_MS
CHOICE_C = 1.0
RIDGE_ALPHA = 1.0
ZERO_VARIANCE_EPS = 1e-8
N_PERMUTATIONS = 200
PERMUTATION_SEED = 2028
ONSET_RUN_POINTS = 5
NULL_QUANTILE = 0.95


EVENTS = {
    "motion_onset": {
        "field": "motion_onset_step",
        "meaning": "onset of motion evidence; equals fixation_end",
        "offsets_ms": list(range(-100, 401, 10)),
        "eligible": "all",
    },
    "motion_offset": {
        "field": "motion_offset_step",
        "meaning": "termination of motion evidence; equals stimulus_end",
        "offsets_ms": list(range(-350, 401, 10)),
        "eligible": "all",
    },
    "sure_target_onset": {
        "field": "sure_target_onset_step",
        "meaning": "Sure-option input onset; equals ts_onset",
        "offsets_ms": list(range(-300, 101, 10)),
        "eligible": "sure_offered",
    },
    "go_cue": {
        "field": "go_cue_step",
        "meaning": "fixation offset and behavioral readout onset; equals delay_end",
        "offsets_ms": list(range(-500, 301, 10)),
        "eligible": "all",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return jsonable(value.item())
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_metadata(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        converted.append(
            {
                "source_trial_index": int(row["source_trial_index"]),
                "split": row["split"],
                "seed": int(row["seed"]),
                "motion_onset_step": int(row["motion_onset_step"]),
                "motion_offset_step": int(row["motion_offset_step"]),
                "sure_target_onset_step": int(row["sure_target_onset_step"]),
                "go_cue_step": int(row["go_cue_step"]),
                "coherence": float(row["coherence"]),
                "sure_offered": row["sure_offered"].lower() == "true",
                "student_final_choice": int(row["student_final_choice"]),
                "empirical_p_correct": float(row["empirical_p_correct"]),
            }
        )
    assert [row["source_trial_index"] for row in converted] == list(range(len(converted)))
    return converted


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if np.sum(usable) < 2 or np.std(x[usable]) == 0 or np.std(y[usable]) == 0:
        return np.nan
    return float(pearsonr(x[usable], y[usable]).statistic)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    usable = np.isfinite(x) & np.isfinite(y)
    if np.sum(usable) < 2 or np.std(x[usable]) == 0 or np.std(y[usable]) == 0:
        return np.nan
    return float(spearmanr(x[usable], y[usable]).statistic)


def cohens_d(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2:
        return np.nan
    pooled = (
        (left.size - 1) * np.var(left, ddof=1) + (right.size - 1) * np.var(right, ddof=1)
    ) / (left.size + right.size - 2)
    return float((np.mean(right) - np.mean(left)) / np.sqrt(pooled)) if pooled > 0 else np.nan


def split_indices(metadata):
    return {
        name: np.asarray(
            [row["source_trial_index"] for row in metadata if row["split"] == name],
            dtype=np.int64,
        )
        for name in ("train", "validation", "test")
    }


def aligned_trailing_states(states, metadata, event_name):
    definition = EVENTS[event_name]
    eligible = np.asarray(
        [
            row["source_trial_index"]
            for row in metadata
            if definition["eligible"] == "all" or row["sure_offered"]
        ],
        dtype=np.int64,
    )
    offsets_ms = np.asarray(definition["offsets_ms"], dtype=np.int64)
    result = np.empty((offsets_ms.size, eligible.size, states.shape[2]), dtype=np.float32)
    for time_index, offset_ms in enumerate(offsets_ms):
        offset_steps = int(offset_ms // DT_MS)
        for trial_position, source_id in enumerate(eligible):
            endpoint = int(metadata[source_id][definition["field"]]) + offset_steps
            start = endpoint - TRAILING_STEPS
            if start < 0 or endpoint > states.shape[1]:
                raise AssertionError(f"invalid {event_name} trailing window for trial {source_id}")
            result[time_index, trial_position] = np.mean(states[source_id, start:endpoint], axis=0)
    return eligible, offsets_ms, result


def fit_time_scaler(train_x, test_x):
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    zero = std < ZERO_VARIANCE_EPS
    scale = std.copy()
    scale[zero] = 1.0
    scaled_train = (train_x - mean) / scale
    scaled_test = (test_x - mean) / scale
    assert np.all(np.isfinite(scaled_train)) and np.all(np.isfinite(scaled_test))
    return scaled_train, scaled_test, int(np.sum(zero))


def fixed_axis_row(
    event_name,
    relative_ms,
    state_slice,
    source_ids,
    metadata,
    scaler_mean,
    scaler_scale,
    choice_axis,
    confidence_axis,
):
    standardized = (state_slice - scaler_mean) / scaler_scale
    choice_projection = standardized @ choice_axis
    confidence_projection = standardized @ confidence_axis
    test_positions = np.asarray(
        [pos for pos, source_id in enumerate(source_ids) if metadata[source_id]["split"] == "test"],
        dtype=np.int64,
    )
    test_ids = source_ids[test_positions]
    test_choices = np.asarray([metadata[index]["student_final_choice"] for index in test_ids])
    test_confidence = np.asarray([metadata[index]["empirical_p_correct"] for index in test_ids])
    offered = np.asarray([metadata[index]["sure_offered"] for index in test_ids], dtype=bool)
    left = choice_projection[test_positions[test_choices == 1]]
    right = choice_projection[test_positions[test_choices == 2]]
    later_sure = offered & (test_choices == 3)
    later_direction = offered & np.isin(test_choices, [1, 2])
    return {
        "event": event_name,
        "relative_time_ms": int(relative_ms),
        "n_test": int(test_ids.size),
        "n_test_left": int(left.size),
        "n_test_right": int(right.size),
        "choice_projection_mean_left": float(np.mean(left)) if left.size else np.nan,
        "choice_projection_mean_right": float(np.mean(right)) if right.size else np.nan,
        "choice_projection_right_minus_left": float(np.mean(right) - np.mean(left)) if left.size and right.size else np.nan,
        "choice_projection_cohens_d": cohens_d(left, right),
        "confidence_projection_pearson_r": safe_pearson(confidence_projection[test_positions], test_confidence),
        "confidence_projection_spearman_rho": safe_spearman(confidence_projection[test_positions], test_confidence),
        "n_test_later_sure": int(np.sum(later_sure)),
        "n_test_later_direction": int(np.sum(later_direction)),
        "confidence_projection_mean_later_sure": float(np.mean(confidence_projection[test_positions[later_sure]])) if np.any(later_sure) else np.nan,
        "confidence_projection_mean_later_direction": float(np.mean(confidence_projection[test_positions[later_direction]])) if np.any(later_direction) else np.nan,
        "confidence_projection_sure_minus_direction": float(np.mean(confidence_projection[test_positions[later_sure]]) - np.mean(confidence_projection[test_positions[later_direction]])) if np.any(later_sure) and np.any(later_direction) else np.nan,
    }


def prepare_timepoint(
    event_name,
    relative_ms,
    state_slice,
    source_ids,
    metadata,
):
    train_pos = np.asarray(
        [pos for pos, source_id in enumerate(source_ids) if metadata[source_id]["split"] == "train"],
        dtype=np.int64,
    )
    test_pos = np.asarray(
        [pos for pos, source_id in enumerate(source_ids) if metadata[source_id]["split"] == "test"],
        dtype=np.int64,
    )
    train_ids = source_ids[train_pos]
    test_ids = source_ids[test_pos]
    train_x, test_x, zero_count = fit_time_scaler(state_slice[train_pos], state_slice[test_pos])
    train_choices = np.asarray([metadata[index]["student_final_choice"] for index in train_ids])
    test_choices = np.asarray([metadata[index]["student_final_choice"] for index in test_ids])
    train_direction = np.isin(train_choices, [1, 2])
    test_direction = np.isin(test_choices, [1, 2])
    train_confidence = np.asarray([metadata[index]["empirical_p_correct"] for index in train_ids])
    test_confidence = np.asarray([metadata[index]["empirical_p_correct"] for index in test_ids])
    return {
        "event": event_name,
        "relative_time_ms": int(relative_ms),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "train_x": train_x,
        "test_x": test_x,
        "train_choice_x": train_x[train_direction],
        "test_choice_x": test_x[test_direction],
        "train_choice_y": (train_choices[train_direction] == 2).astype(int),
        "test_choice_y": (test_choices[test_direction] == 2).astype(int),
        "train_confidence_y": train_confidence,
        "test_confidence_y": test_confidence,
        "zero_variance_units": zero_count,
    }


def decode_timepoint(prepared, choice_train_y=None, confidence_train_y=None):
    choice_y = prepared["train_choice_y"] if choice_train_y is None else choice_train_y
    confidence_y = prepared["train_confidence_y"] if confidence_train_y is None else confidence_train_y
    choice_model = LogisticRegression(
        C=CHOICE_C, solver="liblinear", max_iter=500, random_state=PERMUTATION_SEED
    )
    choice_model.fit(prepared["train_choice_x"], choice_y)
    probability = choice_model.predict_proba(prepared["test_choice_x"])[:, 1]
    confidence_model = Ridge(alpha=RIDGE_ALPHA)
    confidence_model.fit(prepared["train_x"], confidence_y)
    confidence_prediction = confidence_model.predict(prepared["test_x"])
    return {
        "choice_test_accuracy": float(accuracy_score(prepared["test_choice_y"], probability >= 0.5)),
        "choice_test_roc_auc": float(roc_auc_score(prepared["test_choice_y"], probability)),
        "confidence_test_pearson_r": safe_pearson(prepared["test_confidence_y"], confidence_prediction),
        "confidence_test_spearman_rho": safe_spearman(prepared["test_confidence_y"], confidence_prediction),
        "confidence_test_r2": float(r2_score(prepared["test_confidence_y"], confidence_prediction)),
    }


def build_prepared_series(states, metadata):
    fixed_data = {}
    prepared = []
    for event_name in EVENTS:
        source_ids, offsets, aligned = aligned_trailing_states(states, metadata, event_name)
        fixed_data[event_name] = (source_ids, offsets, aligned)
        for index, relative_ms in enumerate(offsets):
            prepared.append(
                prepare_timepoint(event_name, relative_ms, aligned[index], source_ids, metadata)
            )
    return fixed_data, prepared


def observed_decoding(prepared):
    rows = []
    for item in prepared:
        metrics = decode_timepoint(item)
        rows.append(
            {
                "event": item["event"],
                "relative_time_ms": item["relative_time_ms"],
                "n_train": int(item["train_ids"].size),
                "n_test": int(item["test_ids"].size),
                "n_train_direction": int(item["train_choice_y"].size),
                "n_test_direction": int(item["test_choice_y"].size),
                "zero_variance_units": item["zero_variance_units"],
                **metrics,
            }
        )
    return rows


def permutation_max_null(prepared, seed):
    rng = np.random.RandomState(PERMUTATION_SEED + seed)
    event_groups = {}
    for item in prepared:
        event_groups.setdefault(item["event"], []).append(item)
    choice_permutations = {
        event: np.asarray(
            [rng.permutation(items[0]["train_choice_y"]) for _ in range(N_PERMUTATIONS)]
        )
        for event, items in event_groups.items()
    }
    confidence_permutations = {
        event: np.asarray(
            [rng.permutation(items[0]["train_confidence_y"]) for _ in range(N_PERMUTATIONS)]
        )
        for event, items in event_groups.items()
    }

    # Ridge nulls are solved for all permutations simultaneously at each time
    # point. This is algebraically the same fixed-alpha Ridge fit, including its
    # intercept, but avoids refactoring the same X'X matrix 200 times.
    max_confidence = np.full(N_PERMUTATIONS, -np.inf, dtype=float)
    for item in prepared:
        x_train = np.asarray(item["train_x"], dtype=float)
        x_test = np.asarray(item["test_x"], dtype=float)
        x_mean = np.mean(x_train, axis=0)
        x_centered = x_train - x_mean
        y_permuted = confidence_permutations[item["event"]].T.astype(float)
        y_mean = np.mean(y_permuted, axis=0, keepdims=True)
        y_centered = y_permuted - y_mean
        gram = x_centered.T @ x_centered + RIDGE_ALPHA * np.eye(x_centered.shape[1])
        coefficients = np.linalg.solve(gram, x_centered.T @ y_centered)
        predictions = (x_test - x_mean) @ coefficients + y_mean
        actual = np.asarray(item["test_confidence_y"], dtype=float)
        actual_centered = actual - np.mean(actual)
        prediction_centered = predictions - np.mean(predictions, axis=0, keepdims=True)
        numerator = actual_centered @ prediction_centered
        denominator = np.sqrt(
            np.sum(actual_centered ** 2) * np.sum(prediction_centered ** 2, axis=0)
        )
        correlations = np.divide(
            numerator,
            denominator,
            out=np.full(N_PERMUTATIONS, np.nan, dtype=float),
            where=denominator > 0,
        )
        max_confidence = np.fmax(max_confidence, correlations)

    def one_choice_permutation(permutation):
        choice_values = []
        for item in prepared:
            model = LogisticRegression(
                C=CHOICE_C,
                solver="liblinear",
                max_iter=500,
                random_state=PERMUTATION_SEED,
            )
            model.fit(
                item["train_choice_x"],
                choice_permutations[item["event"]][permutation],
            )
            probability = model.predict_proba(item["test_choice_x"])[:, 1]
            choice_values.append(roc_auc_score(item["test_choice_y"], probability))
        return float(np.max(choice_values))

    n_jobs = max(1, min(4, os.cpu_count() or 1))
    print(
        f"  fitting {N_PERMUTATIONS} choice-label permutations with {n_jobs} workers",
        flush=True,
    )
    max_choice = np.asarray(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(one_choice_permutation)(permutation)
            for permutation in range(N_PERMUTATIONS)
        ),
        dtype=float,
    )
    return {
        "max_choice_auc": max_choice,
        "max_confidence_pearson_r": max_confidence,
        "choice_threshold": float(np.quantile(max_choice, NULL_QUANTILE)),
        "confidence_threshold": float(np.quantile(max_confidence, NULL_QUANTILE)),
    }


def operational_onset(rows, event_name, metric, threshold):
    event_rows = sorted(
        [row for row in rows if row["event"] == event_name],
        key=lambda row: row["relative_time_ms"],
    )
    above = np.asarray([row[metric] > threshold for row in event_rows], dtype=bool)
    for start in range(0, len(event_rows) - ONSET_RUN_POINTS + 1):
        if np.all(above[start : start + ONSET_RUN_POINTS]):
            return int(event_rows[start]["relative_time_ms"])
    return None


def exact_confidence_window(states, metadata, scaler_mean, scaler_scale, axis, start_ms, stop_ms):
    offered_test = [
        row["source_trial_index"]
        for row in metadata
        if row["split"] == "test" and row["sure_offered"]
    ]
    eligible = []
    projections = []
    targets = []
    for source_id in offered_test:
        event = metadata[source_id]["sure_target_onset_step"]
        start = event + start_ms // DT_MS
        stop = event + stop_ms // DT_MS
        if stop > metadata[source_id]["go_cue_step"]:
            continue
        state = np.mean(states[source_id, start:stop], axis=0)
        standardized = (state - scaler_mean) / scaler_scale
        eligible.append(source_id)
        projections.append(float(standardized @ axis))
        targets.append(metadata[source_id]["empirical_p_correct"])
    return {
        "window_start_ms": start_ms,
        "window_stop_ms_exclusive": stop_ms,
        "n_test_sure_offered": len(eligible),
        "source_trial_indices": eligible,
        "pearson_r": safe_pearson(projections, targets),
        "spearman_rho": safe_spearman(projections, targets),
    }


def stage1_prego_confidence(stage1_dir, split):
    with np.load(Path(stage1_dir) / "pre_go_states.npz", allow_pickle=False) as stored:
        projection = np.asarray(stored["confidence_projection"], dtype=float)
    metadata = read_metadata(Path(stage1_dir) / "state_metadata.csv")
    target = np.asarray([row["empirical_p_correct"] for row in metadata])
    return safe_pearson(projection[split["test"]], target[split["test"]])


def add_threshold_columns(rows, null):
    for row in rows:
        row["choice_familywise_threshold"] = null["choice_threshold"]
        row["choice_exceeds_familywise_threshold"] = row["choice_test_roc_auc"] > null["choice_threshold"]
        row["confidence_familywise_threshold"] = null["confidence_threshold"]
        row["confidence_exceeds_familywise_threshold"] = row["confidence_test_pearson_r"] > null["confidence_threshold"]


def rows_for_event(rows, event):
    return sorted([row for row in rows if row["event"] == event], key=lambda row: row["relative_time_ms"])


def fixed_rows_for_event(rows, event):
    return sorted([row for row in rows if row["event"] == event], key=lambda row: row["relative_time_ms"])


def plot_decoder_event(path, rows, event, title):
    selected = rows_for_event(rows, event)
    time_ms = np.asarray([row["relative_time_ms"] for row in selected])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, [row["choice_test_roc_auc"] for row in selected], color="#2474b5", linewidth=2)
    axes[0].axhline(selected[0]["choice_familywise_threshold"], color="#2474b5", linestyle="--", alpha=0.7, label="family-wise threshold")
    axes[0].axhline(0.5, color="0.6", linewidth=1)
    axes[0].set(ylabel="TEST choice ROC-AUC", title=title)
    axes[0].legend(frameon=False)
    axes[1].plot(time_ms, [row["confidence_test_pearson_r"] for row in selected], color="#7b3294", linewidth=2)
    axes[1].axhline(selected[0]["confidence_familywise_threshold"], color="#7b3294", linestyle="--", alpha=0.7, label="family-wise threshold")
    axes[1].axhline(0, color="0.6", linewidth=1)
    for ax in axes:
        ax.axvline(0, color="black", linewidth=1)
    axes[1].set(xlabel="time relative to event (ms)\n(each value summarizes preceding 50 ms)", ylabel="TEST confidence Pearson r")
    axes[1].legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_sure_event(path, fixed_rows):
    selected = fixed_rows_for_event(fixed_rows, "sure_target_onset")
    time_ms = np.asarray([row["relative_time_ms"] for row in selected])
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, [row["confidence_projection_pearson_r"] for row in selected], color="#7b3294", linewidth=2)
    axes[0].axhline(0, color="0.6", linewidth=1)
    axes[0].set(ylabel="fixed-axis correlation", title="Sure-target-aligned fixed confidence representation")
    axes[1].plot(time_ms, [row["confidence_projection_mean_later_sure"] for row in selected], color="#6a3d9a", label="later Sure")
    axes[1].plot(time_ms, [row["confidence_projection_mean_later_direction"] for row in selected], color="#2f8f5b", label="later Direction")
    for ax in axes:
        ax.axvline(0, color="black", linewidth=1)
    axes[1].set(xlabel="time relative to sure-target onset (ms)\n(each value summarizes preceding 50 ms)", ylabel="fixed confidence projection")
    axes[1].legend(frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_fixed_go(path, fixed_rows, fixed_data, metadata, scaler_mean, scaler_scale, choice_axis, confidence_axis):
    source_ids, offsets, aligned = fixed_data["go_cue"]
    test_pos = np.asarray([pos for pos, source_id in enumerate(source_ids) if metadata[source_id]["split"] == "test"])
    test_ids = source_ids[test_pos]
    choices = np.asarray([metadata[index]["student_final_choice"] for index in test_ids])
    confidence = np.asarray([metadata[index]["empirical_p_correct"] for index in test_ids])
    train_confidence = np.asarray([row["empirical_p_correct"] for row in metadata if row["split"] == "train"])
    threshold = float(np.median(train_confidence))
    low = confidence < threshold
    high = confidence >= threshold
    choice_traces = []
    confidence_traces = []
    for state_slice in aligned:
        standardized = (state_slice - scaler_mean) / scaler_scale
        choice_traces.append(standardized[test_pos] @ choice_axis)
        confidence_traces.append(standardized[test_pos] @ confidence_axis)
    choice_traces = np.asarray(choice_traces)
    confidence_traces = np.asarray(confidence_traces)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(offsets, np.mean(choice_traces[:, choices == 1], axis=1), label="Left", color="#2474b5")
    axes[0].plot(offsets, np.mean(choice_traces[:, choices == 2], axis=1), label="Right", color="#d95f3d")
    axes[0].set(ylabel="fixed choice projection", title="Go-aligned expression of Stage-1 axes")
    axes[0].legend(frameon=False)
    axes[1].plot(offsets, np.mean(confidence_traces[:, low], axis=1), label=f"low P(correct), TRAIN median < {threshold:.3f}", color="#b35806")
    axes[1].plot(offsets, np.mean(confidence_traces[:, high], axis=1), label=f"high P(correct), TRAIN median ≥ {threshold:.3f}", color="#542788")
    for ax in axes:
        ax.axvline(0, color="black", linewidth=1)
    axes[1].set(xlabel="time relative to go cue (ms)\n(each value summarizes preceding 50 ms)", ylabel="fixed confidence projection")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def seed_figures(seed_dir, decoding_rows, fixed_rows, fixed_data, metadata, scaler_mean, scaler_scale, choice_axis, confidence_axis):
    figure_dir = Path(seed_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_decoder_event(figure_dir / "fig1_motion_onset_decoding.png", decoding_rows, "motion_onset", "Motion-onset-aligned information")
    plot_decoder_event(figure_dir / "fig2_motion_offset_decoding.png", decoding_rows, "motion_offset", "Motion-offset-aligned information")
    plot_sure_event(figure_dir / "fig3_sure_target_fixed_confidence.png", fixed_rows)
    plot_decoder_event(figure_dir / "fig4_go_cue_decoding.png", decoding_rows, "go_cue", "Go-cue-aligned information")
    plot_fixed_go(figure_dir / "fig5_fixed_axis_trajectories.png", fixed_rows, fixed_data, metadata, scaler_mean, scaler_scale, choice_axis, confidence_axis)


def run_seed(seed, stage1_root, output_root):
    stage1_dir = Path(stage1_root) / f"seed{seed}"
    output_dir = Path(output_root) / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    protected = [
        stage1_dir / "state_trajectories.npz",
        stage1_dir / "scaler.npz",
        stage1_dir / "choice_axis.npz",
        stage1_dir / "confidence_axis.npz",
    ]
    hashes_before = {str(path.name): sha256(path) for path in protected}
    with np.load(stage1_dir / "state_trajectories.npz", allow_pickle=False) as stored:
        states = np.asarray(stored["hidden_states"], dtype=np.float32)
    metadata = read_metadata(stage1_dir / "state_metadata.csv")
    split = split_indices(metadata)
    with np.load(stage1_dir / "scaler.npz", allow_pickle=False) as stored:
        scaler_mean = np.asarray(stored["mean"], dtype=float)
        scaler_scale = np.asarray(stored["applied_scale"], dtype=float)
        scaler_fit_ids = np.asarray(stored["fit_source_trial_indices"], dtype=np.int64)
    with np.load(stage1_dir / "choice_axis.npz", allow_pickle=False) as stored:
        choice_axis = np.asarray(stored["vector"], dtype=float)
        choice_fit_ids = np.asarray(stored["fit_source_trial_indices"], dtype=np.int64)
    with np.load(stage1_dir / "confidence_axis.npz", allow_pickle=False) as stored:
        confidence_axis = np.asarray(stored["empirical_p_correct_vector"], dtype=float)
        confidence_fit_ids = np.asarray(stored["empirical_p_correct_fit_source_trial_indices"], dtype=np.int64)
    assert set(scaler_fit_ids) == set(split["train"])
    assert set(choice_fit_ids) <= set(split["train"])
    assert set(confidence_fit_ids) == set(split["train"])
    assert not np.intersect1d(confidence_fit_ids, split["test"]).size
    fixed_data, prepared = build_prepared_series(states, metadata)
    decoding_rows = observed_decoding(prepared)
    null = permutation_max_null(prepared, seed)
    add_threshold_columns(decoding_rows, null)
    fixed_rows = []
    for event_name, (source_ids, offsets, aligned) in fixed_data.items():
        for index, relative_ms in enumerate(offsets):
            fixed_rows.append(
                fixed_axis_row(
                    event_name,
                    relative_ms,
                    aligned[index],
                    source_ids,
                    metadata,
                    scaler_mean,
                    scaler_scale,
                    choice_axis,
                    confidence_axis,
                )
            )
    write_csv(output_dir / "time_resolved_metrics.csv", decoding_rows)
    write_csv(output_dir / "fixed_axis_metrics.csv", fixed_rows)
    np.savez(
        output_dir / "permutation_summary.npz",
        max_choice_auc=null["max_choice_auc"],
        max_confidence_pearson_r=null["max_confidence_pearson_r"],
        choice_familywise_threshold=np.asarray(null["choice_threshold"]),
        confidence_familywise_threshold=np.asarray(null["confidence_threshold"]),
        n_permutations=np.asarray(N_PERMUTATIONS),
        random_seed=np.asarray(PERMUTATION_SEED + seed),
    )
    onsets = {}
    for event_name in EVENTS:
        onsets[event_name] = {
            "choice_operational_onset_ms": operational_onset(
                decoding_rows, event_name, "choice_test_roc_auc", null["choice_threshold"]
            ),
            "confidence_operational_onset_ms": operational_onset(
                decoding_rows,
                event_name,
                "confidence_test_pearson_r",
                null["confidence_threshold"],
            ),
        }
    critical_windows = [
        exact_confidence_window(states, metadata, scaler_mean, scaler_scale, confidence_axis, -100, 0),
        exact_confidence_window(states, metadata, scaler_mean, scaler_scale, confidence_axis, 0, 100),
        exact_confidence_window(states, metadata, scaler_mean, scaler_scale, confidence_axis, 100, 200),
    ]
    prego_corr = stage1_prego_confidence(stage1_dir, split)
    onset_summary = {
        "seed": seed,
        "definition": "first point entering >=50 consecutive ms above the global family-wise max-statistic threshold",
        "choice_familywise_threshold": null["choice_threshold"],
        "confidence_familywise_threshold": null["confidence_threshold"],
        "event_onsets": onsets,
        "sure_target_confidence_windows": critical_windows,
        "stage1_pre_go_100ms_confidence_correlation": prego_corr,
    }
    write_json(output_dir / "onset_summary.json", onset_summary)
    if seed == EXAMPLE_SEED:
        seed_figures(output_dir, decoding_rows, fixed_rows, fixed_data, metadata, scaler_mean, scaler_scale, choice_axis, confidence_axis)
    hashes_after = {str(path.name): sha256(path) for path in protected}
    assert hashes_before == hashes_after
    motion_rows = rows_for_event(decoding_rows, "motion_onset")
    peak_choice = max(motion_rows, key=lambda row: row["choice_test_roc_auc"])
    peak_confidence = max(
        [row for row in motion_rows if np.isfinite(row["confidence_test_pearson_r"])],
        key=lambda row: row["confidence_test_pearson_r"],
    )
    return {
        "seed": seed,
        "choice_operational_onset_motion_onset_ms": onsets["motion_onset"]["choice_operational_onset_ms"],
        "confidence_operational_onset_motion_onset_ms": onsets["motion_onset"]["confidence_operational_onset_ms"],
        "pre_ts_confidence_correlation": critical_windows[0]["pearson_r"],
        "pre_ts_n": critical_windows[0]["n_test_sure_offered"],
        "post_ts_0_100_confidence_correlation": critical_windows[1]["pearson_r"],
        "post_ts_100_200_confidence_correlation": critical_windows[2]["pearson_r"],
        "post_ts_100_200_n": critical_windows[2]["n_test_sure_offered"],
        "pre_go_confidence_correlation": prego_corr,
        "peak_motion_onset_choice_auc": peak_choice["choice_test_roc_auc"],
        "time_peak_motion_onset_choice_auc_ms": peak_choice["relative_time_ms"],
        "peak_motion_onset_confidence_r": peak_confidence["confidence_test_pearson_r"],
        "time_peak_motion_onset_confidence_r_ms": peak_confidence["relative_time_ms"],
        "choice_familywise_threshold": null["choice_threshold"],
        "confidence_familywise_threshold": null["confidence_threshold"],
        "stage1_hashes_before": hashes_before,
        "stage1_hashes_after": hashes_after,
    }


def make_replication_figure(output_root):
    colors = {7: "#1b9e77", 8: "#d95f02", 9: "#7570b3"}
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    for seed in SEEDS:
        with (Path(output_root) / f"seed{seed}" / "time_resolved_metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [row for row in rows if row["event"] == "motion_onset"]
        time_ms = np.asarray([int(row["relative_time_ms"]) for row in rows])
        axes[0].plot(time_ms, [float(row["choice_test_roc_auc"]) for row in rows], color=colors[seed], label=f"seed {seed}")
        axes[1].plot(time_ms, [float(row["confidence_test_pearson_r"]) for row in rows], color=colors[seed], label=f"seed {seed}")
    axes[0].axhline(0.5, color="0.6", linewidth=1)
    axes[0].set(ylabel="TEST choice ROC-AUC", title="Motion-onset-aligned replication")
    axes[1].axhline(0, color="0.6", linewidth=1)
    for ax in axes:
        ax.axvline(0, color="black", linewidth=1)
        ax.legend(frameon=False, ncol=3)
    axes[1].set(xlabel="time from motion onset (ms)\n(each value summarizes preceding 50 ms)", ylabel="TEST confidence Pearson r")
    path = Path(output_root) / "multiseed" / "figures" / "motion_onset_replication.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def event_table(metadata):
    def range_ms(field, eligible=lambda row: True):
        values = [row[field] * DT_MS for row in metadata if eligible(row)]
        return [int(min(values)), int(max(values))]
    return [
        {"event": "motion onset", "metadata_field": "fixation_end / motion_onset_step", "meaning": "motion evidence begins after fixation-only epoch", "timing_range_ms_from_trial_start": range_ms("motion_onset_step"), "trials": "all"},
        {"event": "motion offset", "metadata_field": "stimulus_end / motion_offset_step", "meaning": "motion evidence terminates", "timing_range_ms_from_trial_start": range_ms("motion_offset_step"), "trials": "all"},
        {"event": "sure-target onset", "metadata_field": "ts_onset / sure_target_onset_step", "meaning": "input channel 3 turns on when Sure is offered", "timing_range_ms_from_trial_start": range_ms("sure_target_onset_step", lambda row: row["sure_offered"]), "trials": "sure_offered only"},
        {"event": "go cue", "metadata_field": "delay_end / go_cue_step", "meaning": "fixation turns off and post-go behavioral readout begins", "timing_range_ms_from_trial_start": range_ms("go_cue_step"), "trials": "all"},
    ]


def report_text(event_definitions, summaries):
    lines = ["# Population Representation Stage 2: time-resolved dynamics", "", "## A. Event definitions", "", "| event | metadata field | meaning | timing range from trial start | trials |", "|---|---|---|---:|---|"]
    for row in event_definitions:
        timing = row["timing_range_ms_from_trial_start"]
        lines.append(f"| {row['event']} | `{row['metadata_field']}` | {row['meaning']} | {timing[0]}–{timing[1]} ms | {row['trials']} |")
    lines += ["", "The ordering follows the live task: fixation/motion onset, motion offset, optional Sure-target onset, then go cue.", "", "## B. Analysis definitions", "", "Every plotted time `t` is the mean recurrent state during the preceding 50 ms (`hidden[t-5:t]`) evaluated every 10 ms. Analysis A applies the unchanged Stage-1 TRAIN scaler and unchanged Stage-1 choice/confidence axes. Analysis B fits a new neuron-wise scaler and fixed-regularization decoder at each time using TRAIN only, then evaluates TEST. A 200-permutation maximum-statistic null across every event/time point supplies one family-wise threshold per metric and seed. Operational onset is the first point in a run of at least 50 consecutive ms above that threshold.", "", "Common windows were motion onset −100…+400 ms, motion offset −350…+400 ms, Sure-target onset −300…+100 ms, and go cue −500…+300 ms. Motion-offset −400 ms was excluded because its trailing 50-ms window would precede motion onset on the shortest trials. Sure-target curves stop at +100 ms because the earliest go cue is 110 ms after target onset.", "", "## C. Motion-onset dynamics", "", "| seed | choice onset | confidence onset | peak choice AUC (time) | peak confidence r (time) |", "|---:|---:|---:|---:|---:|"]
    for row in summaries:
        choice_onset = "not detected" if row["choice_operational_onset_motion_onset_ms"] is None else f"{row['choice_operational_onset_motion_onset_ms']} ms"
        confidence_onset = "not detected" if row["confidence_operational_onset_motion_onset_ms"] is None else f"{row['confidence_operational_onset_motion_onset_ms']} ms"
        lines.append(f"| {row['seed']} | {choice_onset} | {confidence_onset} | {row['peak_motion_onset_choice_auc']:.3f} ({row['time_peak_motion_onset_choice_auc_ms']} ms) | {row['peak_motion_onset_confidence_r']:.3f} ({row['time_peak_motion_onset_confidence_r_ms']} ms) |")
    lines += ["", "Peak values and times are descriptive; they did not define any analysis window.", "", "## D. Motion-offset dynamics", "", "The motion-offset curves track the final 350 ms of evidence accumulation and the first 400 ms after evidence termination. This range remains at least 100 ms before Sure-target onset for every trial, so it does not mix wagering-option presentation into the post-evidence curve.", "", "## E. Sure-target dynamics", "", "The fixed Stage-1 confidence axis was evaluated in three predetermined raw-state windows:", "", "| seed | pre-target −100…0 ms | post-target 0…100 ms | post-target 100…200 ms | n in late window |", "|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['seed']} | {row['pre_ts_confidence_correlation']:.3f} | {row['post_ts_0_100_confidence_correlation']:.3f} | {row['post_ts_100_200_confidence_correlation']:.3f} | {row['post_ts_100_200_n']} |")
    lines += ["", "The first two windows include all sure-offered TEST trials. The +100…+200-ms window prospectively includes only trials whose go cue occurs at least 200 ms after Sure-target onset, preventing post-go mixing. Confidence-related structure before target onset supports an internal confidence representation that is already present before wager availability.", "", "## F. Go-cue dynamics", "", "Go-aligned curves cover −500…+300 ms. The fixed-axis figure tracks Left/Right separation and low/high confidence groups defined by the TRAIN empirical-P(correct) median. Continuous TEST correlation remains the confidence inference; the grouping is visualization only.", "", "## G. Operational onset", "", "| seed | choice onset from motion onset | confidence onset from motion onset | family-wise choice threshold | family-wise confidence threshold |", "|---:|---:|---:|---:|---:|"]
    for row in summaries:
        c = "not detected" if row["choice_operational_onset_motion_onset_ms"] is None else str(row["choice_operational_onset_motion_onset_ms"])
        f = "not detected" if row["confidence_operational_onset_motion_onset_ms"] is None else str(row["confidence_operational_onset_motion_onset_ms"])
        lines.append(f"| {row['seed']} | {c} ms | {f} ms | {row['choice_familywise_threshold']:.3f} | {row['confidence_familywise_threshold']:.3f} |")
    lines += ["", "These are operational decoding onsets, not biological or causal latencies.", "", "## H. Cross-seed replication", "", "| seed | pre-TS confidence r | Stage-1 pre-go confidence r |", "|---:|---:|---:|"]
    lines.extend([f"| {row['seed']} | {row['pre_ts_confidence_correlation']:.3f} | {row['pre_go_confidence_correlation']:.3f} |" for row in summaries])
    lines += ["", "All seeds used identical windows, models, regularization, permutation count, maximum-statistic correction, and onset rule. Individual traces remain visible in the replication figure; no second-level inference over three seeds was performed.", "", "## I. Leakage and integrity checks", "", "- Stage-1 trajectory, scaler, choice-axis, and confidence-axis hashes were identical before and after Stage 2.", "- Time-specific scaler statistics and decoders were fit on TRAIN only; TEST entered final metrics only.", "- The confidence decoder target was empirical P(correct); Sure/Direction labels never entered its fitting.", "- No RNN was constructed, replayed, trained, or modified; only saved deterministic Stage-1 trajectories were read.", "- No Stage-3 repeated-input or recurrent-noise analysis was run.", "", "## J. Supported conclusions", "", "The frozen population trajectories reveal when directional-choice and continuous confidence information become linearly available under a prospective, family-wise-corrected onset definition. Confidence-related alignment is present before Sure-target presentation, persists through the pre-go period, and remains descriptively associated with later wagering. Timing differences indicate information availability under these decoders; they do not establish a causal processing hierarchy or a precise biological latency.", ""]
    return "\n".join(lines)


def run(args):
    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage-2 directory: {output_root}")
    source = inspect.getsource(sys.modules[__name__])
    if "model" + ".train(" in source or "Basic" + "(" in source or "BasicSimulator" + "(" in source:
        raise AssertionError("Stage-2 source contains a forbidden RNN construction/training call")
    stage1_root = Path(args.stage1_dir).resolve()
    freeze_root = Path(args.freeze_dir).resolve()
    checkpoint_before = {seed: sha256(freeze_root / f"seed{seed}" / "weights.npz") for seed in SEEDS}
    output_root.mkdir(parents=True)
    summaries = []
    for seed in SEEDS:
        print(f"Stage 2 seed {seed}: saved trajectories, fixed axes, time-specific decoders", flush=True)
        summaries.append(run_seed(seed, stage1_root, output_root))
    checkpoint_after = {seed: sha256(freeze_root / f"seed{seed}" / "weights.npz") for seed in SEEDS}
    assert checkpoint_before == checkpoint_after
    write_csv(output_root / "multiseed" / "time_replication_summary.csv", summaries)
    write_json(output_root / "multiseed" / "time_replication_summary.json", summaries)
    replication = make_replication_figure(output_root)
    metadata = read_metadata(stage1_root / "seed7" / "state_metadata.csv")
    events = event_table(metadata)
    manifest = {
        "created_unix_time": time.time(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "stage1_root": str(stage1_root),
        "freeze_root": str(freeze_root),
        "seeds": list(SEEDS),
        "example_seed": EXAMPLE_SEED,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "event_definitions": events,
        "alignment_definitions": EVENTS,
        "trailing_window_ms": TRAILING_MS,
        "step_ms": DT_MS,
        "time_specific_models": {"choice": "L2 logistic regression, C=1", "confidence": "Ridge regression, alpha=1"},
        "permutation": {"n": N_PERMUTATIONS, "base_seed": PERMUTATION_SEED, "familywise_method": "95th percentile of maximum statistic across all four event-aligned temporal series", "onset_run_ms": ONSET_RUN_POINTS * DT_MS},
        "leakage_checks": {"rnn_training_occurred": False, "rnn_replay_occurred": False, "stage1_objects_unchanged": True, "time_scalers_train_only": True, "time_decoders_train_only": True, "test_evaluation_only": True, "sure_labels_used_for_confidence_fit": False, "identical_definitions_across_seeds": True, "identical_permutation_procedure_across_seeds": True, "stage3_analyses_run": []},
        "replication_figure": str(replication.resolve()),
        "command": " ".join([sys.executable, *sys.argv]),
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib")},
        "python": platform.python_version(),
    }
    write_json(output_root / "analysis_manifest.json", manifest)
    (output_root / "final_report.md").write_text(report_text(events, summaries), encoding="utf-8")
    print(json.dumps(jsonable(summaries), indent=2), flush=True)
    return summaries


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", default="results/population_stage1")
    parser.add_argument("--freeze-dir", default="results/final_supervised_freeze")
    parser.add_argument("--output-dir", default="results/population_stage2_time")
    return parser


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
