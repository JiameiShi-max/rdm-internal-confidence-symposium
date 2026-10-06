"""Population Representation Stage 1 for frozen categorical RNN checkpoints.

Inference only: deterministic source-trial replay, train-fitted PCA, and
train-fitted linear choice/confidence axes.  This module intentionally has no
training API and performs no time-resolved decoding or recurrent-noise study.
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

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


SEEDS = (7, 8, 9)
EXAMPLE_SEED = 7
ANALYSIS_RANDOM_SEED = 2027
DT_MS = 10
PRE_GO_MS = 100
PRE_GO_STEPS = PRE_GO_MS // DT_MS
READOUT_STEPS = 10
N_FOLDS = 5
CHOICE_C = 1.0
RIDGE_ALPHA = 1.0
N_CHOICE_PERMUTATIONS = 100
N_BOOTSTRAPS = 5000
ZERO_VARIANCE_EPS = 1e-8


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


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


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
    if np.sum(usable) < 2:
        return np.nan
    return float(spearmanr(x[usable], y[usable]).statistic)


def load_frozen_inputs(dataset_path, split_path):
    with np.load(dataset_path, allow_pickle=True) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    info = [json.loads(str(item)) for item in data["trial_info_json"]]
    with np.load(split_path, allow_pickle=False) as stored:
        split = {
            "train": np.asarray(stored["train_indices"], dtype=np.int64),
            "validation": np.asarray(stored["validation_indices"], dtype=np.int64),
            "test": np.asarray(stored["test_indices"], dtype=np.int64),
        }
    all_ids = np.concatenate(list(split.values()))
    assert not np.intersect1d(split["train"], split["validation"]).size
    assert not np.intersect1d(split["train"], split["test"]).size
    assert not np.intersect1d(split["validation"], split["test"]).size
    assert np.array_equal(np.sort(all_ids), np.arange(data["x"].shape[0]))
    return data, info, split


def deterministic_replay(weights_path, inputs, batch_size=100):
    """Replay only through BasicSimulator; no trainable model is constructed."""
    from psychrnn.backend.simulation import BasicSimulator

    weights = dict(np.load(weights_path, allow_pickle=True))
    assert weights["W_rec"].shape == (50, 50)
    simulator = BasicSimulator(params={"alpha": 0.1, "rec_noise": 0.0}, weights=weights)
    output_batches = []
    state_batches = []
    for start in range(0, inputs.shape[0], int(batch_size)):
        outputs, states = simulator.run_trials(inputs[start : start + batch_size])
        output_batches.append(np.asarray(outputs, dtype=np.float32))
        state_batches.append(np.asarray(states, dtype=np.float32))
    raw_outputs = np.concatenate(output_batches, axis=0)
    states = np.concatenate(state_batches, axis=0)
    assert states.shape == (inputs.shape[0], inputs.shape[1], 50)
    assert np.all(np.isfinite(raw_outputs))
    assert np.all(np.isfinite(states))
    return raw_outputs, states


def categorical_outputs_and_choices(raw_outputs, info):
    logits = raw_outputs[:, :, 1:4].astype(np.float64)
    logits -= np.max(logits, axis=2, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=2, keepdims=True)
    categorical = raw_outputs.copy()
    categorical[:, :, 1:4] = probabilities.astype(np.float32)
    choices = np.empty(len(info), dtype=np.int64)
    post_go = np.empty((len(info), 3), dtype=np.float64)
    for index, trial in enumerate(info):
        start = int(trial["delay_end"])
        stop = min(raw_outputs.shape[1], start + READOUT_STEPS)
        post_go[index] = np.mean(probabilities[index, start:stop], axis=0)
        choices[index] = 1 + int(np.argmax(post_go[index]))
    return categorical, choices, post_go


def extract_primary_states(states, info):
    pre_go = np.empty((len(info), states.shape[2]), dtype=np.float32)
    evidence_end = np.empty_like(pre_go)
    starts = np.empty(len(info), dtype=np.int64)
    stops = np.empty(len(info), dtype=np.int64)
    for index, trial in enumerate(info):
        stop = int(trial["delay_end"])
        start = stop - PRE_GO_STEPS
        if start < 0:
            raise ValueError("pre-go window starts before the trial")
        starts[index] = start
        stops[index] = stop
        pre_go[index] = np.mean(states[index, start:stop], axis=0)
        evidence_end[index] = states[index, int(trial["stimulus_end"]) - 1]
    assert np.all(stops - starts == PRE_GO_STEPS)
    return pre_go, evidence_end, starts, stops


def fit_scaler(pre_go, train_indices):
    train_indices = np.asarray(train_indices, dtype=np.int64)
    mean = np.mean(pre_go[train_indices], axis=0)
    std = np.std(pre_go[train_indices], axis=0)
    zero_variance = std < ZERO_VARIANCE_EPS
    scale = std.copy()
    scale[zero_variance] = 1.0
    standardized = (pre_go - mean) / scale
    assert np.all(np.isfinite(standardized))
    return standardized, mean, std, scale, zero_variance


def fit_pca(standardized, split):
    pca = PCA(n_components=standardized.shape[1], svd_solver="full")
    pca.fit(standardized[split["train"]])
    projections = pca.transform(standardized)
    return pca, projections


def choice_cross_validation(x, choices, train_indices, seed):
    eligible = np.asarray(
        [index for index in train_indices if choices[index] in (1, 2)], dtype=np.int64
    )
    labels = (choices[eligible] == 2).astype(int)
    assert set(np.unique(labels)) == {0, 1}
    folds = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=ANALYSIS_RANDOM_SEED
    )
    rows = []
    for fold, (fit_pos, eval_pos) in enumerate(folds.split(x[eligible], labels), start=1):
        model = LogisticRegression(
            C=CHOICE_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=seed
        )
        model.fit(x[eligible[fit_pos]], labels[fit_pos])
        probability = model.predict_proba(x[eligible[eval_pos]])[:, 1]
        predicted = (probability >= 0.5).astype(int)
        rows.append(
            {
                "fold": fold,
                "n_train": int(fit_pos.size),
                "n_evaluation": int(eval_pos.size),
                "accuracy": float(accuracy_score(labels[eval_pos], predicted)),
                "roc_auc": float(roc_auc_score(labels[eval_pos], probability)),
            }
        )
    observed = float(np.mean([row["accuracy"] for row in rows]))
    rng = np.random.RandomState(ANALYSIS_RANDOM_SEED + seed)
    permutation_scores = []
    fixed_splits = list(folds.split(x[eligible], labels))
    for _ in range(N_CHOICE_PERMUTATIONS):
        permuted = rng.permutation(labels)
        fold_scores = []
        for fit_pos, eval_pos in fixed_splits:
            model = LogisticRegression(
                C=CHOICE_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=seed
            )
            model.fit(x[eligible[fit_pos]], permuted[fit_pos])
            fold_scores.append(
                accuracy_score(permuted[eval_pos], model.predict(x[eligible[eval_pos]]))
            )
        permutation_scores.append(float(np.mean(fold_scores)))
    final = LogisticRegression(
        C=CHOICE_C, penalty="l2", solver="liblinear", max_iter=1000, random_state=seed
    )
    final.fit(x[eligible], labels)
    vector = final.coef_[0].astype(float)
    norm_value = float(np.linalg.norm(vector))
    if norm_value <= 0:
        raise ValueError("choice decoder produced a zero coefficient vector")
    vector /= norm_value
    return {
        "eligible_indices": eligible,
        "labels": labels,
        "folds": rows,
        "accuracy": mean_sd([row["accuracy"] for row in rows]),
        "roc_auc": mean_sd([row["roc_auc"] for row in rows]),
        "permutation": {
            "n": N_CHOICE_PERMUTATIONS,
            "observed_mean_accuracy": observed,
            "null_mean_accuracy": float(np.mean(permutation_scores)),
            "null_sd_accuracy": float(np.std(permutation_scores, ddof=1)),
            "p_one_sided": float(
                (1 + np.sum(np.asarray(permutation_scores) >= observed))
                / (N_CHOICE_PERMUTATIONS + 1)
            ),
        },
        "model": final,
        "vector": vector,
    }


def confidence_cross_validation(x, target, train_indices, seed, target_name):
    eligible = np.asarray(
        [index for index in train_indices if np.isfinite(target[index])], dtype=np.int64
    )
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=ANALYSIS_RANDOM_SEED)
    rows = []
    for fold, (fit_pos, eval_pos) in enumerate(folds.split(eligible), start=1):
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(x[eligible[fit_pos]], target[eligible[fit_pos]])
        predicted = model.predict(x[eligible[eval_pos]])
        actual = target[eligible[eval_pos]]
        rows.append(
            {
                "fold": fold,
                "n_train": int(fit_pos.size),
                "n_evaluation": int(eval_pos.size),
                "pearson_r": safe_pearson(actual, predicted),
                "spearman_rho": safe_spearman(actual, predicted),
                "r2": float(r2_score(actual, predicted)),
            }
        )
    final = Ridge(alpha=RIDGE_ALPHA)
    final.fit(x[eligible], target[eligible])
    vector = final.coef_.astype(float)
    norm_value = float(np.linalg.norm(vector))
    if norm_value <= 0:
        raise ValueError(f"{target_name} decoder produced a zero coefficient vector")
    vector /= norm_value
    train_projection = x[eligible] @ vector
    orientation_flipped = False
    if safe_pearson(train_projection, target[eligible]) < 0:
        vector *= -1.0
        orientation_flipped = True
    assert safe_pearson(x[eligible] @ vector, target[eligible]) >= 0
    return {
        "eligible_indices": eligible,
        "folds": rows,
        "pearson_r": mean_sd([row["pearson_r"] for row in rows]),
        "spearman_rho": mean_sd([row["spearman_rho"] for row in rows]),
        "r2": mean_sd([row["r2"] for row in rows]),
        "model": final,
        "vector": vector,
        "orientation_flipped": orientation_flipped,
        "target_name": target_name,
    }


def unpenalized_logistic_with_uncertainty(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(x.size), x])

    def objective(beta):
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(objective, np.zeros(2), method="BFGS")
    beta = np.asarray(result.x, dtype=float)
    probabilities = expit(design @ beta)
    weights = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
    covariance = np.linalg.pinv(design.T @ (design * weights[:, None]))
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_value = beta[1] / standard_error[1] if standard_error[1] > 0 else np.nan
    return {
        "converged": bool(result.success),
        "intercept": float(beta[0]),
        "coefficient": float(beta[1]),
        "standard_error": float(standard_error[1]),
        "ci95": [float(beta[1] - 1.96 * standard_error[1]), float(beta[1] + 1.96 * standard_error[1])],
        "wald_z": float(z_value),
        "wald_p_two_sided": float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan,
    }


def wager_metrics(projection, choices, offered, test_indices, seed):
    selected = np.asarray(
        [index for index in test_indices if offered[index]], dtype=np.int64
    )
    sure = choices[selected] == 3
    if not np.any(sure) or np.all(sure):
        raise ValueError("held-out offered trials do not contain both Sure and Direction choices")
    sure_values = projection[selected[sure]]
    direction_values = projection[selected[~sure]]
    difference = float(np.mean(sure_values) - np.mean(direction_values))
    pooled_variance = (
        (sure_values.size - 1) * np.var(sure_values, ddof=1)
        + (direction_values.size - 1) * np.var(direction_values, ddof=1)
    ) / (sure_values.size + direction_values.size - 2)
    effect_size = difference / np.sqrt(pooled_variance) if pooled_variance > 0 else np.nan
    rng = np.random.RandomState(ANALYSIS_RANDOM_SEED + 100 + seed)
    boot = np.empty(N_BOOTSTRAPS, dtype=float)
    for index in range(N_BOOTSTRAPS):
        boot[index] = float(
            np.mean(rng.choice(sure_values, sure_values.size, replace=True))
            - np.mean(rng.choice(direction_values, direction_values.size, replace=True))
        )
    logistic = unpenalized_logistic_with_uncertainty(projection[selected], sure.astype(int))
    return {
        "n_sure_offered_test": int(selected.size),
        "n_later_sure": int(sure_values.size),
        "n_later_direction": int(direction_values.size),
        "sure_projection_mean": float(np.mean(sure_values)),
        "sure_projection_median": float(np.median(sure_values)),
        "direction_projection_mean": float(np.mean(direction_values)),
        "direction_projection_median": float(np.median(direction_values)),
        "sure_minus_direction_mean_difference": difference,
        "bootstrap_ci95_mean_difference": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
        "cohens_d_sure_minus_direction": float(effect_size),
        "later_sure_logistic": logistic,
    }


def metadata_rows(seed, info, split, choices, post_go, starts, stops, data):
    split_by_id = {}
    for name, indices in split.items():
        split_by_id.update({int(index): name for index in indices})
    rows = []
    for index, trial in enumerate(info):
        choice = int(choices[index])
        rows.append(
            {
                "source_trial_index": index,
                "split": split_by_id[index],
                "seed": seed,
                "motion_onset_step": int(trial["fixation_end"]),
                "motion_offset_step": int(trial["stimulus_end"]),
                "sure_target_onset_step": int(trial["ts_onset"]),
                "go_cue_step": int(trial["delay_end"]),
                "pre_go_start_step": int(starts[index]),
                "pre_go_stop_step_exclusive": int(stops[index]),
                "coherence": float(trial["coh"]),
                "signed_coherence": float(trial["signed_coherence"]),
                "motion_direction": int(trial.get("true_direction", trial["dir_choice"])),
                "stimulus_duration_ms": int(trial.get("stimulus_duration", trial["stimulus_dur"])),
                "sure_offered": bool(trial["sure_available"]),
                "student_final_choice": choice,
                "student_final_choice_name": {1: "Left", 2: "Right", 3: "Sure"}[choice],
                "deterministic_p_left": float(post_go[index, 0]),
                "deterministic_p_right": float(post_go[index, 1]),
                "deterministic_p_sure": float(post_go[index, 2]),
                "teacher_internal_margin": float(data["teacher_internal_margin"][index]),
                "empirical_p_correct": float(data["teacher_direction_success_proxy"][index]),
                "teacher_sure_strength": float(data["teacher_sure_strength"][index]),
            }
        )
    return rows


def heldout_choice_metrics(choice_result, x, choices, test_indices):
    eligible = np.asarray(
        [index for index in test_indices if choices[index] in (1, 2)], dtype=np.int64
    )
    labels = (choices[eligible] == 2).astype(int)
    probability = choice_result["model"].predict_proba(x[eligible])[:, 1]
    projection = x @ choice_result["vector"]
    return {
        "n_direction_choices": int(eligible.size),
        "accuracy": float(accuracy_score(labels, probability >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "mean_projection_left": float(np.mean(projection[eligible[labels == 0]])),
        "mean_projection_right": float(np.mean(projection[eligible[labels == 1]])),
    }


def make_seed7_figures(seed_dir, metadata, pca_projection, choice_projection, confidence_projection):
    figure_dir = Path(seed_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    split = np.asarray([row["split"] for row in metadata])
    test = split == "test"
    choices = np.asarray([row["student_final_choice"] for row in metadata])
    offered = np.asarray([row["sure_offered"] for row in metadata], dtype=bool)
    p_correct = np.asarray([row["empirical_p_correct"] for row in metadata], dtype=float)
    colors = {1: "#2474b5", 2: "#d95f3d", 3: "#6a3d9a"}
    names = {1: "Left", 2: "Right", 3: "Sure"}

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    for choice in (1, 2, 3):
        selected = test & (choices == choice)
        ax.scatter(pca_projection[selected, 0], pca_projection[selected, 1], s=28, alpha=0.72, c=colors[choice], label=f"{names[choice]} (n={np.sum(selected)})")
    ax.set(xlabel="PC1", ylabel="PC2", title="Held-out pre-go states by deterministic choice")
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "fig1_pca_choice.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    scatter = ax.scatter(pca_projection[test, 0], pca_projection[test, 1], c=p_correct[test], cmap="viridis", s=30, alpha=0.78)
    ax.set(xlabel="PC1", ylabel="PC2", title="Held-out pre-go states by empirical P(correct)")
    fig.colorbar(scatter, ax=ax, label="empirical P(correct)")
    fig.savefig(figure_dir / "fig2_pca_confidence.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    selected_base = test & offered
    for sure_choice, marker, label in ((False, "o", "Direction"), (True, "^", "Sure")):
        selected = selected_base & ((choices == 3) == sure_choice)
        ax.scatter(pca_projection[selected, 0], pca_projection[selected, 1], s=38, alpha=0.75, marker=marker, label=f"{label} (n={np.sum(selected)})")
    ax.set(xlabel="PC1", ylabel="PC2", title="Sure-offered held-out trials")
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "fig2b_pca_sure_behavior.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    data = [choice_projection[test & (choices == 1)], choice_projection[test & (choices == 2)]]
    ax.boxplot(data, tick_labels=["Left", "Right"], showfliers=False)
    rng = np.random.RandomState(ANALYSIS_RANDOM_SEED)
    for pos, values in enumerate(data, start=1):
        ax.scatter(pos + rng.uniform(-0.08, 0.08, len(values)), values, s=18, alpha=0.55)
    ax.set(ylabel="choice-axis projection", title="Held-out directional choices")
    fig.savefig(figure_dir / "fig3_choice_axis.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.7), constrained_layout=True)
    ax.scatter(p_correct[test], confidence_projection[test], c=np.asarray([colors[c] for c in choices[test]]), s=27, alpha=0.7)
    ax.set(xlabel="empirical P(correct)", ylabel="confidence-axis projection", title=f"Held-out confidence generalization (r={safe_pearson(p_correct[test], confidence_projection[test]):.2f})")
    fig.savefig(figure_dir / "fig4_confidence_axis.png", dpi=200)
    plt.close(fig)

    for filename, selection, title in (
        ("fig5_choice_confidence.png", test, "Held-out choice × confidence representation"),
        ("fig5b_choice_confidence_sure_offered.png", test & offered, "Sure-offered held-out representation"),
    ):
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        for choice in (1, 2, 3):
            selected = selection & (choices == choice)
            ax.scatter(choice_projection[selected], confidence_projection[selected], s=30, alpha=0.72, c=colors[choice], label=f"{names[choice]} (n={np.sum(selected)})")
        ax.set(xlabel="choice-axis projection (higher = Right)", ylabel="confidence-axis projection (higher = higher P(correct))", title=title)
        ax.legend(frameon=False)
        fig.savefig(figure_dir / filename, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    data = [confidence_projection[test & offered & (choices == 3)], confidence_projection[test & offered & (choices != 3)]]
    ax.boxplot(data, tick_labels=["Later Sure", "Later Direction"], showfliers=False)
    rng = np.random.RandomState(ANALYSIS_RANDOM_SEED + 1)
    for pos, values in enumerate(data, start=1):
        ax.scatter(pos + rng.uniform(-0.08, 0.08, len(values)), values, s=20, alpha=0.58)
    ax.set(ylabel="confidence-axis projection", title="Sure-offered held-out trials")
    fig.savefig(figure_dir / "fig6_confidence_by_wager.png", dpi=200)
    plt.close(fig)


def run_seed(seed, data, info, split, freeze_dir, output_dir):
    seed_dir = Path(output_dir) / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    weights_path = Path(freeze_dir) / f"seed{seed}" / "weights.npz"
    before_hash = sha256(weights_path)
    raw_outputs, states = deterministic_replay(weights_path, np.asarray(data["x"], dtype=np.float32))
    categorical_outputs, choices, post_go = categorical_outputs_and_choices(raw_outputs, info)
    pre_go, evidence_end, starts, stops = extract_primary_states(states, info)
    standardized, scaler_mean, scaler_std, scaler_scale, zero_variance = fit_scaler(pre_go, split["train"])
    pca, pca_projection = fit_pca(standardized, split)
    p_correct = np.asarray(data["teacher_direction_success_proxy"], dtype=float)
    internal_margin = np.asarray(data["teacher_internal_margin"], dtype=float)
    offered = np.asarray([bool(item["sure_available"]) for item in info], dtype=bool)

    choice = choice_cross_validation(standardized, choices, split["train"], seed)
    confidence = confidence_cross_validation(standardized, p_correct, split["train"], seed, "empirical_p_correct")
    margin_confidence = confidence_cross_validation(standardized, internal_margin, split["train"], seed, "teacher_internal_margin")
    choice_projection = standardized @ choice["vector"]
    confidence_projection = standardized @ confidence["vector"]
    margin_projection = standardized @ margin_confidence["vector"]
    test = split["test"]
    wager = wager_metrics(confidence_projection, choices, offered, test, seed)
    choice_test = heldout_choice_metrics(choice, standardized, choices, test)
    cosine = float(np.dot(choice["vector"], confidence["vector"]))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    metadata = metadata_rows(seed, info, split, choices, post_go, starts, stops, data)
    write_csv(seed_dir / "state_metadata.csv", metadata)
    np.savez_compressed(
        seed_dir / "state_trajectories.npz",
        source_trial_indices=np.arange(states.shape[0], dtype=np.int64),
        hidden_states=states,
        raw_outputs=raw_outputs,
        categorical_outputs=categorical_outputs,
        input_dataset_path=np.asarray(str(Path("data/model_freeze/teacher_seed7.npz"))),
        input_dataset_sha256=np.asarray(sha256("data/model_freeze/teacher_seed7.npz")),
    )
    split_labels = np.empty(states.shape[0], dtype="<U10")
    for name, indices in split.items():
        split_labels[indices] = name
    np.savez(
        seed_dir / "pre_go_states.npz",
        source_trial_indices=np.arange(states.shape[0], dtype=np.int64),
        split=split_labels,
        raw_pre_go_states=pre_go,
        standardized_pre_go_states=standardized,
        end_of_evidence_states=evidence_end,
        pre_go_start_steps=starts,
        pre_go_stop_steps_exclusive=stops,
        choice_projection=choice_projection,
        confidence_projection=confidence_projection,
        internal_margin_projection=margin_projection,
    )
    np.savez(
        seed_dir / "scaler.npz",
        mean=scaler_mean,
        original_std=scaler_std,
        applied_scale=scaler_scale,
        zero_variance_mask=zero_variance,
        fit_source_trial_indices=split["train"],
        zero_variance_epsilon=np.asarray(ZERO_VARIANCE_EPS),
    )
    np.savez(
        seed_dir / "pca.npz",
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        mean=pca.mean_,
        fit_source_trial_indices=split["train"],
        train_source_trial_indices=split["train"],
        train_coordinates=pca_projection[split["train"]],
        validation_source_trial_indices=split["validation"],
        validation_coordinates=pca_projection[split["validation"]],
        test_source_trial_indices=split["test"],
        test_coordinates=pca_projection[split["test"]],
    )
    np.savez(
        seed_dir / "choice_axis.npz",
        vector=choice["vector"],
        intercept=np.asarray(choice["model"].intercept_[0]),
        raw_coefficient=choice["model"].coef_[0],
        fit_source_trial_indices=choice["eligible_indices"],
        positive_class=np.asarray("Right"),
        excluded_class=np.asarray("Sure"),
    )
    np.savez(
        seed_dir / "confidence_axis.npz",
        empirical_p_correct_vector=confidence["vector"],
        empirical_p_correct_intercept=np.asarray(confidence["model"].intercept_),
        empirical_p_correct_raw_coefficient=confidence["model"].coef_,
        empirical_p_correct_fit_source_trial_indices=confidence["eligible_indices"],
        empirical_p_correct_orientation=np.asarray("higher projection = higher empirical P(correct) on TRAIN"),
        internal_margin_vector=margin_confidence["vector"],
        internal_margin_intercept=np.asarray(margin_confidence["model"].intercept_),
        internal_margin_raw_coefficient=margin_confidence["model"].coef_,
        internal_margin_fit_source_trial_indices=margin_confidence["eligible_indices"],
    )

    cv_metrics = {
        "choice": {key: value for key, value in choice.items() if key not in ("model", "vector", "labels", "eligible_indices")},
        "confidence_empirical_p_correct": {key: value for key, value in confidence.items() if key not in ("model", "vector", "eligible_indices")},
        "confidence_internal_margin": {key: value for key, value in margin_confidence.items() if key not in ("model", "vector", "eligible_indices")},
    }
    heldout = {
        "seed": seed,
        "n_test": int(test.size),
        "choice": choice_test,
        "confidence": {
            "corr_projection_empirical_p_correct": safe_pearson(confidence_projection[test], p_correct[test]),
            "spearman_projection_empirical_p_correct": safe_spearman(confidence_projection[test], p_correct[test]),
            "r2_ridge_prediction_empirical_p_correct": float(r2_score(p_correct[test], confidence["model"].predict(standardized[test]))),
            "corr_internal_margin_projection_internal_margin": safe_pearson(margin_projection[test], internal_margin[test]),
        },
        "wager": wager,
        "geometry": {
            "choice_confidence_cosine_similarity": cosine,
            "choice_confidence_absolute_cosine_similarity": abs(cosine),
            "choice_confidence_angle_degrees": angle,
        },
        "pca": {
            "explained_variance_ratio_first_10": pca.explained_variance_ratio_[:10],
            "cumulative_variance_first_2": float(np.sum(pca.explained_variance_ratio_[:2])),
            "cumulative_variance_first_3": float(np.sum(pca.explained_variance_ratio_[:3])),
            "cumulative_variance_first_5": float(np.sum(pca.explained_variance_ratio_[:5])),
            "components_for_80_percent": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.8) + 1),
            "components_for_90_percent": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1),
        },
        "finite_deterministic_outputs_and_states": True,
        "zero_variance_unit_count": int(np.sum(zero_variance)),
    }
    write_json(seed_dir / "cv_metrics.json", cv_metrics)
    write_json(seed_dir / "heldout_metrics.json", heldout)
    if seed == EXAMPLE_SEED:
        make_seed7_figures(seed_dir, metadata, pca_projection, choice_projection, confidence_projection)

    after_hash = sha256(weights_path)
    if before_hash != after_hash:
        raise AssertionError(f"checkpoint changed during analysis: {weights_path}")
    summary = {
        "seed": seed,
        "choice_cv_accuracy": choice["accuracy"]["mean"],
        "choice_cv_accuracy_sd": choice["accuracy"]["sd"],
        "choice_cv_auc": choice["roc_auc"]["mean"],
        "choice_cv_auc_sd": choice["roc_auc"]["sd"],
        "confidence_cv_pearson_r": confidence["pearson_r"]["mean"],
        "confidence_cv_pearson_r_sd": confidence["pearson_r"]["sd"],
        "confidence_cv_spearman_rho": confidence["spearman_rho"]["mean"],
        "confidence_cv_spearman_rho_sd": confidence["spearman_rho"]["sd"],
        "confidence_cv_r2": confidence["r2"]["mean"],
        "confidence_cv_r2_sd": confidence["r2"]["sd"],
        "test_corr_confidence_projection_empirical_p_correct": heldout["confidence"]["corr_projection_empirical_p_correct"],
        "n_test_sure": wager["n_later_sure"],
        "n_test_direction": wager["n_later_direction"],
        "mean_confidence_projection_sure": wager["sure_projection_mean"],
        "mean_confidence_projection_direction": wager["direction_projection_mean"],
        "sure_minus_direction_confidence": wager["sure_minus_direction_mean_difference"],
        "sure_minus_direction_ci95_low": wager["bootstrap_ci95_mean_difference"][0],
        "sure_minus_direction_ci95_high": wager["bootstrap_ci95_mean_difference"][1],
        "later_sure_logistic_coefficient": wager["later_sure_logistic"]["coefficient"],
        "later_sure_logistic_se": wager["later_sure_logistic"]["standard_error"],
        "choice_confidence_cosine_similarity": cosine,
        "choice_confidence_absolute_cosine_similarity": abs(cosine),
        "choice_confidence_angle_degrees": angle,
        "pca_cumulative_variance_first_2": heldout["pca"]["cumulative_variance_first_2"],
        "checkpoint_sha256_before": before_hash,
        "checkpoint_sha256_after": after_hash,
    }
    return summary


def make_replication_figure(output_dir, summaries):
    seeds = [row["seed"] for row in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    axes[0].bar([str(seed) for seed in seeds], [row["confidence_cv_pearson_r"] for row in summaries], color="#4078a8")
    axes[0].set(title="Confidence CV", xlabel="seed", ylabel="Pearson r", ylim=(0, 1))
    differences = np.asarray([row["sure_minus_direction_confidence"] for row in summaries])
    lower = differences - np.asarray([row["sure_minus_direction_ci95_low"] for row in summaries])
    upper = np.asarray([row["sure_minus_direction_ci95_high"] for row in summaries]) - differences
    axes[1].errorbar([str(seed) for seed in seeds], differences, yerr=np.vstack([lower, upper]), fmt="o", color="#7b3294", capsize=4)
    axes[1].axhline(0, color="0.5", linewidth=1)
    axes[1].set(title="Later wager association", xlabel="seed", ylabel="Sure − Direction confidence")
    axes[2].bar([str(seed) for seed in seeds], [row["choice_confidence_absolute_cosine_similarity"] for row in summaries], color="#e08214")
    axes[2].set(title="Axis overlap", xlabel="seed", ylabel="absolute cosine", ylim=(0, 1))
    path = Path(output_dir) / "multiseed" / "replication_summary.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def report_text(manifest, summaries):
    lines = [
        "# Population Representation Stage 1", "",
        "## A. State extraction", "",
        "Each frozen PsychRNN checkpoint was loaded read-only into `BasicSimulator` with `alpha=0.1`, ReLU dynamics, and `rec_noise=0`. The exact 1,000 saved teacher-dataset input tensors were replayed in source-trial order; no task trials were regenerated. Hidden trajectories have 50 units and shape `1000 × 300 × 50` per seed. Raw and categorical outputs, trial metadata, and reconstructable input references were saved. Checkpoint hashes were identical before and after analysis.", "",
        "## B. Primary analysis window", "",
        "`delay_end` is the go cue and post-go behavioral-readout onset. The primary state is the mean of recurrent states over `[delay_end-10, delay_end)`, exactly 100 ms at `dt=10 ms`, excluding post-go activity. Motion onset is `fixation_end`, motion offset is `stimulus_end`, and sure-target onset is `ts_onset` when `sure_available` is true.", "",
        "## C. PCA", "",
        "PCA was fit separately for each seed using standardized TRAIN pre-go states only. PCA is used for visualization, not as the inferential test.", "",
        "| seed | variance PC1+PC2 |", "|---:|---:|",
    ]
    lines.extend([f"| {row['seed']} | {row['pca_cumulative_variance_first_2']:.3f} |" for row in summaries])
    lines += ["", "The seed-7 figures show choice- and confidence-related geometry in the held-out projection, but PCA component separation alone is not treated as evidence for a confidence representation.", "", "## D. Choice representation", "", "The L2 logistic decoder used only TRAIN trials with deterministic Left or Right choices; Sure choices were excluded. Five-fold stratified cross-validation used fixed `C=1`.", "", "| seed | CV accuracy mean ± SD | CV AUC mean ± SD |", "|---:|---:|---:|"]
    lines.extend([f"| {row['seed']} | {row['choice_cv_accuracy']:.3f} ± {row['choice_cv_accuracy_sd']:.3f} | {row['choice_cv_auc']:.3f} ± {row['choice_cv_auc_sd']:.3f} |" for row in summaries])
    lines += ["", "## E. Confidence representation", "", "The primary confidence axis used Ridge regression (`alpha=1`) from TRAIN pre-go states to empirical P(correct), with no Sure/Direction labels. Internal margin was analyzed with the identical complementary procedure.", "", "| seed | CV Pearson r | CV Spearman rho | CV R² | held-out correlation |", "|---:|---:|---:|---:|---:|"]
    lines.extend([f"| {row['seed']} | {row['confidence_cv_pearson_r']:.3f} | {row['confidence_cv_spearman_rho']:.3f} | {row['confidence_cv_r2']:.3f} | {row['test_corr_confidence_projection_empirical_p_correct']:.3f} |" for row in summaries])
    lines += ["", "## F. Wager prediction", "", "The TRAIN-defined confidence axis was evaluated among sure-offered TEST trials. Higher projection is oriented toward higher TRAIN empirical P(correct); orientation never used test wagering.", "", "| seed | n Sure / Direction | mean Sure | mean Direction | Sure − Direction [bootstrap 95% CI] | logistic coefficient ± SE |", "|---:|---:|---:|---:|---:|---:|"]
    lines.extend([f"| {row['seed']} | {row['n_test_sure']} / {row['n_test_direction']} | {row['mean_confidence_projection_sure']:.3f} | {row['mean_confidence_projection_direction']:.3f} | {row['sure_minus_direction_confidence']:.3f} [{row['sure_minus_direction_ci95_low']:.3f}, {row['sure_minus_direction_ci95_high']:.3f}] | {row['later_sure_logistic_coefficient']:.3f} ± {row['later_sure_logistic_se']:.3f} |" for row in summaries])
    lines += ["", "Sample sizes are small for later Sure choices, so these estimates are presented as associations with uncertainty, not causal effects.", "", "## G. Geometry", "", "The independently fit choice and empirical-confidence axes were not orthogonalized.", "", "| seed | cosine | absolute cosine | angle |", "|---:|---:|---:|---:|"]
    lines.extend([f"| {row['seed']} | {row['choice_confidence_cosine_similarity']:.3f} | {row['choice_confidence_absolute_cosine_similarity']:.3f} | {row['choice_confidence_angle_degrees']:.1f}° |" for row in summaries])
    lines += ["", "## H. Cross-seed replication", "", "Identical extraction, preprocessing, folds, fixed regularization, permutation count, bootstrap count, and plotting definitions were applied to seeds 7, 8, and 9. Seed 7 was retained as the predetermined example regardless of result strength.", "", "## I. Leakage checks", "", "- Scaler statistics and PCA were fit only on the 700 TRAIN identities.", "- Choice and confidence decoders were fit and cross-validated only within TRAIN.", "- TEST states and labels entered only final held-out transformations and metrics.", "- The confidence target was continuous empirical P(correct); Sure-vs-Direction labels were never used to fit or orient it.", "- Checkpoint SHA-256 values were unchanged before versus after replay.", "- The analysis imports only the inference simulator and contains no network training call.", "", "## J. Artifact paths", "", "Each `seedN/` directory contains `state_metadata.csv`, full deterministic `state_trajectories.npz`, `pre_go_states.npz`, `scaler.npz`, `pca.npz`, both axes, CV metrics, and held-out metrics. Seed 7 additionally contains the detailed figures. Cross-seed tables and the replication figure are under `multiseed/`; exact paths and hashes are in `analysis_manifest.json`.", "", "## Supported conclusion and next stage", "", "This stage tests whether frozen pre-go population activity contains cross-validated linear information about directional choice and teacher-derived confidence, whether that information generalizes to held-out trials, and whether the independently defined confidence projection is associated with later wagering. Decoding and association do not establish causal use.", "", "Time-resolved decoding, recurrent-noise variability, pre-TS variability, TDR, dPCA, and fixed-point analysis were not performed and remain separate future stages.", ""]
    return "\n".join(lines)


def run(args):
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis directory: {output_dir}")
    source_text = Path(__file__).read_text(encoding="utf-8")
    if "model" + ".train(" in source_text or "Basic" + "(" in source_text:
        raise AssertionError("analysis source contains a forbidden network-training construction")
    freeze_dir = Path(args.freeze_dir).resolve()
    dataset_path = Path(args.dataset).resolve()
    split_path = freeze_dir / "split_indices.npz"
    data, info, split = load_frozen_inputs(dataset_path, split_path)
    checkpoint_before = {seed: sha256(freeze_dir / f"seed{seed}" / "weights.npz") for seed in SEEDS}
    output_dir.mkdir(parents=True)
    summaries = []
    for seed in SEEDS:
        print(f"Population Stage 1: deterministic replay and train-only fits for seed {seed}", flush=True)
        summaries.append(run_seed(seed, data, info, split, freeze_dir, output_dir))
    checkpoint_after = {seed: sha256(freeze_dir / f"seed{seed}" / "weights.npz") for seed in SEEDS}
    assert checkpoint_before == checkpoint_after
    write_csv(output_dir / "multiseed" / "representation_summary.csv", summaries)
    write_json(output_dir / "multiseed" / "representation_summary.json", summaries)
    replication_figure = make_replication_figure(output_dir, summaries)
    manifest = {
        "created_unix_time": time.time(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "split_path": str(split_path),
        "split_sha256": sha256(split_path),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "seeds": list(SEEDS),
        "example_seed": EXAMPLE_SEED,
        "hidden_state_dimensionality": 50,
        "deterministic_replay": {"simulator": "psychrnn.backend.simulation.BasicSimulator", "alpha": 0.1, "rec_noise": 0.0, "source_inputs": "saved dataset x tensor"},
        "primary_window": {"event": "delay_end/go cue", "start_offset_steps": -PRE_GO_STEPS, "stop_offset_steps_exclusive": 0, "duration_ms": PRE_GO_MS},
        "analysis_constants": {"n_folds": N_FOLDS, "choice_C": CHOICE_C, "ridge_alpha": RIDGE_ALPHA, "choice_permutations": N_CHOICE_PERMUTATIONS, "bootstraps": N_BOOTSTRAPS, "random_seed": ANALYSIS_RANDOM_SEED, "zero_variance_epsilon": ZERO_VARIANCE_EPS},
        "split_counts": {name: int(indices.size) for name, indices in split.items()},
        "leakage_checks": {"scaler_train_only": True, "pca_train_only": True, "choice_axis_train_only": True, "confidence_axis_train_only": True, "test_excluded_from_all_fits": True, "sure_labels_excluded_from_confidence_fit": True, "identical_definitions_all_seeds": True, "checkpoints_unchanged": True, "network_training_called": False},
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "psychrnn")},
        "python": platform.python_version(),
        "command": " ".join([sys.executable, *sys.argv]),
        "replication_figure": str(replication_figure.resolve()),
        "stage2_analyses_run": [],
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    (output_dir / "final_report.md").write_text(report_text(manifest, summaries), encoding="utf-8")
    print(json.dumps(jsonable(summaries), indent=2), flush=True)
    return summaries


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/model_freeze/teacher_seed7.npz")
    parser.add_argument("--freeze-dir", default="results/final_supervised_freeze")
    parser.add_argument("--output-dir", default="results/population_stage1")
    return parser


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
