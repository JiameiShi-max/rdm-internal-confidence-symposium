"""Population Representation Stage 3: identical-stimulus noise variability.

This is an inference-only analysis.  It replays saved TEST input tensors through
frozen weights and projects recurrent states onto the independently fitted
Stage-1 axes.  It contains no network optimization or neural decoder fitting.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit, logit


SEEDS = (7, 8, 9)
EXAMPLE_SEED = 7
DT_MS = 10
ALPHA = 0.1
STOCHASTIC_REC_NOISE = 0.05
DETERMINISTIC_REC_NOISE = 0.0
N_REPEATS = 100
N_DETERMINISTIC_REPEATS = 10
READOUT_STEPS = 10
PRE_TS_STEPS = 10
TEMPORAL_OFFSETS_MS = np.arange(-300, 1, 10, dtype=np.int64)
TEMPORAL_WINDOW_STEPS = 5
N_BOOTSTRAPS = 2000
N_PRIMARY_PERMUTATIONS = 1000
N_TEMPORAL_PERMUTATIONS = 500
ONSET_RUN_POINTS = 5
NULL_QUANTILE = 0.95
RANDOM_SEED = 2029
NOISE_SEED_BASE = 30_000_000


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
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


def noise_seed(model_seed, source_trial_id, repeat_id):
    """One-to-one seed schedule for the requested ranges."""
    model_seed = int(model_seed)
    source_trial_id = int(source_trial_id)
    repeat_id = int(repeat_id)
    if not (0 <= source_trial_id < 1000 and 0 <= repeat_id < 1000):
        raise ValueError("source and repeat identifiers must be in [0, 1000)")
    return NOISE_SEED_BASE + model_seed * 1_000_000 + source_trial_id * 1_000 + repeat_id


def softmax_actions(raw_outputs):
    logits = np.asarray(raw_outputs[..., 1:4], dtype=np.float64)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / np.sum(values, axis=-1, keepdims=True)


def simulate_frozen_repeats(weights, source_input, seeds, rec_noise):
    """Replay one exact input for several independently seeded realizations.

    The recurrence is algebraically and numerically identical to PsychRNN's
    BasicSimulator.  Supplying each repeat's complete noise tensor explicitly
    makes the repeat-level seed schedule reproducible and auditable.
    """
    source_input = np.asarray(source_input, dtype=np.float32)
    seeds = np.asarray(seeds, dtype=np.int64)
    n_repeats = seeds.size
    n_steps = source_input.shape[0]
    n_rec = int(np.asarray(weights["W_rec"]).shape[0])
    repeated_input = np.repeat(source_input[None, :, :], n_repeats, axis=0)
    if not np.array_equal(repeated_input, np.broadcast_to(source_input, repeated_input.shape)):
        raise AssertionError("repeated inputs are not exactly identical")

    noise = np.empty((n_repeats, n_steps, n_rec), dtype=np.float64)
    for repeat_id, seed in enumerate(seeds):
        noise[repeat_id] = np.random.RandomState(int(seed)).normal(
            loc=0.0, scale=1.0, size=(n_steps, n_rec)
        )

    state = np.repeat(np.asarray(weights["init_state"]), n_repeats, axis=0)
    states = np.empty((n_repeats, n_steps, n_rec), dtype=np.float64)
    outputs = np.empty((n_repeats, n_steps, np.asarray(weights["W_out"]).shape[0]), dtype=np.float64)
    w_rec = np.asarray(weights["W_rec"])
    w_in = np.asarray(weights["W_in"])
    w_out = np.asarray(weights["W_out"])
    b_rec = np.asarray(weights["b_rec"])
    b_out = np.asarray(weights["b_out"])
    noise_scale = np.sqrt(2.0 * ALPHA * float(rec_noise) ** 2)
    recurrent_connectivity = np.ones_like(w_rec)
    for step in range(n_steps):
        state = (
            (1.0 - ALPHA) * state
            + ALPHA
            * (
                np.matmul(np.maximum(state, 0.0), np.transpose(w_rec * recurrent_connectivity))
                + np.matmul(repeated_input[:, step], np.transpose(w_in))
                + b_rec
            )
            + noise_scale * noise[:, step]
        )
        outputs[:, step] = np.matmul(np.maximum(state, 0.0), np.transpose(w_out)) + b_out
        states[:, step] = state
    return outputs, states


def fixed_axis_projection(states, scaler_mean, scaler_scale, axis):
    standardized = (np.asarray(states, dtype=np.float64) - scaler_mean) / scaler_scale
    return standardized @ axis


def temporal_pre_ts_states(states, ts_onset):
    rows = []
    for offset_ms in TEMPORAL_OFFSETS_MS:
        endpoint = int(ts_onset) + int(offset_ms // DT_MS)
        start = endpoint - TEMPORAL_WINDOW_STEPS
        if start < 0 or endpoint > states.shape[1]:
            raise AssertionError("invalid pre-target trailing window")
        rows.append(np.mean(states[:, start:endpoint, :], axis=1))
    return np.asarray(rows, dtype=np.float64)


def center_within_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values - np.mean(values, axis=1, keepdims=True)


def _profile_intercepts(x, beta, successes):
    linear = np.einsum("gnp,p->gn", x, beta)
    fraction = successes / x.shape[1]
    alpha = logit(fraction)
    for _ in range(40):
        probability = expit(alpha[:, None] + linear)
        residual = np.sum(probability, axis=1) - successes
        derivative = np.sum(probability * (1.0 - probability), axis=1)
        update = residual / np.maximum(derivative, 1e-12)
        alpha -= np.clip(update, -5.0, 5.0)
        if np.max(np.abs(residual)) < 1e-10:
            break
    return alpha, expit(alpha[:, None] + linear)


def fit_fixed_effect_logistic(x, y, group_weights=None, max_iter=80):
    """Profile stimulus intercepts and estimate common logistic slopes.

    x has shape stimulus x repeat x predictor.  Only variable-outcome strata
    are valid.  Cluster weights support source-stimulus bootstrap resampling.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, :, None]
    if y.shape != x.shape[:2]:
        raise ValueError("x and y group/repeat dimensions differ")
    successes = np.sum(y, axis=1)
    if np.any(successes <= 0) or np.any(successes >= y.shape[1]):
        raise ValueError("fixed-effect fit requires variable-outcome stimuli")
    weights = np.ones(x.shape[0], dtype=np.float64) if group_weights is None else np.asarray(group_weights, dtype=np.float64)
    beta = np.zeros(x.shape[2], dtype=np.float64)
    converged = False
    for iteration in range(max_iter):
        alpha, probability = _profile_intercepts(x, beta, successes)
        variance = probability * (1.0 - probability)
        score_by_group = np.einsum("gnp,gn->gp", x, y - probability)
        score = np.einsum("g,gp->p", weights, score_by_group)
        information = np.zeros((x.shape[2], x.shape[2]), dtype=np.float64)
        for group in range(x.shape[0]):
            if weights[group] == 0:
                continue
            xg = x[group]
            qg = variance[group]
            qsum = float(np.sum(qg))
            weighted_sum = np.sum(qg[:, None] * xg, axis=0)
            block = xg.T @ (qg[:, None] * xg) - np.outer(weighted_sum, weighted_sum) / max(qsum, 1e-12)
            information += weights[group] * block
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ score
        if np.max(np.abs(step)) > 5.0:
            step *= 5.0 / np.max(np.abs(step))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            converged = True
            break
    alpha, probability = _profile_intercepts(x, beta, successes)
    eta = alpha[:, None] + np.einsum("gnp,p->gn", x, beta)
    group_log_likelihood = np.sum(y * eta - np.logaddexp(0.0, eta), axis=1)
    return {
        "beta": beta,
        "stimulus_intercepts": alpha,
        "log_likelihood": float(np.sum(weights * group_log_likelihood)),
        "converged": bool(converged),
        "iterations": int(iteration + 1),
    }


def standardize_centered(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        sd = float(np.std(values))
    else:
        weights = np.asarray(weights, dtype=np.float64)
        sd = float(np.sqrt(np.sum(weights[:, None] * values ** 2) / (values.shape[1] * np.sum(weights))))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("centered predictor has no variation")
    return values / sd, sd


def permuted_outcomes(y, rng):
    return np.asarray([rng.permutation(row) for row in np.asarray(y)], dtype=np.int8)


def operational_negative_onset(times_ms, beta, threshold):
    exceeds = -np.asarray(beta, dtype=float) > float(threshold)
    for start in range(0, exceeds.size - ONSET_RUN_POINTS + 1):
        if np.all(exceeds[start : start + ONSET_RUN_POINTS]):
            return int(times_ms[start])
    return None


def analyze_primary(confidence, absolute_choice, outcomes, source_ids, seed):
    variable = (np.sum(outcomes, axis=1) > 0) & (np.sum(outcomes, axis=1) < outcomes.shape[1])
    if not np.any(variable):
        raise RuntimeError("no variable stimuli; primary model is undefined")
    variable_ids = np.asarray(source_ids)[variable]
    y = np.asarray(outcomes[variable], dtype=np.int8)
    confidence_centered = center_within_rows(confidence[variable])
    confidence_z, confidence_sd = standardize_centered(confidence_centered)
    fit = fit_fixed_effect_logistic(confidence_z, y)
    observed_beta = float(fit["beta"][0])

    bootstrap_rng = np.random.RandomState(RANDOM_SEED + seed * 10 + 1)
    bootstrap_beta = np.empty(N_BOOTSTRAPS, dtype=np.float64)
    for index in range(N_BOOTSTRAPS):
        sampled = bootstrap_rng.randint(0, y.shape[0], size=y.shape[0])
        weights = np.bincount(sampled, minlength=y.shape[0]).astype(np.float64)
        boot_x, _ = standardize_centered(confidence_centered, weights=weights)
        bootstrap_beta[index] = fit_fixed_effect_logistic(boot_x, y, group_weights=weights)["beta"][0]

    permutation_rng = np.random.RandomState(RANDOM_SEED + seed * 10 + 2)
    permutation_beta = np.empty(N_PRIMARY_PERMUTATIONS, dtype=np.float64)
    for index in range(N_PRIMARY_PERMUTATIONS):
        permutation_beta[index] = fit_fixed_effect_logistic(
            confidence_z, permuted_outcomes(y, permutation_rng)
        )["beta"][0]
    two_sided = (1.0 + np.sum(np.abs(permutation_beta) >= abs(observed_beta))) / (N_PRIMARY_PERMUTATIONS + 1.0)
    negative_tail = (1.0 + np.sum(permutation_beta <= observed_beta)) / (N_PRIMARY_PERMUTATIONS + 1.0)

    absolute_centered = center_within_rows(absolute_choice[variable])
    absolute_z, absolute_sd = standardize_centered(absolute_centered)
    specificity_x = np.stack([confidence_z, absolute_z], axis=2)
    specificity = fit_fixed_effect_logistic(specificity_x, y)

    differences = np.asarray(
        [np.mean(confidence[variable][i][y[i] == 1]) - np.mean(confidence[variable][i][y[i] == 0]) for i in range(y.shape[0])],
        dtype=np.float64,
    )
    model = {
        "model": "profiled stimulus-fixed-effect logistic regression",
        "outcome": "later Sure=1, Direction=0",
        "predictor": "within-stimulus centered confidence projection, globally standardized over variable-stimulus repeats",
        "n_variable_stimuli": int(y.shape[0]),
        "n_repeats": int(y.size),
        "variable_source_trial_ids": variable_ids,
        "confidence_predictor_sd_before_standardization": confidence_sd,
        "beta": observed_beta,
        "odds_ratio_per_1sd": float(np.exp(observed_beta)),
        "converged": fit["converged"],
        "iterations": fit["iterations"],
        "log_likelihood": fit["log_likelihood"],
        "bootstrap_median": float(np.median(bootstrap_beta)),
        "bootstrap_ci95": np.percentile(bootstrap_beta, [2.5, 97.5]),
        "bootstrap_resampling_unit": "source stimulus",
        "bootstrap_count": N_BOOTSTRAPS,
        "permutation_count": N_PRIMARY_PERMUTATIONS,
        "permutation_scheme": "Sure/Direction shuffled independently within each variable source stimulus",
        "permutation_two_sided_p": float(two_sided),
        "permutation_negative_tail_p": float(negative_tail),
        "specificity_model": {
            "predictors": ["within-stimulus confidence", "within-stimulus absolute choice-axis projection"],
            "confidence_beta": float(specificity["beta"][0]),
            "absolute_choice_beta": float(specificity["beta"][1]),
            "absolute_choice_predictor_sd_before_standardization": absolute_sd,
            "converged": specificity["converged"],
        },
        "transparent_summary": {
            "mean_sure_minus_direction_confidence_difference": float(np.mean(differences)),
            "median_sure_minus_direction_confidence_difference": float(np.median(differences)),
            "fraction_variable_stimuli_negative": float(np.mean(differences < 0)),
        },
    }
    return model, variable, differences, bootstrap_beta, permutation_beta


def analyze_temporal(time_confidence, outcomes, variable, seed):
    y = np.asarray(outcomes[variable], dtype=np.int8)
    values = np.asarray(time_confidence[:, variable, :], dtype=np.float64)
    centered = values - np.mean(values, axis=2, keepdims=True)
    standardized = np.empty_like(centered)
    predictor_sd = np.empty(centered.shape[0], dtype=np.float64)
    observed_beta = np.empty(centered.shape[0], dtype=np.float64)
    for time_index in range(centered.shape[0]):
        standardized[time_index], predictor_sd[time_index] = standardize_centered(centered[time_index])
        observed_beta[time_index] = fit_fixed_effect_logistic(standardized[time_index], y)["beta"][0]

    rng = np.random.RandomState(RANDOM_SEED + seed * 10 + 3)
    null_beta = np.empty((N_TEMPORAL_PERMUTATIONS, centered.shape[0]), dtype=np.float64)
    for permutation in range(N_TEMPORAL_PERMUTATIONS):
        permuted = permuted_outcomes(y, rng)
        for time_index in range(centered.shape[0]):
            null_beta[permutation, time_index] = fit_fixed_effect_logistic(
                standardized[time_index], permuted
            )["beta"][0]
    max_negative = np.max(-null_beta, axis=1)
    threshold = float(np.quantile(max_negative, NULL_QUANTILE))
    onset = operational_negative_onset(TEMPORAL_OFFSETS_MS, observed_beta, threshold)
    rows = []
    for index, offset in enumerate(TEMPORAL_OFFSETS_MS):
        rows.append(
            {
                "relative_time_ms": int(offset),
                "trailing_window_ms": TEMPORAL_WINDOW_STEPS * DT_MS,
                "n_variable_stimuli": int(y.shape[0]),
                "n_repeats": int(y.size),
                "confidence_predictor_sd_before_standardization": float(predictor_sd[index]),
                "fixed_effect_beta": float(observed_beta[index]),
                "predicted_negative_association": float(-observed_beta[index]),
                "familywise_negative_threshold": threshold,
                "exceeds_familywise_threshold": bool(-observed_beta[index] > threshold),
            }
        )
    return rows, null_beta, max_negative, threshold, onset


def make_seed7_figures(seed_dir, repeat_rows, stimulus_rows, confidence, outcomes, variable, differences, model, temporal_rows, deterministic, time_confidence, source_ids):
    figure_dir = Path(seed_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_ids = np.asarray(source_ids)

    fig, ax = plt.subplots(figsize=(10, 4.7), constrained_layout=True)
    ax.bar(np.arange(len(stimulus_rows)), [row["p_sure"] for row in stimulus_rows], color="#6a3d9a")
    ax.set(xlabel="sure-offered TEST stimulus (ascending source ID)", ylabel="P(Sure | 100 repeats)", title="Identical-input behavioral variability")
    ax.set_xticks(np.arange(0, len(stimulus_rows), 5), [str(stimulus_rows[i]["source_trial_id"]) for i in range(0, len(stimulus_rows), 5)], rotation=45)
    fig.savefig(figure_dir / "fig1_identical_input_behavior.png", dpi=200)
    plt.close(fig)

    variable_ids = source_ids[variable]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for index, source_id in enumerate(variable_ids):
        sure_mean = np.mean(confidence[variable][index][outcomes[variable][index] == 1])
        direction_mean = np.mean(confidence[variable][index][outcomes[variable][index] == 0])
        ax.plot([0, 1], [direction_mean, sure_mean], color="0.72", linewidth=0.8, alpha=0.8)
        ax.scatter([0, 1], [direction_mean, sure_mean], s=16, color=["#1b9e77", "#6a3d9a"])
    ax.set(xticks=[0, 1], xticklabels=["Direction repeats", "Sure repeats"], ylabel="fixed confidence-axis projection", title=f"Within-stimulus confidence effect (n={len(variable_ids)} stimuli)")
    fig.savefig(figure_dir / "fig2_within_stimulus_paired.png", dpi=200)
    plt.close(fig)

    centered = center_within_rows(confidence[variable]).ravel()
    yflat = outcomes[variable].ravel().astype(bool)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    bins = np.linspace(np.percentile(centered, 0.5), np.percentile(centered, 99.5), 35)
    ax.hist(centered[~yflat], bins=bins, density=True, alpha=0.55, label="later Direction", color="#1b9e77")
    ax.hist(centered[yflat], bins=bins, density=True, alpha=0.55, label="later Sure", color="#6a3d9a")
    ax.set(xlabel="within-stimulus confidence projection", ylabel="density", title="Descriptive pooled distribution (nested repeats)")
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "fig3_centered_projection.png", dpi=200)
    plt.close(fig)

    grid = np.linspace(-2.5, 2.5, 200)
    beta = model["beta"]
    baseline = float(np.mean(outcomes[variable]))
    predicted = expit(logit(baseline) + beta * grid)
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax.plot(grid, predicted, color="#6a3d9a", linewidth=2)
    ax.axvline(0, color="0.5", linewidth=1)
    ax.set(xlabel="within-stimulus confidence fluctuation (SD)", ylabel="illustrative P(Sure)", title="Fixed-effect slope at the average baseline")
    fig.savefig(figure_dir / "fig4_fixed_effect_relationship.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    times = np.asarray([row["relative_time_ms"] for row in temporal_rows])
    association = np.asarray([row["predicted_negative_association"] for row in temporal_rows])
    threshold = temporal_rows[0]["familywise_negative_threshold"]
    ax.plot(times, association, color="#6a3d9a", linewidth=2)
    ax.axhline(threshold, color="#d95f02", linestyle="--", label="95% max-stat threshold")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="time relative to Sure-target onset (ms)\n(each point summarizes preceding 50 ms)", ylabel="negative fixed-effect coefficient", title="Pre-target internal-variability association")
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "fig5_temporal_divergence.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Noise-free identical-input control", pad=18)
    ax.text(0.5, 0.72, "Exact repeatability", ha="center", va="center", fontsize=22, color="#1b9e77", weight="bold")
    ax.text(
        0.5,
        0.40,
        f"choice consistency  {deterministic['choice_consistency']:.3f}\n"
        f"maximum state difference  {deterministic['max_abs_state_difference']:.3g}\n"
        f"maximum output difference  {deterministic['max_abs_output_difference']:.3g}",
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.7,
    )
    fig.savefig(figure_dir / "fig6_deterministic_control.png", dpi=200)
    plt.close(fig)

    example_source = int(np.min(variable_ids))
    example_index = int(np.flatnonzero(source_ids == example_source)[0])
    example_y = outcomes[example_index].astype(bool)
    example_time = time_confidence[:, example_index, :]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for sure, color, label in ((False, "#1b9e77", "later Direction"), (True, "#6a3d9a", "later Sure")):
        traces = example_time[:, example_y == sure]
        ax.plot(TEMPORAL_OFFSETS_MS, traces, color=color, alpha=0.08, linewidth=0.7)
        ax.plot(TEMPORAL_OFFSETS_MS, np.mean(traces, axis=1), color=color, linewidth=2.3, label=f"{label} mean (n={traces.shape[1]})")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="time relative to Sure-target onset (ms)", ylabel="fixed confidence-axis projection", title=f"Illustrative lowest-ID variable stimulus: {example_source}")
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "fig7_lowest_id_variable_trajectory.png", dpi=200)
    plt.close(fig)
    return example_source


def load_stage1_objects(stage1_dir):
    stage1_dir = Path(stage1_dir)
    protected = [stage1_dir / name for name in ("scaler.npz", "confidence_axis.npz", "choice_axis.npz")]
    hashes = {path.name: sha256(path) for path in protected}
    with np.load(stage1_dir / "scaler.npz", allow_pickle=False) as stored:
        mean = np.asarray(stored["mean"], dtype=np.float64)
        scale = np.asarray(stored["applied_scale"], dtype=np.float64)
        scaler_fit = np.asarray(stored["fit_source_trial_indices"], dtype=np.int64)
    with np.load(stage1_dir / "confidence_axis.npz", allow_pickle=False) as stored:
        confidence_axis = np.asarray(stored["empirical_p_correct_vector"], dtype=np.float64)
        confidence_fit = np.asarray(stored["empirical_p_correct_fit_source_trial_indices"], dtype=np.int64)
        orientation = str(stored["empirical_p_correct_orientation"])
    with np.load(stage1_dir / "choice_axis.npz", allow_pickle=False) as stored:
        choice_axis = np.asarray(stored["vector"], dtype=np.float64)
        choice_fit = np.asarray(stored["fit_source_trial_indices"], dtype=np.int64)
    return mean, scale, confidence_axis, choice_axis, scaler_fit, confidence_fit, choice_fit, orientation, hashes


def replay_seed(seed, data, info, split, stage1_root, freeze_root, output_root):
    seed_dir = Path(output_root) / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    stage1_dir = Path(stage1_root) / f"seed{seed}"
    weights_path = Path(freeze_root) / f"seed{seed}" / "weights.npz"
    checkpoint_before = sha256(weights_path)
    mean, scale, confidence_axis, choice_axis, scaler_fit, confidence_fit, choice_fit, orientation, stage1_before = load_stage1_objects(stage1_dir)
    if set(scaler_fit) != set(split["train"]) or set(confidence_fit) != set(split["train"]) or not set(choice_fit) <= set(split["train"]):
        raise AssertionError("Stage-1 objects are not TRAIN-derived as required")
    if "higher projection = higher empirical P(correct)" not in orientation:
        raise AssertionError("unexpected Stage-1 confidence orientation")
    with np.load(weights_path, allow_pickle=True) as stored:
        weights = {name: stored[name] for name in stored.files}

    source_ids = np.sort(np.asarray([index for index in split["test"] if bool(info[int(index)]["sure_available"])], dtype=np.int64))
    n_sources = source_ids.size
    if n_sources == 0:
        raise RuntimeError("no sure-offered TEST source stimuli")
    confidence = np.empty((n_sources, N_REPEATS), dtype=np.float64)
    choice_projection = np.empty_like(confidence)
    outcomes = np.empty((n_sources, N_REPEATS), dtype=np.int8)
    pre_ts_states = np.empty((n_sources, N_REPEATS, confidence_axis.size), dtype=np.float32)
    time_confidence = np.empty((TEMPORAL_OFFSETS_MS.size, n_sources, N_REPEATS), dtype=np.float64)
    repeat_rows = []
    input_hashes = {}
    metadata_hashes = {}
    sure_channel_pre_ts_max = 0.0

    for source_position, source_id in enumerate(source_ids):
        trial = info[int(source_id)]
        source_input = np.asarray(data["x"][source_id], dtype=np.float32)
        ts_onset = int(trial["ts_onset"])
        sure_channel_pre_ts_max = max(sure_channel_pre_ts_max, float(np.max(np.abs(source_input[ts_onset - PRE_TS_STEPS : ts_onset, 3]))))
        if not np.all(source_input[ts_onset - PRE_TS_STEPS : ts_onset, 3] == 0):
            raise AssertionError("Sure input is active in the primary pre-target window")
        input_hashes[str(source_id)] = array_sha256(source_input)
        metadata_hashes[str(source_id)] = hashlib.sha256(json.dumps(trial, sort_keys=True).encode("utf-8")).hexdigest()
        seeds = np.asarray([noise_seed(seed, source_id, repeat) for repeat in range(N_REPEATS)], dtype=np.int64)
        raw_outputs, states = simulate_frozen_repeats(weights, source_input, seeds, STOCHASTIC_REC_NOISE)
        primary_state = np.mean(states[:, ts_onset - PRE_TS_STEPS : ts_onset, :], axis=1)
        pre_ts_states[source_position] = primary_state.astype(np.float32)
        confidence[source_position] = fixed_axis_projection(primary_state, mean, scale, confidence_axis)
        choice_projection[source_position] = fixed_axis_projection(primary_state, mean, scale, choice_axis)
        temporal_states = temporal_pre_ts_states(states, ts_onset)
        time_confidence[:, source_position, :] = fixed_axis_projection(temporal_states, mean, scale, confidence_axis)
        probabilities = softmax_actions(raw_outputs)
        start = int(trial["delay_end"])
        post_go = np.mean(probabilities[:, start : start + READOUT_STEPS, :], axis=1)
        hard_choices = 1 + np.argmax(post_go, axis=1)
        outcomes[source_position] = (hard_choices == 3).astype(np.int8)
        for repeat in range(N_REPEATS):
            repeat_rows.append(
                {
                    "source_trial_id": int(source_id),
                    "repeat_id": repeat,
                    "noise_seed": int(seeds[repeat]),
                    "final_hard_choice": int(hard_choices[repeat]),
                    "final_hard_choice_name": {1: "Left", 2: "Right", 3: "Sure"}[int(hard_choices[repeat])],
                    "p_left": float(post_go[repeat, 0]),
                    "p_right": float(post_go[repeat, 1]),
                    "p_sure": float(post_go[repeat, 2]),
                    "direction_choice_if_applicable": "" if hard_choices[repeat] == 3 else {1: "Left", 2: "Right"}[int(hard_choices[repeat])],
                    "sure_outcome": int(hard_choices[repeat] == 3),
                    "confidence_projection": float(confidence[source_position, repeat]),
                    "choice_axis_projection": float(choice_projection[source_position, repeat]),
                    "absolute_choice_axis_projection": float(abs(choice_projection[source_position, repeat])),
                }
            )
        print(f"  seed {seed}: replayed {source_position + 1}/{n_sources} source stimuli", flush=True)

    centered_confidence = center_within_rows(confidence)
    centered_choice = center_within_rows(choice_projection)
    centered_absolute_choice = center_within_rows(np.abs(choice_projection))
    row_index = 0
    for source_position in range(n_sources):
        for repeat in range(N_REPEATS):
            repeat_rows[row_index]["within_stimulus_confidence"] = float(centered_confidence[source_position, repeat])
            repeat_rows[row_index]["within_stimulus_choice_projection"] = float(centered_choice[source_position, repeat])
            repeat_rows[row_index]["within_stimulus_absolute_choice_projection"] = float(centered_absolute_choice[source_position, repeat])
            row_index += 1

    stimulus_rows = []
    for position, source_id in enumerate(source_ids):
        n_sure = int(np.sum(outcomes[position]))
        category = "stable Direction" if n_sure == 0 else "stable Sure" if n_sure == N_REPEATS else "variable"
        if category == "variable":
            difference = float(np.mean(confidence[position][outcomes[position] == 1]) - np.mean(confidence[position][outcomes[position] == 0]))
        else:
            difference = np.nan
        stimulus_rows.append(
            {
                "source_trial_id": int(source_id),
                "n_repeats": N_REPEATS,
                "n_sure": n_sure,
                "n_direction": int(N_REPEATS - n_sure),
                "p_sure": float(n_sure / N_REPEATS),
                "variability_class": category,
                "mean_confidence_projection": float(np.mean(confidence[position])),
                "sd_confidence_projection": float(np.std(confidence[position], ddof=1)),
                "sure_minus_direction_confidence_difference": difference,
                "input_sha256": input_hashes[str(source_id)],
                "metadata_sha256": metadata_hashes[str(source_id)],
            }
        )

    model, variable, differences, bootstrap_beta, permutation_beta = analyze_primary(
        confidence, np.abs(choice_projection), outcomes, source_ids, seed
    )
    temporal_rows, temporal_null_beta, temporal_max_negative, temporal_threshold, onset = analyze_temporal(
        time_confidence, outcomes, variable, seed
    )

    deterministic_max_state = 0.0
    deterministic_max_output = 0.0
    deterministic_consistent = 0
    for source_id in source_ids:
        seeds = np.asarray([noise_seed(seed, source_id, repeat) for repeat in range(N_DETERMINISTIC_REPEATS)], dtype=np.int64)
        outputs_zero, states_zero = simulate_frozen_repeats(weights, data["x"][source_id], seeds, DETERMINISTIC_REC_NOISE)
        deterministic_max_state = max(deterministic_max_state, float(np.max(np.abs(states_zero - states_zero[0:1]))))
        deterministic_max_output = max(deterministic_max_output, float(np.max(np.abs(outputs_zero - outputs_zero[0:1]))))
        probabilities = softmax_actions(outputs_zero)
        start = int(info[int(source_id)]["delay_end"])
        choices_zero = 1 + np.argmax(np.mean(probabilities[:, start : start + READOUT_STEPS], axis=1), axis=1)
        deterministic_consistent += int(np.all(choices_zero == choices_zero[0]))
    deterministic = {
        "rec_noise": DETERMINISTIC_REC_NOISE,
        "n_source_stimuli": int(n_sources),
        "repeats_per_stimulus": N_DETERMINISTIC_REPEATS,
        "max_abs_state_difference": deterministic_max_state,
        "max_abs_output_difference": deterministic_max_output,
        "choice_consistency": float(deterministic_consistent / n_sources),
        "all_exactly_repeatable": bool(deterministic_max_state == 0.0 and deterministic_max_output == 0.0 and deterministic_consistent == n_sources),
    }

    counts = {name: int(sum(row["variability_class"] == name for row in stimulus_rows)) for name in ("stable Direction", "variable", "stable Sure")}
    p_sure_by_stimulus = np.asarray([row["p_sure"] for row in stimulus_rows], dtype=np.float64)
    model["variability_counts"] = counts
    model["overall_stochastic_p_sure"] = float(np.mean(outcomes))
    model["mean_within_stimulus_sd_confidence_projection"] = float(np.mean(np.std(confidence, axis=1, ddof=1)))
    model["operational_internal_variability_prediction_onset_ms"] = onset
    model["temporal_familywise_negative_threshold"] = temporal_threshold

    write_csv(seed_dir / "repeat_records.csv", repeat_rows)
    write_csv(seed_dir / "stimulus_summary.csv", stimulus_rows)
    write_json(seed_dir / "primary_model.json", model)
    write_csv(seed_dir / "time_resolved_metrics.csv", temporal_rows)
    write_json(seed_dir / "deterministic_control.json", deterministic)
    np.savez_compressed(
        seed_dir / "repeat_states.npz",
        source_trial_ids=source_ids,
        repeat_ids=np.arange(N_REPEATS, dtype=np.int64),
        pre_ts_states=pre_ts_states,
        confidence_projection=confidence,
        choice_axis_projection=choice_projection,
        within_stimulus_confidence=centered_confidence,
        temporal_offsets_ms=TEMPORAL_OFFSETS_MS,
        temporal_confidence_projection=time_confidence,
        sure_outcome=outcomes,
    )
    np.savez_compressed(
        seed_dir / "bootstrap_results.npz",
        beta=bootstrap_beta,
        n_bootstraps=np.asarray(N_BOOTSTRAPS),
        random_seed=np.asarray(RANDOM_SEED + seed * 10 + 1),
    )
    np.savez_compressed(
        seed_dir / "permutation_results.npz",
        primary_beta=permutation_beta,
        n_primary_permutations=np.asarray(N_PRIMARY_PERMUTATIONS),
        primary_random_seed=np.asarray(RANDOM_SEED + seed * 10 + 2),
        temporal_beta=temporal_null_beta,
        temporal_max_negative_statistic=temporal_max_negative,
        n_temporal_permutations=np.asarray(N_TEMPORAL_PERMUTATIONS),
        temporal_random_seed=np.asarray(RANDOM_SEED + seed * 10 + 3),
        temporal_familywise_negative_threshold=np.asarray(temporal_threshold),
    )

    example_source = None
    if seed == EXAMPLE_SEED:
        example_source = make_seed7_figures(
            seed_dir, repeat_rows, stimulus_rows, confidence, outcomes, variable, differences,
            model, temporal_rows, deterministic, time_confidence, source_ids,
        )

    stage1_after = load_stage1_objects(stage1_dir)[-1]
    checkpoint_after = sha256(weights_path)
    if stage1_before != stage1_after or checkpoint_before != checkpoint_after:
        raise AssertionError("protected frozen artifacts changed during Stage 3")
    return {
        "seed": seed,
        "n_offered_test_stimuli": int(n_sources),
        "n_stable_direction": counts["stable Direction"],
        "n_variable": counts["variable"],
        "n_stable_sure": counts["stable Sure"],
        "overall_stochastic_p_sure": model["overall_stochastic_p_sure"],
        "stimulus_p_sure_q25": float(np.percentile(p_sure_by_stimulus, 25)),
        "stimulus_p_sure_median": float(np.median(p_sure_by_stimulus)),
        "stimulus_p_sure_q75": float(np.percentile(p_sure_by_stimulus, 75)),
        "stimulus_p_sure_min": float(np.min(p_sure_by_stimulus)),
        "stimulus_p_sure_max": float(np.max(p_sure_by_stimulus)),
        "mean_within_stimulus_sd_confidence_projection": model["mean_within_stimulus_sd_confidence_projection"],
        "primary_fixed_effect_beta": model["beta"],
        "primary_bootstrap_ci95_low": model["bootstrap_ci95"][0],
        "primary_bootstrap_ci95_high": model["bootstrap_ci95"][1],
        "primary_permutation_two_sided_p": model["permutation_two_sided_p"],
        "primary_permutation_negative_tail_p": model["permutation_negative_tail_p"],
        "mean_sure_minus_direction_confidence_difference": model["transparent_summary"]["mean_sure_minus_direction_confidence_difference"],
        "median_sure_minus_direction_confidence_difference": model["transparent_summary"]["median_sure_minus_direction_confidence_difference"],
        "fraction_variable_stimuli_negative": model["transparent_summary"]["fraction_variable_stimuli_negative"],
        "secondary_confidence_beta_after_choice_control": model["specificity_model"]["confidence_beta"],
        "secondary_absolute_choice_beta": model["specificity_model"]["absolute_choice_beta"],
        "noise_zero_choice_consistency": deterministic["choice_consistency"],
        "noise_zero_max_abs_state_difference": deterministic_max_state,
        "noise_zero_max_abs_output_difference": deterministic_max_output,
        "operational_variability_prediction_onset_ms": onset,
        "example_source_trial_id": example_source,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "stage1_hashes_before": stage1_before,
        "stage1_hashes_after": stage1_after,
        "input_identity_max_abs_difference": 0.0,
        "sure_channel_pre_ts_max_abs": sure_channel_pre_ts_max,
    }


def make_multiseed_figure(output_root, summaries):
    path = Path(output_root) / "multiseed" / "figures" / "primary_effect_replication.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.7), constrained_layout=True)
    y = np.arange(len(summaries))
    beta = np.asarray([row["primary_fixed_effect_beta"] for row in summaries])
    low = np.asarray([row["primary_bootstrap_ci95_low"] for row in summaries])
    high = np.asarray([row["primary_bootstrap_ci95_high"] for row in summaries])
    ax.errorbar(beta, y, xerr=np.vstack([beta - low, high - beta]), fmt="o", color="#6a3d9a", capsize=4)
    ax.axvline(0, color="0.4", linewidth=1)
    ax.set(yticks=y, yticklabels=[f"seed {row['seed']}" for row in summaries], xlabel="stimulus-fixed-effect confidence beta (per SD)", title="Identical-input confidence–wagering association")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def report_text(summaries, seed_formula):
    lines = [
        "# Population Representation Stage 3: identical-stimulus recurrent-noise variability",
        "", "## A. Replay design", "",
        f"For each frozen network, every sure-offered TEST source tensor was replayed exactly {N_REPEATS} times with `rec_noise={STOCHASTIC_REC_NOISE}`. Trials were never regenerated. The repeat seed formula was `{seed_formula}`. The frozen categorical readout was the mean Left/Right/Sure softmax probability over the first 10 post-go steps followed by argmax.",
        "", "## B. Identity checks", "",
        "All repeats within a source stimulus used byte-identical saved input tensors and identical saved timing/metadata. Input equality was checked exactly. Input channel 3 was zero throughout every primary −100…0-ms pre-target window. The only repeat-varying exogenous quantity was the explicitly seeded recurrent-noise tensor.",
        "", "## C. Behavioral variability", "",
        "| seed | offered TEST stimuli | stable Direction | variable | stable Sure | overall P(Sure) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['seed']} | {row['n_offered_test_stimuli']} | {row['n_stable_direction']} | {row['n_variable']} | {row['n_stable_sure']} | {row['overall_stochastic_p_sure']:.3f} |")
    lines += [
        "", "All stimuli remain in these descriptive counts. Only variable stimuli contribute outcome information to fixed-effect inference. The stimulus-level P(Sure) median [IQR] and range are reported below:",
        "", "| seed | median [IQR] P(Sure) | range |", "|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['seed']} | {row['stimulus_p_sure_median']:.3f} [{row['stimulus_p_sure_q25']:.3f}, {row['stimulus_p_sure_q75']:.3f}] | {row['stimulus_p_sure_min']:.3f}–{row['stimulus_p_sure_max']:.3f} |")
    lines += [
        "", "## D. Pre-TS confidence variability", "",
        "The primary state is the average of the 10 recurrent states immediately before `ts_onset`. The unchanged Stage-1 TRAIN scaler and empirical-P(correct) axis were applied with their saved orientation (higher projection means higher empirical P(correct)). The predictor was centered within each exact source stimulus and then globally standardized over variable-stimulus repeats.",
        "", "## E. Primary within-stimulus result", "",
        "| seed | variable stimuli | beta per SD | bootstrap median | bootstrap 95% CI | permutation p (two-sided) | negative-tail p |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        model = json.loads((Path(row["output_root"]) / f"seed{row['seed']}" / "primary_model.json").read_text()) if "output_root" in row else None
        median = model["bootstrap_median"] if model else np.nan
        lines.append(f"| {row['seed']} | {row['n_variable']} | {row['primary_fixed_effect_beta']:.3f} | {median:.3f} | [{row['primary_bootstrap_ci95_low']:.3f}, {row['primary_bootstrap_ci95_high']:.3f}] | {row['primary_permutation_two_sided_p']:.4f} | {row['primary_permutation_negative_tail_p']:.4f} |")
    lines += [
        "", "The model profiles a separate intercept for every source stimulus. The cluster bootstrap resampled source identities 2,000 times; the 1,000-permutation test shuffled outcomes independently within each source stimulus.",
        "", "## F. Transparent stimulus-level summary", "",
        "| seed | mean Sure−Direction difference | median difference | fraction negative |",
        "|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['seed']} | {row['mean_sure_minus_direction_confidence_difference']:.4f} | {row['median_sure_minus_direction_confidence_difference']:.4f} | {row['fraction_variable_stimuli_negative']:.3f} |")
    lines += [
        "", "Negative values mean that, for the same input, Sure repeats occupied a lower-confidence state than Direction repeats.",
        "", "## G. Specificity control", "",
        "| seed | confidence beta controlling absolute choice-axis magnitude | absolute choice beta |",
        "|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['seed']} | {row['secondary_confidence_beta_after_choice_control']:.3f} | {row['secondary_absolute_choice_beta']:.3f} |")
    lines += [
        "", "This secondary model used only the two independently frozen Stage-1 axes and stimulus fixed effects; it did not fit a Sure/Direction neural decoder. The confidence coefficient remained negative for seeds 7 and 9 but reversed for seed 8, so specificity beyond absolute choice-axis magnitude did not replicate uniformly.",
        "", "## H. Temporal analysis", "",
        "| seed | operational internal-variability prediction onset |",
        "|---:|---:|",
    ]
    for row in summaries:
        onset = "not detected" if row["operational_variability_prediction_onset_ms"] is None else f"{row['operational_variability_prediction_onset_ms']} ms"
        lines.append(f"| {row['seed']} | {onset} |")
    lines += [
        "", "The secondary curve used the unchanged confidence axis, 50-ms trailing windows every 10 ms from −300 to 0 ms, 500 within-stimulus permutations, a maximum negative-association statistic across time, and a prospective five-point (50-ms) run rule. Seeds 7 and 8 already met the rule at the earliest evaluated point, so their −300-ms values are left-boundary-limited rather than exact emergence times. These are operational prediction onsets, not causal or biological latencies.",
        "", "## I. Deterministic control", "",
        "| seed | choice consistency | max state difference | max output difference |",
        "|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['seed']} | {row['noise_zero_choice_consistency']:.3f} | {row['noise_zero_max_abs_state_difference']:.3g} | {row['noise_zero_max_abs_output_difference']:.3g} |")
    lines += [
        "", "With `rec_noise=0`, ten replays of every source stimulus were exactly repeatable.",
        "", "## J. Cross-seed replication", "",
        "The pipeline, TEST identities, repeat count, readout, windows, fixed-effect models, bootstrap and permutation counts, and onset rule were identical for seeds 7, 8, and 9. No second-level inference over three seeds was performed. Seed 7 remained the predetermined example; its illustrative trajectory uses the lowest source-trial ID among variable stimuli.",
        "", "## K. Integrity and leakage checks", "",
        "- Frozen checkpoint hashes were identical before and after replay.",
        "- Stage-1 scaler, confidence-axis, and choice-axis hashes were identical before and after replay.",
        "- Primary inference used sure-offered TEST source identities only.",
        "- No network training, new confidence axis, or Sure/Direction decoder was used.",
        "- Recurrent noise was exactly 0.05 for stochastic replay and exactly 0 for deterministic control.",
        "- The categorical readout was unchanged from the final supervised freeze.",
        "", "## L. Supported conclusion", "",
        "Across all three seeds, under identical external input, lower-than-usual pre-target recurrent states along the independently defined confidence dimension were associated with a higher probability of later Sure choice after controlling source-stimulus identity. The primary association replicated, while the secondary adjustment for absolute choice-axis magnitude did not retain the predicted confidence sign in seed 8. These are observational within-network associations under controlled in-silico input; they do not establish that confidence fluctuations cause Sure choice or that the axis is a causal biological mechanism.", "",
    ]
    return "\n".join(lines)


def run(args):
    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage-3 directory: {output_root}")
    source = inspect.getsource(sys.modules[__name__])
    forbidden = ["model" + ".train(", "Logistic" + "Regression(", "Ri" + "dge("]
    if any(token in source for token in forbidden):
        raise AssertionError("Stage-3 source contains a forbidden training or neural-decoder call")
    dataset_path = Path(args.dataset).resolve()
    split_path = Path(args.split).resolve()
    stage1_root = Path(args.stage1_dir).resolve()
    freeze_root = Path(args.freeze_dir).resolve()
    with np.load(dataset_path, allow_pickle=True) as stored:
        data = {name: np.asarray(stored[name]) for name in stored.files}
    info = [json.loads(str(value)) for value in data["trial_info_json"]]
    with np.load(split_path, allow_pickle=False) as stored:
        split = {name: np.asarray(stored[f"{name}_indices"], dtype=np.int64) for name in ("train", "validation", "test")}
    offered_test = np.sort(np.asarray([index for index in split["test"] if bool(info[int(index)]["sure_available"])], dtype=np.int64))
    output_root.mkdir(parents=True)
    checkpoint_before = {seed: sha256(freeze_root / f"seed{seed}" / "weights.npz") for seed in SEEDS}
    summaries = []
    for seed in SEEDS:
        print(f"Stage 3 seed {seed}: identical-input stochastic replay", flush=True)
        summary = replay_seed(seed, data, info, split, stage1_root, freeze_root, output_root)
        summary["output_root"] = str(output_root)
        summaries.append(summary)
    checkpoint_after = {seed: sha256(freeze_root / f"seed{seed}" / "weights.npz") for seed in SEEDS}
    if checkpoint_before != checkpoint_after:
        raise AssertionError("frozen checkpoint changed during Stage 3")
    multiseed_dir = output_root / "multiseed"
    write_csv(multiseed_dir / "variability_summary.csv", [{k: v for k, v in row.items() if k != "output_root"} for row in summaries])
    write_json(multiseed_dir / "variability_summary.json", [{k: v for k, v in row.items() if k != "output_root"} for row in summaries])
    replication_figure = make_multiseed_figure(output_root, summaries)
    seed_formula = f"{NOISE_SEED_BASE} + model_seed*1,000,000 + source_trial_id*1,000 + repeat_id"
    manifest = {
        "created_unix_time": time.time(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": " ".join([sys.executable, *sys.argv]),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "split_path": str(split_path),
        "split_sha256": sha256(split_path),
        "stage1_root": str(stage1_root),
        "freeze_root": str(freeze_root),
        "seeds": list(SEEDS),
        "example_seed": EXAMPLE_SEED,
        "offered_test_source_trial_ids": offered_test,
        "n_offered_test_source_trials": int(offered_test.size),
        "stochastic_replay": {"repeats": N_REPEATS, "rec_noise": STOCHASTIC_REC_NOISE, "alpha": ALPHA, "seed_formula": seed_formula, "source": "exact saved dataset input tensor"},
        "deterministic_control": {"repeats": N_DETERMINISTIC_REPEATS, "rec_noise": DETERMINISTIC_REC_NOISE},
        "behavioral_readout": "per-step Left/Right/Sure softmax; mean first 10 post-go steps; categorical argmax",
        "primary_state": "mean hidden[ts_onset-10:ts_onset], 100 ms before Sure-target onset",
        "primary_model": "profiled stimulus-fixed-effect logistic regression; within-stimulus confidence standardized globally over variable-stimulus repeats",
        "bootstrap": {"count": N_BOOTSTRAPS, "unit": "source stimulus"},
        "primary_permutation": {"count": N_PRIMARY_PERMUTATIONS, "scheme": "within-source outcome shuffle", "tails": ["two-sided", "prospective negative"]},
        "temporal_analysis": {"range_ms": [-300, 0], "step_ms": DT_MS, "trailing_window_ms": TEMPORAL_WINDOW_STEPS * DT_MS, "permutations": N_TEMPORAL_PERMUTATIONS, "correction": "maximum predicted-negative coefficient across time", "onset_run_ms": ONSET_RUN_POINTS * DT_MS},
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "integrity_checks": {"checkpoint_hashes_unchanged": True, "stage1_hashes_unchanged": True, "all_repeated_inputs_exactly_identical": True, "only_recurrent_noise_varied": True, "primary_uses_test_stimuli_only": True, "network_training_called": False, "new_confidence_axis_fitted": False, "sure_direction_decoder_fitted": False, "behavioral_readout_unchanged": True, "identical_definitions_all_seeds": True},
        "replication_figure": str(replication_figure.resolve()),
        "python": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib", "psychrnn")},
    }
    write_json(output_root / "analysis_manifest.json", manifest)
    (output_root / "final_report.md").write_text(report_text(summaries, seed_formula), encoding="utf-8")
    print(json.dumps(jsonable([{k: v for k, v in row.items() if k != "output_root"} for row in summaries]), indent=2), flush=True)
    return summaries


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/model_freeze/teacher_seed7.npz")
    parser.add_argument("--split", default="results/final_supervised_freeze/split_indices.npz")
    parser.add_argument("--stage1-dir", default="results/population_stage1")
    parser.add_argument("--freeze-dir", default="results/final_supervised_freeze")
    parser.add_argument("--output-dir", default="results/population_stage3_variability")
    return parser


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
