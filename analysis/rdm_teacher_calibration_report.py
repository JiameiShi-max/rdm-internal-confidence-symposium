import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET = "stage9b_timed_teacher_dataset_50k_fixed_auto.npz"


BIN_FIELD_ORDER = [
    "bin_index",
    "n_trials",
    "internal_margin_min",
    "internal_margin_max",
    "internal_margin_mean",
    "p_correct_proxy_mean",
    "empirical_accuracy",
    "expected_direction_value_mean",
    "sure_target_mean",
    "external_sensory_margin_mean",
    "external_sure_strength_mean",
    "coherence_mean",
    "sure_available_rate",
]


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 2:
        return None
    x = x[ok]
    y = y[ok]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def _parse_trial_info(raw_info):
    parsed = []
    for item in raw_info:
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        parsed.append(json.loads(str(item)))
    return parsed


def _optional_array(npz, key, n_trials, default=np.nan):
    if key in npz.files:
        return np.asarray(npz[key], dtype=float)
    return np.full(n_trials, default, dtype=float)


def _scalar(npz, key, default=None):
    if key not in npz.files:
        return default
    value = npz[key]
    if np.ndim(value) == 0:
        return value.item()
    return value


def load_teacher_dataset(dataset_path):
    npz = np.load(dataset_path, allow_pickle=True)
    trial_info = _parse_trial_info(npz["trial_info_json"])
    n_trials = len(trial_info)
    sure_available = np.asarray(
        [bool(info.get("sure_available", False)) for info in trial_info],
        dtype=bool,
    )
    coherence = np.asarray(
        [_safe_float(info.get("coh")) if _safe_float(info.get("coh")) is not None else np.nan for info in trial_info],
        dtype=float,
    )
    stimulus_dur = np.asarray(
        [
            _safe_float(info.get("stimulus_dur"))
            if _safe_float(info.get("stimulus_dur")) is not None
            else np.nan
            for info in trial_info
        ],
        dtype=float,
    )
    return {
        "dataset_path": os.path.abspath(dataset_path),
        "n_trials": int(n_trials),
        "trial_info": trial_info,
        "sure_available": sure_available,
        "coherence": coherence,
        "stimulus_dur": stimulus_dur,
        "teacher_internal_margin": np.asarray(npz["teacher_internal_margin"], dtype=float),
        "teacher_direction_success_proxy": np.asarray(
            npz["teacher_direction_success_proxy"], dtype=float
        ),
        "teacher_correct": (
            np.asarray(npz["teacher_correct"], dtype=bool)
            if "teacher_correct" in npz.files
            else None
        ),
        "teacher_direction_choice": (
            np.asarray(npz["teacher_direction_choice"], dtype=int)
            if "teacher_direction_choice" in npz.files
            else None
        ),
        "teacher_expected_direction_value": np.asarray(
            npz["teacher_expected_direction_value"], dtype=float
        ),
        "teacher_sure_strength": np.asarray(npz["teacher_sure_strength"], dtype=float),
        "external_sensory_margin": _optional_array(npz, "external_sensory_margin", n_trials),
        "external_sure_strength": _optional_array(npz, "external_sure_strength", n_trials),
        "internal_margin_center_used": _safe_float(
            _scalar(npz, "internal_margin_center_used", default=np.nan)
        ),
        "internal_margin_temp_used": _safe_float(
            _scalar(npz, "internal_margin_temp_used", default=np.nan)
        ),
        "internal_margin_calibration_mode": str(
            _scalar(npz, "internal_margin_calibration_mode", default="")
        ),
        "internal_margin_n_calibration_trials": int(
            _scalar(npz, "internal_margin_n_calibration_trials", default=0)
        ),
    }


def _quantile_bin_edges(values, n_bins):
    quantiles = np.linspace(0.0, 100.0, int(n_bins) + 1)
    edges = np.percentile(values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def build_calibration_bins(
    internal_margin,
    p_correct_proxy,
    expected_direction_value,
    sure_strength,
    coherence,
    sure_available,
    n_bins=10,
    external_sensory_margin=None,
    external_sure_strength=None,
    teacher_correct=None,
):
    internal_margin = np.asarray(internal_margin, dtype=float)
    p_correct_proxy = np.asarray(p_correct_proxy, dtype=float)
    expected_direction_value = np.asarray(expected_direction_value, dtype=float)
    sure_strength = np.asarray(sure_strength, dtype=float)
    coherence = np.asarray(coherence, dtype=float)
    sure_available = np.asarray(sure_available, dtype=bool)
    if external_sensory_margin is None:
        external_sensory_margin = np.full(internal_margin.shape, np.nan, dtype=float)
    if external_sure_strength is None:
        external_sure_strength = np.full(internal_margin.shape, np.nan, dtype=float)
    external_sensory_margin = np.asarray(external_sensory_margin, dtype=float)
    external_sure_strength = np.asarray(external_sure_strength, dtype=float)
    if teacher_correct is not None:
        teacher_correct = np.asarray(teacher_correct, dtype=float)

    ok = np.isfinite(internal_margin)
    valid_margin = internal_margin[ok]
    if valid_margin.size == 0:
        return []

    edges = _quantile_bin_edges(valid_margin, n_bins)
    rows = []
    for bin_index in range(int(n_bins)):
        in_bin = ok & (internal_margin >= edges[bin_index]) & (internal_margin < edges[bin_index + 1])
        if not np.any(in_bin):
            continue
        rows.append(
            {
                "bin_index": int(bin_index),
                "n_trials": int(np.sum(in_bin)),
                "internal_margin_min": float(np.min(internal_margin[in_bin])),
                "internal_margin_max": float(np.max(internal_margin[in_bin])),
                "internal_margin_mean": _mean(internal_margin[in_bin]),
                "p_correct_proxy_mean": _mean(p_correct_proxy[in_bin]),
                "empirical_accuracy": (
                    _mean(teacher_correct[in_bin]) if teacher_correct is not None else None
                ),
                "expected_direction_value_mean": _mean(expected_direction_value[in_bin]),
                "sure_target_mean": _mean(sure_strength[in_bin & sure_available]),
                "external_sensory_margin_mean": _mean(external_sensory_margin[in_bin]),
                "external_sure_strength_mean": _mean(external_sure_strength[in_bin & sure_available]),
                "coherence_mean": _mean(coherence[in_bin]),
                "sure_available_rate": float(np.mean(sure_available[in_bin])),
            }
        )
    return rows


def summarize_by_coherence(dataset):
    rows = []
    coherence = dataset["coherence"]
    for coh in sorted(set(float(v) for v in coherence[np.isfinite(coherence)])):
        in_coh = coherence == coh
        offered = in_coh & dataset["sure_available"]
        rows.append(
            {
                "coh": float(coh),
                "n_trials": int(np.sum(in_coh)),
                "n_sure_offered": int(np.sum(offered)),
                "internal_margin_mean": _mean(dataset["teacher_internal_margin"][in_coh]),
                "p_correct_proxy_mean": _mean(dataset["teacher_direction_success_proxy"][in_coh]),
                "empirical_accuracy": (
                    _mean(dataset["teacher_correct"][in_coh])
                    if dataset["teacher_correct"] is not None
                    else None
                ),
                "expected_direction_value_mean": _mean(
                    dataset["teacher_expected_direction_value"][in_coh]
                ),
                "sure_target_mean_offered": _mean(dataset["teacher_sure_strength"][offered]),
                "external_sensory_margin_mean": _mean(dataset["external_sensory_margin"][in_coh]),
            }
        )
    return rows


def _binary_metrics(y, probability):
    y = np.asarray(y, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    ok = np.isfinite(y) & np.isfinite(probability)
    y = y[ok]
    probability = probability[ok]
    if y.size == 0:
        return {"n": 0, "brier_score": None, "log_loss": None, "auc": None}
    auc = None
    positive = probability[y == 1]
    negative = probability[y == 0]
    if positive.size and negative.size:
        auc = float(np.mean(positive[:, None] > negative[None, :]) + 0.5 * np.mean(positive[:, None] == negative[None, :]))
    return {
        "n": int(y.size),
        "brier_score": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))),
        "auc": auc,
    }


def _fit_logistic(x, y, max_iter=100):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(x.shape[0]), x])
    coef = np.zeros(design.shape[1], dtype=float)
    for _ in range(int(max_iter)):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ coef, -30, 30)))
        w = np.maximum(p * (1.0 - p), 1e-7)
        hessian = design.T @ (design * w[:, None]) + 1e-7 * np.eye(design.shape[1])
        next_coef = coef + np.linalg.solve(hessian, design.T @ (y - p))
        if np.max(np.abs(next_coef - coef)) < 1e-9:
            coef = next_coef
            break
        coef = next_coef
    p = 1.0 / (1.0 + np.exp(-np.clip(design @ coef, -30, 30)))
    w = np.maximum(p * (1.0 - p), 1e-7)
    covariance = np.linalg.pinv(design.T @ (design * w[:, None]) + 1e-7 * np.eye(design.shape[1]))
    return coef, np.sqrt(np.maximum(np.diag(covariance), 0.0))


def _predict_logistic(coef, x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.ones(x.shape[0]), x])
    return 1.0 / (1.0 + np.exp(-np.clip(design @ np.asarray(coef), -30, 30)))


def compare_confidence_models(margin, duration_ms, teacher_correct, n_folds=5, seed=7):
    margin = np.asarray(margin, dtype=float)
    duration_s = np.asarray(duration_ms, dtype=float) / 1000.0
    y = np.asarray(teacher_correct, dtype=float)
    ok = np.isfinite(margin) & np.isfinite(duration_s) & np.isfinite(y)
    margin, duration_s, y = margin[ok], duration_s[ok], y[ok]
    models = {
        "margin": np.column_stack([margin]),
        "margin_duration": np.column_stack([margin, duration_s]),
        "margin_duration_interaction": np.column_stack([margin, duration_s, margin * duration_s]),
    }
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(y.size)
    folds = np.array_split(order, min(int(n_folds), max(2, y.size)))
    results = {}
    for name, features in models.items():
        predictions = np.full(y.size, np.nan, dtype=float)
        for test_idx in folds:
            train_mask = np.ones(y.size, dtype=bool)
            train_mask[test_idx] = False
            if len(np.unique(y[train_mask])) < 2:
                predictions[test_idx] = np.mean(y[train_mask])
            else:
                coef, _ = _fit_logistic(features[train_mask], y[train_mask])
                predictions[test_idx] = _predict_logistic(coef, features[test_idx])
        coef, se = _fit_logistic(features, y)
        results[name] = {
            **_binary_metrics(y, predictions),
            "coefficients": [float(v) for v in coef],
            "coefficient_se": [float(v) for v in se],
        }
    baseline = results["margin"]
    duration = results["margin_duration"]
    improvement = float(baseline["log_loss"] - duration["log_loss"])
    results["decision"] = {
        "duration_log_loss_improvement": improvement,
        "duration_adds_meaningful_information": bool(improvement >= 0.002),
        "recommended_mapping": (
            "empirical_margin_duration" if improvement >= 0.002 else "legacy_margin"
        ),
        "threshold": 0.002,
    }
    return results


def empirical_accuracy_tables(margin, duration_ms, teacher_correct, n_bins=5):
    margin = np.asarray(margin, dtype=float)
    duration_ms = np.asarray(duration_ms, dtype=float)
    correct = np.asarray(teacher_correct, dtype=float)
    ok = np.isfinite(margin) & np.isfinite(duration_ms) & np.isfinite(correct)
    margin, duration_ms, correct = margin[ok], duration_ms[ok], correct[ok]
    if correct.size == 0:
        return [], []
    duration_edges = np.unique(np.quantile(duration_ms, np.linspace(0, 1, int(n_bins) + 1)))
    margin_edges = np.unique(np.quantile(margin, np.linspace(0, 1, int(n_bins) + 1)))
    duration_rows = []
    joint_rows = []
    for d_idx in range(max(0, duration_edges.size - 1)):
        d_select = (duration_ms >= duration_edges[d_idx]) & (
            (duration_ms <= duration_edges[d_idx + 1])
            if d_idx == duration_edges.size - 2
            else (duration_ms < duration_edges[d_idx + 1])
        )
        duration_rows.append({
            "duration_bin": int(d_idx),
            "duration_low_ms": float(duration_edges[d_idx]),
            "duration_high_ms": float(duration_edges[d_idx + 1]),
            "duration_mean_ms": _mean(duration_ms[d_select]),
            "n_trials": int(np.sum(d_select)),
            "empirical_accuracy": _mean(correct[d_select]),
        })
        for m_idx in range(max(0, margin_edges.size - 1)):
            m_select = (margin >= margin_edges[m_idx]) & (
                (margin <= margin_edges[m_idx + 1])
                if m_idx == margin_edges.size - 2
                else (margin < margin_edges[m_idx + 1])
            )
            select = d_select & m_select
            joint_rows.append({
                "duration_bin": int(d_idx),
                "margin_bin": int(m_idx),
                "duration_mean_ms": _mean(duration_ms[select]),
                "margin_mean": _mean(margin[select]),
                "n_trials": int(np.sum(select)),
                "empirical_accuracy": _mean(correct[select]),
            })
    return duration_rows, joint_rows


def summarize_teacher_calibration(dataset, bins):
    n_trials = int(dataset["n_trials"])
    n_offered = int(np.sum(dataset["sure_available"]))
    empirical_available = dataset.get("teacher_correct") is not None
    summary = {
        "dataset": dataset["dataset_path"],
        "n_trials": n_trials,
        "n_sure_offered": n_offered,
        "sure_offered_rate": float(n_offered / n_trials) if n_trials else None,
        "calibration_target": (
            "empirical_teacher_direction_correctness"
            if empirical_available
            else "teacher_direction_success_proxy"
        ),
        "empirical_teacher_accuracy_available": bool(empirical_available),
        "interpretation_boundary": (
            "Calibration uses actual correctness of the teacher's left/right preference."
            if empirical_available
            else "Legacy artifact is incompatible with empirical calibration because teacher_correct is absent."
        ),
        "internal_margin_center_used": dataset["internal_margin_center_used"],
        "internal_margin_temp_used": dataset["internal_margin_temp_used"],
        "internal_margin_calibration_mode": dataset["internal_margin_calibration_mode"],
        "internal_margin_n_calibration_trials": dataset[
            "internal_margin_n_calibration_trials"
        ],
        "n_bins": int(len(bins)),
        "corr_internal_margin_p_correct_proxy": _safe_corr(
            dataset["teacher_internal_margin"], dataset["teacher_direction_success_proxy"]
        ),
        "corr_internal_margin_expected_value": _safe_corr(
            dataset["teacher_internal_margin"], dataset["teacher_expected_direction_value"]
        ),
        "corr_internal_margin_sure_target_offered": _safe_corr(
            dataset["teacher_internal_margin"][dataset["sure_available"]],
            dataset["teacher_sure_strength"][dataset["sure_available"]],
        ),
        "corr_external_margin_internal_margin": _safe_corr(
            dataset["external_sensory_margin"], dataset["teacher_internal_margin"]
        ),
        "mean_teacher_internal_margin": _mean(dataset["teacher_internal_margin"]),
        "mean_p_correct_proxy": _mean(dataset["teacher_direction_success_proxy"]),
        "mean_sure_target_offered": _mean(
            dataset["teacher_sure_strength"][dataset["sure_available"]]
        ),
        "coherence_summary": summarize_by_coherence(dataset),
    }
    if empirical_available:
        y = np.asarray(dataset["teacher_correct"], dtype=float)
        proxy = np.asarray(dataset["teacher_direction_success_proxy"], dtype=float)
        summary["proxy_metrics"] = _binary_metrics(y, proxy)
        proxy_logit = np.log(np.clip(proxy, 1e-6, 1 - 1e-6) / np.clip(1 - proxy, 1e-6, 1))
        calibration_coef, calibration_se = _fit_logistic(proxy_logit, y)
        summary["calibration_intercept"] = float(calibration_coef[0])
        summary["calibration_slope"] = float(calibration_coef[1])
        summary["calibration_intercept_se"] = float(calibration_se[0])
        summary["calibration_slope_se"] = float(calibration_se[1])
        summary["confidence_model_comparison"] = compare_confidence_models(
            dataset["teacher_internal_margin"],
            dataset["stimulus_dur"],
            y,
        )
        duration_rows, joint_rows = empirical_accuracy_tables(
            dataset["teacher_internal_margin"],
            dataset["stimulus_dur"],
            y,
        )
        summary["duration_accuracy"] = duration_rows
        summary["margin_x_duration_accuracy"] = joint_rows
    return summary


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean(val) for val in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(payload), f, indent=2)
    return path


def _write_csv(path, rows, field_order):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [key for key in field_order if any(key in row for row in rows)]
    fieldnames += sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_clean(rows))
    return path


def plot_teacher_calibration(path, summary, bins):
    margin = np.asarray([row["internal_margin_mean"] for row in bins], dtype=float)
    p_correct = np.asarray([row["p_correct_proxy_mean"] for row in bins], dtype=float)
    empirical = np.asarray([
        np.nan if row.get("empirical_accuracy") is None else row["empirical_accuracy"]
        for row in bins
    ], dtype=float)
    expected_value = np.asarray(
        [row["expected_direction_value_mean"] for row in bins], dtype=float
    )
    sure_target = np.asarray([row["sure_target_mean"] for row in bins], dtype=float)

    coh_rows = summary.get("coherence_summary", [])
    coh = np.asarray([row["coh"] for row in coh_rows], dtype=float)
    coh_p = np.asarray([row["p_correct_proxy_mean"] for row in coh_rows], dtype=float)
    coh_sure = np.asarray([row["sure_target_mean_offered"] for row in coh_rows], dtype=float)

    empirical_available = bool(summary.get("empirical_teacher_accuracy_available"))
    if empirical_available:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    fig.suptitle("Teacher internal-confidence proxy calibration", fontsize=14, fontweight="bold")

    if empirical_available:
        axes[0].plot(p_correct, empirical, marker="o", lw=2.3, color="#1F77B4")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="#666666")
        axes[0].set_xlabel("predicted P(correct)")
        axes[0].set_ylabel("empirical accuracy")
        axes[0].set_title("Reliability")
        axes[1].plot(margin, empirical, marker="s", lw=2.0, color="#E45756")
        axes[1].set_xlabel("teacher internal margin")
        axes[1].set_ylabel("empirical accuracy")
        axes[1].set_title("Accuracy vs margin")
        duration_rows = summary.get("duration_accuracy", [])
        axes[2].plot(
            [row["duration_mean_ms"] for row in duration_rows],
            [row["empirical_accuracy"] for row in duration_rows],
            marker="o",
            color="#54A24B",
        )
        axes[2].set_xlabel("stimulus duration (ms)")
        axes[2].set_ylabel("empirical accuracy")
        axes[2].set_title("Accuracy vs duration")
        joint = summary.get("margin_x_duration_accuracy", [])
        n_d = max([row["duration_bin"] for row in joint], default=-1) + 1
        n_m = max([row["margin_bin"] for row in joint], default=-1) + 1
        heat = np.full((n_d, n_m), np.nan)
        for row in joint:
            heat[row["duration_bin"], row["margin_bin"]] = row["empirical_accuracy"]
        image = axes[3].imshow(heat, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axes[3].set_xlabel("margin quantile bin")
        axes[3].set_ylabel("duration quantile bin")
        axes[3].set_title("Accuracy: margin x duration")
        fig.colorbar(image, ax=axes[3], label="accuracy")
        for ax in axes[:3]:
            ax.set_ylim(0, 1.03)
            ax.grid(True, alpha=0.25)
    else:
        axes[0].plot(margin, p_correct, marker="o", lw=2.3, color="#1F77B4")
        axes[0].axvline(
            summary.get("internal_margin_center_used"),
            color="#666666",
            ls="--",
            lw=1.1,
            label="mapping center",
        )
        axes[0].set_xlabel("Teacher internal output margin")
        axes[0].set_ylabel("P(correct) proxy")
        axes[0].set_ylim(0.0, 1.03)
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(margin, expected_value, marker="s", lw=2.1, color="#4C78A8", label="Expected direction value")
        axes[1].plot(margin, sure_target, marker="^", lw=2.1, color="#E45756", label="Sure target")
        axes[1].set_xlabel("Teacher internal output margin")
        axes[1].set_ylabel("Target value")
        axes[1].set_ylim(0.0, 1.03)
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(coh, coh_p, marker="o", lw=2.1, color="#1F77B4", label="P(correct) proxy")
        axes[2].plot(coh, coh_sure, marker="^", lw=2.1, color="#E45756", label="Sure target | offered")
        axes[2].set_xlabel("Motion coherence")
        axes[2].set_ylabel("Mean proxy / target")
        axes[2].set_ylim(0.0, 1.03)
        axes[2].set_xscale("symlog", linthresh=0.032)
        axes[2].legend(frameon=False, fontsize=8)
        axes[2].grid(True, alpha=0.25)

    fig.text(
        0.02,
        0.01,
        summary.get("interpretation_boundary", ""),
        fontsize=8,
        color="#555555",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_teacher_calibration_outputs(output_dir, figure_dir, summary, bins):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    outputs = {
        "summary_json": _write_json(
            os.path.join(output_dir, "teacher_calibration_summary.json"), summary
        ),
        "bins_csv": _write_csv(
            os.path.join(output_dir, "teacher_calibration_bins.csv"), bins, BIN_FIELD_ORDER
        ),
        "coherence_csv": _write_csv(
            os.path.join(output_dir, "teacher_calibration_by_coherence.csv"),
            summary.get("coherence_summary", []),
            [
                "coh",
                "n_trials",
                "n_sure_offered",
                "internal_margin_mean",
                "p_correct_proxy_mean",
                "expected_direction_value_mean",
                "sure_target_mean_offered",
                "external_sensory_margin_mean",
            ],
        ),
        "figure": plot_teacher_calibration(
            os.path.join(figure_dir, "fig5_teacher_calibration.png"), summary, bins
        ),
    }
    if summary.get("empirical_teacher_accuracy_available"):
        outputs["duration_csv"] = _write_csv(
            os.path.join(output_dir, "teacher_accuracy_by_duration.csv"),
            summary.get("duration_accuracy", []),
            ["duration_bin", "duration_low_ms", "duration_high_ms", "duration_mean_ms", "n_trials", "empirical_accuracy"],
        )
        outputs["margin_duration_csv"] = _write_csv(
            os.path.join(output_dir, "teacher_accuracy_by_margin_duration.csv"),
            summary.get("margin_x_duration_accuracy", []),
            ["duration_bin", "margin_bin", "duration_mean_ms", "margin_mean", "n_trials", "empirical_accuracy"],
        )
    return outputs


def build_report(dataset_path, n_bins=10):
    dataset = load_teacher_dataset(dataset_path)
    bins = build_calibration_bins(
        dataset["teacher_internal_margin"],
        dataset["teacher_direction_success_proxy"],
        dataset["teacher_expected_direction_value"],
        dataset["teacher_sure_strength"],
        dataset["coherence"],
        dataset["sure_available"],
        n_bins=n_bins,
        external_sensory_margin=dataset["external_sensory_margin"],
        external_sure_strength=dataset["external_sure_strength"],
        teacher_correct=dataset["teacher_correct"],
    )
    summary = summarize_teacher_calibration(dataset, bins)
    return summary, bins


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build teacher internal-confidence proxy calibration report."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output-dir", default="paper_exports")
    parser.add_argument("--figure-dir", default="paper_figures")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    summary, bins = build_report(args.dataset, n_bins=args.n_bins)
    paths = write_teacher_calibration_outputs(args.output_dir, args.figure_dir, summary, bins)
    print(json.dumps({"outputs": paths, "summary": summary}, indent=2), flush=True)
    return paths


if __name__ == "__main__":
    main()
