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
                "expected_direction_value_mean": _mean(
                    dataset["teacher_expected_direction_value"][in_coh]
                ),
                "sure_target_mean_offered": _mean(dataset["teacher_sure_strength"][offered]),
                "external_sensory_margin_mean": _mean(dataset["external_sensory_margin"][in_coh]),
            }
        )
    return rows


def summarize_teacher_calibration(dataset, bins):
    n_trials = int(dataset["n_trials"])
    n_offered = int(np.sum(dataset["sure_available"]))
    return {
        "dataset": dataset["dataset_path"],
        "n_trials": n_trials,
        "n_sure_offered": n_offered,
        "sure_offered_rate": float(n_offered / n_trials) if n_trials else None,
        "calibration_target": "teacher_direction_success_proxy",
        "empirical_teacher_accuracy_available": False,
        "interpretation_boundary": (
            "This report validates the saved internal-confidence proxy mapping; "
            "the exported dataset does not contain empirical teacher trial correctness."
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
    expected_value = np.asarray(
        [row["expected_direction_value_mean"] for row in bins], dtype=float
    )
    sure_target = np.asarray([row["sure_target_mean"] for row in bins], dtype=float)

    coh_rows = summary.get("coherence_summary", [])
    coh = np.asarray([row["coh"] for row in coh_rows], dtype=float)
    coh_p = np.asarray([row["p_correct_proxy_mean"] for row in coh_rows], dtype=float)
    coh_sure = np.asarray([row["sure_target_mean_offered"] for row in coh_rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))
    fig.suptitle("Teacher internal-confidence proxy calibration", fontsize=14, fontweight="bold")

    axes[0].plot(margin, p_correct, marker="o", lw=2.3, color="#1F77B4")
    axes[0].axvline(
        summary.get("internal_margin_center_used"),
        color="#666666",
        ls="--",
        lw=1.1,
        label="auto center",
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
        "Boundary: exported datasets contain proxy P(correct), not empirical teacher trial accuracy.",
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
    return {
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
