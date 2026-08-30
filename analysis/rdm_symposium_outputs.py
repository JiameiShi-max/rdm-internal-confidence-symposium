import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


DEFAULT_ORIGINAL_SUMMARY = "stage9_internal_readout_50k_auto_summary.json"
DEFAULT_TIMED_SUMMARY = "stage9b_timed_readout_50k_fixed_auto_summary.json"
DEFAULT_TIMED_MULTISEED = "multiseed_validation_timed_5seed/multiseed_validation_summary.json"


DEFAULT_PARAMETER_SPECS = [
    (
        "student_75k",
        "param_validation_timed_hpc/timed_student_75k/timed_student_75k_seed7_summary.json",
        "parameter",
    ),
    (
        "reward_057",
        "param_validation_timed_hpc/timed_reward_057/timed_reward_057_seed7_summary.json",
        "parameter",
    ),
    (
        "temp_012",
        "param_validation_timed_hpc/timed_contrast_temp012/timed_contrast_temp012_seed7_summary.json",
        "parameter",
    ),
    (
        "blend_090",
        "param_validation_timed_hpc/timed_contrast_blend090/timed_contrast_blend090_seed7_summary.json",
        "parameter",
    ),
]


METRIC_FIELD_ORDER = [
    "model",
    "model_group",
    "n_seeds",
    "pass",
    "n_trials",
    "n_sure_offered",
    "n_sure_choice",
    "sure_offered_rate",
    "sure_choice_rate_offered",
    "residual_corr",
    "residual_r2",
    "residual_r2_std",
    "evidence_r2_p_sure",
    "evidence_r2_p_sure_std",
    "evidence_r2_confidence",
    "sure_axis_r2_p_sure",
    "pre_ts_corr",
    "pre_ts_corr_max",
    "post_ts_corr",
    "deterministic_stability",
    "population_peak_abs_diff",
    "dataset",
    "summary_path",
]


COHERENCE_FIELD_ORDER = [
    "model",
    "model_group",
    "coh",
    "n_trials",
    "n_sure_offered",
    "p_sure",
    "no_sure_accuracy",
    "waived_sure_accuracy",
    "n_waived_sure",
]


def _load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sibling_json(summary_path, suffix):
    root, _ = os.path.splitext(summary_path)
    return _load_json(f"{root}_{suffix}.json")


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _ratio(num, den):
    num = _finite_float(num)
    den = _finite_float(den)
    if num is None or den is None or den == 0.0:
        return None
    return num / den


def _metric_stat(aggregate, metric, stat):
    return _finite_float(aggregate.get("metrics", {}).get(metric, {}).get(stat))


def _clean_for_json(value):
    if isinstance(value, dict):
        return {key: _clean_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(val) for val in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_model(model, summary_path, model_group):
    summary = _load_json(summary_path)
    pre_ts = _sibling_json(summary_path, "pre_ts_leakage_summary")
    deterministic = _sibling_json(summary_path, "deterministic_repeated_stimulus_summary")
    population = _sibling_json(summary_path, "population_dynamics_summary")
    n_trials = summary.get("n_trials")
    n_offered = summary.get("n_sure_offered")
    n_sure = summary.get("n_sure_choice")
    return {
        "model": model,
        "model_group": model_group,
        "n_trials": n_trials,
        "n_sure_offered": n_offered,
        "n_sure_choice": n_sure,
        "sure_offered_rate": _ratio(n_offered, n_trials),
        "sure_choice_rate_offered": _ratio(n_sure, n_offered),
        "residual_corr": _finite_float(
            summary.get("corr_sure_axis_with_sure_residual_after_evidence")
        ),
        "residual_r2": _finite_float(
            summary.get("r2_sure_axis_explains_residual_p_sure_after_evidence")
        ),
        "evidence_r2_p_sure": _finite_float(summary.get("r2_evidence_axis_explains_p_sure")),
        "evidence_r2_confidence": _finite_float(
            summary.get("r2_evidence_axis_explains_confidence")
        ),
        "sure_axis_r2_p_sure": _finite_float(summary.get("r2_sure_axis_explains_p_sure")),
        "pre_ts_corr": _finite_float(pre_ts.get("corr_pre_ts_sure_axis_with_sure_choice")),
        "post_ts_corr": _finite_float(post_ts.get("corr_post_ts_sure_axis_with_sure_choice"))
        if (post_ts := pre_ts)
        else None,
        "deterministic_stability": _finite_float(
            deterministic.get("all_repeats_same_choice_fraction")
        ),
        "population_peak_abs_diff": _finite_float(population.get("sure_axis_peak_abs_difference")),
        "dataset": os.path.basename(str(summary.get("dataset", ""))),
        "summary_path": os.path.abspath(summary_path),
    }


def summarize_multiseed(aggregate_path, model="timed_5seed"):
    aggregate = _load_json(aggregate_path)
    n_offered = _metric_stat(aggregate, "n_sure_offered", "mean")
    n_sure = _metric_stat(aggregate, "n_sure_choice", "mean")
    return {
        "model": model,
        "model_group": "multiseed",
        "n_seeds": aggregate.get("n_seeds"),
        "pass": aggregate.get("pass"),
        "n_sure_offered": n_offered,
        "n_sure_choice": n_sure,
        "sure_choice_rate_offered": _ratio(n_sure, n_offered),
        "residual_corr": _metric_stat(
            aggregate, "corr_sure_axis_with_sure_residual_after_evidence", "mean"
        ),
        "residual_r2": _metric_stat(
            aggregate, "r2_sure_axis_explains_residual_p_sure_after_evidence", "mean"
        ),
        "residual_r2_std": _metric_stat(
            aggregate, "r2_sure_axis_explains_residual_p_sure_after_evidence", "std"
        ),
        "evidence_r2_p_sure": _metric_stat(
            aggregate, "r2_evidence_axis_explains_p_sure", "mean"
        ),
        "evidence_r2_p_sure_std": _metric_stat(
            aggregate, "r2_evidence_axis_explains_p_sure", "std"
        ),
        "pre_ts_corr": _metric_stat(
            aggregate, "pre_ts_leakage.corr_pre_ts_sure_axis_with_sure_choice", "mean"
        ),
        "pre_ts_corr_max": _metric_stat(
            aggregate, "pre_ts_leakage.corr_pre_ts_sure_axis_with_sure_choice", "max"
        ),
        "deterministic_stability": _metric_stat(
            aggregate, "deterministic_repeated.all_repeats_same_choice_fraction", "min"
        ),
        "population_peak_abs_diff": _metric_stat(
            aggregate, "population_dynamics.sure_axis_peak_abs_difference", "mean"
        ),
        "summary_path": os.path.abspath(aggregate_path),
    }


def behavior_rows_for_model(model, summary_path, model_group):
    behavior = _sibling_json(summary_path, "behavior")
    rows = []
    for row in behavior.get("coherence", []):
        rows.append(
            {
                "model": model,
                "model_group": model_group,
                "coh": _finite_float(row.get("coh")),
                "n_trials": row.get("n_trials"),
                "n_sure_offered": row.get("n_sure_offered"),
                "p_sure": _finite_float(row.get("p_sure")),
                "no_sure_accuracy": _finite_float(row.get("no_sure_accuracy")),
                "waived_sure_accuracy": _finite_float(row.get("waived_sure_accuracy")),
                "n_waived_sure": row.get("n_waived_sure"),
            }
        )
    return rows


def build_symposium_payload(
    model_specs=None,
    parameter_specs=None,
    multiseed_summary=DEFAULT_TIMED_MULTISEED,
):
    if model_specs is None:
        model_specs = [
            ("original", DEFAULT_ORIGINAL_SUMMARY, "single"),
            ("baseline_timed", DEFAULT_TIMED_SUMMARY, "main"),
        ]
    if parameter_specs is None:
        parameter_specs = DEFAULT_PARAMETER_SPECS

    metrics_rows = []
    coherence_rows = []
    for model, summary_path, model_group in list(model_specs):
        if os.path.exists(summary_path):
            metrics_rows.append(summarize_model(model, summary_path, model_group))
            coherence_rows.extend(behavior_rows_for_model(model, summary_path, model_group))

    if multiseed_summary and os.path.exists(multiseed_summary):
        metrics_rows.append(summarize_multiseed(multiseed_summary))

    for model, summary_path, model_group in list(parameter_specs):
        if os.path.exists(summary_path):
            metrics_rows.append(summarize_model(model, summary_path, model_group))
            coherence_rows.extend(behavior_rows_for_model(model, summary_path, model_group))

    return _clean_for_json(
        {
            "metrics_rows": metrics_rows,
            "coherence_rows": coherence_rows,
        }
    )


def _ordered_fieldnames(rows, preferred):
    keys = []
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    extras = sorted({key for row in rows for key in row.keys()} - set(keys))
    return keys + extras


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean_for_json(payload), f, indent=2)
    return path


def _write_csv(path, rows, preferred_fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = _ordered_fieldnames(rows, preferred_fieldnames)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_clean_for_json(rows))
    return path


def _model_label(name):
    labels = {
        "original": "Original",
        "baseline_timed": "Timed baseline",
        "timed_5seed": "Timed 5-seed",
        "student_75k": "Student 75k",
        "reward_057": "Reward 0.57",
        "temp_012": "Temp 0.12",
        "blend_090": "Blend 0.90",
    }
    return labels.get(name, name)


def _value(row, key):
    value = row.get(key)
    if value is None:
        return np.nan
    return float(value)


def _finish_figure(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_task_mechanism(path):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), gridspec_kw={"height_ratios": [1.0, 1.3]})
    fig.suptitle("Controlled timed task and internal-confidence supervision", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    segments = [
        (4, 16, "Fixation", "#B8D8BA"),
        (20, 46, "Variable stimulus", "#9EC5E8"),
        (50, 68, "Delay", "#F2C078"),
        (72, 88, "Choice / TS", "#D7BDE2"),
    ]
    for x0, x1, label, color in segments:
        ax.add_patch(Rectangle((x0, 0.36), x1 - x0, 0.28, facecolor=color, edgecolor="#333333", lw=1.2))
        ax.text((x0 + x1) / 2.0, 0.50, label, ha="center", va="center", fontsize=10)
    ax.text(80, 0.78, "sure offered on 50% trials", ha="center", fontsize=10)
    ax.plot([88, 96], [0.50, 0.50], color="#555555", lw=1.5)
    ax.text(96, 0.50, "direction or sure", va="center", fontsize=10)

    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (3, 0.58, 18, 0.22, "Teacher RNN\nchoice outputs", "#A9CCE3"),
        (29, 0.58, 18, 0.22, "Internal readout\nmargin", "#AED6F1"),
        (55, 0.58, 18, 0.22, "Calibrated\nP(correct)", "#F9E79F"),
        (79, 0.58, 18, 0.22, "Expected value\nsure target", "#F5B7B1"),
        (32, 0.18, 24, 0.22, "Student RNN\nsupervised outputs", "#ABEBC6"),
        (68, 0.18, 24, 0.22, "Behavior +\npopulation axes", "#D2B4DE"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", lw=1.2))
        ax.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center", fontsize=10)
    arrows = [((21, 0.69), (29, 0.69)), ((47, 0.69), (55, 0.69)), ((73, 0.69), (79, 0.69)), ((88, 0.58), (46, 0.40)), ((56, 0.29), (68, 0.29))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333333"))
    return _finish_figure(fig, path)


def plot_behavior(path, payload):
    rows = payload.get("coherence_rows", [])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharex=False)
    fig.suptitle("Behavior follows uncertainty in the controlled timed task", fontsize=14, fontweight="bold")

    colors = {
        "baseline_timed": "#1F77B4",
        "original": "#7F7F7F",
        "student_75k": "#2CA02C",
        "reward_057": "#D62728",
        "temp_012": "#9467BD",
        "blend_090": "#FF7F0E",
    }
    for model in ["baseline_timed", "original", "student_75k", "reward_057", "blend_090", "temp_012"]:
        model_rows = sorted([row for row in rows if row.get("model") == model], key=lambda row: row["coh"])
        if not model_rows:
            continue
        lw = 2.8 if model == "baseline_timed" else 1.5
        alpha = 1.0 if model == "baseline_timed" else 0.72
        axes[0].plot(
            [row["coh"] for row in model_rows],
            [row["p_sure"] for row in model_rows],
            marker="o",
            lw=lw,
            alpha=alpha,
            color=colors.get(model),
            label=_model_label(model),
        )

    baseline_rows = sorted(
        [row for row in rows if row.get("model") == "baseline_timed"], key=lambda row: row["coh"]
    )
    if baseline_rows:
        axes[1].plot(
            [row["coh"] for row in baseline_rows],
            [row["no_sure_accuracy"] for row in baseline_rows],
            marker="s",
            lw=2.5,
            color="#1F77B4",
            label="No-sure accuracy",
        )
        axes[1].plot(
            [row["coh"] for row in baseline_rows],
            [row["waived_sure_accuracy"] for row in baseline_rows],
            marker="^",
            lw=2.5,
            color="#2CA02C",
            label="Waived-sure accuracy",
        )

    axes[0].set_xlabel("Motion coherence")
    axes[0].set_ylabel("P(sure | offered)")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_xscale("symlog", linthresh=0.032)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Motion coherence")
    axes[1].set_ylabel("Direction accuracy")
    axes[1].set_ylim(0.35, 1.04)
    axes[1].set_xscale("symlog", linthresh=0.032)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(True, alpha=0.25)
    return _finish_figure(fig, path)


def plot_population_metrics(path, payload):
    rows = [
        row
        for row in payload.get("metrics_rows", [])
        if row.get("model") in {"original", "baseline_timed", "timed_5seed"}
    ]
    labels = [_model_label(row["model"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    ax.bar(x - width, [_value(row, "evidence_r2_confidence") for row in rows], width, label="Evidence axis -> confidence", color="#4C78A8")
    ax.bar(x, [_value(row, "evidence_r2_p_sure") for row in rows], width, label="Evidence axis -> P(sure)", color="#F58518")
    ax.bar(x + width, [_value(row, "residual_r2") for row in rows], width, label="Sure axis -> residual P(sure)", color="#54A24B")
    for idx, row in enumerate(rows):
        std = row.get("residual_r2_std")
        if std is not None:
            ax.errorbar(idx + width, row["residual_r2"], yerr=std, color="#333333", capsize=3, lw=1.0)
    ax.set_title("Population readouts explain sure tendency beyond evidence", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("R2 / variance explained")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    return _finish_figure(fig, path)


def plot_leakage_parameter_validation(path, payload):
    rows = [
        row
        for row in payload.get("metrics_rows", [])
        if row.get("model") in {"baseline_timed", "student_75k", "reward_057", "temp_012", "blend_090"}
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    group_colors = {"main": "#1F77B4", "parameter": "#D62728"}
    for row in rows:
        x = _value(row, "residual_r2")
        y = _value(row, "pre_ts_corr")
        size = 170 if row.get("model") == "baseline_timed" else 95
        ax.scatter(x, y, s=size, color=group_colors.get(row.get("model_group"), "#7F7F7F"), alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.text(x + 0.012, y + 0.012, _model_label(row["model"]), fontsize=9)
    ax.axhline(0.35, color="#444444", lw=1.2, ls="--", alpha=0.75)
    ax.text(0.02, 0.365, "pre-TS diagnostic threshold", fontsize=9, color="#444444")
    ax.set_title("Stronger sure targets trade signal strength for leakage", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual P(sure) R2 explained by sure axis")
    ax.set_ylabel("pre-TS sure-axis correlation")
    ax.set_xlim(0, max(0.95, max([_value(row, "residual_r2") for row in rows] + [0.0]) + 0.08))
    ax.set_ylim(0, max(0.95, max([_value(row, "pre_ts_corr") for row in rows] + [0.0]) + 0.08))
    ax.grid(True, alpha=0.25)
    return _finish_figure(fig, path)


def write_symposium_outputs(output_dir, figure_dir, payload):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    paths = {
        "payload_json": _write_json(os.path.join(output_dir, "symposium_tables.json"), payload),
        "metrics_csv": _write_csv(
            os.path.join(output_dir, "symposium_model_metrics.csv"),
            payload.get("metrics_rows", []),
            METRIC_FIELD_ORDER,
        ),
        "coherence_csv": _write_csv(
            os.path.join(output_dir, "symposium_behavior_by_coherence.csv"),
            payload.get("coherence_rows", []),
            COHERENCE_FIELD_ORDER,
        ),
        "figures": {},
    }
    paths["figures"]["fig1_task_mechanism"] = plot_task_mechanism(
        os.path.join(figure_dir, "fig1_task_mechanism.png")
    )
    paths["figures"]["fig2_behavior"] = plot_behavior(
        os.path.join(figure_dir, "fig2_behavior.png"), payload
    )
    paths["figures"]["fig3_population_r2"] = plot_population_metrics(
        os.path.join(figure_dir, "fig3_population_r2.png"), payload
    )
    paths["figures"]["fig4_leakage_parameter_validation"] = plot_leakage_parameter_validation(
        os.path.join(figure_dir, "fig4_leakage_parameter_validation.png"), payload
    )
    return paths


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build RDM symposium-ready tables and figures.")
    parser.add_argument("--output-dir", default="paper_exports")
    parser.add_argument("--figure-dir", default="paper_figures")
    parser.add_argument("--original-summary", default=DEFAULT_ORIGINAL_SUMMARY)
    parser.add_argument("--timed-summary", default=DEFAULT_TIMED_SUMMARY)
    parser.add_argument("--timed-multiseed-summary", default=DEFAULT_TIMED_MULTISEED)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    payload = build_symposium_payload(
        model_specs=[
            ("original", args.original_summary, "single"),
            ("baseline_timed", args.timed_summary, "main"),
        ],
        parameter_specs=DEFAULT_PARAMETER_SPECS,
        multiseed_summary=args.timed_multiseed_summary,
    )
    paths = write_symposium_outputs(args.output_dir, args.figure_dir, payload)
    print(json.dumps({"outputs": paths}, indent=2), flush=True)
    return paths


if __name__ == "__main__":
    main()
