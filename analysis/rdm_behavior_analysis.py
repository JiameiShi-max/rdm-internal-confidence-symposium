import csv
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")


def direction_choice_correct(choice, dir_choice):
    choice = int(choice)
    dir_choice = int(dir_choice)
    return bool((choice == 1 and dir_choice == 0) or (choice == 2 and dir_choice == 1))


def build_trial_eval_records(trial_info, choices, conf_values=None, axes_data=None):
    choices = np.asarray(choices, dtype=int)
    if conf_values is None:
        conf_values = np.full(len(trial_info), np.nan, dtype=float)
    else:
        conf_values = np.asarray(conf_values, dtype=float)

    evidence_proj = np.full(len(trial_info), np.nan, dtype=float)
    sure_proj = np.full(len(trial_info), np.nan, dtype=float)
    if axes_data is not None:
        if "evidence_proj" in axes_data:
            evidence_proj = np.abs(np.asarray(axes_data["evidence_proj"], dtype=float))
        if "sure_proj" in axes_data:
            sure_proj = np.asarray(axes_data["sure_proj"], dtype=float)

    records = []
    for idx, info in enumerate(trial_info):
        choice = int(choices[idx])
        sure_available = bool(info.get("sure_available", False))
        chose_sure = choice == 3
        chose_direction = choice in (1, 2)
        correct = direction_choice_correct(choice, info.get("dir_choice", -1)) if chose_direction else False
        records.append(
            {
                "trial_index": idx,
                "dataset_index": int(info.get("dataset_index", idx)),
                "coh": float(info.get("coh", np.nan)),
                "signed_coherence": float(info.get("signed_coherence", np.nan)),
                "dir_choice": int(info.get("dir_choice", -1)),
                "true_direction": int(info.get("true_direction", info.get("dir_choice", -1))),
                "sure_available": sure_available,
                "choice": choice,
                "final_network_choice": choice,
                "chose_sure": chose_sure,
                "chose_direction": chose_direction,
                "correct": correct,
                "direction_correct": correct if chose_direction else False,
                "waived_sure": bool(sure_available and chose_direction),
                "stimulus_duration_ms": float(info.get("stimulus_duration", info.get("stimulus_dur", np.nan))),
                "delay_duration_ms": float(info.get("delay_duration", info.get("delay_dur", np.nan))),
                "ts_latency_from_motion_offset_ms": float(info.get("ts_latency_from_motion_offset", info.get("ts_delay", np.nan))),
                "ts_onset_step": int(info.get("ts_onset", -1)),
                "ts_onset_ms": float(info.get("ts_onset_ms", np.nan)),
                "delay_end_step": int(info.get("delay_end", -1)),
                "go_cue_ms": float(info.get("go_cue_ms", info.get("delay_end_ms", np.nan))),
                "confidence_near_ts": float(conf_values[idx]),
                "evidence_axis": float(evidence_proj[idx]),
                "sure_axis": float(sure_proj[idx]),
                "sensory_margin": float(info.get("sensory_margin", np.nan)),
                "mean_sensory_margin": float(info.get("mean_sensory_margin", np.nan)),
                "teacher_internal_margin": float(info.get("teacher_internal_margin", np.nan)),
                "teacher_sure_strength": float(info.get("teacher_sure_strength", np.nan)),
                "teacher_correct": bool(info.get("teacher_correct", False)),
                "teacher_direction_choice": int(info.get("teacher_direction_choice", -1)),
                "teacher_p_correct_proxy": float(info.get("teacher_direction_success_proxy", np.nan)),
                "teacher_expected_direction_value": float(info.get("teacher_expected_direction_value", np.nan)),
                "teacher_sure_advantage": float(info.get("teacher_sure_advantage", info.get("sure_advantage", np.nan))),
                "external_sensory_margin": float(info.get("external_sensory_margin", np.nan)),
                "external_sure_strength": float(info.get("external_sure_strength", np.nan)),
            }
        )
    return records


def _fit_logistic_irls(x, y, max_iter=100, ridge=1e-8):
    """Small dependency-free logistic fit with Wald standard errors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(x.shape[0]), x])
    coef = np.zeros(design.shape[1], dtype=float)
    for _ in range(int(max_iter)):
        eta = np.clip(design @ coef, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1.0 - p), 1e-7)
        hessian = design.T @ (design * w[:, None]) + ridge * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, design.T @ (y - p))
        coef_next = coef + step
        if np.max(np.abs(coef_next - coef)) < 1e-9:
            coef = coef_next
            break
        coef = coef_next
    eta = np.clip(design @ coef, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.maximum(p * (1.0 - p), 1e-7)
    covariance = np.linalg.pinv(design.T @ (design * w[:, None]) + ridge * np.eye(design.shape[1]))
    return coef, np.sqrt(np.maximum(np.diag(covariance), 0.0))


def summarize_sure_logistic(records, include_interaction=False):
    offered = [r for r in records if r["sure_available"]]
    usable = [
        r for r in offered
        if np.isfinite(r["coh"]) and np.isfinite(r["stimulus_duration_ms"])
    ]
    names = ["intercept", "coherence", "duration_s"]
    if len(usable) < 4 or len({bool(r["chose_sure"]) for r in usable}) < 2:
        return {"n": len(usable), "available": False, "reason": "insufficient outcome variation"}
    x = []
    for r in usable:
        duration_s = float(r["stimulus_duration_ms"]) / 1000.0
        row = [float(r["coh"]), duration_s]
        if include_interaction:
            row.append(float(r["coh"]) * duration_s)
        x.append(row)
    if include_interaction:
        names.append("coherence_x_duration_s")
    coef, se = _fit_logistic_irls(x, [r["chose_sure"] for r in usable])
    return {
        "n": len(usable),
        "available": True,
        "coefficients": {
            name: {
                "estimate": float(value),
                "standard_error": float(error),
                "ci95_low": float(value - 1.96 * error),
                "ci95_high": float(value + 1.96 * error),
            }
            for name, value, error in zip(names, coef, se)
        },
    }


def _binned_rows(records, field, n_bins=4):
    values = np.asarray([r[field] for r in records], dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return [], np.asarray([], dtype=float)
    edges = np.unique(np.quantile(values[finite], np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size < 2:
        edges = np.asarray([values[finite][0] - 0.5, values[finite][0] + 0.5])
    rows = []
    for idx in range(edges.size - 1):
        upper_ok = values <= edges[idx + 1] if idx == edges.size - 2 else values < edges[idx + 1]
        select = finite & (values >= edges[idx]) & upper_ok
        subset = [records[i] for i in np.flatnonzero(select)]
        rows.append({
            "bin_index": int(idx),
            "low": float(edges[idx]),
            "high": float(edges[idx + 1]),
            "center": float(np.mean(values[select])),
            "n": int(len(subset)),
            "p_sure": mean_bool(subset, lambda r: r["sure_available"], "chose_sure"),
            "no_sure_accuracy": mean_bool(subset, lambda r: not r["sure_available"], "correct"),
            "waived_sure_accuracy": mean_bool(subset, lambda r: r["waived_sure"], "correct"),
        })
    return rows, edges


def mean_bool(records, predicate, value_key):
    values = [float(bool(record[value_key])) for record in records if predicate(record)]
    if not values:
        return np.nan
    return float(np.mean(values))


def summarize_behavior(records):
    offered = [record for record in records if record["sure_available"]]
    no_sure = [record for record in records if not record["sure_available"]]
    waived = [record for record in records if record["waived_sure"]]

    coherence_rows = []
    coherences = sorted({record["coh"] for record in records if np.isfinite(record["coh"])})
    for coh in coherences:
        coh_records = [record for record in records if record["coh"] == coh]
        offered_coh = [record for record in coh_records if record["sure_available"]]
        no_sure_coh = [record for record in coh_records if not record["sure_available"]]
        waived_coh = [record for record in coh_records if record["waived_sure"]]
        coherence_rows.append(
            {
                "coh": float(coh),
                "n_trials": int(len(coh_records)),
                "n_sure_offered": int(len(offered_coh)),
                "p_sure": mean_bool(offered_coh, lambda _: True, "chose_sure"),
                "no_sure_accuracy": mean_bool(no_sure_coh, lambda _: True, "correct"),
                "waived_sure_accuracy": mean_bool(waived_coh, lambda _: True, "correct"),
                "n_waived_sure": int(len(waived_coh)),
            }
        )

    duration_rows, duration_edges = _binned_rows(records, "stimulus_duration_ms")
    joint_rows = []
    for coh in coherences:
        coh_records = [record for record in records if record["coh"] == coh]
        for duration_row in duration_rows:
            subset = [
                r for r in coh_records
                if r["stimulus_duration_ms"] >= duration_row["low"]
                and (
                    r["stimulus_duration_ms"] <= duration_row["high"]
                    if duration_row["bin_index"] == len(duration_rows) - 1
                    else r["stimulus_duration_ms"] < duration_row["high"]
                )
            ]
            offered_subset = [r for r in subset if r["sure_available"]]
            joint_rows.append({
                "coh": float(coh),
                "duration_bin": int(duration_row["bin_index"]),
                "duration_low_ms": duration_row["low"],
                "duration_high_ms": duration_row["high"],
                "n": int(len(subset)),
                "n_offered": int(len(offered_subset)),
                "p_sure": mean_bool(offered_subset, lambda _: True, "chose_sure"),
            })

    return {
        "n_trials": int(len(records)),
        "n_sure_offered": int(len(offered)),
        "n_sure_choice": int(sum(record["chose_sure"] for record in records)),
        "overall_p_sure": mean_bool(offered, lambda _: True, "chose_sure"),
        "no_sure_accuracy": mean_bool(no_sure, lambda _: True, "correct"),
        "waived_sure_accuracy": mean_bool(waived, lambda _: True, "correct"),
        "n_waived_sure": int(len(waived)),
        "coherence": coherence_rows,
        "duration": duration_rows,
        "coherence_x_duration": joint_rows,
        "sure_logistic": summarize_sure_logistic(records),
    }


def write_trial_records_csv(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def write_behavior_summary_json(path, summary):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def plot_behavior_summary(path, behavior_summary, title="RDM behavior"):
    import matplotlib.pyplot as plt

    rows = behavior_summary["coherence"]
    coh = np.asarray([row["coh"] for row in rows], dtype=float)
    p_sure = np.asarray([row["p_sure"] for row in rows], dtype=float)
    no_sure_acc = np.asarray([row["no_sure_accuracy"] for row in rows], dtype=float)
    waived_acc = np.asarray([row["waived_sure_accuracy"] for row in rows], dtype=float)

    duration_rows = behavior_summary.get("duration", [])
    duration = np.asarray([row["center"] for row in duration_rows], dtype=float)
    duration_p_sure = np.asarray([row["p_sure"] for row in duration_rows], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    axes[0].plot(coh, p_sure, marker="o", color="#2f6f9f")
    axes[0].set_title("P(sure)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].plot(coh, no_sure_acc, marker="o", color="#2f8f5b")
    axes[1].set_title("No-sure accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].plot(coh, waived_acc, marker="o", color="#9a5c9f")
    axes[2].set_title("Waived-sure accuracy")
    axes[2].set_ylim(-0.05, 1.05)
    axes[3].plot(duration, duration_p_sure, marker="o", color="#d17c25")
    axes[3].set_title("P(sure) vs duration")
    axes[3].set_xlabel("stimulus duration (ms)")
    axes[3].set_ylim(-0.05, 1.05)
    duration_no_sure = np.asarray([row["no_sure_accuracy"] for row in duration_rows], dtype=float)
    duration_waived = np.asarray([row["waived_sure_accuracy"] for row in duration_rows], dtype=float)
    axes[4].plot(duration, duration_no_sure, marker="o", label="no sure", color="#2f8f5b")
    axes[4].plot(duration, duration_waived, marker="s", label="waived sure", color="#9a5c9f")
    axes[4].set_title("Accuracy vs duration")
    axes[4].set_xlabel("stimulus duration (ms)")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].legend(frameon=False)
    joint = behavior_summary.get("coherence_x_duration", [])
    n_d = max([row["duration_bin"] for row in joint], default=-1) + 1
    heat = np.full((n_d, len(coh)), np.nan)
    coh_index = {float(value): idx for idx, value in enumerate(coh)}
    for row in joint:
        if row["coh"] in coh_index:
            heat[row["duration_bin"], coh_index[row["coh"]]] = row["p_sure"]
    if heat.size:
        image = axes[5].imshow(heat, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axes[5].set_title("P(sure): coherence x duration")
        axes[5].set_xlabel("coherence bin")
        axes[5].set_ylabel("duration bin")
        fig.colorbar(image, ax=axes[5], label="P(sure | offered)")
    else:
        axes[5].text(0.5, 0.5, "duration metadata unavailable", ha="center", va="center")
        axes[5].set_axis_off()
    for ax in axes[:3]:
        ax.set_xlabel("coherence")
    for ax in axes[:5]:
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("probability / accuracy")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def default_behavior_paths(summary_path):
    root, _ = os.path.splitext(summary_path)
    return {
        "trial_csv": f"{root}_trials.csv",
        "behavior_summary": f"{root}_behavior.json",
        "behavior_fig": f"{root}_behavior.png",
    }
