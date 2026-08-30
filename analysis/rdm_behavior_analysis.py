import csv
import json
import os

import numpy as np


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
                "dir_choice": int(info.get("dir_choice", -1)),
                "sure_available": sure_available,
                "choice": choice,
                "chose_sure": chose_sure,
                "chose_direction": chose_direction,
                "correct": correct,
                "waived_sure": bool(sure_available and chose_direction),
                "confidence_near_ts": float(conf_values[idx]),
                "evidence_axis": float(evidence_proj[idx]),
                "sure_axis": float(sure_proj[idx]),
                "sensory_margin": float(info.get("sensory_margin", np.nan)),
                "mean_sensory_margin": float(info.get("mean_sensory_margin", np.nan)),
                "teacher_internal_margin": float(info.get("teacher_internal_margin", np.nan)),
                "teacher_sure_strength": float(info.get("teacher_sure_strength", np.nan)),
                "external_sensory_margin": float(info.get("external_sensory_margin", np.nan)),
                "external_sure_strength": float(info.get("external_sure_strength", np.nan)),
            }
        )
    return records


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

    return {
        "n_trials": int(len(records)),
        "n_sure_offered": int(len(offered)),
        "n_sure_choice": int(sum(record["chose_sure"] for record in records)),
        "overall_p_sure": mean_bool(offered, lambda _: True, "chose_sure"),
        "no_sure_accuracy": mean_bool(no_sure, lambda _: True, "correct"),
        "waived_sure_accuracy": mean_bool(waived, lambda _: True, "correct"),
        "n_waived_sure": int(len(waived)),
        "coherence": coherence_rows,
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

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True)
    axes[0].plot(coh, p_sure, marker="o", color="#2f6f9f")
    axes[0].set_title("P(sure)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].plot(coh, no_sure_acc, marker="o", color="#2f8f5b")
    axes[1].set_title("No-sure accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].plot(coh, waived_acc, marker="o", color="#9a5c9f")
    axes[2].set_title("Waived-sure accuracy")
    axes[2].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.set_xlabel("coherence")
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
