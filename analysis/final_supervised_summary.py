"""Assemble frozen supervised-RNN results into final figures and prose.

This module is synthesis only. It reads completed Stage 0-3 artifacts, copies
their reported quantities into compact source-data tables, and renders figures.
It performs no model replay, fitting, decoding, axis construction, or inference.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SEEDS = (7, 8, 9)
EXAMPLE_SEED = 7
SEED_COLORS = {7: "#1b9e77", 8: "#d95f02", 9: "#7570b3"}
CHOICE_COLOR = "#2474b5"
CONFIDENCE_COLOR = "#6a3d9a"
LEFT_COLOR = "#2474b5"
RIGHT_COLOR = "#d95f3d"
SURE_COLOR = "#6a3d9a"
GRAY = "#5f6368"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def by_seed(rows):
    return {int(row["seed"]): row for row in rows}


def load_sources(results_root):
    results_root = Path(results_root)
    paths = {
        "freeze_report": results_root / "final_supervised_freeze" / "final_report.md",
        "freeze_manifest": results_root / "final_supervised_freeze" / "manifest.json",
        "freeze_summary": results_root / "final_supervised_freeze" / "multiseed" / "freeze_summary.json",
        "stage1_report": results_root / "population_stage1" / "final_report.md",
        "stage1_manifest": results_root / "population_stage1" / "analysis_manifest.json",
        "stage1_summary": results_root / "population_stage1" / "multiseed" / "representation_summary.json",
        "stage1_seed7_states": results_root / "population_stage1" / "seed7" / "pre_go_states.npz",
        "stage1_seed7_pca": results_root / "population_stage1" / "seed7" / "pca.npz",
        "stage1_seed7_metadata": results_root / "population_stage1" / "seed7" / "state_metadata.csv",
        "stage2_report": results_root / "population_stage2_time" / "final_report.md",
        "stage2_manifest": results_root / "population_stage2_time" / "analysis_manifest.json",
        "stage2_summary": results_root / "population_stage2_time" / "multiseed" / "time_replication_summary.json",
        "stage2_seed7_time": results_root / "population_stage2_time" / "seed7" / "time_resolved_metrics.csv",
        "stage2_seed7_fixed": results_root / "population_stage2_time" / "seed7" / "fixed_axis_metrics.csv",
        "stage3_report": results_root / "population_stage3_variability" / "final_report.md",
        "stage3_manifest": results_root / "population_stage3_variability" / "analysis_manifest.json",
        "stage3_summary": results_root / "population_stage3_variability" / "multiseed" / "variability_summary.json",
        "stage3_seed7_repeats": results_root / "population_stage3_variability" / "seed7" / "repeat_records.csv",
        "stage3_seed7_stimuli": results_root / "population_stage3_variability" / "seed7" / "stimulus_summary.csv",
    }
    for seed in SEEDS:
        paths[f"freeze_seed{seed}_test"] = results_root / "final_supervised_freeze" / f"seed{seed}" / "test_summary.json"
        paths[f"stage3_seed{seed}_control"] = results_root / "population_stage3_variability" / f"seed{seed}" / "deterministic_control.json"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen source artifacts: " + ", ".join(missing))
    return paths


def assemble_key_results(paths):
    freeze = by_seed(load_json(paths["freeze_summary"])["individual_seed_metrics"])
    stage1 = by_seed(load_json(paths["stage1_summary"]))
    stage2 = by_seed(load_json(paths["stage2_summary"]))
    stage3 = by_seed(load_json(paths["stage3_summary"]))
    rows = []
    for seed in SEEDS:
        rows.append(
            {
                "seed": seed,
                "test_no_sure_direction_accuracy": freeze[seed]["test_direction_accuracy"],
                "test_p_sure_given_offered": freeze[seed]["p_sure_given_offered"],
                "stage1_choice_cv_accuracy": stage1[seed]["choice_cv_accuracy"],
                "stage1_confidence_cv_pearson_r": stage1[seed]["confidence_cv_pearson_r"],
                "stage1_confidence_test_pearson_r": stage1[seed]["test_corr_confidence_projection_empirical_p_correct"],
                "stage1_choice_confidence_absolute_cosine": stage1[seed]["choice_confidence_absolute_cosine_similarity"],
                "stage2_choice_operational_onset_ms": stage2[seed]["choice_operational_onset_motion_onset_ms"],
                "stage2_confidence_operational_onset_ms": stage2[seed]["confidence_operational_onset_motion_onset_ms"],
                "stage2_pre_ts_confidence_r": stage2[seed]["pre_ts_confidence_correlation"],
                "stage2_pre_go_confidence_r": stage2[seed]["pre_go_confidence_correlation"],
                "stage3_n_variable_stimuli": stage3[seed]["n_variable"],
                "stage3_overall_stochastic_p_sure": stage3[seed]["overall_stochastic_p_sure"],
                "stage3_primary_beta": stage3[seed]["primary_fixed_effect_beta"],
                "stage3_bootstrap_ci95_low": stage3[seed]["primary_bootstrap_ci95_low"],
                "stage3_bootstrap_ci95_high": stage3[seed]["primary_bootstrap_ci95_high"],
                "stage3_permutation_two_sided_p": stage3[seed]["primary_permutation_two_sided_p"],
                "stage3_specificity_confidence_beta": stage3[seed]["secondary_confidence_beta_after_choice_control"],
                "stage3_zero_noise_choice_consistency": stage3[seed]["noise_zero_choice_consistency"],
            }
        )
    return rows


def stage_audit_rows():
    return [
        {
            "analysis_stage": "Final supervised freeze",
            "scientific_question": "Does the frozen categorical RNN reproduce confidence-dependent wagering?",
            "primary_metric": "Held-out no-Sure direction accuracy; P(Sure | offered)",
            "seed7_result": "0.880; 0.160",
            "seed8_result": "0.840; 0.093",
            "seed9_result": "0.827; 0.107",
            "main_limitation": "Student wagering shows quantitative mismatch from the teacher policy.",
            "artifact_source": "results/final_supervised_freeze/multiseed/freeze_summary.json",
        },
        {
            "analysis_stage": "Population Stage 1",
            "scientific_question": "Are directional choice and graded confidence linearly represented in population activity?",
            "primary_metric": "Choice CV accuracy; confidence CV/test r; |choice-confidence cosine|",
            "seed7_result": "0.995; 0.768/0.776; 0.133",
            "seed8_result": "0.997; 0.755/0.749; 0.040",
            "seed9_result": "0.997; 0.748/0.740; 0.090",
            "main_limitation": "Low overlap is approximate separability, not mathematical orthogonality or causality.",
            "artifact_source": "results/population_stage1/multiseed/representation_summary.json",
        },
        {
            "analysis_stage": "Population Stage 2",
            "scientific_question": "When do choice and confidence information become linearly available?",
            "primary_metric": "Permutation-corrected motion-onset decoding onset; pre-TS confidence r",
            "seed7_result": "choice 80 ms; confidence 140 ms; pre-TS r=0.590",
            "seed8_result": "choice 80 ms; confidence 180 ms; pre-TS r=0.615",
            "seed9_result": "choice 30 ms; confidence 80 ms; pre-TS r=0.665",
            "main_limitation": "Operational decoder onsets are not biological or causal latencies.",
            "artifact_source": "results/population_stage2_time/multiseed/time_replication_summary.json",
        },
        {
            "analysis_stage": "Population Stage 3",
            "scientific_question": "Under identical input, do confidence-axis fluctuations predict later wagering?",
            "primary_metric": "Stimulus-fixed-effect logistic beta with cluster-bootstrap CI and within-stimulus permutation p",
            "seed7_result": "β=-0.581 [-0.744,-0.441], p=0.001",
            "seed8_result": "β=-0.391 [-0.572,-0.219], p=0.001",
            "seed9_result": "β=-0.336 [-0.481,-0.222], p=0.001",
            "main_limitation": "Specificity after absolute choice-axis control was not uniform (seed 8 confidence β>0).",
            "artifact_source": "results/population_stage3_variability/multiseed/variability_summary.json",
        },
    ]


def create_source_data(paths, output_root):
    source_dir = Path(output_root) / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    freeze = load_json(paths["freeze_summary"])
    behavior = []
    for row in freeze["p_sure_given_offered_by_coherence"]:
        behavior.append(
            {
                "coherence": row["coherence"],
                "seed7_p_sure": row["individual_seeds"]["7"],
                "seed8_p_sure": row["individual_seeds"]["8"],
                "seed9_p_sure": row["individual_seeds"]["9"],
                "mean_p_sure": row["mean"],
                "sd_p_sure": row["sd"],
            }
        )
    write_csv(source_dir / "panel_b_behavior.csv", behavior)

    metadata = read_csv(paths["stage1_seed7_metadata"])
    with np.load(paths["stage1_seed7_states"], allow_pickle=False) as stored:
        choice_projection = np.asarray(stored["choice_projection"], dtype=float)
        confidence_projection = np.asarray(stored["confidence_projection"], dtype=float)
    geometry = []
    for row in metadata:
        if row["split"] != "test":
            continue
        index = int(row["source_trial_index"])
        geometry.append(
            {
                "source_trial_id": index,
                "final_choice": int(row["student_final_choice"]),
                "final_choice_name": row["student_final_choice_name"],
                "choice_axis_projection": choice_projection[index],
                "confidence_axis_projection": confidence_projection[index],
            }
        )
    write_csv(source_dir / "panel_c_geometry.csv", geometry)

    stage2_rows = read_csv(paths["stage2_seed7_time"])
    motion = [row for row in stage2_rows if row["event"] == "motion_onset"]
    dynamics = [
        {
            "relative_time_ms": int(row["relative_time_ms"]),
            "choice_test_roc_auc": row["choice_test_roc_auc"],
            "confidence_test_pearson_r": row["confidence_test_pearson_r"],
            "choice_familywise_threshold": row["choice_familywise_threshold"],
            "confidence_familywise_threshold": row["confidence_familywise_threshold"],
            "choice_exceeds_threshold": row["choice_exceeds_familywise_threshold"],
            "confidence_exceeds_threshold": row["confidence_exceeds_familywise_threshold"],
        }
        for row in motion
    ]
    write_csv(source_dir / "panel_d_motion_onset.csv", dynamics)

    # Reproduce the already-defined Stage-3 paired summary from saved repeat records.
    repeat_rows = read_csv(paths["stage3_seed7_repeats"])
    grouped = {}
    for row in repeat_rows:
        source = int(row["source_trial_id"])
        grouped.setdefault(source, {"Sure": [], "Direction": []})
        group = "Sure" if int(row["sure_outcome"]) == 1 else "Direction"
        grouped[source][group].append(float(row["confidence_projection"]))
    paired = []
    for source in sorted(grouped):
        sure = grouped[source]["Sure"]
        direction = grouped[source]["Direction"]
        if sure and direction:
            paired.append(
                {
                    "source_trial_id": source,
                    "n_sure_repeats": len(sure),
                    "n_direction_repeats": len(direction),
                    "mean_confidence_sure": float(np.mean(sure)),
                    "mean_confidence_direction": float(np.mean(direction)),
                    "sure_minus_direction": float(np.mean(sure) - np.mean(direction)),
                }
            )
    write_csv(source_dir / "panel_e_identical_stimulus_pairs.csv", paired)

    stage3 = by_seed(load_json(paths["stage3_summary"]))
    replication = [
        {
            "seed": seed,
            "primary_beta": stage3[seed]["primary_fixed_effect_beta"],
            "bootstrap_ci95_low": stage3[seed]["primary_bootstrap_ci95_low"],
            "bootstrap_ci95_high": stage3[seed]["primary_bootstrap_ci95_high"],
            "permutation_two_sided_p": stage3[seed]["primary_permutation_two_sided_p"],
        }
        for seed in SEEDS
    ]
    write_csv(source_dir / "panel_f_stage3_replication.csv", replication)

    with np.load(paths["stage1_seed7_pca"], allow_pickle=False) as stored:
        test_ids = np.asarray(stored["test_source_trial_indices"], dtype=int)
        test_coordinates = np.asarray(stored["test_coordinates"], dtype=float)
    choice_by_id = {int(row["source_trial_index"]): row["student_final_choice_name"] for row in metadata}
    pca_rows = [
        {"source_trial_id": int(source), "final_choice_name": choice_by_id[int(source)], "pc1": coord[0], "pc2": coord[1]}
        for source, coord in zip(test_ids, test_coordinates)
    ]
    write_csv(source_dir / "supp_stage1_pca.csv", pca_rows)

    full_dynamics = [
        {
            "event": row["event"],
            "relative_time_ms": int(row["relative_time_ms"]),
            "choice_test_roc_auc": row["choice_test_roc_auc"],
            "confidence_test_pearson_r": row["confidence_test_pearson_r"],
            "choice_familywise_threshold": row["choice_familywise_threshold"],
            "confidence_familywise_threshold": row["confidence_familywise_threshold"],
        }
        for row in stage2_rows
    ]
    write_csv(source_dir / "supp_stage2_all_events.csv", full_dynamics)

    stimulus_rows = read_csv(paths["stage3_seed7_stimuli"])
    p_sure_rows = [
        {"source_trial_id": int(row["source_trial_id"]), "p_sure": row["p_sure"], "variability_class": row["variability_class"]}
        for row in stimulus_rows
    ]
    write_csv(source_dir / "supp_stage3_p_sure_by_stimulus.csv", p_sure_rows)

    specificity = [
        {
            "seed": seed,
            "confidence_beta_after_choice_control": stage3[seed]["secondary_confidence_beta_after_choice_control"],
            "absolute_choice_beta": stage3[seed]["secondary_absolute_choice_beta"],
        }
        for seed in SEEDS
    ]
    write_csv(source_dir / "supp_stage3_specificity.csv", specificity)

    controls = []
    for seed in SEEDS:
        control = load_json(paths[f"stage3_seed{seed}_control"])
        controls.append(
            {
                "seed": seed,
                "choice_consistency": control["choice_consistency"],
                "max_abs_state_difference": control["max_abs_state_difference"],
                "max_abs_output_difference": control["max_abs_output_difference"],
            }
        )
    write_csv(source_dir / "supp_stage3_deterministic_control.csv", controls)
    return {
        "behavior": behavior,
        "geometry": geometry,
        "dynamics": dynamics,
        "paired": paired,
        "replication": replication,
        "pca": pca_rows,
        "full_dynamics": full_dynamics,
        "p_sure_stimulus": p_sure_rows,
        "specificity": specificity,
        "controls": controls,
    }


def panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=17, weight="bold", va="top")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)


def draw_schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    panel_label(ax, "A")
    ax.text(0.0, 9.5, "Task and frozen recurrent network", fontsize=13, weight="bold", va="top")
    boxes = [
        (0.3, 5.3, 2.1, 1.25, "motion\nevidence", "#dbe9f6"),
        (3.4, 5.0, 2.8, 1.85, "50-unit recurrent\nnetwork", "#e5d9f2"),
        (7.2, 5.3, 2.35, 1.25, "Left  Right  Sure\ncategorical readout", "#f5dfd5"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", facecolor=color, edgecolor=GRAY, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for start, end in [((2.45, 5.93), (3.3, 5.93)), ((6.25, 5.93), (7.1, 5.93))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=GRAY))
    y = 2.35
    ax.plot([0.6, 9.4], [y, y], color=GRAY, linewidth=1.4)
    events = [(1.4, "motion\nonset"), (4.1, "motion\noffset"), (6.65, "Sure-target\nonset"), (8.75, "go cue")]
    for x, text in events:
        ax.plot([x, x], [y - 0.2, y + 0.35], color=GRAY, linewidth=1.2)
        ax.text(x, y - 0.45, text, ha="center", va="top", fontsize=9)
    ax.text(0.6, 3.05, "variable trial epochs", fontsize=9, color=GRAY)
    ax.text(5.0, 0.35, "Seed 7: predetermined example  •  Seeds 8–9: replication", ha="center", fontsize=9, color=GRAY)


def main_figure(data, key_results, output_dir):
    key = by_seed(key_results)
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    outer = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.15, 1.15])

    ax_a = fig.add_subplot(outer[0, 0])
    draw_schematic(ax_a)

    ax_b = fig.add_subplot(outer[0, 1])
    panel_label(ax_b, "B")
    coherence = np.asarray([float(row["coherence"]) for row in data["behavior"]])
    for seed in SEEDS:
        values = np.asarray([float(row[f"seed{seed}_p_sure"]) for row in data["behavior"]])
        ax_b.plot(coherence, values, marker="o", markersize=4, linewidth=1.2, alpha=0.75, color=SEED_COLORS[seed], label=f"seed {seed}")
    mean = np.asarray([float(row["mean_p_sure"]) for row in data["behavior"]])
    sd = np.asarray([float(row["sd_p_sure"]) for row in data["behavior"]])
    ax_b.plot(coherence, mean, color="black", linewidth=2.6, marker="o", label="seed mean")
    ax_b.fill_between(coherence, np.maximum(0, mean - sd), np.minimum(1, mean + sd), color="0.5", alpha=0.14, linewidth=0)
    ax_b.set(xlabel="motion coherence", ylabel="P(Sure | offered)", title="Held-out confidence-dependent wagering", ylim=(-0.03, 0.78))
    ax_b.legend(frameon=False, ncol=2, fontsize=9)
    style_axis(ax_b)

    ax_c = fig.add_subplot(outer[1, 0])
    panel_label(ax_c, "C")
    for name, color, marker in [("Left", LEFT_COLOR, "o"), ("Right", RIGHT_COLOR, "o"), ("Sure", SURE_COLOR, "^")]:
        selected = [row for row in data["geometry"] if row["final_choice_name"] == name]
        ax_c.scatter(
            [float(row["choice_axis_projection"]) for row in selected],
            [float(row["confidence_axis_projection"]) for row in selected],
            s=34 if name != "Sure" else 48,
            alpha=0.72,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.4,
            label=f"{name} (n={len(selected)})",
        )
    ax_c.axhline(0, color="0.85", linewidth=0.8)
    ax_c.axvline(0, color="0.85", linewidth=0.8)
    ax_c.set(xlabel="choice-axis projection  (higher = Right)", ylabel="confidence-axis projection\n(higher = higher empirical P(correct))", title="Held-out choice × confidence geometry")
    ax_c.text(0.03, 0.97, f"|cosine| = {key[7]['stage1_choice_confidence_absolute_cosine']:.3f}\napproximately distinct axes", transform=ax_c.transAxes, va="top", fontsize=10, bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9})
    ax_c.legend(frameon=False, fontsize=9, loc="lower right")
    style_axis(ax_c)

    sub_d = outer[1, 1].subgridspec(2, 1, hspace=0.08)
    ax_d1 = fig.add_subplot(sub_d[0, 0])
    ax_d2 = fig.add_subplot(sub_d[1, 0], sharex=ax_d1)
    panel_label(ax_d1, "D")
    times = np.asarray([int(row["relative_time_ms"]) for row in data["dynamics"]])
    auc = np.asarray([float(row["choice_test_roc_auc"]) for row in data["dynamics"]])
    conf = np.asarray([float(row["confidence_test_pearson_r"]) if row["confidence_test_pearson_r"] != "nan" else np.nan for row in data["dynamics"]])
    choice_threshold = float(data["dynamics"][0]["choice_familywise_threshold"])
    confidence_threshold = float(data["dynamics"][0]["confidence_familywise_threshold"])
    ax_d1.plot(times, auc, color=CHOICE_COLOR, linewidth=2.4)
    ax_d1.axhline(choice_threshold, color=CHOICE_COLOR, linestyle="--", alpha=0.55)
    ax_d1.axvline(0, color="black", linewidth=1)
    ax_d1.axvline(key[7]["stage2_choice_operational_onset_ms"], color=CHOICE_COLOR, linestyle=":", linewidth=1.8)
    ax_d1.set(ylabel="choice TEST AUC", title="Motion-onset-aligned information (seed 7)", ylim=(0.45, 1.03))
    ax_d1.text(key[7]["stage2_choice_operational_onset_ms"] + 7, 0.56, "onset 80 ms", color=CHOICE_COLOR, fontsize=9)
    ax_d2.plot(times, conf, color=CONFIDENCE_COLOR, linewidth=2.4)
    ax_d2.axhline(confidence_threshold, color=CONFIDENCE_COLOR, linestyle="--", alpha=0.55)
    ax_d2.axvline(0, color="black", linewidth=1)
    ax_d2.axvline(key[7]["stage2_confidence_operational_onset_ms"], color=CONFIDENCE_COLOR, linestyle=":", linewidth=1.8)
    ax_d2.set(xlabel="time from motion onset (ms)\n(each point summarizes preceding 50 ms)", ylabel="confidence TEST r", ylim=(-0.05, 0.72))
    ax_d2.text(key[7]["stage2_confidence_operational_onset_ms"] + 7, 0.08, "onset 140 ms", color=CONFIDENCE_COLOR, fontsize=9)
    plt.setp(ax_d1.get_xticklabels(), visible=False)
    style_axis(ax_d1)
    style_axis(ax_d2)

    ax_e = fig.add_subplot(outer[2, 0])
    panel_label(ax_e, "E")
    for row in data["paired"]:
        direction = float(row["mean_confidence_direction"])
        sure = float(row["mean_confidence_sure"])
        ax_e.plot([0, 1], [direction, sure], color="0.75", linewidth=0.8, alpha=0.8)
        ax_e.scatter([0, 1], [direction, sure], color=[SEED_COLORS[7], SURE_COLOR], s=24, zorder=3)
    ax_e.set(xticks=[0, 1], xticklabels=["Direction repeats", "Sure repeats"], ylabel="confidence-axis projection", title=f"Identical input, different recurrent noise (seed 7; {len(data['paired'])} variable stimuli)")
    ax_e.text(0.03, 0.04, "paired within source stimulus\n96.2% of differences < 0", transform=ax_e.transAxes, fontsize=10, va="bottom", bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9})
    style_axis(ax_e)

    ax_f = fig.add_subplot(outer[2, 1])
    panel_label(ax_f, "F")
    y = np.arange(len(SEEDS))[::-1]
    for yi, row in zip(y, data["replication"]):
        seed = int(row["seed"])
        beta = float(row["primary_beta"])
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        ax_f.errorbar(beta, yi, xerr=[[beta - low], [high - beta]], fmt="o", markersize=8, color=SEED_COLORS[seed], capsize=4, linewidth=2)
        ax_f.text(low - 0.015, yi + 0.18, "p = 0.001", ha="left", fontsize=9, color=SEED_COLORS[seed])
    ax_f.axvline(0, color=GRAY, linewidth=1.2)
    ax_f.set(yticks=y, yticklabels=[f"seed {seed}" for seed in SEEDS], xlabel="stimulus-fixed-effect confidence β (per SD)", title="Identical-stimulus replication", ylim=(-0.65, 2.65), xlim=(-0.82, 0.08))
    ax_f.text(0.03, 0.05, "lower confidence → greater P(Sure)", transform=ax_f.transAxes, fontsize=10, color=CONFIDENCE_COLOR)
    style_axis(ax_f)

    fig.suptitle("Confidence-dependent wagering in frozen supervised recurrent networks", fontsize=20, weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"main_figure.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def supplementary_figure(data, output_dir):
    figure_dir = Path(output_dir) / "figures"
    fig = plt.figure(figsize=(20, 10.5), constrained_layout=True)
    outer = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0])

    ax_a = fig.add_subplot(outer[0, 0])
    panel_label(ax_a, "A")
    for name, color, marker in [("Left", LEFT_COLOR, "o"), ("Right", RIGHT_COLOR, "o"), ("Sure", SURE_COLOR, "^")]:
        selected = [row for row in data["pca"] if row["final_choice_name"] == name]
        ax_a.scatter([float(row["pc1"]) for row in selected], [float(row["pc2"]) for row in selected], s=24, alpha=0.65, color=color, marker=marker, label=name)
    ax_a.set(xlabel="PC1", ylabel="PC2", title="Stage 1: held-out PCA (seed 7)")
    ax_a.legend(frameon=False, fontsize=8)
    style_axis(ax_a)

    ax_b = fig.add_subplot(outer[0, 1])
    panel_label(ax_b, "B")
    stim = sorted(data["p_sure_stimulus"], key=lambda row: int(row["source_trial_id"]))
    ax_b.bar(np.arange(len(stim)), [float(row["p_sure"]) for row in stim], color=CONFIDENCE_COLOR, width=0.9)
    ax_b.set(xlabel="sure-offered TEST stimulus (source-ID order)", ylabel="P(Sure | 100 repeats)", title="Stage 3: identical-input behavior")
    ax_b.set_xticks([])
    style_axis(ax_b)

    # Four compact event-aligned panels, each with metric-specific stacked axes.
    event_titles = {
        "motion_onset": "motion onset",
        "motion_offset": "motion offset",
        "sure_target_onset": "Sure-target onset",
        "go_cue": "go cue",
    }
    for position, event in enumerate(event_titles):
        cell = outer[1, position].subgridspec(2, 1, hspace=0.08)
        top = fig.add_subplot(cell[0, 0])
        bottom = fig.add_subplot(cell[1, 0], sharex=top)
        panel_label(top, chr(ord("C") + position))
        rows = [row for row in data["full_dynamics"] if row["event"] == event]
        t = np.asarray([int(row["relative_time_ms"]) for row in rows])
        choice = np.asarray([float(row["choice_test_roc_auc"]) for row in rows])
        confidence = np.asarray([float(row["confidence_test_pearson_r"]) if row["confidence_test_pearson_r"] != "nan" else np.nan for row in rows])
        top.plot(t, choice, color=CHOICE_COLOR, linewidth=1.8)
        top.axhline(float(rows[0]["choice_familywise_threshold"]), color=CHOICE_COLOR, linestyle="--", alpha=0.45)
        bottom.plot(t, confidence, color=CONFIDENCE_COLOR, linewidth=1.8)
        bottom.axhline(float(rows[0]["confidence_familywise_threshold"]), color=CONFIDENCE_COLOR, linestyle="--", alpha=0.45)
        for ax in (top, bottom):
            ax.axvline(0, color="black", linewidth=0.8)
            style_axis(ax)
        top.set(ylabel="AUC", title=event_titles[event])
        bottom.set(xlabel="relative time (ms)", ylabel="r")
        plt.setp(top.get_xticklabels(), visible=False)

    # Inset overlays occupy unused top-row cells 3 and 4.
    ax_g = fig.add_subplot(outer[0, 2])
    panel_label(ax_g, "G")
    x = np.arange(len(SEEDS))
    width = 0.34
    confidence_beta = [float(row["confidence_beta_after_choice_control"]) for row in data["specificity"]]
    choice_beta = [float(row["absolute_choice_beta"]) for row in data["specificity"]]
    ax_g.bar(x - width / 2, confidence_beta, width, label="confidence β", color=CONFIDENCE_COLOR)
    ax_g.bar(x + width / 2, choice_beta, width, label="absolute choice-axis β", color=CHOICE_COLOR)
    ax_g.axhline(0, color=GRAY, linewidth=1)
    ax_g.set(xticks=x, xticklabels=[f"seed {seed}" for seed in SEEDS], ylabel="fixed-effect coefficient", title="Stage 3 specificity control")
    ax_g.legend(frameon=False, fontsize=8)
    style_axis(ax_g)

    ax_h = fig.add_subplot(outer[0, 3])
    panel_label(ax_h, "H")
    ax_h.axis("off")
    ax_h.set_title("Noise-free control", pad=12)
    ax_h.text(0.5, 0.72, "Exact repeatability", ha="center", va="center", fontsize=20, color=SEED_COLORS[7], weight="bold")
    ax_h.text(0.5, 0.42, "all seeds\nchoice consistency = 1.000\nmax state difference = 0\nmax output difference = 0", ha="center", va="center", fontsize=12, linespacing=1.5)

    fig.suptitle("Supplementary frozen-analysis overview", fontsize=19, weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"supplementary_figure.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def results_summary_text(key_results):
    key = by_seed(key_results)
    return f"""# Final supervised-RNN results

## 1. Supervised RNN reproduces confidence-dependent wagering behavior

Three categorical recurrent networks were evaluated on the same 150 held-out TEST trials. No-Sure direction accuracy was {key[7]['test_no_sure_direction_accuracy']:.3f}, {key[8]['test_no_sure_direction_accuracy']:.3f}, and {key[9]['test_no_sure_direction_accuracy']:.3f} for seeds 7–9. When Sure was offered, the networks chose it on {key[7]['test_p_sure_given_offered']:.3f}, {key[8]['test_p_sure_given_offered']:.3f}, and {key[9]['test_p_sure_given_offered']:.3f} of trials. Hard P(Sure) decreased from 0.385–0.692 at zero coherence to zero at coherence ≥0.064. Predicted Sure probability was negatively correlated with empirical P(correct) (r = −0.548 to −0.656) and internal margin (r = −0.726 to −0.802). Thus wagering was confidence-dependent and non-degenerate, but its quantitative policy did not exactly recover the teacher.

## 2. Recurrent population activity contains separable choice- and confidence-related dimensions

Stage 1 used deterministic pre-go states and TRAIN-only linear models. Choice-decoder cross-validation accuracy was {key[7]['stage1_choice_cv_accuracy']:.3f}–{key[8]['stage1_choice_cv_accuracy']:.3f}. Confidence decoding reached CV Pearson r = {key[9]['stage1_confidence_cv_pearson_r']:.3f}–{key[7]['stage1_confidence_cv_pearson_r']:.3f} and held-out r = {key[9]['stage1_confidence_test_pearson_r']:.3f}–{key[7]['stage1_confidence_test_pearson_r']:.3f}. Absolute choice–confidence axis cosine was only {key[8]['stage1_choice_confidence_absolute_cosine']:.3f}–{key[7]['stage1_choice_confidence_absolute_cosine']:.3f}. The independently TRAIN-defined confidence projection was lower for later Sure than Direction choices in every seed. Population activity therefore contained approximately distinct, low-overlap choice- and confidence-related linear dimensions; the axes were not mathematically orthogonalized and the association does not establish causal use.

## 3. Confidence information emerges during evidence accumulation before wager availability

Motion-onset-aligned Stage-2 decoding used a prospective 50-ms sustained-run criterion and a family-wise permutation threshold. Operational choice onsets were {key[7]['stage2_choice_operational_onset_ms']}, {key[8]['stage2_choice_operational_onset_ms']}, and {key[9]['stage2_choice_operational_onset_ms']} ms; confidence onsets were {key[7]['stage2_confidence_operational_onset_ms']}, {key[8]['stage2_confidence_operational_onset_ms']}, and {key[9]['stage2_confidence_operational_onset_ms']} ms. Immediately before Sure-target onset, the fixed confidence-axis correlation with empirical P(correct) was {key[7]['stage2_pre_ts_confidence_r']:.3f}, {key[8]['stage2_pre_ts_confidence_r']:.3f}, and {key[9]['stage2_pre_ts_confidence_r']:.3f}. Confidence-related population structure was therefore present during evidence accumulation and before the wagering option appeared. These are operational information-availability estimates, not biological or causal latencies.

## 4. Endogenous confidence-state variability predicts wagering under identical sensory input

Stage 3 replayed each of 75 sure-offered TEST inputs 100 times while changing only the recurrent-noise realization. After within-stimulus centering, lower pre-target confidence projection predicted later Sure choice in all networks: seed 7 β = {key[7]['stage3_primary_beta']:.3f}, 95% bootstrap CI [{key[7]['stage3_bootstrap_ci95_low']:.3f}, {key[7]['stage3_bootstrap_ci95_high']:.3f}]; seed 8 β = {key[8]['stage3_primary_beta']:.3f} [{key[8]['stage3_bootstrap_ci95_low']:.3f}, {key[8]['stage3_bootstrap_ci95_high']:.3f}]; seed 9 β = {key[9]['stage3_primary_beta']:.3f} [{key[9]['stage3_bootstrap_ci95_low']:.3f}, {key[9]['stage3_bootstrap_ci95_high']:.3f}]. Every within-stimulus permutation p was 0.001. Accordingly, the same exact external input combined with a different recurrent-noise realization produced different internal confidence states associated with different probabilities of later Sure choice. This is a stimulus-controlled predictive association, not a causal effect.

## 5. Limitations

The student networks captured the qualitative confidence-dependent wagering pattern but did not exactly reproduce the teacher policy. Linear axes and decoding quantify information and association rather than causal implementation. Most importantly, after adding within-stimulus absolute choice-axis magnitude to the Stage-3 model, the confidence coefficient was negative for seeds 7 and 9 ({key[7]['stage3_specificity_confidence_beta']:.3f}, {key[9]['stage3_specificity_confidence_beta']:.3f}) but positive for seed 8 ({key[8]['stage3_specificity_confidence_beta']:.3f}). The primary confidence–wager association replicated across all three networks, but its specificity relative to absolute choice-axis magnitude was not uniform across seeds.

## Conclusion

A supervised recurrent neural network trained to perform confidence-dependent post-decision wagering developed approximately distinct population dimensions related to directional choice and graded confidence. Confidence-related activity emerged during evidence accumulation and was already present before the wagering option appeared. Under identical external input, endogenous fluctuations toward lower-confidence states predicted a greater probability of later Sure choice across all three trained networks.

These analyses establish robust representational and predictive associations within the trained RNN, but do not demonstrate that the identified confidence axis is a causal decision variable. Specificity relative to absolute choice-axis magnitude was not uniform across seeds.
"""


def poster_summary_text(key_results):
    key = by_seed(key_results)
    return f"""# Poster-ready supervised-RNN summary

### Question

Do recurrent networks trained for post-decision wagering develop an internal confidence representation that predicts later Sure choices beyond differences in external stimuli?

### Approach

Three frozen 50-unit categorical RNNs were evaluated on a fixed held-out split. TRAIN-defined choice and empirical-P(correct) axes were tracked through time, then each sure-offered TEST input was replayed 100 times with independent recurrent noise to isolate within-stimulus state variability.

### Key findings

- Held-out no-Sure direction accuracy was {min(row['test_no_sure_direction_accuracy'] for row in key_results):.3f}–{max(row['test_no_sure_direction_accuracy'] for row in key_results):.3f}; P(Sure | offered) was non-degenerate ({min(row['test_p_sure_given_offered'] for row in key_results):.3f}–{max(row['test_p_sure_given_offered'] for row in key_results):.3f}) and decreased with coherence.
- Population activity contained approximately distinct choice and confidence dimensions: choice CV accuracy was {min(row['stage1_choice_cv_accuracy'] for row in key_results):.3f}–{max(row['stage1_choice_cv_accuracy'] for row in key_results):.3f}, confidence held-out r was {min(row['stage1_confidence_test_pearson_r'] for row in key_results):.3f}–{max(row['stage1_confidence_test_pearson_r'] for row in key_results):.3f}, and absolute axis cosine was {min(row['stage1_choice_confidence_absolute_cosine'] for row in key_results):.3f}–{max(row['stage1_choice_confidence_absolute_cosine'] for row in key_results):.3f}.
- Confidence information became operationally decodable 80–180 ms after motion onset and remained correlated with empirical P(correct) before Sure-target onset (r = {min(row['stage2_pre_ts_confidence_r'] for row in key_results):.3f}–{max(row['stage2_pre_ts_confidence_r'] for row in key_results):.3f}).
- Under identical input, lower-confidence fluctuations predicted later Sure choice in every seed (β = {key[7]['stage3_primary_beta']:.3f}, {key[8]['stage3_primary_beta']:.3f}, {key[9]['stage3_primary_beta']:.3f}; all permutation p = 0.001).

### Conclusion

The networks developed low-overlap linear dimensions related to directional choice and graded confidence. Confidence-related population activity preceded wager availability, and endogenous movement toward lower-confidence states predicted a greater probability of later Sure wagering under identical sensory input.

### Limitation

These results are representational and predictive, not causal. The Stage-3 confidence effect remained negative after controlling absolute choice-axis magnitude in seeds 7 and 9 but not seed 8, so specificity was not uniform across networks.
"""


def build_manifest(paths, output_root):
    panel_sources = {
        "A_task_model_schematic": ["freeze_manifest", "stage2_manifest"],
        "B_behavior": ["freeze_summary"],
        "C_choice_confidence_geometry": ["stage1_seed7_states", "stage1_seed7_metadata", "stage1_summary"],
        "D_time_resolved_dynamics": ["stage2_seed7_time", "stage2_summary"],
        "E_identical_stimulus_pairs": ["stage3_seed7_repeats", "stage3_summary"],
        "F_stage3_replication": ["stage3_summary"],
        "supplementary": ["stage1_seed7_pca", "stage1_seed7_metadata", "stage2_seed7_time", "stage3_seed7_stimuli", "stage3_summary", "stage3_seed7_control", "stage3_seed8_control", "stage3_seed9_control"],
    }
    return {
        "created_unix_time": time.time(),
        "figure_script": str(Path(__file__).resolve()),
        "command": " ".join([sys.executable, *sys.argv]),
        "analysis_stages_used": ["final_supervised_freeze", "population_stage1", "population_stage2_time", "population_stage3_variability"],
        "seed_definitions": {"example": 7, "replication": [8, 9], "selection": "predetermined; no seed selected from results"},
        "source_artifacts": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()},
        "panel_to_source_mapping": {panel: [str(paths[name].resolve()) for name in names] for panel, names in panel_sources.items()},
        "output_figures": {
            name: {
                "path": str((Path(output_root) / "figures" / name).resolve()),
                "sha256": sha256(Path(output_root) / "figures" / name),
            }
            for name in ("main_figure.png", "main_figure.pdf", "supplementary_figure.png", "supplementary_figure.pdf")
        },
        "synthesis_constraints": {"new_scientific_analyses": False, "new_model_fits": False, "new_decoders": False, "new_axes": False, "network_replay": False, "network_training": False},
        "python": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")},
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }


def run(args):
    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing final summary directory: {output_root}")
    paths = load_sources(args.results_root)
    output_root.mkdir(parents=True)
    key_results = assemble_key_results(paths)
    write_csv(output_root / "key_results.csv", key_results)
    write_csv(output_root / "stage_audit.csv", stage_audit_rows())
    data = create_source_data(paths, output_root)
    main_figure(data, key_results, output_root)
    supplementary_figure(data, output_root)
    (output_root / "results_summary.md").write_text(results_summary_text(key_results), encoding="utf-8")
    (output_root / "poster_summary.md").write_text(poster_summary_text(key_results), encoding="utf-8")
    write_json(output_root / "figure_manifest.json", build_manifest(paths, output_root))
    return key_results


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/final_supervised_summary")
    return parser


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
