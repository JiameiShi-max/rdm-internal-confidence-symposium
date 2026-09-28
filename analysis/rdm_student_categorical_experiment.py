import argparse
import csv
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def floats(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def bools(rows, key):
    return np.asarray([str(row[key]).lower() in ("true", "1") for row in rows], dtype=bool)


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 2 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def normalize_legacy_actions(actions):
    """Project independent nonnegative scores to the simplex for CE diagnostics."""
    projected = np.maximum(np.asarray(actions, dtype=float), 0.0)
    totals = np.sum(projected, axis=1, keepdims=True)
    empty = totals[:, 0] <= 1e-12
    projected[empty] = 1.0
    totals = np.sum(projected, axis=1, keepdims=True)
    return projected / totals


def distribution_metrics(target, predicted):
    target = np.asarray(target, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    eps = 1e-12
    ce = -np.sum(target * np.log(np.maximum(predicted, eps)), axis=1)
    kl_terms = np.where(
        target > 0,
        target * np.log(np.maximum(target, eps) / np.maximum(predicted, eps)),
        0.0,
    )
    return {
        "cross_entropy": float(np.mean(ce)),
        "kl_divergence": float(np.mean(np.sum(kl_terms, axis=1))),
        "action_distribution_mae": float(np.mean(np.abs(target - predicted))),
    }


def standard_categorical_metrics(rows):
    offered = bools(rows, "sure_available")
    direction = floats(rows, "dir_choice").astype(int)
    target_left = floats(rows, "target_left")
    target_right = floats(rows, "target_right")
    target_sure = floats(rows, "target_sure")
    pred_left = floats(rows, "predicted_left")
    pred_right = floats(rows, "predicted_right")
    pred_sure = floats(rows, "predicted_sure")
    target_correct = np.where(direction == 0, target_left, target_right)
    pred_correct = np.where(direction == 0, pred_left, pred_right)
    pred_incorrect = np.where(direction == 0, pred_right, pred_left)
    target_margin = target_sure - target_correct
    pred_margin = pred_sure - pred_correct
    choices = floats(rows, "final_argmax_choice").astype(int)
    target = np.column_stack([target_left, target_right, target_sure])
    predicted = np.column_stack([pred_left, pred_right, pred_sure])
    metrics = distribution_metrics(target, predicted)
    metrics.update(
        {
            "n_trials": int(len(rows)),
            "n_offered": int(np.sum(offered)),
            "target_sure_favoring_count": int(np.sum(offered & (target_margin > 0))),
            "target_sure_favoring_fraction": float(np.mean(target_margin[offered] > 0)),
            "predicted_sure_favoring_count": int(np.sum(offered & (pred_margin > 0))),
            "predicted_sure_favoring_fraction": float(np.mean(pred_margin[offered] > 0)),
            "hard_sure_count": int(np.sum(offered & (choices == 3))),
            "hard_p_sure_offered": float(np.mean(choices[offered] == 3)),
            "target_predicted_sure_correlation": corr(target_sure[offered], pred_sure[offered]),
            "target_predicted_action_margin_correlation": corr(
                target_margin[offered], pred_margin[offered]
            ),
            "action_margin_bias": float(np.mean(pred_margin[offered] - target_margin[offered])),
            "action_margin_sign_agreement": float(
                np.mean(np.sign(pred_margin[offered]) == np.sign(target_margin[offered]))
            ),
            "incorrect_direction_offered": float(np.mean(pred_incorrect[offered])),
            "incorrect_direction_no_sure": float(np.mean(pred_incorrect[~offered])),
            "incorrect_direction_sure_favoring": float(
                np.mean(pred_incorrect[offered & (target_margin > 0)])
            ),
            "incorrect_direction_direction_favoring": float(
                np.mean(pred_incorrect[offered & (target_margin < 0)])
            ),
            "post_go_fixation_mean": float(np.mean(floats(rows, "predicted_fixation"))),
            "post_go_fixation_mean_abs": float(np.mean(np.abs(floats(rows, "predicted_fixation")))),
            "p_left": float(np.mean(choices == 1)),
            "p_right": float(np.mean(choices == 2)),
            "p_sure": float(np.mean(choices == 3)),
        }
    )
    favored = offered & (target_margin > 0)
    metrics["sure_favoring_subset"] = {
        "n": int(np.sum(favored)),
        "mean_target_sure": float(np.mean(target_sure[favored])),
        "mean_predicted_sure": float(np.mean(pred_sure[favored])),
        "mean_predicted_correct_direction": float(np.mean(pred_correct[favored])),
        "fraction_expressed_as_sure_preference": float(np.mean(pred_margin[favored] > 0)),
    }
    metrics["soft_correlations_offered"] = {
        "teacher_sure_strength": corr(
            floats(rows, "teacher_sure_strength")[offered], pred_sure[offered]
        ),
        "empirical_p_correct": corr(floats(rows, "empirical_p_correct")[offered], pred_sure[offered]),
        "internal_margin": corr(floats(rows, "internal_margin")[offered], pred_sure[offered]),
        "coherence": corr(floats(rows, "stimulus_coherence")[offered], pred_sure[offered]),
        "stimulus_duration_ms": corr(
            floats(rows, "stimulus_duration_ms")[offered], pred_sure[offered]
        ),
    }
    arrays = {
        "offered": offered,
        "target": target,
        "predicted": predicted,
        "target_correct": target_correct,
        "pred_correct": pred_correct,
        "pred_incorrect": pred_incorrect,
        "target_margin": target_margin,
        "pred_margin": pred_margin,
        "choices": choices,
    }
    return metrics, arrays


def repeat_geometry(rows, dataset):
    info = [json.loads(str(item)) for item in dataset["trial_info_json"]]
    strengths = np.asarray(dataset["teacher_sure_strength"], dtype=float)
    target = []
    predicted = []
    raw_correct = []
    raw_incorrect = []
    raw_sure = []
    target_margin = []
    predicted_margin = []
    for row in rows:
        idx = int(row["stimulus_id"])
        direction = int(info[idx]["dir_choice"])
        sure = float(strengths[idx])
        q = np.asarray([1.0 - sure, 0.0, sure]) if direction == 0 else np.asarray([0.0, 1.0 - sure, sure])
        action = np.asarray([float(row["left_output"]), float(row["right_output"]), float(row["sure_output"])])
        correct = action[direction]
        incorrect = action[1 - direction]
        target.append(q)
        predicted.append(action)
        raw_correct.append(correct)
        raw_incorrect.append(incorrect)
        raw_sure.append(action[2])
        target_margin.append(sure - (1.0 - sure))
        predicted_margin.append(action[2] - correct)
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    return {
        "target": target,
        "predicted": predicted,
        "correct": np.asarray(raw_correct),
        "incorrect": np.asarray(raw_incorrect),
        "sure": np.asarray(raw_sure),
        "target_margin": np.asarray(target_margin),
        "predicted_margin": np.asarray(predicted_margin),
        "choice": np.asarray([int(row["choice"]) for row in rows]),
    }


def binned_soft(rows, condition, field, edges):
    offered = bools(rows, "sure_available") if "sure_available" in rows[0] else np.ones(len(rows), dtype=bool)
    values = floats(rows, field)
    predicted = floats(rows, "predicted_sure")
    target = floats(rows, "target_sure")
    result = []
    for idx in range(len(edges) - 1):
        selected = offered & (values >= edges[idx]) & (
            values <= edges[idx + 1] if idx == len(edges) - 2 else values < edges[idx + 1]
        )
        if not np.any(selected):
            continue
        result.append(
            {
                "condition": condition,
                "field": field,
                "bin_index": idx,
                "low": float(edges[idx]),
                "high": float(edges[idx + 1]),
                "center": float(np.mean(values[selected])),
                "n": int(np.sum(selected)),
                "mean_target_sure": float(np.mean(target[selected])),
                "mean_predicted_sure": float(np.mean(predicted[selected])),
            }
        )
    return result


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(output_dir, cat_rows, legacy_zero_rows, cat_arrays, legacy_repeat, cat_repeat, legacy_behavior, cat_behavior):
    os.makedirs(output_dir, exist_ok=True)
    figure_paths = []

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    labels = ["LEFT", "RIGHT", "SURE"]
    for idx, ax in enumerate(axes):
        ax.scatter(cat_arrays["target"][:, idx], cat_arrays["predicted"][:, idx], s=10, alpha=0.45)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(title=labels[idx], xlabel="target probability", ylabel="predicted probability")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    path = os.path.join(output_dir, "fig1_target_vs_predicted_actions.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for ax, title, data in [
        (axes[0], "Legacy repeat set", legacy_repeat),
        (axes[1], "Categorical repeat set", cat_repeat),
    ]:
        ax.scatter(data["target_margin"], data["predicted_margin"], s=9, alpha=0.35)
        ax.axhline(0, color="0.5", linewidth=1)
        ax.axvline(0, color="0.5", linewidth=1)
        ax.set(title=f"{title}\nr={corr(data['target_margin'], data['predicted_margin']):.3f}", xlabel="target sure − correct", ylabel="predicted sure − correct")
    path = os.path.join(output_dir, "fig2_target_vs_predicted_margin.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, [np.mean(legacy_repeat[k]) for k in ("correct", "incorrect", "sure")], width, label="Legacy raw output")
    ax.bar(x + width / 2, [np.mean(cat_repeat[k]) for k in ("correct", "incorrect", "sure")], width, label="Categorical probability")
    ax.set_xticks(x, ["correct direction", "incorrect direction", "sure"])
    ax.set_ylabel("mean post-go output")
    ax.legend(frameon=False)
    path = os.path.join(output_dir, "fig3_action_output_geometry.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    legacy_coh = {float(item["coh"]): float(item["p_sure"]) for item in legacy_behavior["coherence"]}
    cat_coh = {float(item["coh"]): float(item["p_sure"]) for item in cat_behavior["coherence"]}
    coh = sorted(set(legacy_coh) & set(cat_coh))
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    ax.plot(coh, [legacy_coh[x] for x in coh], "o--", label="Legacy hard P(sure)")
    ax.plot(coh, [cat_coh[x] for x in coh], "o-", label="Categorical hard P(sure)")
    for rows, label, style in [(legacy_zero_rows, "Legacy mean sure output", "s--"), (cat_rows, "Categorical mean sure probability", "s-")]:
        offered = bools(rows, "sure_available") if "sure_available" in rows[0] else np.ones(len(rows), dtype=bool)
        values = floats(rows, "coherence" if "coherence" in rows[0] else "stimulus_coherence")
        sure = floats(rows, "predicted_sure")
        ax.plot(coh, [np.mean(sure[offered & (values == x)]) for x in coh], style, label=label)
    ax.set(xlabel="coherence", ylabel="probability / mean sure output", ylim=(-0.02, 0.62))
    ax.legend(frameon=False, fontsize=8)
    path = os.path.join(output_dir, "fig4_p_sure_vs_coherence.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    for behavior, label, style in [(legacy_behavior, "Legacy hard P(sure)", "o--"), (cat_behavior, "Categorical hard P(sure)", "o-")]:
        ax.plot([x["center"] for x in behavior["duration"]], [x["p_sure"] for x in behavior["duration"]], style, label=label)
    duration_edges = np.asarray([400.0, 600.0, 800.0, 1000.0, 1190.0])
    for rows, label, style, field in [
        (legacy_zero_rows, "Legacy mean sure output", "s--", "stimulus_duration_ms"),
        (cat_rows, "Categorical mean sure probability", "s-", "stimulus_duration_ms"),
    ]:
        bins = binned_soft(rows, label, field, duration_edges)
        ax.plot([x["center"] for x in bins], [x["mean_predicted_sure"] for x in bins], style, label=label)
    ax.set(xlabel="stimulus duration (ms)", ylabel="probability / mean sure output", ylim=(-0.02, 0.35))
    ax.legend(frameon=False, fontsize=8)
    path = os.path.join(output_dir, "fig5_p_sure_vs_duration.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)
    return figure_paths


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--categorical-dir", default="results/model_freeze/student_categorical_seed7")
    parser.add_argument("--legacy-dir", default="results/model_freeze/multiseed")
    parser.add_argument("--dataset", default="data/model_freeze/teacher_seed7.npz")
    parser.add_argument("--zero-diagnostic", default="results/model_freeze/zero_sure_diagnostic.csv")
    args = parser.parse_args(argv)

    cat_rows = read_csv(os.path.join(args.categorical_dir, "post_go_outputs.csv"))
    legacy_zero_rows = read_csv(args.zero_diagnostic)
    cat_metrics, cat_arrays = standard_categorical_metrics(cat_rows)
    with open(os.path.join(args.categorical_dir, "summary_behavior.json"), encoding="utf-8") as handle:
        cat_behavior = json.load(handle)
    with open(os.path.join(args.legacy_dir, "timed_seed7_summary_behavior.json"), encoding="utf-8") as handle:
        legacy_behavior = json.load(handle)
    with open(os.path.join(args.categorical_dir, "summary_repeated_stimulus_summary.json"), encoding="utf-8") as handle:
        cat_repeat_summary = json.load(handle)
    with open(os.path.join(args.categorical_dir, "summary_deterministic_repeated_stimulus_summary.json"), encoding="utf-8") as handle:
        cat_det_summary = json.load(handle)
    with open(os.path.join(args.legacy_dir, "timed_seed7_summary_repeated_stimulus_summary.json"), encoding="utf-8") as handle:
        legacy_repeat_summary = json.load(handle)
    with open(os.path.join(args.legacy_dir, "timed_seed7_summary_deterministic_repeated_stimulus_summary.json"), encoding="utf-8") as handle:
        legacy_det_summary = json.load(handle)

    dataset = np.load(args.dataset, allow_pickle=True)
    legacy_repeat_rows = read_csv(os.path.join(args.legacy_dir, "timed_seed7_summary_repeated_stimulus_trials.csv"))
    cat_repeat_rows = read_csv(os.path.join(args.categorical_dir, "summary_repeated_stimulus_trials.csv"))
    legacy_repeat = repeat_geometry(legacy_repeat_rows, dataset)
    cat_repeat = repeat_geometry(cat_repeat_rows, dataset)
    legacy_repeat_distribution = distribution_metrics(
        legacy_repeat["target"], normalize_legacy_actions(legacy_repeat["predicted"])
    )
    cat_repeat_distribution = distribution_metrics(cat_repeat["target"], cat_repeat["predicted"])

    legacy_target_sure = floats(legacy_zero_rows, "target_sure")
    legacy_pred_sure = floats(legacy_zero_rows, "predicted_sure")
    legacy_target_margin = floats(legacy_zero_rows, "target_choice_margin")
    legacy_choice = floats(legacy_zero_rows, "final_argmax_choice").astype(int)
    legacy = {
        "n_offered": len(legacy_zero_rows),
        "target_sure_favoring_count": int(np.sum(legacy_target_margin > 0)),
        "target_sure_favoring_fraction": float(np.mean(legacy_target_margin > 0)),
        "predicted_sure_favoring_count": None,
        "predicted_sure_favoring_fraction": None,
        "hard_sure_count": int(np.sum(legacy_choice == 3)),
        "hard_p_sure_offered": float(np.mean(legacy_choice == 3)),
        "target_predicted_sure_correlation": corr(legacy_target_sure, legacy_pred_sure),
        "target_predicted_action_margin_correlation": corr(
            legacy_repeat["target_margin"], legacy_repeat["predicted_margin"]
        ),
        "action_margin_sign_agreement": float(
            np.mean(np.sign(legacy_repeat["target_margin"]) == np.sign(legacy_repeat["predicted_margin"]))
        ),
        "incorrect_direction_residual": float(np.mean(legacy_repeat["incorrect"])),
        "post_go_fixation_mean": None,
        "no_sure_accuracy": float(legacy_behavior["no_sure_accuracy"]),
        "waived_sure_accuracy": float(legacy_behavior["waived_sure_accuracy"]),
        "repeat_distribution_diagnostic": legacy_repeat_distribution,
    }
    categorical = dict(cat_metrics)
    categorical.update(
        {
            "no_sure_accuracy": float(cat_behavior["no_sure_accuracy"]),
            "waived_sure_accuracy": float(cat_behavior["waived_sure_accuracy"]),
            "repeat_distribution": cat_repeat_distribution,
        }
    )
    categorical["repeated_stimulus"] = {
        "stochastic_hard_sure_frequency": float(np.mean(cat_repeat["choice"] == 3)),
        "n_stimuli_with_sure_direction_switching": int(cat_repeat_summary["n_stimuli_with_choice_variability"]),
        "target_predicted_action_margin_correlation": corr(
            cat_repeat["target_margin"], cat_repeat["predicted_margin"]
        ),
        "action_margin_bias": float(
            np.mean(cat_repeat["predicted_margin"] - cat_repeat["target_margin"])
        ),
        "action_margin_sign_agreement": float(
            np.mean(np.sign(cat_repeat["target_margin"]) == np.sign(cat_repeat["predicted_margin"]))
        ),
        "incorrect_direction_residual": float(np.mean(cat_repeat["incorrect"])),
        "within_stimulus_pre_ts_margin_logit_coefficient": cat_repeat_summary["within_stimulus_pre_ts_margin_logit_coefficient"],
        "within_stimulus_pre_ts_hidden_pc1_logit_coefficient": cat_repeat_summary.get("within_stimulus_pre_ts_hidden_pc1_logit_coefficient"),
        "deterministic_all_repeats_same_choice_fraction": cat_det_summary["all_repeats_same_choice_fraction"],
        "deterministic_max_output_abs_diff": cat_det_summary["max_output_abs_diff"],
        "deterministic_max_state_abs_diff": cat_det_summary["max_state_abs_diff"],
    }
    legacy["repeated_stimulus"] = {
        "stochastic_hard_sure_frequency": float(np.mean(legacy_repeat["choice"] == 3)),
        "n_stimuli_with_sure_direction_switching": int(legacy_repeat_summary["n_stimuli_with_choice_variability"]),
        "target_predicted_action_margin_correlation": corr(
            legacy_repeat["target_margin"], legacy_repeat["predicted_margin"]
        ),
        "action_margin_bias": float(
            np.mean(legacy_repeat["predicted_margin"] - legacy_repeat["target_margin"])
        ),
        "action_margin_sign_agreement": float(
            np.mean(np.sign(legacy_repeat["target_margin"]) == np.sign(legacy_repeat["predicted_margin"]))
        ),
        "incorrect_direction_residual": float(np.mean(legacy_repeat["incorrect"])),
        "within_stimulus_pre_ts_margin_logit_coefficient": legacy_repeat_summary["within_stimulus_pre_ts_margin_logit_coefficient"],
        "deterministic_all_repeats_same_choice_fraction": legacy_det_summary["all_repeats_same_choice_fraction"],
        "deterministic_max_output_abs_diff": legacy_det_summary["max_output_abs_diff"],
        "deterministic_max_state_abs_diff": legacy_det_summary["max_state_abs_diff"],
    }

    soft_bins = []
    coherence_edges = np.asarray([-1e-12, 0.016, 0.048, 0.096, 0.192, 0.384, 0.513])
    duration_edges = np.asarray([400.0, 600.0, 800.0, 1000.0, 1190.0])
    legacy_soft_rows = [dict(row, sure_available="True") for row in legacy_zero_rows]
    for rows, condition, coh_field in [
        (legacy_soft_rows, "legacy", "coherence"),
        (cat_rows, "categorical", "stimulus_coherence"),
    ]:
        soft_bins += binned_soft(rows, condition, coh_field, coherence_edges)
        soft_bins += binned_soft(rows, condition, "stimulus_duration_ms", duration_edges)

    figures = make_figures(
        os.path.join(args.categorical_dir, "figures"),
        cat_rows,
        legacy_soft_rows,
        cat_arrays,
        legacy_repeat,
        cat_repeat,
        legacy_behavior,
        cat_behavior,
    )
    result = {
        "legacy": legacy,
        "categorical": categorical,
        "comparison_scope": {
            "standard_trials": "same first 400 sequential dataset trials; legacy lacks saved direction/fixation outputs",
            "action_geometry": "same 48 selected offered stimuli x 20 stochastic repeats",
            "legacy_distribution_projection": "clip independent scores at zero and renormalize only for CE/KL diagnostic",
        },
        "figures": [os.path.abspath(path) for path in figures],
    }
    summary_path = os.path.join(args.categorical_dir, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    write_csv(os.path.join(args.categorical_dir, "soft_structure_bins.csv"), soft_bins)
    metric_rows = []
    for metric in [
        "target_sure_favoring_fraction", "hard_p_sure_offered", "target_predicted_sure_correlation",
        "no_sure_accuracy", "waived_sure_accuracy",
    ]:
        metric_rows.append({"metric": metric, "legacy": legacy.get(metric), "categorical": categorical.get(metric)})
    metric_rows += [
        {"metric": "target_predicted_action_margin_correlation_matched_repeat", "legacy": legacy["repeated_stimulus"]["target_predicted_action_margin_correlation"], "categorical": categorical["repeated_stimulus"]["target_predicted_action_margin_correlation"]},
        {"metric": "action_margin_sign_agreement_matched_repeat", "legacy": legacy["repeated_stimulus"]["action_margin_sign_agreement"], "categorical": categorical["repeated_stimulus"]["action_margin_sign_agreement"]},
        {"metric": "incorrect_direction_residual_repeat", "legacy": legacy["incorrect_direction_residual"], "categorical": float(np.mean(cat_repeat["incorrect"]))},
        {"metric": "post_go_fixation_mean_standard", "legacy": None, "categorical": categorical["post_go_fixation_mean"]},
    ]
    write_csv(os.path.join(args.categorical_dir, "comparison_metrics.csv"), metric_rows)
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
