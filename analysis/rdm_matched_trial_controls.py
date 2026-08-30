import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRIAL_CSV = "stage9b_timed_readout_50k_fixed_auto_summary_trials.csv"


BOOL_FIELDS = {
    "sure_available",
    "chose_sure",
    "chose_direction",
    "correct",
    "waived_sure",
}


FLOAT_FIELDS = {
    "coh",
    "confidence_near_ts",
    "evidence_axis",
    "sure_axis",
    "sensory_margin",
    "mean_sensory_margin",
    "teacher_internal_margin",
    "teacher_sure_strength",
    "external_sensory_margin",
    "external_sure_strength",
}


INT_FIELDS = {
    "trial_index",
    "dataset_index",
    "dir_choice",
    "choice",
}


MATCH_FIELD_ORDER = [
    "match_type",
    "coh",
    "match_key",
    "match_delta",
    "offered_trial_index",
    "offered_choice_type",
    "offered_correct",
    "offered_evidence_axis",
    "offered_sure_axis",
    "offered_confidence_near_ts",
    "matched_no_sure_trial_index",
    "matched_no_sure_correct",
    "matched_no_sure_evidence_axis",
    "matched_no_sure_sure_axis",
    "matched_no_sure_confidence_near_ts",
    "sure_trial_index",
    "waived_trial_index",
    "sure_axis_delta_sure_minus_waived",
    "confidence_delta_sure_minus_waived",
]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed


def _parse_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def load_trial_records(trial_csv):
    with open(trial_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    records = []
    for row in rows:
        record = dict(row)
        for key in BOOL_FIELDS:
            if key in record:
                record[key] = _parse_bool(record[key])
        for key in FLOAT_FIELDS:
            if key in record:
                record[key] = _parse_float(record[key])
        for key in INT_FIELDS:
            if key in record:
                record[key] = _parse_int(record[key])
        records.append(record)
    return records


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def _mean(values):
    values = [float(value) for value in values if _finite(value)]
    if not values:
        return None
    return float(np.mean(values))


def _fraction(records, key):
    if not records:
        return None
    return float(np.mean([bool(record.get(key, False)) for record in records]))


def compute_delta_inference(deltas, n_bootstrap=1000, n_permutations=1000, seed=7):
    values = np.asarray([float(value) for value in deltas if _finite(value)], dtype=float)
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "bootstrap_ci95_low": None,
            "bootstrap_ci95_high": None,
            "permutation_p_two_sided": None,
        }

    rng = np.random.RandomState(int(seed))
    mean_value = float(np.mean(values))
    boot_means = []
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means.append(float(np.mean(sample)))

    abs_observed = abs(mean_value)
    perm_extreme = 0
    for _ in range(int(n_permutations)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=values.size, replace=True)
        perm_mean = float(np.mean(values * signs))
        if abs(perm_mean) >= abs_observed:
            perm_extreme += 1

    return {
        "n": int(values.size),
        "mean": mean_value,
        "bootstrap_ci95_low": float(np.percentile(boot_means, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(boot_means, 97.5)),
        "permutation_p_two_sided": float((perm_extreme + 1) / (int(n_permutations) + 1)),
    }


def _nearest(record, candidates, match_key):
    if not _finite(record.get(match_key)):
        return None, None
    usable = [candidate for candidate in candidates if _finite(candidate.get(match_key))]
    if not usable:
        return None, None
    value = float(record[match_key])
    best = min(usable, key=lambda candidate: abs(float(candidate[match_key]) - value))
    return best, abs(float(best[match_key]) - value)


def _choice_type(record):
    if record.get("chose_sure"):
        return "sure"
    if record.get("chose_direction"):
        return "direction"
    return "other"


def build_matched_controls(records, match_key="evidence_axis"):
    offered = [record for record in records if record.get("sure_available")]
    no_sure = [record for record in records if not record.get("sure_available")]
    sure_chosen = [record for record in offered if record.get("chose_sure")]
    waived = [record for record in offered if record.get("waived_sure")]

    no_sure_by_coh = {}
    waived_by_coh = {}
    for record in no_sure:
        no_sure_by_coh.setdefault(record.get("coh"), []).append(record)
    for record in waived:
        waived_by_coh.setdefault(record.get("coh"), []).append(record)

    offered_matches = []
    for record in offered:
        matched, delta = _nearest(record, no_sure_by_coh.get(record.get("coh"), []), match_key)
        if matched is None:
            continue
        offered_matches.append(
            {
                "match_type": "offered_to_no_sure",
                "coh": record.get("coh"),
                "match_key": match_key,
                "match_delta": delta,
                "offered_trial_index": record.get("trial_index"),
                "offered_choice_type": _choice_type(record),
                "offered_correct": bool(record.get("correct")),
                "offered_evidence_axis": record.get("evidence_axis"),
                "offered_sure_axis": record.get("sure_axis"),
                "offered_confidence_near_ts": record.get("confidence_near_ts"),
                "matched_no_sure_trial_index": matched.get("trial_index"),
                "matched_no_sure_correct": bool(matched.get("correct")),
                "matched_no_sure_evidence_axis": matched.get("evidence_axis"),
                "matched_no_sure_sure_axis": matched.get("sure_axis"),
                "matched_no_sure_confidence_near_ts": matched.get("confidence_near_ts"),
            }
        )

    sure_to_waived_matches = []
    for record in sure_chosen:
        matched, delta = _nearest(record, waived_by_coh.get(record.get("coh"), []), match_key)
        if matched is None:
            continue
        sure_axis_delta = (
            float(record["sure_axis"]) - float(matched["sure_axis"])
            if _finite(record.get("sure_axis")) and _finite(matched.get("sure_axis"))
            else math.nan
        )
        confidence_delta = (
            float(record["confidence_near_ts"]) - float(matched["confidence_near_ts"])
            if _finite(record.get("confidence_near_ts"))
            and _finite(matched.get("confidence_near_ts"))
            else math.nan
        )
        sure_to_waived_matches.append(
            {
                "match_type": "sure_to_waived",
                "coh": record.get("coh"),
                "match_key": match_key,
                "match_delta": delta,
                "sure_trial_index": record.get("trial_index"),
                "waived_trial_index": matched.get("trial_index"),
                "sure_axis_delta_sure_minus_waived": sure_axis_delta,
                "confidence_delta_sure_minus_waived": confidence_delta,
                "offered_evidence_axis": record.get("evidence_axis"),
                "matched_no_sure_evidence_axis": matched.get("evidence_axis"),
            }
        )

    return {
        "offered_to_no_sure_matches": offered_matches,
        "sure_to_waived_matches": sure_to_waived_matches,
    }


def summarize_by_coherence(records):
    rows = []
    coherences = sorted({record.get("coh") for record in records if _finite(record.get("coh"))})
    for coh in coherences:
        in_coh = [record for record in records if record.get("coh") == coh]
        offered = [record for record in in_coh if record.get("sure_available")]
        no_sure = [record for record in in_coh if not record.get("sure_available")]
        waived = [record for record in offered if record.get("waived_sure")]
        rows.append(
            {
                "coh": float(coh),
                "n_trials": int(len(in_coh)),
                "n_sure_offered": int(len(offered)),
                "p_sure": _fraction(offered, "chose_sure"),
                "no_sure_accuracy": _fraction(no_sure, "correct"),
                "waived_accuracy": _fraction(waived, "correct"),
                "mean_evidence_axis_offered": _mean([record.get("evidence_axis") for record in offered]),
                "mean_evidence_axis_no_sure": _mean([record.get("evidence_axis") for record in no_sure]),
            }
        )
    return rows


def summarize_matched_controls(records, controls, match_key="evidence_axis"):
    offered = [record for record in records if record.get("sure_available")]
    no_sure = [record for record in records if not record.get("sure_available")]
    waived = [record for record in offered if record.get("waived_sure")]
    sure_chosen = [record for record in offered if record.get("chose_sure")]
    offered_matches = controls.get("offered_to_no_sure_matches", [])
    sure_to_waived = controls.get("sure_to_waived_matches", [])
    waived_matches = [
        match for match in offered_matches if match.get("offered_choice_type") == "direction"
    ]
    sure_axis_deltas = [match.get("sure_axis_delta_sure_minus_waived") for match in sure_to_waived]
    confidence_deltas = [
        match.get("confidence_delta_sure_minus_waived") for match in sure_to_waived
    ]

    return {
        "n_trials": int(len(records)),
        "n_sure_offered": int(len(offered)),
        "n_sure_choice": int(len(sure_chosen)),
        "n_no_sure_trials": int(len(no_sure)),
        "n_waived_sure": int(len(waived)),
        "match_key": match_key,
        "n_offered_to_no_sure_matches": int(len(offered_matches)),
        "n_sure_to_waived_matches": int(len(sure_to_waived)),
        "mean_offered_to_no_sure_match_delta": _mean(
            [match.get("match_delta") for match in offered_matches]
        ),
        "overall_p_sure_offered": _fraction(offered, "chose_sure"),
        "no_sure_accuracy": _fraction(no_sure, "correct"),
        "waived_accuracy": _fraction(waived, "correct"),
        "matched_no_sure_accuracy_for_waived": _fraction(
            [{"correct": match.get("matched_no_sure_correct")} for match in waived_matches],
            "correct",
        ),
        "mean_sure_axis_sure_choice": _mean([record.get("sure_axis") for record in sure_chosen]),
        "mean_sure_axis_waived": _mean([record.get("sure_axis") for record in waived]),
        "mean_sure_axis_delta_sure_minus_waived": _mean(
            sure_axis_deltas
        ),
        "mean_confidence_delta_sure_minus_waived": _mean(
            confidence_deltas
        ),
        "sure_axis_delta_inference": compute_delta_inference(sure_axis_deltas, seed=7),
        "confidence_delta_inference": compute_delta_inference(confidence_deltas, seed=11),
        "coherence_summary": summarize_by_coherence(records),
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


def _ordered_fields(rows):
    fields = [field for field in MATCH_FIELD_ORDER if any(field in row for row in rows)]
    fields.extend(sorted({key for row in rows for key in row.keys()} - set(fields)))
    return fields


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = _ordered_fields(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_clean(rows))
    return path


def _flatten_matches(controls):
    return controls.get("offered_to_no_sure_matches", []) + controls.get(
        "sure_to_waived_matches", []
    )


def plot_matched_controls(path, summary):
    coh_rows = summary.get("coherence_summary", [])
    coh = np.asarray([row["coh"] for row in coh_rows], dtype=float)
    p_sure = np.asarray([row["p_sure"] for row in coh_rows], dtype=float)
    no_sure_acc = np.asarray([row["no_sure_accuracy"] for row in coh_rows], dtype=float)
    waived_acc = np.asarray([row["waived_accuracy"] for row in coh_rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle("Matched-trial controls for sure behavior", fontsize=14, fontweight="bold")

    axes[0].plot(coh, p_sure, marker="o", lw=2.2, color="#1F77B4")
    axes[0].set_title("P(sure) by coherence")
    axes[0].set_xlabel("Motion coherence")
    axes[0].set_ylabel("P(sure | offered)")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_xscale("symlog", linthresh=0.032)
    axes[0].grid(True, alpha=0.25)

    labels = ["Waived sure", "Matched no-sure"]
    values = [
        summary.get("waived_accuracy"),
        summary.get("matched_no_sure_accuracy_for_waived"),
    ]
    axes[1].bar(labels, values, color=["#54A24B", "#4C78A8"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Accuracy after evidence matching")
    axes[1].set_ylabel("Direction accuracy")
    axes[1].tick_params(axis="x", labelrotation=15)
    axes[1].grid(True, axis="y", alpha=0.25)

    labels = ["Sure chosen", "Waived sure"]
    values = [
        summary.get("mean_sure_axis_sure_choice"),
        summary.get("mean_sure_axis_waived"),
    ]
    axes[2].bar(labels, values, color=["#E45756", "#54A24B"])
    axes[2].set_title("Sure axis within offered trials")
    axes[2].set_ylabel("Mean sure-axis projection")
    axes[2].tick_params(axis="x", labelrotation=15)
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.text(
        0.02,
        0.01,
        f"Matching: same coherence, nearest {summary.get('match_key')}.",
        fontsize=8,
        color="#555555",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_matched_control_outputs(output_dir, figure_dir, summary, controls):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    return {
        "summary_json": _write_json(
            os.path.join(output_dir, "matched_trial_controls_summary.json"), summary
        ),
        "matches_csv": _write_csv(
            os.path.join(output_dir, "matched_trial_controls.csv"),
            _flatten_matches(controls),
        ),
        "figure": plot_matched_controls(
            os.path.join(figure_dir, "fig6_matched_trial_controls.png"), summary
        ),
    }


def build_report(trial_csv, match_key="evidence_axis"):
    records = load_trial_records(trial_csv)
    controls = build_matched_controls(records, match_key=match_key)
    summary = summarize_matched_controls(records, controls, match_key=match_key)
    return summary, controls


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build matched-trial controls for RDM sure behavior.")
    parser.add_argument("--trials", default=DEFAULT_TRIAL_CSV)
    parser.add_argument("--match-key", default="evidence_axis")
    parser.add_argument("--output-dir", default="paper_exports")
    parser.add_argument("--figure-dir", default="paper_figures")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    summary, controls = build_report(args.trials, match_key=args.match_key)
    paths = write_matched_control_outputs(args.output_dir, args.figure_dir, summary, controls)
    print(json.dumps({"outputs": paths, "summary": summary}, indent=2), flush=True)
    return paths


if __name__ == "__main__":
    main()
