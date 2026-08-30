import argparse
import csv
import json
import os

import numpy as np


TIMED_DATASET = "stage9b_timed_teacher_dataset_50k_fixed_auto.npz"
ORIGINAL_DATASET = "stage9_internal_teacher_dataset_50k_auto.npz"


def default_validation_metrics():
    return [
        "n_sure_offered",
        "n_sure_choice",
        "corr_sure_axis_with_sure_residual_after_evidence",
        "r2_sure_axis_explains_residual_p_sure_after_evidence",
        "corr_psure_vs_evidence_axis",
        "r2_evidence_axis_explains_p_sure",
        "pre_ts_leakage.corr_pre_ts_sure_axis_with_sure_choice",
        "pre_ts_leakage.pre_ts_sure_axis_mean_difference",
        "pre_ts_leakage.post_ts_sure_axis_mean_difference",
        "deterministic_repeated.all_repeats_same_choice_fraction",
        "deterministic_repeated.max_output_abs_diff",
        "deterministic_repeated.max_state_abs_diff",
        "population_dynamics.sure_axis_pre_event_mean_difference",
        "population_dynamics.sure_axis_post_event_mean_difference",
        "population_dynamics.sure_axis_peak_abs_difference",
    ]


def nested_get(payload, dotted_key, default=np.nan):
    current = payload
    for part in str(dotted_key).split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def metric_stats(values):
    values = np.asarray([float_or_nan(value) for value in values], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def aggregate_seed_summaries(
    summaries,
    metrics=None,
    min_residual_corr=0.45,
    max_pre_ts_corr=0.35,
    min_deterministic_stability=0.999,
):
    metrics = list(default_validation_metrics() if metrics is None else metrics)
    required_finite_metrics = [
        "corr_sure_axis_with_sure_residual_after_evidence",
        "pre_ts_leakage.corr_pre_ts_sure_axis_with_sure_choice",
        "deterministic_repeated.all_repeats_same_choice_fraction",
    ]
    per_seed = []
    for item in summaries:
        row = {"seed": int(item["seed"])}
        for metric in metrics:
            row[metric] = nested_get(item, metric)
        per_seed.append(row)

    metric_summary = {
        metric: metric_stats([row.get(metric, np.nan) for row in per_seed])
        for metric in metrics
    }
    residual_corr_min = metric_summary["corr_sure_axis_with_sure_residual_after_evidence"]["min"]
    pre_ts_corr_max = metric_summary["pre_ts_leakage.corr_pre_ts_sure_axis_with_sure_choice"]["max"]
    deterministic_stability_min = metric_summary[
        "deterministic_repeated.all_repeats_same_choice_fraction"
    ]["min"]
    missing_required = []
    for row in per_seed:
        for metric in required_finite_metrics:
            if not np.isfinite(float_or_nan(row.get(metric, np.nan))):
                missing_required.append({"seed": int(row["seed"]), "metric": metric})
    criteria = {
        "all_required_metrics_finite": {
            "required_metrics": required_finite_metrics,
            "missing": missing_required,
            "pass": len(missing_required) == 0,
        },
        "min_residual_corr": {
            "threshold": float(min_residual_corr),
            "observed": residual_corr_min,
            "pass": bool(np.isfinite(residual_corr_min) and residual_corr_min >= float(min_residual_corr)),
        },
        "max_pre_ts_corr": {
            "threshold": float(max_pre_ts_corr),
            "observed": pre_ts_corr_max,
            "pass": bool(np.isfinite(pre_ts_corr_max) and pre_ts_corr_max <= float(max_pre_ts_corr)),
        },
        "min_deterministic_stability": {
            "threshold": float(min_deterministic_stability),
            "observed": deterministic_stability_min,
            "pass": bool(
                np.isfinite(deterministic_stability_min)
                and deterministic_stability_min >= float(min_deterministic_stability)
            ),
        },
    }
    return {
        "n_seeds": int(len(per_seed)),
        "pass": bool(all(item["pass"] for item in criteria.values())),
        "criteria": criteria,
        "metrics": metric_summary,
        "per_seed": per_seed,
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_per_seed_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_aggregate_outputs(output_dir, aggregate):
    os.makedirs(output_dir, exist_ok=True)
    return {
        "summary_json": write_json(
            os.path.join(output_dir, "multiseed_validation_summary.json"),
            aggregate,
        ),
        "per_seed_csv": write_per_seed_csv(
            os.path.join(output_dir, "multiseed_validation_per_seed.csv"),
            aggregate["per_seed"],
        ),
    }


def load_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_seed_outputs(summary_path):
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    root, _ = os.path.splitext(summary_path)
    summary["pre_ts_leakage"] = load_json_if_exists(f"{root}_pre_ts_leakage_summary.json")
    summary["deterministic_repeated"] = load_json_if_exists(
        f"{root}_deterministic_repeated_stimulus_summary.json"
    )
    summary["population_dynamics"] = load_json_if_exists(f"{root}_population_dynamics_summary.json")
    return summary


def build_seed_summary_path(output_dir, model_name, seed):
    return os.path.join(output_dir, f"{model_name}_seed{int(seed)}_summary.json")


def run_seed(model_name, seed, output_dir, args):
    from sure_target_stage9_internal_readout import main as run_student

    dataset = TIMED_DATASET if model_name == "timed" else ORIGINAL_DATASET
    summary_path = build_seed_summary_path(output_dir, model_name, seed)
    argv = [
        "--dataset",
        dataset,
        "--summary",
        summary_path,
        "--seed",
        str(int(seed)),
        "--training-iters",
        str(int(args.training_iters)),
        "--loss-epoch",
        str(int(args.loss_epoch)),
        "--n-batch",
        str(int(args.n_batch)),
        "--n-eval-batches",
        str(int(args.n_eval_batches)),
        "--repeated-n-stimuli",
        str(int(args.repeated_n_stimuli)),
        "--repeated-n-repeats",
        str(int(args.repeated_n_repeats)),
        "--population-pre-window",
        str(int(args.population_pre_window)),
        "--population-post-window",
        str(int(args.population_post_window)),
    ]
    run_student(argv)
    loaded = load_seed_outputs(summary_path)
    loaded["seed"] = int(seed)
    loaded["model_name"] = model_name
    loaded["summary_path"] = os.path.abspath(summary_path)
    return loaded


def parse_seeds(seed_text):
    return [int(item.strip()) for item in str(seed_text).split(",") if item.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run multi-seed RDM timed/original validation.")
    parser.add_argument("--model", choices=["timed", "original"], default="timed")
    parser.add_argument("--seeds", default="7,8,9")
    parser.add_argument("--output-dir", default="multiseed_validation_timed")
    parser.add_argument("--training-iters", type=int, default=50000)
    parser.add_argument("--loss-epoch", type=int, default=1000)
    parser.add_argument("--n-batch", type=int, default=50)
    parser.add_argument("--n-eval-batches", type=int, default=8)
    parser.add_argument("--repeated-n-stimuli", type=int, default=24)
    parser.add_argument("--repeated-n-repeats", type=int, default=5)
    parser.add_argument("--population-pre-window", type=int, default=60)
    parser.add_argument("--population-post-window", type=int, default=40)
    parser.add_argument("--min-residual-corr", type=float, default=0.45)
    parser.add_argument("--max-pre-ts-corr", type=float, default=0.35)
    parser.add_argument("--min-deterministic-stability", type=float, default=0.999)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    os.makedirs(args.output_dir, exist_ok=True)
    summaries = []
    for seed in seeds:
        summary_path = build_seed_summary_path(args.output_dir, args.model, seed)
        if args.aggregate_only:
            loaded = load_seed_outputs(summary_path)
            loaded["seed"] = int(seed)
            loaded["model_name"] = args.model
            loaded["summary_path"] = os.path.abspath(summary_path)
            summaries.append(loaded)
        else:
            summaries.append(run_seed(args.model, seed, args.output_dir, args))

    aggregate = aggregate_seed_summaries(
        summaries,
        min_residual_corr=args.min_residual_corr,
        max_pre_ts_corr=args.max_pre_ts_corr,
        min_deterministic_stability=args.min_deterministic_stability,
    )
    paths = write_aggregate_outputs(args.output_dir, aggregate)
    print(json.dumps({"outputs": paths, "aggregate": aggregate}, indent=2), flush=True)
    return aggregate


if __name__ == "__main__":
    main()
