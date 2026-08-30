import argparse
import csv
import json
import os

import numpy as np


DEFAULT_ORIGINAL_SUMMARY = "stage9_internal_readout_50k_auto_summary.json"
DEFAULT_TIMED_SUMMARY = "stage9b_timed_readout_50k_fixed_auto_summary.json"
DEFAULT_TIMED_MULTISEED = "multiseed_validation_timed_5seed/multiseed_validation_summary.json"


def _load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sibling_json(summary_path, suffix):
    root, _ = os.path.splitext(summary_path)
    return _load_json(f"{root}_{suffix}.json")


def _float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _ratio(num, den):
    num = _float_or_none(num)
    den = _float_or_none(den)
    if num is None or den is None or den == 0.0:
        return None
    return float(num / den)


def _metric_stats(aggregate, metric):
    stats = aggregate.get("metrics", {}).get(metric, {})
    return {
        "mean": _float_or_none(stats.get("mean")),
        "std": _float_or_none(stats.get("std")),
        "min": _float_or_none(stats.get("min")),
        "max": _float_or_none(stats.get("max")),
    }


def _load_single_bundle(summary_path):
    summary = _load_json(summary_path)
    return {
        "summary": summary,
        "pre_ts": _sibling_json(summary_path, "pre_ts_leakage_summary"),
        "deterministic": _sibling_json(summary_path, "deterministic_repeated_stimulus_summary"),
        "population": _sibling_json(summary_path, "population_dynamics_summary"),
    }


def summarize_single_model(model_name, summary_path):
    bundle = _load_single_bundle(summary_path)
    summary = bundle["summary"]
    pre_ts = bundle["pre_ts"]
    deterministic = bundle["deterministic"]
    population = bundle["population"]
    n_trials = summary.get("n_trials")
    n_offered = summary.get("n_sure_offered")
    n_sure = summary.get("n_sure_choice")
    return {
        "model": model_name,
        "summary_path": os.path.abspath(summary_path),
        "dataset": os.path.basename(str(summary.get("dataset", ""))),
        "n_trials": n_trials,
        "n_sure_offered": n_offered,
        "n_sure_choice": n_sure,
        "sure_offered_rate": _ratio(n_offered, n_trials),
        "sure_choice_rate_offered": _ratio(n_sure, n_offered),
        "residual_corr": _float_or_none(
            summary.get("corr_sure_axis_with_sure_residual_after_evidence")
        ),
        "residual_r2": _float_or_none(
            summary.get("r2_sure_axis_explains_residual_p_sure_after_evidence")
        ),
        "evidence_r2_p_sure": _float_or_none(summary.get("r2_evidence_axis_explains_p_sure")),
        "pre_ts_corr": _float_or_none(pre_ts.get("corr_pre_ts_sure_axis_with_sure_choice")),
        "deterministic_stability": _float_or_none(
            deterministic.get("all_repeats_same_choice_fraction")
        ),
        "population_pre_event_diff": _float_or_none(
            population.get("sure_axis_pre_event_mean_difference")
        ),
        "population_post_event_diff": _float_or_none(
            population.get("sure_axis_post_event_mean_difference")
        ),
        "population_peak_abs_diff": _float_or_none(population.get("sure_axis_peak_abs_difference")),
    }


def summarize_multiseed(model_name, aggregate_path):
    aggregate = _load_json(aggregate_path)
    residual = _metric_stats(aggregate, "corr_sure_axis_with_sure_residual_after_evidence")
    residual_r2 = _metric_stats(
        aggregate, "r2_sure_axis_explains_residual_p_sure_after_evidence"
    )
    evidence_r2 = _metric_stats(aggregate, "r2_evidence_axis_explains_p_sure")
    pre_ts = _metric_stats(aggregate, "pre_ts_leakage.corr_pre_ts_sure_axis_with_sure_choice")
    deterministic = _metric_stats(
        aggregate, "deterministic_repeated.all_repeats_same_choice_fraction"
    )
    n_sure_choice = _metric_stats(aggregate, "n_sure_choice")
    return {
        "model": model_name,
        "aggregate_path": os.path.abspath(aggregate_path),
        "n_seeds": aggregate.get("n_seeds"),
        "pass": aggregate.get("pass"),
        "n_sure_choice_mean": n_sure_choice["mean"],
        "n_sure_choice_std": n_sure_choice["std"],
        "residual_corr_mean": residual["mean"],
        "residual_corr_std": residual["std"],
        "residual_corr_min": residual["min"],
        "residual_corr_max": residual["max"],
        "residual_r2_mean": residual_r2["mean"],
        "residual_r2_std": residual_r2["std"],
        "evidence_r2_p_sure_mean": evidence_r2["mean"],
        "evidence_r2_p_sure_std": evidence_r2["std"],
        "pre_ts_corr_mean": pre_ts["mean"],
        "pre_ts_corr_std": pre_ts["std"],
        "pre_ts_corr_max": pre_ts["max"],
        "deterministic_stability_min": deterministic["min"],
    }


def build_report_payload(
    original_summary=DEFAULT_ORIGINAL_SUMMARY,
    timed_summary=DEFAULT_TIMED_SUMMARY,
    timed_multiseed_summary=DEFAULT_TIMED_MULTISEED,
):
    single_rows = []
    if original_summary and os.path.exists(original_summary):
        single_rows.append(summarize_single_model("original_single_seed", original_summary))
    if timed_summary and os.path.exists(timed_summary):
        single_rows.append(summarize_single_model("timed_single_seed", timed_summary))

    multiseed_rows = []
    if timed_multiseed_summary and os.path.exists(timed_multiseed_summary):
        multiseed_rows.append(summarize_multiseed("timed_5seed", timed_multiseed_summary))

    return {
        "single_model_rows": single_rows,
        "multiseed_rows": multiseed_rows,
    }


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt(value, digits=3):
    value = _float_or_none(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _markdown_table(rows, columns):
    lines = []
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(_fmt(value))
            else:
                values.append(str(value) if value is not None else "NA")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_markdown(payload):
    lines = [
        "# RDM internal-confidence 当前验证汇总",
        "",
        "## Single model 对照",
    ]
    single_columns = [
        ("model", "model"),
        ("n_trials", "n"),
        ("sure_offered_rate", "sure offered"),
        ("sure_choice_rate_offered", "sure choice/offered"),
        ("residual_corr", "residual corr"),
        ("residual_r2", "residual R2"),
        ("evidence_r2_p_sure", "evidence R2 P(sure)"),
        ("pre_ts_corr", "pre-TS corr"),
        ("deterministic_stability", "deterministic stability"),
        ("population_peak_abs_diff", "population peak diff"),
    ]
    lines.append(_markdown_table(payload.get("single_model_rows", []), single_columns))
    lines.extend(["", "## Multi-seed 稳健性"])
    multiseed_columns = [
        ("model", "model"),
        ("n_seeds", "seeds"),
        ("pass", "strict pass"),
        ("residual_corr_mean", "residual corr mean"),
        ("residual_corr_std", "residual corr std"),
        ("residual_corr_min", "residual corr min"),
        ("residual_r2_mean", "residual R2 mean"),
        ("pre_ts_corr_mean", "pre-TS corr mean"),
        ("pre_ts_corr_max", "pre-TS corr max"),
        ("deterministic_stability_min", "deterministic stability min"),
    ]
    lines.append(_markdown_table(payload.get("multiseed_rows", []), multiseed_columns))
    lines.extend(
        [
            "",
            "## 汇报口径",
            "",
            "- timed 版作为主模型：timing 更严格，sure offered 接近 50%，适合承载主要结论。",
            "- original 版作为对照：sure-axis signal 更强，但 timing/sure availability confound 更多。",
            "- 当前 timed 5-seed 显示 residual sure signal 稳定；pre-TS leakage 有 seed 依赖，应作为限制和后续优化点报告。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report_outputs(output_dir, payload):
    os.makedirs(output_dir, exist_ok=True)
    report_md = os.path.join(output_dir, "rdm_current_validation_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(_build_markdown(payload))
    return {
        "report_md": report_md,
        "report_json": _write_json(os.path.join(output_dir, "rdm_current_validation_report.json"), payload),
        "single_model_csv": _write_csv(
            os.path.join(output_dir, "rdm_single_model_comparison.csv"),
            payload.get("single_model_rows", []),
        ),
        "multiseed_csv": _write_csv(
            os.path.join(output_dir, "rdm_multiseed_summary.csv"),
            payload.get("multiseed_rows", []),
        ),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build report-ready RDM validation tables.")
    parser.add_argument("--original-summary", default=DEFAULT_ORIGINAL_SUMMARY)
    parser.add_argument("--timed-summary", default=DEFAULT_TIMED_SUMMARY)
    parser.add_argument("--timed-multiseed-summary", default=DEFAULT_TIMED_MULTISEED)
    parser.add_argument("--output-dir", default="analysis_reports")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    payload = build_report_payload(
        original_summary=args.original_summary,
        timed_summary=args.timed_summary,
        timed_multiseed_summary=args.timed_multiseed_summary,
    )
    paths = write_report_outputs(args.output_dir, payload)
    print(json.dumps({"outputs": paths, "report": payload}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
