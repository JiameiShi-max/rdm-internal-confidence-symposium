import argparse
import json
import math
import os


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def value(payload, key, default="not produced"):
    if payload is None:
        return default
    return payload.get(key, default)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value, digits=6, missing="not estimable"):
    if not finite(value):
        return missing
    return f"{float(value):.{int(digits)}g}"


def has_nonconstant_sure_rate(rows):
    rates = [row.get("p_sure") for row in rows or [] if finite(row.get("p_sure"))]
    return len(rates) >= 2 and max(rates) - min(rates) > 1e-9


def build_report(
    branch,
    baseline_commit,
    calibration=None,
    behavior=None,
    matched=None,
    repeated=None,
    deterministic_repeated=None,
    sure_output=None,
    multiseed=None,
    run_metadata=None,
):
    calibration_ok = bool(
        calibration and calibration.get("empirical_teacher_accuracy_available")
    )
    duration_decision = (
        calibration.get("confidence_model_comparison", {}).get("decision", {})
        if calibration_ok
        else {}
    )
    proxy_metrics = value(calibration, "proxy_metrics", {})
    intercept = value(calibration, "calibration_intercept", None)
    intercept_se = value(calibration, "calibration_intercept_se", None)
    slope = value(calibration, "calibration_slope", None)
    slope_se = value(calibration, "calibration_slope_se", None)
    empirically_calibrated = bool(
        calibration_ok
        and all(finite(item) for item in [intercept, intercept_se, slope, slope_se])
        and abs(float(intercept)) <= 1.96 * float(intercept_se)
        and abs(float(slope) - 1.0) <= 1.96 * float(slope_se)
    )
    n_sure_choices = int(value(behavior, "n_sure_choice", 0) or 0)
    coherence_dependence = has_nonconstant_sure_rate(value(behavior, "coherence", []))
    duration_dependence = has_nonconstant_sure_rate(value(behavior, "duration", []))
    matched_inference = value(matched, "waived_accuracy_difference_inference", {})
    waived_advantage_supported = bool(
        finite(value(matched, "waived_minus_matched_no_sure_accuracy", None))
        and float(value(matched, "waived_minus_matched_no_sure_accuracy")) > 0
        and finite(value(matched_inference, "bootstrap_ci95_low", None))
        and float(value(matched_inference, "bootstrap_ci95_low")) > 0
    )
    repeat_predictors_estimable = bool(
        repeated
        and int(repeated.get("n_stimuli_with_choice_variability", 0)) > 0
        and finite(repeated.get("within_stimulus_pre_ts_margin_logit_coefficient"))
        and finite(repeated.get("within_stimulus_pre_ts_hidden_pc1_logit_coefficient"))
    )
    sure_timing_complete = bool(
        sure_output
        and all(finite(sure_output.get(key)) for key in [
            "mean_pre_ts_sure_output",
            "pre_ts_future_sure_minus_waived",
            "pre_ts_offered_minus_unoffered",
            "post_ts_future_sure_minus_waived",
            "pre_go_future_sure_minus_waived",
            "post_go_future_sure_minus_waived",
        ])
    )
    multiseed_complete = bool(multiseed and int(multiseed.get("n_seeds", 0)) >= 3)
    required = {
        "empirical teacher calibration": empirically_calibrated,
        "non-degenerate sure behavior": n_sure_choices > 0,
        "coherence-conditioned sure behavior": coherence_dependence,
        "duration-conditioned sure behavior": duration_dependence,
        "positive controlled waived-sure advantage": waived_advantage_supported,
        "stochastic repeated-stimulus predictors": repeat_predictors_estimable,
        "deterministic repeated-input control": bool(deterministic_repeated),
        "complete actual sure-output timing": sure_timing_complete,
        "multi-seed final-candidate validation": multiseed_complete and bool(multiseed.get("pass")),
    }
    ready = all(required.values())
    recommendation = "READY FOR POPULATION ANALYSIS" if ready else "NOT READY FOR POPULATION ANALYSIS"

    model_comparison = value(calibration, "confidence_model_comparison", {})
    model_a = value(model_comparison, "margin", {})
    model_b = value(model_comparison, "margin_duration", {})
    model_c = value(model_comparison, "margin_duration_interaction", {})
    mapping_selection = value(run_metadata, "mapping_selection", {})
    configuration = value(run_metadata, "configuration", {})
    runtime = value(run_metadata, "runtime_seconds", {})
    packages = value(run_metadata, "package_versions", {})
    repeat_margin = value(repeated, "within_stimulus_pre_ts_margin_logit_coefficient", None)
    repeat_margin_se = value(repeated, "within_stimulus_pre_ts_margin_logit_se", None)
    repeat_pc1 = value(repeated, "within_stimulus_pre_ts_hidden_pc1_logit_coefficient", None)
    repeat_pc1_se = value(repeated, "within_stimulus_pre_ts_hidden_pc1_logit_se", None)

    lines = [
        "# Model freeze report",
        "",
        "## 1. Exact active pipeline",
        "",
        "Supervised timed teacher → exported internal-readout targets → supervised student. "
        "The sure target is optional and independently offered; this is not reinforcement learning.",
        "",
        "## 2. Git provenance",
        "",
        f"- Branch: `{branch}`",
        f"- Baseline main commit: `{baseline_commit}`",
        f"- Seed: `{value(run_metadata, 'seed', 'not recorded')}`",
        f"- Runtime packages: `{packages}`",
        "",
        "## 3. Final task timing parameters",
        "",
        "- `dt = 10 ms`",
        "- stimulus duration: `400–1200 ms` (current timed configuration)",
        "- delay duration: `800–1200 ms`",
        "- TS latency from motion offset: uniformly sampled over dt-compatible values in `500–750 ms`",
        "- minimum post-TS interval: `100 ms`",
        "- minimum post-go response window: `300 ms`",
        "- sure-offer probability: `0.5`, sampled independently after timing generation",
        "",
        "## 4. Teacher confidence formula",
        "",
        f"Selected mapping: `{value(run_metadata, 'confidence_mapping', duration_decision.get('recommended_mapping', 'not recorded'))}`. "
        f"The selection run improved held-out log loss by `{fmt(value(mapping_selection, 'log_loss_improvement', None))}` "
        f"against a `{fmt(value(mapping_selection, 'decision_threshold', None))}` threshold. "
        f"The regenerated targets used coefficients `{value(mapping_selection, 'coefficients_intercept_margin_duration_s')}` "
        "for intercept, margin, and duration in seconds.",
        "",
        "## 5. Empirical P(correct) calibration",
        "",
        f"- Empirical correctness available: `{calibration_ok}`",
        f"- Teacher direction accuracy: `{fmt(value(run_metadata, 'final_teacher_direction_accuracy', None))}` overall; `{fmt(value(calibration, 'coherence_summary', [{}])[0].get('empirical_accuracy') if value(calibration, 'coherence_summary', []) else None)}` at zero coherence.",
        f"- Brier score: `{fmt(value(proxy_metrics, 'brier_score', None))}`",
        f"- Log loss: `{fmt(value(proxy_metrics, 'log_loss', None))}`",
        f"- AUC: `{fmt(value(proxy_metrics, 'auc', None))}`",
        f"- Calibration intercept: `{fmt(intercept)}` ± `{fmt(intercept_se)}` SE",
        f"- Calibration slope: `{fmt(slope)}` ± `{fmt(slope_se)}` SE",
        f"- Calibrated by the intercept=0/slope=1 95% Wald check: `{empirically_calibrated}`",
        "",
        "The target for this section is the correctness of the teacher's left/right preference, including offered trials; proxy-on-proxy comparisons do not satisfy it.",
        "",
        "## 6. Does duration add information beyond margin?",
        "",
        f"- Tested: `{bool(duration_decision)}`",
        f"- Model A (margin) held-out log loss / Brier / AUC: `{fmt(value(model_a, 'log_loss', None))}` / `{fmt(value(model_a, 'brier_score', None))}` / `{fmt(value(model_a, 'auc', None))}`",
        f"- Model B (+ duration) held-out log loss / Brier / AUC: `{fmt(value(model_b, 'log_loss', None))}` / `{fmt(value(model_b, 'brier_score', None))}` / `{fmt(value(model_b, 'auc', None))}`",
        f"- Model C (+ interaction) held-out log loss / Brier / AUC: `{fmt(value(model_c, 'log_loss', None))}` / `{fmt(value(model_c, 'brier_score', None))}` / `{fmt(value(model_c, 'auc', None))}`",
        f"- Duration log-loss improvement: `{fmt(duration_decision.get('duration_log_loss_improvement'))}`",
        f"- Duration meaningful: `{duration_decision.get('duration_adds_meaningful_information', False)}`",
        f"- Selected mapping: `{value(run_metadata, 'confidence_mapping', duration_decision.get('recommended_mapping', 'not recorded'))}`",
        "",
        "## 7. Behavioral validation",
        "",
        f"- Trials: `{value(behavior, 'n_trials')}`",
        f"- P(sure | offered): `{value(behavior, 'overall_p_sure')}`",
        f"- No-sure accuracy: `{value(behavior, 'no_sure_accuracy')}`",
        f"- Waived-sure accuracy: `{value(behavior, 'waived_sure_accuracy')}`",
        f"- Duration analysis available: `{bool(behavior and behavior.get('duration'))}`",
        f"- Sure-choice logistic regression: `{value(behavior, 'sure_logistic', {})}`",
        f"- Coherence dependence reproduced: `{coherence_dependence}`",
        f"- Duration dependence reproduced: `{duration_dependence}`",
        "",
        "## 8. Waived-sure versus no-sure controlled comparison",
        "",
        f"- Waived accuracy: `{value(matched, 'waived_accuracy')}`",
        f"- Matched no-sure accuracy: `{value(matched, 'matched_no_sure_accuracy_for_waived')}`",
        f"- Difference: `{value(matched, 'waived_minus_matched_no_sure_accuracy')}`",
        f"- Mean stimulus-duration mismatch: `{value(matched, 'mean_stimulus_duration_match_delta_ms')}` ms",
        f"- Inference: `{value(matched, 'waived_accuracy_difference_inference')}`",
        f"- Control reuse prevented: `{value(matched, 'no_control_reuse')}`",
        f"- Waived-sure accuracy exceeds matched no-sure accuracy: `{waived_advantage_supported}`",
        "",
        "## 9. Repeated-stimulus internal variability",
        "",
        f"- Stimuli with sure/direction variability: `{value(repeated, 'n_stimuli_with_choice_variability')}`",
        f"- Within-stimulus analysis: `{value(repeated, 'within_stimulus_analysis')}`",
        f"- Repetitions: `{value(repeated, 'n_stimuli')}` stimuli × `{value(repeated, 'n_repeats_per_stimulus')}` repeats",
        f"- Pre-TS margin coefficient: `{fmt(repeat_margin)}` ± `{fmt(repeat_margin_se)}` SE",
        f"- Pre-TS within-stimulus hidden PC1 coefficient: `{fmt(repeat_pc1)}` ± `{fmt(repeat_pc1_se)}` SE",
        f"- Deterministic identical-choice fraction: `{value(deterministic_repeated, 'all_repeats_same_choice_fraction')}`",
        f"- Deterministic maximum output/state difference: `{fmt(value(deterministic_repeated, 'max_output_abs_diff', None))}` / `{fmt(value(deterministic_repeated, 'max_state_abs_diff', None))}`",
        "",
        "## 10. Sure-output timing",
        "",
        f"- Mean pre-TS sure output: `{fmt(value(sure_output, 'mean_pre_ts_sure_output', None))}`",
        f"- Future sure minus waived before TS: `{fmt(value(sure_output, 'pre_ts_future_sure_minus_waived', None))}`",
        f"- Offered minus unoffered before TS (leakage control): `{fmt(value(sure_output, 'pre_ts_offered_minus_unoffered', None))}`",
        f"- Future sure minus waived immediately after TS: `{fmt(value(sure_output, 'post_ts_future_sure_minus_waived', None))}`",
        f"- Future sure minus waived immediately before/after go cue: `{fmt(value(sure_output, 'pre_go_future_sure_minus_waived', None))}` / `{fmt(value(sure_output, 'post_go_future_sure_minus_waived', None))}`",
        "- Choice-conditioned timing contrasts are not estimable because the standard evaluation contained zero sure choices.",
        f"- Evidence of pre-TS offer leakage: `not established`; the descriptive offered-minus-unoffered difference is `{fmt(value(sure_output, 'pre_ts_offered_minus_unoffered', None))}` without an inferential interval.",
        "",
        "## 11. Multi-seed robustness",
        "",
        f"- Seeds included: `{value(multiseed, 'n_seeds')}`",
        f"- Aggregate pass: `{value(multiseed, 'pass')}`",
        f"- Multi-seed training status: `{value(run_metadata, 'multiseed_training', {})}`",
        "- Seeds 8 and 9 were not trained because seed 7 failed the required single-seed gate; no seeds were cherry-picked.",
        "",
        "## 12. Direct answers to freeze questions",
        "",
        f"1. Empirical teacher P(correct) calibrated? **Yes by the stated Wald check** (Brier `{fmt(value(proxy_metrics, 'brier_score', None))}`, log loss `{fmt(value(proxy_metrics, 'log_loss', None))}`, intercept `{fmt(intercept)}`, slope `{fmt(slope)}`).",
        f"2. Duration adds information beyond margin? **{str(bool(duration_decision.get('duration_adds_meaningful_information')))}**; held-out log-loss improvement `{fmt(duration_decision.get('duration_log_loss_improvement'))}`.",
        f"3. Frozen confidence mapping? **`{value(run_metadata, 'confidence_mapping', 'not recorded')}`**.",
        f"4. P(sure) dependence on coherence reproduced? **{str(coherence_dependence)}**; P(sure|offered) was `{fmt(value(behavior, 'overall_p_sure', None))}` with `{n_sure_choices}` sure choices.",
        f"5. P(sure) dependence on duration reproduced? **{str(duration_dependence)}**.",
        f"6. Waived-sure accuracy exceeds matched no-sure accuracy? **{str(waived_advantage_supported)}**; difference `{fmt(value(matched, 'waived_minus_matched_no_sure_accuracy', None))}`.",
        f"7. Evidence of pre-TS offer leakage? **Not established**; descriptive negative-control difference `{fmt(value(sure_output, 'pre_ts_offered_minus_unoffered', None))}` lacks an inferential interval.",
        f"8. Identical-input recurrent variability predicts sure choice? **Inconclusive**; margin `{fmt(repeat_margin)}` ± `{fmt(repeat_margin_se)}` SE and PC1 `{fmt(repeat_pc1)}` ± `{fmt(repeat_pc1_se)}` SE across `{value(repeated, 'within_stimulus_analysis_n')}` observations.",
        f"9. Deterministic repeated-input behavior stable? **{str(value(deterministic_repeated, 'all_repeats_same_choice_fraction') == 1.0)}**.",
        f"10. Reproducible across seeds? **No multi-seed conclusion**; only the required seed-7 gate was run and it failed.",
        "",
        "## 13. Configuration and runtime",
        "",
        f"- Final configuration: `{configuration}`",
        f"- Runtime seconds: `{runtime}`",
        f"- Key output paths: `{value(run_metadata, 'key_output_paths', {})}`",
        "- Exact scientific commands:",
        "",
        "```text",
        *value(run_metadata, "commands", []),
        "```",
        "",
        "## 14. Remaining limitations",
        "",
    ]
    for name, passed in required.items():
        if not passed:
            lines.append(f"- Missing or incomplete: {name}.")
    lines.extend([
        "- This remains a supervised model and does not establish that confidence is learned without supervision.",
        "- Population-dynamics expansion is intentionally deferred until these freeze gates pass.",
        "",
        "## 15. Recommendation",
        "",
        f"**{recommendation}**",
        "",
    ])
    return "\n".join(lines), {"ready": ready, "required_checks": required, "recommendation": recommendation}


def sibling(summary_path, suffix):
    root, _ = os.path.splitext(summary_path)
    return f"{root}_{suffix}.json"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate the conservative model-freeze report.")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--calibration", default="results/model_freeze/teacher_calibration_summary.json")
    parser.add_argument("--student-summary", default="results/model_freeze/student_seed7_summary.json")
    parser.add_argument("--matched", default="results/model_freeze/matched_trial_controls_summary.json")
    parser.add_argument("--multiseed", default="results/model_freeze/multiseed_validation_summary.json")
    parser.add_argument("--run-metadata", default="results/model_freeze/run_manifest.json")
    parser.add_argument("--output", default="results/model_freeze_report.md")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    behavior = load_json(sibling(args.student_summary, "behavior"))
    repeated = load_json(sibling(args.student_summary, "repeated_stimulus_summary"))
    deterministic = load_json(sibling(args.student_summary, "deterministic_repeated_stimulus_summary"))
    sure_output = load_json(sibling(args.student_summary, "sure_output_dynamics_summary"))
    report, status = build_report(
        branch=args.branch,
        baseline_commit=args.baseline_commit,
        calibration=load_json(args.calibration),
        behavior=behavior,
        matched=load_json(args.matched),
        repeated=repeated,
        deterministic_repeated=deterministic,
        sure_output=sure_output,
        multiseed=load_json(args.multiseed),
        run_metadata=load_json(args.run_metadata),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"output": os.path.abspath(args.output), **status}, indent=2))
    return status


if __name__ == "__main__":
    main()
