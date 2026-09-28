import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np

from rdm_population_diagnostics import (
    collect_repeated_stimulus_evaluation,
    simulator_test,
    softmax_action_outputs,
    summarize_repeated_stimulus_trials,
    write_json,
    write_repeated_trials_csv,
)
from rdm_task_params import task_params_from_task
from sure_target_stage9_internal_readout import (
    RDM_SureTarget_InternalReadoutDatasetTask,
    build_post_go_output_records,
    collect_eval_batches,
    configure_student_action_objective,
    set_global_seed,
)


TEACHER_SHA256 = "8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d"
BASELINE_NO_SURE_ACCURACY = 0.934010152284264
VALID_NO_SURE_FLOOR = BASELINE_NO_SURE_ACCURACY - 0.02
STAGE_A_CE_GUARDRAIL = 0.02
STAGE_A_MARGIN_R_GUARDRAIL = 0.05
STAGE_A_SIGN_GUARDRAIL = 0.03


def _corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 2 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _direction_correct(choice, direction):
    return (choice == 1 and direction == 0) or (choice == 2 and direction == 1)


def summarize_records(records):
    offered = np.asarray([bool(row["sure_available"]) for row in records])
    direction = np.asarray([int(row["dir_choice"]) for row in records])
    choices = np.asarray([int(row["final_argmax_choice"]) for row in records])
    target = np.asarray(
        [[row["target_left"], row["target_right"], row["target_sure"]] for row in records],
        dtype=float,
    )
    predicted = np.asarray(
        [[row["p_left"], row["p_right"], row["p_sure"]] for row in records],
        dtype=float,
    )
    correct_index = direction
    incorrect_index = 1 - direction
    rows_index = np.arange(len(records))
    target_correct = target[rows_index, correct_index]
    predicted_correct = predicted[rows_index, correct_index]
    predicted_incorrect = predicted[rows_index, incorrect_index]
    target_margin = target[:, 2] - target_correct
    predicted_margin = predicted[:, 2] - predicted_correct
    teacher_sure_favoring = offered & (target_margin > 0)
    teacher_direction_favoring = offered & (target_margin < 0)
    eps = 1e-12
    ce = -np.sum(target * np.log(np.maximum(predicted, eps)), axis=1)
    kl = np.sum(
        np.where(
            target > 0,
            target * np.log(np.maximum(target, eps) / np.maximum(predicted, eps)),
            0.0,
        ),
        axis=1,
    )
    no_sure = ~offered
    offered_direction_choice = offered & np.isin(choices, [1, 2])
    fixation = np.asarray([row["raw_post_go_fixation"] for row in records], dtype=float)

    def accuracy(mask):
        if not np.any(mask):
            return np.nan
        return float(
            np.mean(
                [_direction_correct(choices[idx], direction[idx]) for idx in np.flatnonzero(mask)]
            )
        )

    def subset(mask):
        if not np.any(mask):
            return {"n": 0}
        return {
            "n": int(np.sum(mask)),
            "predicted_p_sure": float(np.mean(predicted[mask, 2])),
            "predicted_p_correct": float(np.mean(predicted_correct[mask])),
            "predicted_p_incorrect": float(np.mean(predicted_incorrect[mask])),
            "target_p_sure": float(np.mean(target[mask, 2])),
            "predicted_action_margin": float(np.mean(predicted_margin[mask])),
            "fraction_sure_over_correct": float(np.mean(predicted_margin[mask] > 0)),
            "fraction_sure_final_argmax": float(np.mean(choices[mask] == 3)),
        }

    metrics = {
        "n_trials": int(len(records)),
        "n_offered": int(np.sum(offered)),
        "mean_abs_postgo_fixation": float(np.mean(np.abs(fixation))),
        "postgo_fixation_rms": float(np.sqrt(np.mean(np.square(fixation)))),
        "no_sure_accuracy": accuracy(no_sure),
        "offered_direction_accuracy": accuracy(offered_direction_choice),
        "hard_p_sure_offered": float(np.mean(choices[offered] == 3)),
        "teacher_sure_favoring_fraction": float(np.mean(target_margin[offered] > 0)),
        "target_predicted_sure_correlation": _corr(target[offered, 2], predicted[offered, 2]),
        "action_margin_correlation": _corr(target_margin[offered], predicted_margin[offered]),
        "action_margin_sign_agreement": float(
            np.mean(np.sign(target_margin[offered]) == np.sign(predicted_margin[offered]))
        ),
        "action_cross_entropy": float(np.mean(ce)),
        "action_kl": float(np.mean(kl)),
        "action_mae": float(np.mean(np.abs(target - predicted))),
        "incorrect_direction_probability": float(np.mean(predicted_incorrect[offered])),
        "sure_favoring_target_sign_agreement": float(
            np.mean(predicted_margin[teacher_sure_favoring] > 0)
        ),
        "all_finite": bool(np.all(np.isfinite(predicted)) and np.all(np.isfinite(fixation))),
        "teacher_sure_favoring": subset(teacher_sure_favoring),
        "teacher_direction_favoring": subset(teacher_direction_favoring),
        "soft_correlations": {
            "teacher_sure_strength": _corr(
                [row["teacher_sure_strength"] for row in np.asarray(records, dtype=object)[offered]],
                predicted[offered, 2],
            ),
            "empirical_p_correct": _corr(
                [row["empirical_p_correct"] for row in np.asarray(records, dtype=object)[offered]],
                predicted[offered, 2],
            ),
            "internal_margin": _corr(
                [row["internal_margin"] for row in np.asarray(records, dtype=object)[offered]],
                predicted[offered, 2],
            ),
            "coherence": _corr(
                [row["stimulus_coherence"] for row in np.asarray(records, dtype=object)[offered]],
                predicted[offered, 2],
            ),
            "duration": _corr(
                [row["stimulus_duration_ms"] for row in np.asarray(records, dtype=object)[offered]],
                predicted[offered, 2],
            ),
        },
    }
    return metrics


def _network_params(task, lambda_fix_postgo, lambda_action, load_weights_path=None):
    params = task_params_from_task(task)
    params.update(
        {
            "name": "RDM_SureTarget_Stage9_InternalReadoutStudent",
            "N_rec": 50,
            "rec_noise": 0.05,
            "activation": "rectified_linear",
            "alpha": 0.1,
        }
    )
    if load_weights_path is not None:
        params["load_weights_path"] = load_weights_path
    configure_student_action_objective(
        params,
        "categorical_soft",
        lambda_fix_postgo=lambda_fix_postgo,
        lambda_action=lambda_action,
    )
    return params


def train_candidate(output_dir, lambda_fix_postgo, lambda_action):
    import tensorflow as tf
    from psychrnn.backend.models.basic import Basic

    os.makedirs(output_dir, exist_ok=True)
    tf.compat.v1.reset_default_graph()
    set_global_seed(7)
    task = RDM_SureTarget_InternalReadoutDatasetTask(
        dataset_path="data/model_freeze/teacher_seed7.npz",
        dt=10,
        tau=100,
        N_batch=50,
        sample_mode="random",
        seed=7,
    )
    model = Basic(_network_params(task, lambda_fix_postgo, lambda_action))
    weights_path = os.path.join(output_dir, "weights.npz")
    started = time.time()
    _, training_seconds, initialization_seconds = model.train(
        task,
        train_params={
            "training_iters": 50000,
            "loss_epoch": 1000,
            "save_weights_path": weights_path,
            "verbosity": False,
        },
    )
    training_wall_seconds = time.time() - started
    task.sample_mode = "sequential"
    task.next_index = 0
    raw_outputs, _, targets, eval_inputs, trial_info = collect_eval_batches(
        model, task, n_eval_batches=8
    )
    probabilities = softmax_action_outputs(raw_outputs)
    stochastic_records = build_post_go_output_records(
        probabilities,
        targets,
        trial_info,
        objective="categorical_soft",
        readout_window=10,
        raw_outputs=raw_outputs,
    )
    deterministic_raw, _ = simulator_test(model, eval_inputs, rec_noise=0.0)
    deterministic_probabilities = softmax_action_outputs(deterministic_raw)
    deterministic_records = build_post_go_output_records(
        deterministic_probabilities,
        targets,
        trial_info,
        objective="categorical_soft",
        readout_window=10,
        raw_outputs=deterministic_raw,
    )
    stochastic_path = os.path.join(output_dir, "standard_stochastic.csv")
    deterministic_path = os.path.join(output_dir, "standard_deterministic.csv")
    write_repeated_trials_csv(stochastic_path, stochastic_records)
    write_repeated_trials_csv(deterministic_path, deterministic_records)
    stochastic_metrics = summarize_records(stochastic_records)
    deterministic_metrics = summarize_records(deterministic_records)
    dataset_indices = [int(row["dataset_index"]) for row in stochastic_records]
    summary = {
        "lambda_fix_postgo": float(lambda_fix_postgo),
        "lambda_action": float(lambda_action),
        "weights": os.path.abspath(weights_path),
        "standard_stochastic_csv": os.path.abspath(stochastic_path),
        "standard_deterministic_csv": os.path.abspath(deterministic_path),
        "dataset_indices": dataset_indices,
        "training_wall_seconds": float(training_wall_seconds),
        "psychrnn_training_seconds": float(training_seconds),
        "psychrnn_initialization_seconds": float(initialization_seconds),
        "stochastic": stochastic_metrics,
        "deterministic": deterministic_metrics,
    }
    summary["valid"] = bool(
        stochastic_metrics["all_finite"]
        and deterministic_metrics["all_finite"]
        and stochastic_metrics["no_sure_accuracy"] >= VALID_NO_SURE_FLOOR
    )
    write_json(os.path.join(output_dir, "summary.json"), summary)
    model.destruct()
    return summary


def flatten_sweep_row(stage, summary):
    det = summary["deterministic"]
    sto = summary["stochastic"]
    return {
        "stage": stage,
        "lambda_fix_postgo": summary["lambda_fix_postgo"],
        "lambda_action": summary["lambda_action"],
        "valid": summary["valid"],
        "det_fixation_abs": det["mean_abs_postgo_fixation"],
        "det_fixation_rms": det["postgo_fixation_rms"],
        "det_action_ce": det["action_cross_entropy"],
        "det_action_kl": det["action_kl"],
        "det_action_mae": det["action_mae"],
        "det_margin_r": det["action_margin_correlation"],
        "det_margin_sign": det["action_margin_sign_agreement"],
        "det_incorrect_probability": det["incorrect_direction_probability"],
        "det_sure_subset_sign": det["sure_favoring_target_sign_agreement"],
        "det_hard_p_sure_offered": det["hard_p_sure_offered"],
        "det_target_sure_r": det["target_predicted_sure_correlation"],
        "sto_no_sure_accuracy": sto["no_sure_accuracy"],
        "sto_offered_direction_accuracy": sto["offered_direction_accuracy"],
        "sto_hard_p_sure_offered": sto["hard_p_sure_offered"],
        "training_wall_seconds": summary["training_wall_seconds"],
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_stage_a(candidates):
    baseline = next(item for item in candidates if item["lambda_fix_postgo"] == 1.0)
    base = baseline["deterministic"]
    guarded = [
        item
        for item in candidates
        if item["valid"]
        and item["deterministic"]["action_cross_entropy"] <= base["action_cross_entropy"] + STAGE_A_CE_GUARDRAIL
        and item["deterministic"]["action_margin_correlation"] >= base["action_margin_correlation"] - STAGE_A_MARGIN_R_GUARDRAIL
        and item["deterministic"]["action_margin_sign_agreement"] >= base["action_margin_sign_agreement"] - STAGE_A_SIGN_GUARDRAIL
    ]
    if not guarded:
        return baseline
    return min(guarded, key=lambda item: item["deterministic"]["mean_abs_postgo_fixation"])


def select_stage_b(candidates):
    valid = [item for item in candidates if item["valid"]]
    if not valid:
        raise RuntimeError("No valid Stage-B candidate")
    return min(
        valid,
        key=lambda item: (
            item["deterministic"]["action_cross_entropy"],
            item["deterministic"]["action_kl"],
            -item["deterministic"]["action_margin_correlation"],
            -item["deterministic"]["action_margin_sign_agreement"],
            -item["deterministic"]["sure_favoring_target_sign_agreement"],
            item["deterministic"]["incorrect_direction_probability"],
            item["deterministic"]["mean_abs_postgo_fixation"],
        ),
    )


def value_distance_rows(records):
    offered = [row for row in records if bool(row["sure_available"])]
    bins = [
        ("strongly_direction", -np.inf, -0.20),
        ("weakly_direction", -0.20, -0.05),
        ("near_boundary", -0.05, 0.05),
        ("weakly_sure", 0.05, 0.15),
        ("strongly_sure", 0.15, np.inf),
    ]
    rows = []
    for label, low, high in bins:
        selected = [row for row in offered if float(row["delta_value"]) >= low and float(row["delta_value"]) < high]
        if not selected:
            rows.append({"bin": label, "low": low, "high": high, "n": 0, "hard_disagreement": np.nan, "mean_abs_margin_error": np.nan})
            continue
        rows.append(
            {
                "bin": label,
                "low": low,
                "high": high,
                "n": len(selected),
                "hard_disagreement": float(np.mean([int(row["final_argmax_choice"]) != int(row["teacher_preferred_action"]) for row in selected])),
                "mean_abs_margin_error": float(np.mean([abs(float(row["predicted_action_margin"]) - float(row["target_action_margin"])) for row in selected])),
            }
        )
    return rows


def run_finalist_repeats(output_dir, selected):
    import tensorflow as tf
    from psychrnn.backend.models.basic import Basic

    tf.compat.v1.reset_default_graph()
    set_global_seed(7)
    task = RDM_SureTarget_InternalReadoutDatasetTask(
        dataset_path="data/model_freeze/teacher_seed7.npz",
        dt=10,
        tau=100,
        N_batch=50,
        sample_mode="random",
        seed=7,
    )
    model = Basic(
        _network_params(
            task,
            selected["lambda_fix_postgo"],
            selected["lambda_action"],
            load_weights_path=selected["weights"],
        )
    )
    records, outputs, states = collect_repeated_stimulus_evaluation(
        model, task, n_stimuli=48, n_repeats=20, readout_window=10,
        ts_half_window=5, action_only=True, action_softmax=True,
    )
    det_records, det_outputs, det_states = collect_repeated_stimulus_evaluation(
        model, task, n_stimuli=48, n_repeats=20, readout_window=10,
        ts_half_window=5, deterministic=True, rec_noise=0.0,
        action_only=True, action_softmax=True,
    )
    write_repeated_trials_csv(os.path.join(output_dir, "finalist_repeated_stochastic.csv"), records)
    write_repeated_trials_csv(os.path.join(output_dir, "finalist_repeated_deterministic.csv"), det_records)
    stochastic = summarize_repeated_stimulus_trials(records, outputs, states)
    deterministic = summarize_repeated_stimulus_trials(det_records, det_outputs, det_states)
    stochastic["hard_p_sure"] = float(np.mean([int(row["choice"]) == 3 for row in records]))
    deterministic["hard_p_sure"] = float(np.mean([int(row["choice"]) == 3 for row in det_records]))
    summary = {"stochastic": stochastic, "deterministic": deterministic}
    write_json(os.path.join(output_dir, "finalist_repeated_summary.json"), summary)
    model.destruct()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/model_freeze/student_head_calibration")
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    with open("data/model_freeze/teacher_seed7.npz", "rb") as handle:
        observed_hash = hashlib.sha256(handle.read()).hexdigest()
    if observed_hash != TEACHER_SHA256:
        raise RuntimeError(f"Teacher dataset hash mismatch: {observed_hash}")

    total_started = time.time()
    stage_a = []
    reference_indices = None
    for value in [1.0, 2.0, 5.0, 10.0]:
        print(f"Stage A: lambda_fix_postgo={value:g}, lambda_action=1", flush=True)
        candidate = train_candidate(
            os.path.join(args.output_dir, "stage_a", f"fix_{value:g}_action_1"), value, 1.0
        )
        if reference_indices is None:
            reference_indices = candidate["dataset_indices"]
        elif candidate["dataset_indices"] != reference_indices:
            raise RuntimeError("Candidate standard evaluation trials differ")
        stage_a.append(candidate)
        print(json.dumps(flatten_sweep_row("A", candidate), indent=2), flush=True)
    selected_a = select_stage_a(stage_a)
    print(f"Selected Stage A lambda_fix_postgo={selected_a['lambda_fix_postgo']:g}", flush=True)

    stage_b = []
    selected_fix = selected_a["lambda_fix_postgo"]
    for value in [0.5, 1.0, 2.0, 4.0]:
        print(f"Stage B: lambda_fix_postgo={selected_fix:g}, lambda_action={value:g}", flush=True)
        candidate = train_candidate(
            os.path.join(args.output_dir, "stage_b", f"fix_{selected_fix:g}_action_{value:g}"),
            selected_fix,
            value,
        )
        if candidate["dataset_indices"] != reference_indices:
            raise RuntimeError("Candidate standard evaluation trials differ")
        stage_b.append(candidate)
        print(json.dumps(flatten_sweep_row("B", candidate), indent=2), flush=True)
    selected_b = select_stage_b(stage_b)
    print(
        f"Selected Stage B lambda_fix_postgo={selected_b['lambda_fix_postgo']:g}, "
        f"lambda_action={selected_b['lambda_action']:g}",
        flush=True,
    )

    with open(selected_b["standard_deterministic_csv"], newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        finalist_records = []
        for row in reader:
            converted = dict(row)
            for key in row:
                if key in ("sure_available",):
                    converted[key] = row[key].lower() == "true"
                elif key not in ("student_action_objective",):
                    try:
                        converted[key] = float(row[key])
                    except ValueError:
                        pass
            finalist_records.append(converted)
    boundary_rows = value_distance_rows(finalist_records)
    write_csv(os.path.join(args.output_dir, "value_distance.csv"), boundary_rows)
    repeat_summary = run_finalist_repeats(args.output_dir, selected_b)
    write_csv(
        os.path.join(args.output_dir, "stage_a_sweep.csv"),
        [flatten_sweep_row("A", item) for item in stage_a],
    )
    write_csv(
        os.path.join(args.output_dir, "stage_b_sweep.csv"),
        [flatten_sweep_row("B", item) for item in stage_b],
    )
    result = {
        "teacher_dataset_sha256_before": observed_hash,
        "selection_rule": {
            "invalid_no_sure_accuracy_floor": VALID_NO_SURE_FLOOR,
            "stage_a": "minimum deterministic fixation among valid candidates within predeclared CE (+0.02), margin-r (-0.05), and sign-agreement (-0.03) guardrails from fix=1",
            "stage_b": "lexicographic deterministic CE, KL, margin r, sign agreement, sure-subset fidelity, incorrect probability, fixation",
        },
        "selected_stage_a_lambda_fix_postgo": selected_a["lambda_fix_postgo"],
        "selected_lambda_fix_postgo": selected_b["lambda_fix_postgo"],
        "selected_lambda_action": selected_b["lambda_action"],
        "selected_candidate": selected_b,
        "value_distance": boundary_rows,
        "finalist_repeated": repeat_summary,
        "total_runtime_seconds": float(time.time() - total_started),
    }
    with open("data/model_freeze/teacher_seed7.npz", "rb") as handle:
        result["teacher_dataset_sha256_after"] = hashlib.sha256(handle.read()).hexdigest()
    write_json(os.path.join(args.output_dir, "selection_summary.json"), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
