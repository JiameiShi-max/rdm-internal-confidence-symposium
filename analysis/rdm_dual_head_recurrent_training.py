import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dual_head_student import (
    configure_dual_head_objective,
    dual_head_loss_numpy,
    dual_head_probabilities,
    regularization_configuration,
    split_output_head_weights,
)
from rdm_frozen_readout_diagnostic import (
    action_metrics,
    build_epoch_features,
    build_targets,
    cross_validated_action_probe,
    safe_corr,
    sha256,
    write_csv,
)
from rdm_population_diagnostics import (
    collect_repeated_stimulus_evaluation,
    summarize_repeated_stimulus_trials,
    write_json,
    write_repeated_trials_csv,
)
from rdm_staged_output_heads import (
    action_subset_metrics,
    coherence_accuracy_rows,
    counterfactual_targets,
    fixation_metrics,
    hard_behavior_rows,
    phase_samples,
    soft_confidence_correlations,
    value_distance_rows,
)
from rdm_task_params import task_params_from_task
from sure_target_stage9_internal_readout import (
    RDM_SureTarget_InternalReadoutDatasetTask,
    set_global_seed,
)


EXPECTED_TEACHER_SHA256 = "8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d"
SEED = 7
N_REC = 50
REC_NOISE = 0.05
DT = 10
TAU = 100
N_BATCH = 50
TRAINING_ITERS = 50000
WINDOW = 10
N_STANDARD_TEST = 400

# Declared before the seed-7 training run. Success requires a broad improvement
# over the frozen staged-head ceiling, not a change in sure-choice frequency.
SUCCESS_RULE = {
    "fixation_pre_go_min": 0.95,
    "fixation_post_go_max": 0.05,
    "fixation_auc_min": 0.99,
    "ce_relative_improvement_over_staged_min": 0.05,
    "kl_relative_improvement_over_staged_min": 0.10,
    "margin_r_gain_over_staged_min": 0.05,
    "teacher_policy_accuracy_gain_min": 0.02,
    "incorrect_direction_tolerance": 0.0,
    "sure_fidelity_tolerance": 0.0,
    "direction_accuracy_tolerance": 0.01,
}


def network_params(task, load_weights_path=None):
    params = task_params_from_task(task)
    params.update(
        {
            "name": "RDM_SureTarget_DualHead_Seed7",
            "N_rec": N_REC,
            "rec_noise": REC_NOISE,
            "activation": "rectified_linear",
            "alpha": DT / TAU,
            # These are also PsychRNN's defaults. They are explicit here so the
            # recurrent-regularization audit is serialized and unambiguous.
            "L1_in": 0.0,
            "L1_rec": 0.0,
            "L1_out": 0.0,
            "L2_in": 0.0,
            "L2_rec": 0.0,
            "L2_out": 0.0,
            "L2_firing_rate": 0.0,
        }
    )
    if load_weights_path is not None:
        params["load_weights_path"] = load_weights_path
    configure_dual_head_objective(params)
    return params


def trial_action_predictions(action_probabilities, trial_info, indices, window=WINDOW):
    predictions = []
    for idx in np.asarray(indices, dtype=int):
        go = int(trial_info[idx]["delay_end"])
        predictions.append(np.mean(action_probabilities[idx, go : go + window], axis=0))
    return np.asarray(predictions, dtype=float)


def time_resolved_probe_rows(epoch_features, trial_info, test_indices):
    test_indices = np.asarray(test_indices, dtype=int)
    test_info = [trial_info[idx] for idx in test_indices]
    targets = counterfactual_targets(test_info)
    direction = np.asarray([int(info["dir_choice"]) for info in test_info])
    row = np.arange(len(test_info))
    target_margin = targets[:, 2] - targets[row, direction]
    target_sure = target_margin > 0
    preferred = np.argmax(targets, axis=1)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = [(train, test) for train, test in splitter.split(test_indices, preferred)]
    rows = []
    for epoch in ("late_motion", "pre_ts", "pre_go", "post_go"):
        predictions = cross_validated_action_probe(
            epoch_features[epoch][test_indices], targets, folds
        )
        predicted_margin = predictions[:, 2] - predictions[row, direction]
        rows.append(
            {
                "epoch": epoch,
                "target_definition": "counterfactual value preference independent of sure offer",
                "action_margin_correlation": safe_corr(target_margin, predicted_margin),
                "sure_vs_direction_auc": float(roc_auc_score(target_sure, predicted_margin)),
                "direction_accuracy": float(
                    np.mean(np.argmax(predictions[:, :2], axis=1) == direction)
                ),
            }
        )
    return rows


def find_hard_row(summary, label):
    for row in summary["hard_behavior"]:
        if row["model"] == label:
            return row
    raise KeyError(label)


def classify_result(staged_summary, fixation, action, hard_behavior, losses):
    staged_action = staged_summary["action"]["staged"]
    staged_hard = find_hard_row(staged_summary, "staged")
    finite = bool(
        np.all(np.isfinite(losses))
        and all(np.isfinite(value) for value in fixation.values() if isinstance(value, float))
        and all(np.isfinite(value) for value in action.values() if isinstance(value, float))
    )
    if not finite:
        return {
            "classification": "C. TRAINING UNSTABLE / IMPLEMENTATION ISSUE",
            "stable": False,
        }
    fixation_clean = bool(
        fixation["pre_go_mean"] >= SUCCESS_RULE["fixation_pre_go_min"]
        and fixation["post_go_mean"] <= SUCCESS_RULE["fixation_post_go_max"]
        and fixation["auc"] >= SUCCESS_RULE["fixation_auc_min"]
    )
    ce_gain = 1.0 - action["action_cross_entropy"] / staged_action["action_cross_entropy"]
    kl_gain = 1.0 - action["action_kl"] / staged_action["action_kl"]
    margin_gain = (
        action["action_margin_correlation"]
        - staged_action["action_margin_correlation"]
    )
    policy_gain = (
        hard_behavior["teacher_policy_accuracy"]
        - staged_hard["teacher_policy_accuracy"]
    )
    broad_action_improvement = bool(
        ce_gain >= SUCCESS_RULE["ce_relative_improvement_over_staged_min"]
        and kl_gain >= SUCCESS_RULE["kl_relative_improvement_over_staged_min"]
        and margin_gain >= SUCCESS_RULE["margin_r_gain_over_staged_min"]
        and policy_gain >= SUCCESS_RULE["teacher_policy_accuracy_gain_min"]
        and action["incorrect_direction_probability"]
        <= staged_action["incorrect_direction_probability"]
        + SUCCESS_RULE["incorrect_direction_tolerance"]
        and action["teacher_sure_favoring_fidelity"]
        >= staged_action["teacher_sure_favoring_fidelity"]
        - SUCCESS_RULE["sure_fidelity_tolerance"]
        and hard_behavior["no_sure_direction_accuracy"]
        >= staged_hard["no_sure_direction_accuracy"]
        - SUCCESS_RULE["direction_accuracy_tolerance"]
        and hard_behavior["offered_teacher_direction_accuracy"]
        >= staged_hard["offered_teacher_direction_accuracy"]
        - SUCCESS_RULE["direction_accuracy_tolerance"]
    )
    classification = (
        "A. RECURRENT OBJECTIVE IMPROVED REPRESENTATION"
        if fixation_clean and broad_action_improvement
        else "B. RECURRENT REPRESENTATION STILL LIMITED"
    )
    return {
        "classification": classification,
        "stable": True,
        "fixation_clean": fixation_clean,
        "broad_action_improvement": broad_action_improvement,
        "ce_relative_improvement_over_staged": float(ce_gain),
        "kl_relative_improvement_over_staged": float(kl_gain),
        "margin_r_gain_over_staged": float(margin_gain),
        "teacher_policy_accuracy_gain_over_staged": float(policy_gain),
    }


def save_split_weights(output_dir, weights):
    fixation, action = split_output_head_weights(weights)
    output_keys = {"W_out", "b_out", "output_connectivity", "Dale_out"}
    recurrent = {key: value for key, value in weights.items() if key not in output_keys}
    recurrent["source_full_checkpoint"] = np.asarray("weights.npz")
    recurrent_path = os.path.join(output_dir, "recurrent_backbone_weights.npz")
    fixation_path = os.path.join(output_dir, "fixation_head_weights.npz")
    action_path = os.path.join(output_dir, "action_head_weights.npz")
    np.savez(recurrent_path, **recurrent)
    np.savez(fixation_path, **fixation)
    np.savez(action_path, **action)
    return recurrent_path, fixation_path, action_path


def repeated_stimulus_evaluation(model, task, output_dir):
    stochastic_records, stochastic_outputs, stochastic_states = (
        collect_repeated_stimulus_evaluation(
            model,
            task,
            n_stimuli=48,
            n_repeats=20,
            readout_window=WINDOW,
            ts_half_window=5,
            action_only=True,
            action_softmax=True,
        )
    )
    deterministic_records, deterministic_outputs, deterministic_states = (
        collect_repeated_stimulus_evaluation(
            model,
            task,
            n_stimuli=48,
            n_repeats=20,
            readout_window=WINDOW,
            ts_half_window=5,
            deterministic=True,
            rec_noise=0.0,
            action_only=True,
            action_softmax=True,
        )
    )
    stochastic = summarize_repeated_stimulus_trials(
        stochastic_records, stochastic_outputs, stochastic_states
    )
    deterministic = summarize_repeated_stimulus_trials(
        deterministic_records, deterministic_outputs, deterministic_states
    )
    stochastic["hard_p_sure"] = float(
        np.mean([int(row["choice"]) == 3 for row in stochastic_records])
    )
    deterministic["hard_p_sure"] = float(
        np.mean([int(row["choice"]) == 3 for row in deterministic_records])
    )
    write_repeated_trials_csv(
        os.path.join(output_dir, "repeated_stochastic.csv"), stochastic_records
    )
    write_repeated_trials_csv(
        os.path.join(output_dir, "repeated_deterministic.csv"), deterministic_records
    )
    return {"run": True, "stochastic": stochastic, "deterministic": deterministic}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/model_freeze/teacher_seed7.npz")
    parser.add_argument(
        "--staged-summary",
        default="results/model_freeze/staged_output_head_experiment/summary.json",
    )
    parser.add_argument("--output-dir", default="results/model_freeze/dual_head_seed7")
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    wall_started = time.time()
    teacher_hash_before = sha256(args.dataset)
    if teacher_hash_before != EXPECTED_TEACHER_SHA256:
        raise RuntimeError(f"Teacher dataset hash mismatch: {teacher_hash_before}")
    with open(args.staged_summary, encoding="utf-8") as handle:
        staged_summary = json.load(handle)

    import tensorflow as tf
    from psychrnn.backend.models.basic import Basic
    from psychrnn.backend.simulation import BasicSimulator

    tf.compat.v1.reset_default_graph()
    set_global_seed(SEED)
    task = RDM_SureTarget_InternalReadoutDatasetTask(
        dataset_path=args.dataset,
        dt=DT,
        tau=TAU,
        N_batch=N_BATCH,
        sample_mode="random",
        seed=SEED,
    )
    params = network_params(task)
    regularization = regularization_configuration(params)
    if any(value != 0.0 for value in regularization.values()):
        raise RuntimeError(f"Unexpected nonzero recurrent regularization: {regularization}")
    model = Basic(params)
    checkpoint_path = os.path.join(args.output_dir, "weights.npz")
    train_started = time.time()
    losses, training_seconds, initialization_seconds = model.train(
        task,
        train_params={
            "training_iters": TRAINING_ITERS,
            "loss_epoch": 100,
            "save_weights_path": checkpoint_path,
            "verbosity": False,
        },
    )
    training_wall_seconds = time.time() - train_started
    weights = model.get_weights()
    recurrent_path, fixation_path, action_path = save_split_weights(
        args.output_dir, weights
    )
    write_csv(
        os.path.join(args.output_dir, "training_history.csv"),
        [
            {"logged_update": idx + 1, "training_examples_seen": (idx + 1) * 5000, "loss": float(loss)}
            for idx, loss in enumerate(losses)
        ],
    )

    dataset = np.load(args.dataset, allow_pickle=True)
    trial_info = [json.loads(str(item)) for item in dataset["trial_info_json"]]
    for idx, info in enumerate(trial_info):
        info["teacher_internal_margin"] = float(dataset["teacher_internal_margin"][idx])
        info["teacher_sure_strength"] = float(dataset["teacher_sure_strength"][idx])
        info["teacher_direction_success_proxy"] = float(
            dataset["teacher_direction_success_proxy"][idx]
        )
    targets = build_targets(dataset, trial_info, window=WINDOW)
    test_indices = np.arange(N_STANDARD_TEST, dtype=int)
    test_info = [trial_info[idx] for idx in test_indices]
    test_targets = targets[test_indices]

    simulator = BasicSimulator(
        params={"alpha": DT / TAU, "rec_noise": 0.0},
        weights=dict(np.load(checkpoint_path, allow_pickle=True)),
    )
    raw_outputs, states = simulator.run_trials(
        np.asarray(dataset["x"][test_indices], dtype=np.float32)
    )
    fixation_probabilities, action_probabilities = dual_head_probabilities(raw_outputs)
    local_info = test_info
    fix_x, fix_y, fix_trial, _, fix_steps = phase_samples(
        states, local_info, np.arange(len(test_indices)), window=WINDOW
    )
    del fix_x
    staged_fixation_samples = fixation_probabilities[fix_trial, fix_steps]
    fixation = fixation_metrics(fix_y, staged_fixation_samples)
    action_predictions = trial_action_predictions(
        action_probabilities, local_info, np.arange(len(test_indices))
    )
    action = action_metrics(test_targets, action_predictions, test_info)
    hard_row = hard_behavior_rows(
        {"end_to_end": action_predictions}, test_targets, test_info
    )[0]
    subsets = action_subset_metrics(test_targets, action_predictions, test_info)
    value_rows = value_distance_rows(
        test_targets,
        np.asarray(
            [
                [float(row[f"native_{name}"]) for name in ("left", "right", "sure")]
                for row in _read_csv(
                    "results/model_freeze/staged_output_head_experiment/test_trial_predictions.csv"
                )
            ]
        ),
        action_predictions,
        test_info,
    )
    for row in value_rows:
        row["end_to_end_disagreement"] = row.pop("staged_disagreement")
    correlations = soft_confidence_correlations(
        {"end_to_end": action_predictions}, test_info
    )["end_to_end"]
    coherence_rows = coherence_accuracy_rows(
        {"end_to_end": action_predictions}, test_targets, test_info
    )

    epoch_features = build_epoch_features(states, local_info, window=WINDOW)
    preferred = np.argmax(test_targets, axis=1)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = [(train, test) for train, test in splitter.split(test_indices, preferred)]
    probe_predictions = cross_validated_action_probe(
        epoch_features["post_go"], test_targets, folds
    )
    probe = action_metrics(test_targets, probe_predictions, test_info)
    time_rows = time_resolved_probe_rows(epoch_features, local_info, test_indices)

    decision = classify_result(staged_summary, fixation, action, hard_row, losses)
    repeated = {"run": False, "reason": "standard behavior did not meet success rule"}
    if decision["classification"].startswith("A."):
        repeated = repeated_stimulus_evaluation(model, task, args.output_dir)

    raw_loss_audit = dual_head_loss_numpy(
        raw_outputs,
        np.asarray(dataset["y_internal"][:N_STANDARD_TEST]),
        np.asarray(dataset["mask_internal"][:N_STANDARD_TEST]),
    )
    np.savez(
        os.path.join(args.output_dir, "reconstruction_reference.npz"),
        dataset_indices=np.arange(5, dtype=int),
        raw_outputs=raw_outputs[:5],
        states=states[:5],
    )
    configuration = {
        "seed": SEED,
        "N_rec": N_REC,
        "rec_noise": REC_NOISE,
        "dt": DT,
        "tau": TAU,
        "alpha": DT / TAU,
        "activation": "rectified_linear",
        "batch_size": N_BATCH,
        "training_iters": TRAINING_ITERS,
        "optimizer": "PsychRNN default Adam",
        "learning_rate": 0.001,
        "gradient_clipping": "per-gradient norm 1.0 (PsychRNN default)",
        "training_sample_mode": "random over all 1000 frozen teacher trials",
        "standard_evaluation_indices": [0, 399],
        "loss": "active-mean fixation BCE + active-mean post-go 3-way soft CE",
        "loss_weights": {"fixation": 1.0, "action": 1.0},
        "regularization": regularization,
        "teacher_dataset": os.path.abspath(args.dataset),
        "teacher_sha256": teacher_hash_before,
    }
    write_json(os.path.join(args.output_dir, "configuration.json"), configuration)

    comparison = {
        "initial_categorical": {
            "fixation": staged_summary["fixation"]["native"],
            "action": staged_summary["action"]["native"],
            "hard_behavior": find_hard_row(staged_summary, "native"),
        },
        "staged_frozen_head": {
            "fixation": staged_summary["fixation"]["staged"],
            "action": staged_summary["action"]["staged"],
            "hard_behavior": find_hard_row(staged_summary, "staged"),
        },
        "end_to_end_dual_head": {
            "fixation": fixation,
            "action": action,
            "hard_behavior": hard_row,
        },
    }
    summary = {
        "success_rule_declared_before_training": SUCCESS_RULE,
        "decision": decision,
        "architecture": (
            "50-unit shared ReLU recurrent backbone; one sigmoid fixation logit; "
            "independent LEFT/RIGHT/SURE softmax logits"
        ),
        "configuration": configuration,
        "training": {
            "wall_seconds": float(training_wall_seconds),
            "psychrnn_training_seconds": float(training_seconds),
            "psychrnn_initialization_seconds": float(initialization_seconds),
            "logged_losses": [float(value) for value in losses],
            "all_losses_finite": bool(np.all(np.isfinite(losses))),
            "final_standard_loss_components": raw_loss_audit,
        },
        "comparison": comparison,
        "teacher_subsets": subsets,
        "value_distance": value_rows,
        "soft_confidence_correlations": correlations,
        "coherence_accuracy": coherence_rows,
        "representation_probe": {
            "native_dual_head": action,
            "held_out_linear_probe": probe,
            "ce_gap_probe_minus_native": float(
                probe["action_cross_entropy"] - action["action_cross_entropy"]
            ),
        },
        "time_resolved_held_out": time_rows,
        "time_resolved_target_definition": (
            "counterfactual value preference from 0.55 - empirical_P(correct), "
            "independent of actual sure-offer status"
        ),
        "repeated_stimulus": repeated,
        "artifacts": {
            "full_checkpoint": os.path.abspath(checkpoint_path),
            "recurrent_backbone": os.path.abspath(recurrent_path),
            "fixation_head": os.path.abspath(fixation_path),
            "action_head": os.path.abspath(action_path),
            "configuration": os.path.abspath(
                os.path.join(args.output_dir, "configuration.json")
            ),
            "training_history": os.path.abspath(
                os.path.join(args.output_dir, "training_history.csv")
            ),
        },
        "runtime_seconds": float(time.time() - wall_started),
        "teacher_sha256_before": teacher_hash_before,
        "teacher_sha256_after": sha256(args.dataset),
        "teacher_retrained": False,
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    model.destruct()
    print(json.dumps(summary, indent=2), flush=True)


def _read_csv(path):
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
