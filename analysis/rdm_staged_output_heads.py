import argparse
import csv
import json
import os
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, roc_auc_score

from rdm_frozen_readout_diagnostic import (
    action_metrics,
    average_window,
    build_epoch_features,
    build_targets,
    fit_softmax_linear,
    safe_corr,
    sha256,
    softmax,
    write_csv,
)
from rdm_population_diagnostics import write_json


EXPECTED_BACKBONE_SHA256 = "bab443333a2b962a10bee9ed675e13216eb352ca76114f5052a2db133b5a1329"
EXPECTED_TEACHER_SHA256 = "8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d"
L2 = 1e-4
WINDOW = 10
STANDARD_TEST_STOP = 400

# Declared before running the experiment. A requires a clean fixation result,
# material soft-policy gains, and no meaningful hard-policy tradeoff.
SUCCESS_RULE = {
    "fixation_postgo_max": 0.05,
    "fixation_prego_min": 0.95,
    "fixation_auc_min": 0.99,
    "action_ce_relative_reduction_min": 0.10,
    "action_kl_relative_reduction_min": 0.15,
    "action_margin_r_gain_min": 0.05,
    "direction_accuracy_tolerance": 0.005,
}


def sigmoid(logits):
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def standardization(features):
    features = np.asarray(features, dtype=float)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def fit_logistic_linear(features, labels, l2=L2, max_iter=3000):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n_features = features.shape[1]

    def objective(flat):
        weights = flat[:n_features]
        bias = flat[-1]
        probabilities = sigmoid(features @ weights + bias)
        eps = 1e-12
        loss = -np.mean(
            labels * np.log(np.maximum(probabilities, eps))
            + (1.0 - labels) * np.log(np.maximum(1.0 - probabilities, eps))
        )
        loss += 0.5 * float(l2) * np.sum(weights * weights)
        residual = (probabilities - labels) / len(labels)
        gradient = np.concatenate(
            [features.T @ residual + float(l2) * weights, [np.sum(residual)]]
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(n_features + 1, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Fixation-head optimization failed: {result.message}")
    return result.x[:n_features], float(result.x[-1])


def fit_fixation_head_from_samples(features, labels, l2=L2):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    mean, scale = standardization(features)
    weights, bias = fit_logistic_linear((features - mean) / scale, labels, l2=l2)
    return {
        "weights": weights,
        "bias": np.asarray(bias),
        "mean": mean,
        "scale": scale,
        "l2": np.asarray(float(l2)),
    }


def fit_action_head_from_samples(features, targets, l2=L2):
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    mean, scale = standardization(features)
    weights, bias = fit_softmax_linear((features - mean) / scale, targets, l2=l2)
    return {
        "weights": weights,
        "bias": bias,
        "mean": mean,
        "scale": scale,
        "l2": np.asarray(float(l2)),
    }


def predict_fixation_head(features, head):
    features = np.asarray(features, dtype=float)
    standardized = (features - head["mean"]) / head["scale"]
    return sigmoid(standardized @ head["weights"] + float(head["bias"]))


def predict_action_head(features, head):
    features = np.asarray(features, dtype=float)
    standardized = (features - head["mean"]) / head["scale"]
    return softmax(standardized @ head["weights"] + head["bias"])


def save_head(path, head):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **head)


def load_head(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def fixed_train_test_split(n_trials, test_stop=STANDARD_TEST_STOP):
    if int(n_trials) <= int(test_stop):
        raise ValueError("Dataset is too small for the fixed standard-evaluation split")
    test = np.arange(0, int(test_stop), dtype=int)
    train = np.arange(int(test_stop), int(n_trials), dtype=int)
    if np.intersect1d(train, test).size:
        raise RuntimeError("Train/test trial overlap")
    return train, test


def phase_samples(states, trial_info, indices, window=WINDOW):
    features = []
    labels = []
    trial_indices = []
    phases = []
    time_steps = []
    for idx in np.asarray(indices, dtype=int):
        go = int(trial_info[idx]["delay_end"])
        for phase, label, start, end in (
            ("pre_go", 1.0, go - window, go),
            ("post_go", 0.0, go, go + window),
        ):
            start = max(0, start)
            end = min(states.shape[1], end)
            for step in range(start, end):
                features.append(states[idx, step])
                labels.append(label)
                trial_indices.append(idx)
                phases.append(phase)
                time_steps.append(step)
    return (
        np.asarray(features, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(trial_indices, dtype=int),
        np.asarray(phases),
        np.asarray(time_steps, dtype=int),
    )


def action_samples(states, targets, trial_info, indices, window=WINDOW):
    features = []
    soft_targets = []
    trial_indices = []
    time_steps = []
    for idx in np.asarray(indices, dtype=int):
        go = int(trial_info[idx]["delay_end"])
        end = min(states.shape[1], go + window)
        for step in range(go, end):
            features.append(states[idx, step])
            soft_targets.append(targets[idx])
            trial_indices.append(idx)
            time_steps.append(step)
    return (
        np.asarray(features, dtype=float),
        np.asarray(soft_targets, dtype=float),
        np.asarray(trial_indices, dtype=int),
        np.asarray(time_steps, dtype=int),
    )


def trial_action_predictions(states, head, trial_info, indices, window=WINDOW):
    predictions = []
    for idx in np.asarray(indices, dtype=int):
        go = int(trial_info[idx]["delay_end"])
        end = min(states.shape[1], go + window)
        predictions.append(np.mean(predict_action_head(states[idx, go:end], head), axis=0))
    return np.asarray(predictions, dtype=float)


def native_trial_predictions(raw_outputs, trial_info, indices, window=WINDOW):
    action = []
    for idx in np.asarray(indices, dtype=int):
        go = int(trial_info[idx]["delay_end"])
        end = min(raw_outputs.shape[1], go + window)
        action.append(np.mean(softmax(raw_outputs[idx, go:end, 1:4]), axis=0))
    return np.asarray(action, dtype=float)


def fixation_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    pre = labels == 1
    post = labels == 0
    return {
        "pre_go_mean": float(np.mean(predictions[pre])),
        "post_go_mean": float(np.mean(predictions[post])),
        "rmse": float(np.sqrt(np.mean((predictions - labels) ** 2))),
        "correlation": safe_corr(labels, predictions),
        "classification_accuracy": float(accuracy_score(labels, predictions >= 0.5)),
        "auc": float(roc_auc_score(labels, predictions)),
        "n_samples": int(len(labels)),
    }


def action_subset_metrics(targets, predictions, trial_info):
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    offered = np.asarray([bool(info["sure_available"]) for info in trial_info])
    direction = np.asarray([int(info["dir_choice"]) for info in trial_info])
    row = np.arange(len(trial_info))
    target_margin = targets[:, 2] - targets[row, direction]
    predicted_margin = predictions[:, 2] - predictions[row, direction]
    hard = np.argmax(predictions, axis=1)

    def summarize(mask):
        if not np.any(mask):
            return {"n": 0}
        return {
            "n": int(np.sum(mask)),
            "target_p_sure": float(np.mean(targets[mask, 2])),
            "predicted_p_sure": float(np.mean(predictions[mask, 2])),
            "predicted_p_correct": float(np.mean(predictions[row, direction][mask])),
            "predicted_p_incorrect": float(np.mean(predictions[row, 1 - direction][mask])),
            "predicted_margin": float(np.mean(predicted_margin[mask])),
            "sure_over_correct": float(np.mean(predicted_margin[mask] > 0)),
            "teacher_argmax_fidelity": float(
                np.mean(hard[mask] == np.argmax(targets[mask], axis=1))
            ),
        }

    return {
        "teacher_sure_favoring": summarize(offered & (target_margin > 0)),
        "teacher_direction_favoring": summarize(offered & (target_margin < 0)),
    }


def hard_behavior_rows(predictions_by_model, targets, trial_info):
    offered = np.asarray([bool(info["sure_available"]) for info in trial_info])
    direction = np.asarray([int(info["dir_choice"]) for info in trial_info])
    teacher = np.argmax(targets, axis=1)
    offered_direction = offered & (teacher != 2)
    rows = []
    for label, probabilities in predictions_by_model.items():
        hard = np.argmax(probabilities, axis=1)
        rows.append(
            {
                "model": label,
                "p_left": float(np.mean(hard == 0)),
                "p_right": float(np.mean(hard == 1)),
                "p_sure": float(np.mean(hard == 2)),
                "p_sure_offered": float(np.mean(hard[offered] == 2)),
                "offered_teacher_direction_accuracy": float(
                    np.mean(hard[offered_direction] == direction[offered_direction])
                ),
                "no_sure_direction_accuracy": float(
                    np.mean(hard[~offered] == direction[~offered])
                ),
                "teacher_policy_accuracy": float(np.mean(hard == teacher)),
            }
        )
    return rows


def coherence_accuracy_rows(predictions_by_model, targets, trial_info):
    offered = np.asarray([bool(info["sure_available"]) for info in trial_info])
    direction = np.asarray([int(info["dir_choice"]) for info in trial_info])
    coherence = np.asarray([float(info["coh"]) for info in trial_info])
    rows = []
    for value in sorted(np.unique(coherence[~offered])):
        selected = (~offered) & np.isclose(coherence, value)
        row = {"coherence": float(value), "n": int(np.sum(selected))}
        for label, probabilities in predictions_by_model.items():
            hard = np.argmax(probabilities, axis=1)
            row[f"{label}_accuracy"] = float(np.mean(hard[selected] == direction[selected]))
        rows.append(row)
    return rows


def value_distance_rows(targets, native, staged, trial_info):
    offered = np.asarray([bool(info["sure_available"]) for info in trial_info])
    empirical = np.asarray(
        [float(info["teacher_direction_success_proxy"]) for info in trial_info]
    )
    delta = 0.55 - empirical
    teacher = np.argmax(targets, axis=1)
    native_hard = np.argmax(native, axis=1)
    staged_hard = np.argmax(staged, axis=1)
    bins = [
        ("strongly_direction", -np.inf, -0.20),
        ("weakly_direction", -0.20, -0.05),
        ("near_boundary", -0.05, 0.05),
        ("weakly_sure", 0.05, 0.15),
        ("strongly_sure", 0.15, np.inf),
    ]
    rows = []
    for label, low, high in bins:
        selected = offered & (delta >= low) & (delta < high)
        rows.append(
            {
                "bin": label,
                "low": "-inf" if not np.isfinite(low) else float(low),
                "high": "inf" if not np.isfinite(high) else float(high),
                "n": int(np.sum(selected)),
                "native_disagreement": float(np.mean(native_hard[selected] != teacher[selected]))
                if np.any(selected)
                else np.nan,
                "staged_disagreement": float(np.mean(staged_hard[selected] != teacher[selected]))
                if np.any(selected)
                else np.nan,
            }
        )
    return rows


def soft_confidence_correlations(predictions_by_model, trial_info):
    offered = np.asarray([bool(info["sure_available"]) for info in trial_info])
    variables = {
        "teacher_sure_strength": np.asarray(
            [float(info["teacher_sure_strength"]) for info in trial_info]
        ),
        "empirical_p_correct": np.asarray(
            [float(info["teacher_direction_success_proxy"]) for info in trial_info]
        ),
        "internal_margin": np.asarray(
            [float(info["teacher_internal_margin"]) for info in trial_info]
        ),
        "coherence": np.asarray([float(info["coh"]) for info in trial_info]),
        "stimulus_duration": np.asarray(
            [float(info["stimulus_duration"]) for info in trial_info]
        ),
    }
    result = {}
    for label, probabilities in predictions_by_model.items():
        result[label] = {
            name: safe_corr(values[offered], probabilities[offered, 2])
            for name, values in variables.items()
        }
    return result


def counterfactual_targets(trial_info):
    """Sure preference from value only, independent of future offer status."""
    targets = np.zeros((len(trial_info), 3), dtype=float)
    for idx, info in enumerate(trial_info):
        p_correct = float(info["teacher_direction_success_proxy"])
        advantage = 0.55 - p_correct
        raw = float(sigmoid(advantage / 0.15))
        sure_strength = float(np.clip(0.5 + 0.8 * (raw - 0.5), 0.05, 0.95))
        direction = int(info["dir_choice"])
        targets[idx, direction] = 1.0 - sure_strength
        targets[idx, 2] = sure_strength
    return targets


def time_resolved_rows(epoch_features, action_head, trial_info, indices):
    indices = np.asarray(indices, dtype=int)
    info = [trial_info[idx] for idx in indices]
    targets = counterfactual_targets(info)
    direction = np.asarray([int(item["dir_choice"]) for item in info])
    row = np.arange(len(info))
    target_margin = targets[:, 2] - targets[row, direction]
    target_sure = target_margin > 0
    rows = []
    for epoch in ("late_motion", "pre_ts", "pre_go", "post_go"):
        probabilities = predict_action_head(epoch_features[epoch][indices], action_head)
        predicted_margin = probabilities[:, 2] - probabilities[row, direction]
        rows.append(
            {
                "epoch": epoch,
                "target_definition": "counterfactual value preference independent of sure offer",
                "action_margin_correlation": safe_corr(target_margin, predicted_margin),
                "sure_vs_direction_auc": float(roc_auc_score(target_sure, predicted_margin)),
                "direction_accuracy": float(
                    np.mean(np.argmax(probabilities[:, :2], axis=1) == direction)
                ),
            }
        )
    return rows


def load_previous_probe_predictions(path, indices):
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["trial_index"])] = np.asarray(
                [float(row["probe_left"]), float(row["probe_right"]), float(row["probe_sure"])]
            )
    return np.asarray([rows[int(idx)] for idx in indices], dtype=float)


def classify_result(native, staged, fixation):
    fixation_clean = bool(
        fixation["post_go_mean"] <= SUCCESS_RULE["fixation_postgo_max"]
        and fixation["pre_go_mean"] >= SUCCESS_RULE["fixation_prego_min"]
        and fixation["auc"] >= SUCCESS_RULE["fixation_auc_min"]
    )
    ce_gain = 1.0 - staged["action_cross_entropy"] / native["action_cross_entropy"]
    kl_gain = 1.0 - staged["action_kl"] / native["action_kl"]
    margin_gain = staged["action_margin_correlation"] - native["action_margin_correlation"]
    action_material = bool(
        ce_gain >= SUCCESS_RULE["action_ce_relative_reduction_min"]
        and kl_gain >= SUCCESS_RULE["action_kl_relative_reduction_min"]
        and margin_gain >= SUCCESS_RULE["action_margin_r_gain_min"]
        and staged["incorrect_direction_probability"]
        <= native["incorrect_direction_probability"]
        and staged["teacher_sure_favoring_fidelity"]
        >= native["teacher_sure_favoring_fidelity"]
        and staged["direction_decoding_accuracy"]
        >= native["direction_decoding_accuracy"]
        - SUCCESS_RULE["direction_accuracy_tolerance"]
    )
    if fixation_clean and action_material:
        decision = "A. OUTPUT-HEAD CLEANUP SUCCESSFUL"
    elif fixation_clean:
        decision = "B. FIXATION SOLVED, ACTION REPRESENTATION/TRAINING STILL LIMITED"
    else:
        decision = "C. STAGED READOUT FAILED"
    return {
        "decision": decision,
        "fixation_clean": fixation_clean,
        "action_material": action_material,
        "action_ce_relative_reduction": float(ce_gain),
        "action_kl_relative_reduction": float(kl_gain),
        "action_margin_r_gain": float(margin_gain),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/model_freeze/teacher_seed7.npz")
    parser.add_argument(
        "--checkpoint",
        default="results/model_freeze/student_head_calibration_v2/stage_a/fix_1_action_1/weights.npz",
    )
    parser.add_argument(
        "--previous-probe-csv",
        default="results/model_freeze/frozen_readout_diagnostic/trial_predictions.csv",
    )
    parser.add_argument(
        "--output-dir", default="results/model_freeze/staged_output_head_experiment"
    )
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.time()

    backbone_hash_before = sha256(args.checkpoint)
    teacher_hash_before = sha256(args.dataset)
    if backbone_hash_before != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError(f"Unexpected backbone hash: {backbone_hash_before}")
    if teacher_hash_before != EXPECTED_TEACHER_SHA256:
        raise RuntimeError(f"Unexpected teacher dataset hash: {teacher_hash_before}")

    dataset = np.load(args.dataset, allow_pickle=True)
    trial_info = [json.loads(str(item)) for item in dataset["trial_info_json"]]
    for idx, info in enumerate(trial_info):
        info["teacher_internal_margin"] = float(dataset["teacher_internal_margin"][idx])
        info["teacher_sure_strength"] = float(dataset["teacher_sure_strength"][idx])
        info["teacher_direction_success_proxy"] = float(
            dataset["teacher_direction_success_proxy"][idx]
        )

    from psychrnn.backend.simulation import BasicSimulator

    frozen_weights = dict(np.load(args.checkpoint, allow_pickle=True))
    simulator = BasicSimulator(
        params={"alpha": 0.1, "rec_noise": 0.0}, weights=frozen_weights
    )
    raw_outputs, states = simulator.run_trials(np.asarray(dataset["x"], dtype=np.float32))
    targets = build_targets(dataset, trial_info, window=WINDOW)
    epoch_features = build_epoch_features(states, trial_info, window=WINDOW)
    train_indices, test_indices = fixed_train_test_split(len(trial_info))
    if np.intersect1d(train_indices, test_indices).size:
        raise RuntimeError("Train and test trials overlap")

    fix_x_train, fix_y_train, fix_trial_train, _, _ = phase_samples(
        states, trial_info, train_indices
    )
    action_x_train, action_y_train, action_trial_train, _ = action_samples(
        states, targets, trial_info, train_indices
    )
    if np.intersect1d(np.unique(fix_trial_train), test_indices).size:
        raise RuntimeError("Fixation training includes test trials")
    if np.intersect1d(np.unique(action_trial_train), test_indices).size:
        raise RuntimeError("Action training includes test trials")

    fixation_head = fit_fixation_head_from_samples(fix_x_train, fix_y_train)
    action_head = fit_action_head_from_samples(action_x_train, action_y_train)

    fixation_path = os.path.join(args.output_dir, "fixation_head_weights.npz")
    action_path = os.path.join(args.output_dir, "action_head_weights.npz")
    save_head(fixation_path, fixation_head)
    save_head(action_path, action_head)
    reconstructed_fixation_head = load_head(fixation_path)
    reconstructed_action_head = load_head(action_path)

    fix_x_test, fix_y_test, fix_trial_test, fix_phase_test, fix_steps_test = phase_samples(
        states, trial_info, test_indices
    )
    staged_fixation = predict_fixation_head(fix_x_test, reconstructed_fixation_head)
    native_fixation = raw_outputs[fix_trial_test, fix_steps_test, 0]
    staged_fixation_metrics = fixation_metrics(fix_y_test, staged_fixation)
    native_fixation_metrics = fixation_metrics(fix_y_test, native_fixation)

    test_info = [trial_info[idx] for idx in test_indices]
    test_targets = targets[test_indices]
    native_probabilities = native_trial_predictions(raw_outputs, trial_info, test_indices)
    staged_probabilities = trial_action_predictions(
        states, reconstructed_action_head, trial_info, test_indices
    )

    # Split-matched version of the prior linear ceiling: average each post-go
    # hidden window, fit on trials 400:999, evaluate only on 0:399.
    averaged_postgo = epoch_features["post_go"]
    ceiling_head = fit_action_head_from_samples(
        averaged_postgo[train_indices], targets[train_indices]
    )
    ceiling_probabilities = predict_action_head(averaged_postgo[test_indices], ceiling_head)
    previous_probe_probabilities = load_previous_probe_predictions(
        args.previous_probe_csv, test_indices
    )

    native_metrics = action_metrics(test_targets, native_probabilities, test_info)
    ceiling_metrics = action_metrics(test_targets, ceiling_probabilities, test_info)
    staged_metrics = action_metrics(test_targets, staged_probabilities, test_info)
    previous_probe_metrics = action_metrics(
        test_targets, previous_probe_probabilities, test_info
    )
    subsets = {
        label: action_subset_metrics(test_targets, probabilities, test_info)
        for label, probabilities in {
            "native": native_probabilities,
            "previous_oof_probe": previous_probe_probabilities,
            "split_matched_ceiling": ceiling_probabilities,
            "staged": staged_probabilities,
        }.items()
    }
    comparable_predictions = {
        "native": native_probabilities,
        "ceiling": ceiling_probabilities,
        "staged": staged_probabilities,
    }
    hard_rows = hard_behavior_rows(comparable_predictions, test_targets, test_info)
    coherence_rows = coherence_accuracy_rows(
        comparable_predictions, test_targets, test_info
    )
    value_rows = value_distance_rows(
        test_targets, native_probabilities, staged_probabilities, test_info
    )
    correlation_predictions = {
        "native": native_probabilities,
        "previous_oof_probe": previous_probe_probabilities,
        "split_matched_ceiling": ceiling_probabilities,
        "staged": staged_probabilities,
    }
    correlations = soft_confidence_correlations(correlation_predictions, test_info)
    time_rows = time_resolved_rows(
        epoch_features, reconstructed_action_head, trial_info, test_indices
    )
    decision = classify_result(native_metrics, staged_metrics, staged_fixation_metrics)

    trial_rows = []
    for local_idx, dataset_idx in enumerate(test_indices):
        info = test_info[local_idx]
        row = {
            "dataset_index": int(dataset_idx),
            "sure_available": bool(info["sure_available"]),
            "coherence": float(info["coh"]),
            "duration_ms": float(info["stimulus_duration"]),
            "dir_choice": int(info["dir_choice"]),
            "teacher_sure_strength": float(info["teacher_sure_strength"]),
            "empirical_p_correct": float(info["teacher_direction_success_proxy"]),
            "internal_margin": float(info["teacher_internal_margin"]),
            "delta_value": float(0.55 - info["teacher_direction_success_proxy"]),
        }
        for prefix, values in (
            ("target", test_targets[local_idx]),
            ("native", native_probabilities[local_idx]),
            ("previous_probe", previous_probe_probabilities[local_idx]),
            ("ceiling", ceiling_probabilities[local_idx]),
            ("staged", staged_probabilities[local_idx]),
        ):
            row[f"{prefix}_left"] = float(values[0])
            row[f"{prefix}_right"] = float(values[1])
            row[f"{prefix}_sure"] = float(values[2])
        trial_rows.append(row)

    fixation_rows = []
    for idx in range(len(fix_y_test)):
        fixation_rows.append(
            {
                "dataset_index": int(fix_trial_test[idx]),
                "time_step": int(fix_steps_test[idx]),
                "phase": str(fix_phase_test[idx]),
                "target_fixation": float(fix_y_test[idx]),
                "native_fixation": float(native_fixation[idx]),
                "staged_fixation_probability": float(staged_fixation[idx]),
            }
        )

    write_csv(os.path.join(args.output_dir, "test_trial_predictions.csv"), trial_rows)
    write_csv(os.path.join(args.output_dir, "fixation_predictions.csv"), fixation_rows)
    write_csv(os.path.join(args.output_dir, "hard_behavior.csv"), hard_rows)
    write_csv(os.path.join(args.output_dir, "coherence_accuracy.csv"), coherence_rows)
    write_csv(os.path.join(args.output_dir, "value_distance.csv"), value_rows)
    write_csv(os.path.join(args.output_dir, "time_resolved.csv"), time_rows)

    metadata = {
        "model": "frozen recurrent backbone with independent fixation and action heads",
        "backbone_checkpoint": os.path.abspath(args.checkpoint),
        "backbone_sha256": backbone_hash_before,
        "teacher_dataset": os.path.abspath(args.dataset),
        "teacher_dataset_sha256": teacher_hash_before,
        "fixation_head_weights": os.path.abspath(fixation_path),
        "action_head_weights": os.path.abspath(action_path),
        "normalization_stored_with_each_head": True,
        "window_steps": WINDOW,
        "train_trial_range": [400, 999],
        "validation_trials": [],
        "test_trial_range": [0, 399],
        "action_order": ["LEFT", "RIGHT", "SURE"],
        "fixation_never_enters_action_softmax": True,
    }
    write_json(os.path.join(args.output_dir, "model_metadata.json"), metadata)

    summary = {
        "decision_rule_declared_before_results": SUCCESS_RULE,
        "decision": decision,
        "split": {
            "source": "preserved standard evaluation trials 0:399",
            "unit": "whole trial",
            "train_indices": [400, 999],
            "n_train_trials": int(len(train_indices)),
            "validation_indices": None,
            "n_validation_trials": 0,
            "validation_reason": "fixed L2 and no tuning or early stopping",
            "test_indices": [0, 399],
            "n_test_trials": int(len(test_indices)),
            "overlap_count": int(np.intersect1d(train_indices, test_indices).size),
        },
        "objectives": {
            "fixation": "mean binary cross entropy plus 0.5*1e-4*||w_fix||^2",
            "action": "mean soft-label categorical cross entropy plus 0.5*1e-4*||W_action||^2",
            "target_weighting": "none",
        },
        "sample_counts": {
            "fixation_training_samples": int(len(fix_y_train)),
            "fixation_training_trials": int(len(np.unique(fix_trial_train))),
            "action_training_samples": int(len(action_y_train)),
            "action_training_trials": int(len(np.unique(action_trial_train))),
            "fixation_test_samples": int(len(fix_y_test)),
            "action_test_trials": int(len(test_indices)),
        },
        "fixation": {
            "native": native_fixation_metrics,
            "staged": staged_fixation_metrics,
        },
        "action": {
            "native": native_metrics,
            "previous_oof_probe_on_test_subset": previous_probe_metrics,
            "split_matched_linear_ceiling": ceiling_metrics,
            "staged": staged_metrics,
        },
        "subsets": subsets,
        "hard_behavior": hard_rows,
        "coherence_accuracy": coherence_rows,
        "value_distance": value_rows,
        "soft_confidence_correlations": correlations,
        "time_resolved": time_rows,
        "time_resolved_target_definition": (
            "counterfactual sure preference computed from 0.55 - empirical_P(correct), "
            "then the unchanged sigmoid/blend/clip mapping; independent of offer status"
        ),
        "artifacts": metadata,
        "reconstruction": {
            "fixation_max_abs_difference": float(
                np.max(
                    np.abs(
                        staged_fixation
                        - predict_fixation_head(fix_x_test, load_head(fixation_path))
                    )
                )
            ),
            "action_max_abs_difference": float(
                np.max(
                    np.abs(
                        staged_probabilities
                        - trial_action_predictions(
                            states, load_head(action_path), trial_info, test_indices
                        )
                    )
                )
            ),
        },
        "runtime_seconds": float(time.time() - started),
        "backbone_sha256_before": backbone_hash_before,
        "backbone_sha256_after": sha256(args.checkpoint),
        "teacher_dataset_sha256_before": teacher_hash_before,
        "teacher_dataset_sha256_after": sha256(args.dataset),
        "teacher_retrained": False,
        "recurrent_backbone_retrained_or_modified": False,
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
