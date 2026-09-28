import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from rdm_population_diagnostics import write_json


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 2 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return np.nan
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def average_window(values, start, end):
    start = max(0, int(start))
    end = min(values.shape[0], int(end))
    if end <= start:
        end = min(values.shape[0], start + 1)
    return np.asarray(values[start:end], dtype=float).mean(axis=0)


def build_epoch_features(states, trial_info, window=10):
    features = {key: [] for key in ("late_motion", "pre_ts", "pre_go", "post_go")}
    for idx, info in enumerate(trial_info):
        stimulus_end = int(info["stimulus_end"])
        ts_onset = int(info["ts_onset"])
        delay_end = int(info["delay_end"])
        features["late_motion"].append(
            average_window(states[idx], stimulus_end - window, stimulus_end)
        )
        features["pre_ts"].append(average_window(states[idx], ts_onset - window, ts_onset))
        features["pre_go"].append(average_window(states[idx], delay_end - window, delay_end))
        features["post_go"].append(average_window(states[idx], delay_end, delay_end + window))
    return {key: np.asarray(value) for key, value in features.items()}


def build_targets(dataset, trial_info, window=10):
    targets = []
    for idx, info in enumerate(trial_info):
        delay_end = int(info["delay_end"])
        targets.append(
            average_window(dataset["y_internal"][idx, :, 1:4], delay_end, delay_end + window)
        )
    return np.asarray(targets, dtype=float)


def fit_softmax_linear(x_train, q_train, l2=1e-4, max_iter=3000):
    x_train = np.asarray(x_train, dtype=float)
    q_train = np.asarray(q_train, dtype=float)
    n_features = x_train.shape[1]
    n_classes = q_train.shape[1]

    def objective(flat):
        weights = flat[: n_features * n_classes].reshape(n_features, n_classes)
        bias = flat[n_features * n_classes :]
        logits = x_train @ weights + bias
        probabilities = softmax(logits)
        loss = -np.mean(np.sum(q_train * np.log(np.maximum(probabilities, 1e-12)), axis=1))
        loss += 0.5 * float(l2) * np.sum(weights * weights)
        residual = (probabilities - q_train) / x_train.shape[0]
        grad_weights = x_train.T @ residual + float(l2) * weights
        grad_bias = np.sum(residual, axis=0)
        return loss, np.concatenate([grad_weights.ravel(), grad_bias])

    initial = np.zeros(n_features * n_classes + n_classes, dtype=float)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Linear softmax optimization failed: {result.message}")
    weights = result.x[: n_features * n_classes].reshape(n_features, n_classes)
    bias = result.x[n_features * n_classes :]
    return weights, bias


def cross_validated_action_probe(features, targets, folds):
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    predictions = np.full_like(targets, np.nan, dtype=float)
    for train_idx, test_idx in folds:
        mean = np.mean(features[train_idx], axis=0)
        scale = np.std(features[train_idx], axis=0)
        scale[scale < 1e-8] = 1.0
        x_train = (features[train_idx] - mean) / scale
        x_test = (features[test_idx] - mean) / scale
        weights, bias = fit_softmax_linear(x_train, targets[train_idx])
        predictions[test_idx] = softmax(x_test @ weights + bias)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Action probe produced non-finite or missing held-out predictions")
    return predictions


def cross_validated_nonlinear_probe(features, targets, folds):
    """Fit one deliberately small nonlinear readout with soft-label weighting."""
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    predictions = np.full_like(targets, np.nan, dtype=float)
    classes = np.arange(targets.shape[1], dtype=int)
    for fold_number, (train_idx, test_idx) in enumerate(folds):
        mean = np.mean(features[train_idx], axis=0)
        scale = np.std(features[train_idx], axis=0)
        scale[scale < 1e-8] = 1.0
        x_train = (features[train_idx] - mean) / scale
        x_test = (features[test_idx] - mean) / scale

        # Expanding each trial once per action and weighting by target
        # probability yields the desired soft-target cross entropy.
        x_expanded = np.repeat(x_train, targets.shape[1], axis=0)
        y_expanded = np.tile(classes, len(train_idx))
        sample_weight = targets[train_idx].reshape(-1)
        classifier = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=2000,
            n_iter_no_change=50,
            tol=1e-6,
            random_state=7 + fold_number,
        )
        classifier.fit(x_expanded, y_expanded, sample_weight=sample_weight)
        if not np.array_equal(classifier.classes_, classes):
            raise RuntimeError("Nonlinear probe did not retain all three action classes")
        predictions[test_idx] = classifier.predict_proba(x_test)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Nonlinear probe produced non-finite or missing held-out predictions")
    return predictions


def action_metrics(targets, predictions, trial_info):
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    offered = np.asarray([bool(info["sure_available"]) for info in trial_info])
    direction = np.asarray([int(info["dir_choice"]) for info in trial_info])
    row = np.arange(len(trial_info))
    target_correct = targets[row, direction]
    predicted_correct = predictions[row, direction]
    predicted_incorrect = predictions[row, 1 - direction]
    target_margin = targets[:, 2] - target_correct
    predicted_margin = predictions[:, 2] - predicted_correct
    hard = np.argmax(predictions, axis=1)
    sure_favoring = offered & (target_margin > 0)
    eps = 1e-12
    ce = -np.sum(targets * np.log(np.maximum(predictions, eps)), axis=1)
    kl = np.sum(
        np.where(
            targets > 0,
            targets * np.log(np.maximum(targets, eps) / np.maximum(predictions, eps)),
            0.0,
        ),
        axis=1,
    )
    sure_auc = np.nan
    if len(np.unique(sure_favoring[offered])) == 2:
        sure_auc = float(roc_auc_score(sure_favoring[offered], predicted_margin[offered]))
    return {
        "action_cross_entropy": float(np.mean(ce)),
        "action_kl": float(np.mean(kl)),
        "per_action_mae": float(np.mean(np.abs(targets - predictions))),
        "action_margin_correlation": safe_corr(target_margin[offered], predicted_margin[offered]),
        "action_margin_sign_agreement": float(
            np.mean(np.sign(target_margin[offered]) == np.sign(predicted_margin[offered]))
        ),
        "teacher_sure_favoring_fidelity": float(np.mean(predicted_margin[sure_favoring] > 0)),
        "hard_p_sure_offered": float(np.mean(hard[offered] == 2)),
        "incorrect_direction_probability": float(np.mean(predicted_incorrect[offered])),
        "direction_decoding_accuracy": float(np.mean(np.argmax(predictions[:, :2], axis=1) == direction)),
        "no_sure_policy_accuracy": float(np.mean(hard[~offered] == direction[~offered])),
        "sure_vs_direction_auc": sure_auc,
        "n_trials": int(len(trial_info)),
        "n_offered": int(np.sum(offered)),
        "n_teacher_sure_favoring": int(np.sum(sure_favoring)),
    }


def cross_validated_phase_probes(pre_features, post_features, folds):
    pre_features = np.asarray(pre_features, dtype=float)
    post_features = np.asarray(post_features, dtype=float)
    n_trials = pre_features.shape[0]
    features = np.concatenate([pre_features, post_features], axis=0)
    labels = np.concatenate([np.ones(n_trials), np.zeros(n_trials)])
    phase_probability = np.full(2 * n_trials, np.nan)
    fixation_prediction = np.full(2 * n_trials, np.nan)
    fold_id = np.full(2 * n_trials, -1, dtype=int)
    for fold_number, (train_trials, test_trials) in enumerate(folds):
        train = np.concatenate([train_trials, train_trials + n_trials])
        test = np.concatenate([test_trials, test_trials + n_trials])
        mean = np.mean(features[train], axis=0)
        scale = np.std(features[train], axis=0)
        scale[scale < 1e-8] = 1.0
        x_train = (features[train] - mean) / scale
        x_test = (features[test] - mean) / scale
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=7)
        classifier.fit(x_train, labels[train])
        phase_probability[test] = classifier.predict_proba(x_test)[:, 1]
        regressor = Ridge(alpha=1.0)
        regressor.fit(x_train, labels[train])
        fixation_prediction[test] = regressor.predict(x_test)
        fold_id[test] = fold_number
    return {
        "phase_accuracy": float(accuracy_score(labels, phase_probability >= 0.5)),
        "phase_auc": float(roc_auc_score(labels, phase_probability)),
        "fixation_rmse": float(np.sqrt(np.mean((fixation_prediction - labels) ** 2))),
        "fixation_correlation": safe_corr(labels, fixation_prediction),
        "predicted_fixation_pre_go": float(np.mean(fixation_prediction[:n_trials])),
        "predicted_fixation_post_go": float(np.mean(fixation_prediction[n_trials:])),
        "phase_probability": phase_probability,
        "fixation_prediction": fixation_prediction,
        "labels": labels,
        "fold_id": fold_id,
    }


def value_distance_rows(targets, native, probe, trial_info):
    offered = np.asarray([bool(info["sure_available"]) for info in trial_info])
    direction = np.asarray([int(info["dir_choice"]) for info in trial_info])
    empirical = np.asarray([float(info["teacher_direction_success_proxy"]) for info in trial_info])
    delta = 0.55 - empirical
    teacher_class = np.argmax(targets, axis=1)
    native_class = np.argmax(native, axis=1)
    probe_class = np.argmax(probe, axis=1)
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
                "native_hard_disagreement": float(np.mean(native_class[selected] != teacher_class[selected])) if np.any(selected) else np.nan,
                "probe_hard_disagreement": float(np.mean(probe_class[selected] != teacher_class[selected])) if np.any(selected) else np.nan,
                "direction_fraction": float(np.mean(direction[selected] >= 0)) if np.any(selected) else np.nan,
            }
        )
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/model_freeze/teacher_seed7.npz")
    parser.add_argument(
        "--checkpoint",
        default="results/model_freeze/student_head_calibration_v2/stage_a/fix_1_action_1/weights.npz",
    )
    parser.add_argument("--output-dir", default="results/model_freeze/frozen_readout_diagnostic")
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.time()
    dataset_hash_before = sha256(args.dataset)
    checkpoint_hash_before = sha256(args.checkpoint)
    dataset = np.load(args.dataset, allow_pickle=True)
    trial_info = [json.loads(str(item)) for item in dataset["trial_info_json"]]
    for idx, info in enumerate(trial_info):
        for key in (
            "teacher_internal_margin",
            "teacher_sure_strength",
            "teacher_direction_success_proxy",
        ):
            info[key] = float(dataset[key][idx])

    from psychrnn.backend.simulation import BasicSimulator

    weights = dict(np.load(args.checkpoint, allow_pickle=True))
    simulator = BasicSimulator(params={"alpha": 0.1, "rec_noise": 0.0}, weights=weights)
    raw_outputs, states = simulator.run_trials(np.asarray(dataset["x"], dtype=np.float32))
    epoch_features = build_epoch_features(states, trial_info, window=10)
    targets = build_targets(dataset, trial_info, window=10)
    native_probabilities = []
    native_fixation = []
    for idx, info in enumerate(trial_info):
        start = int(info["delay_end"])
        end = min(raw_outputs.shape[1], start + 10)
        native_probabilities.append(np.mean(softmax(raw_outputs[idx, start:end, 1:4]), axis=0))
        native_fixation.append(float(np.mean(raw_outputs[idx, start:end, 0])))
    native_probabilities = np.asarray(native_probabilities)
    native_fixation = np.asarray(native_fixation)

    preferred = np.argmax(targets, axis=1)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    folds = [(train.copy(), test.copy()) for train, test in splitter.split(np.arange(len(trial_info)), preferred)]
    fold_for_trial = np.full(len(trial_info), -1, dtype=int)
    for fold_number, (_, test_idx) in enumerate(folds):
        fold_for_trial[test_idx] = fold_number

    epoch_predictions = {}
    epoch_metrics = []
    for epoch in ("late_motion", "pre_ts", "pre_go", "post_go"):
        predictions = cross_validated_action_probe(epoch_features[epoch], targets, folds)
        epoch_predictions[epoch] = predictions
        metrics = action_metrics(targets, predictions, trial_info)
        epoch_metrics.append(
            {
                "epoch": epoch,
                "action_margin_correlation": metrics["action_margin_correlation"],
                "sure_vs_direction_auc": metrics["sure_vs_direction_auc"],
                "direction_decoding_accuracy": metrics["direction_decoding_accuracy"],
            }
        )
    probe_probabilities = epoch_predictions["post_go"]
    native_metrics = action_metrics(targets, native_probabilities, trial_info)
    probe_metrics = action_metrics(targets, probe_probabilities, trial_info)
    ce_reduction = 1.0 - probe_metrics["action_cross_entropy"] / native_metrics["action_cross_entropy"]
    linear_probe_good = bool(
        ce_reduction >= 0.10
        and probe_metrics["action_margin_correlation"]
        >= native_metrics["action_margin_correlation"] + 0.10
    )
    nonlinear_probabilities = None
    nonlinear_metrics = None
    if not linear_probe_good:
        nonlinear_probabilities = cross_validated_nonlinear_probe(
            epoch_features["post_go"], targets, folds
        )
        nonlinear_metrics = action_metrics(targets, nonlinear_probabilities, trial_info)
    phase = cross_validated_phase_probes(
        epoch_features["pre_go"], epoch_features["post_go"], folds
    )
    value_rows = value_distance_rows(
        targets, native_probabilities, probe_probabilities, trial_info
    )

    trial_rows = []
    direction = np.asarray([int(info["dir_choice"]) for info in trial_info])
    for idx, info in enumerate(trial_info):
        correct = direction[idx]
        row = {
                "trial_index": idx,
                "fold": int(fold_for_trial[idx]),
                "sure_available": bool(info["sure_available"]),
                "dir_choice": int(direction[idx]),
                "teacher_sure_strength": float(info["teacher_sure_strength"]),
                "empirical_p_correct": float(info["teacher_direction_success_proxy"]),
                "delta_value": float(0.55 - info["teacher_direction_success_proxy"]),
                "target_left": float(targets[idx, 0]),
                "target_right": float(targets[idx, 1]),
                "target_sure": float(targets[idx, 2]),
                "native_left": float(native_probabilities[idx, 0]),
                "native_right": float(native_probabilities[idx, 1]),
                "native_sure": float(native_probabilities[idx, 2]),
                "probe_left": float(probe_probabilities[idx, 0]),
                "probe_right": float(probe_probabilities[idx, 1]),
                "probe_sure": float(probe_probabilities[idx, 2]),
                "native_margin": float(native_probabilities[idx, 2] - native_probabilities[idx, correct]),
                "probe_margin": float(probe_probabilities[idx, 2] - probe_probabilities[idx, correct]),
                "native_fixation_postgo": float(native_fixation[idx]),
            }
        if nonlinear_probabilities is not None:
            row.update(
                {
                    "nonlinear_left": float(nonlinear_probabilities[idx, 0]),
                    "nonlinear_right": float(nonlinear_probabilities[idx, 1]),
                    "nonlinear_sure": float(nonlinear_probabilities[idx, 2]),
                    "nonlinear_margin": float(
                        nonlinear_probabilities[idx, 2]
                        - nonlinear_probabilities[idx, correct]
                    ),
                }
            )
        trial_rows.append(row)
    phase_rows = []
    n_trials = len(trial_info)
    for idx in range(n_trials):
        for offset, phase_name in ((0, "pre_go"), (n_trials, "post_go")):
            row_idx = idx + offset
            phase_rows.append(
                {
                    "trial_index": idx,
                    "fold": int(phase["fold_id"][row_idx]),
                    "phase": phase_name,
                    "target_fixation": float(phase["labels"][row_idx]),
                    "phase_probability_pre_go": float(phase["phase_probability"][row_idx]),
                    "predicted_fixation": float(phase["fixation_prediction"][row_idx]),
                }
            )
    write_csv(os.path.join(args.output_dir, "trial_predictions.csv"), trial_rows)
    write_csv(os.path.join(args.output_dir, "phase_predictions.csv"), phase_rows)
    write_csv(os.path.join(args.output_dir, "time_resolved.csv"), epoch_metrics)
    write_csv(os.path.join(args.output_dir, "value_distance.csv"), value_rows)

    summary = {
        "dataset": os.path.abspath(args.dataset),
        "checkpoint": os.path.abspath(args.checkpoint),
        "dataset_sha256_before": dataset_hash_before,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "inference": "deterministic recurrent noise=0; no model training",
        "cross_validation": {
            "n_folds": 5,
            "split_unit": "trial",
            "stratified_by": "teacher preferred LEFT/RIGHT/SURE class",
            "all_trials_have_one_held_out_prediction": bool(np.all(fold_for_trial >= 0)),
        },
        "native": native_metrics,
        "linear_probe": probe_metrics,
        "readout_ceiling": {
            "cross_entropy_relative_reduction": float(ce_reduction),
            "kl_relative_reduction": float(
                1.0 - probe_metrics["action_kl"] / native_metrics["action_kl"]
            ),
            "linear_probe_meets_substantial_improvement_threshold": linear_probe_good,
        },
        "native_postgo_fixation": {
            "mean": float(np.mean(native_fixation)),
            "mean_abs": float(np.mean(np.abs(native_fixation))),
            "rms": float(np.sqrt(np.mean(native_fixation ** 2))),
        },
        "phase_probe": {key: value for key, value in phase.items() if not isinstance(value, np.ndarray)},
        "time_resolved": epoch_metrics,
        "value_distance": value_rows,
        "nonlinear_probe": {
            "run": nonlinear_metrics is not None,
            "architecture": "one hidden ReLU layer, 16 units, Adam, alpha=1e-4"
            if nonlinear_metrics is not None
            else None,
            "metrics": nonlinear_metrics,
            "reason": "linear probe met the predeclared substantial-improvement threshold"
            if linear_probe_good
            else "linear probe did not meet threshold; ran the single allowed small nonlinear probe",
        },
        "runtime_seconds": float(time.time() - started),
        "dataset_sha256_after": sha256(args.dataset),
        "checkpoint_sha256_after": sha256(args.checkpoint),
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
