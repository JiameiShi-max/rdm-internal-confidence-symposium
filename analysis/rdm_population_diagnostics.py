import csv
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y) if x.ndim == 2 else np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def linear_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x, y = finite_xy(x, y)
    if y.size < 2 or np.var(y) < 1e-12:
        return np.nan
    design = np.column_stack([np.ones(x.shape[0]), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-12:
        return np.nan
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def residual_after_linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x_valid, y_valid = finite_xy(x, y)
    residual = np.full(y.shape[0], np.nan, dtype=float)
    if y_valid.size < 2:
        return residual
    design = np.column_stack([np.ones(x_valid.shape[0]), x_valid])
    coef, _, _, _ = np.linalg.lstsq(design, y_valid, rcond=None)
    valid_mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    residual[valid_mask] = y_valid - design @ coef
    return residual


def safe_corrcoef(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize_axis_r2_metrics(evidence_axis, sure_axis, p_sure, confidence, offered=None):
    evidence_axis = np.abs(np.asarray(evidence_axis, dtype=float))
    sure_axis = np.asarray(sure_axis, dtype=float)
    p_sure = np.asarray(p_sure, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if offered is None:
        offered = np.ones(p_sure.shape[0], dtype=bool)
    else:
        offered = np.asarray(offered, dtype=bool)

    residual_p_sure = residual_after_linear_regression(evidence_axis[offered], p_sure[offered])
    return {
        "r2_evidence_axis_explains_confidence": linear_r2(evidence_axis, confidence),
        "r2_evidence_axis_explains_p_sure": linear_r2(evidence_axis[offered], p_sure[offered]),
        "r2_sure_axis_explains_p_sure": linear_r2(sure_axis[offered], p_sure[offered]),
        "r2_sure_axis_explains_residual_p_sure_after_evidence": linear_r2(
            sure_axis[offered],
            residual_p_sure,
        ),
        "corr_evidence_axis_confidence": safe_corrcoef(evidence_axis, confidence),
        "corr_evidence_axis_p_sure": safe_corrcoef(evidence_axis[offered], p_sure[offered]),
        "corr_sure_axis_p_sure": safe_corrcoef(sure_axis[offered], p_sure[offered]),
        "corr_sure_axis_residual_p_sure_after_evidence": safe_corrcoef(
            sure_axis[offered],
            residual_p_sure,
        ),
    }


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_repeated_trials_csv(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def repeated_stimulus_paths(summary_path):
    root, _ = os.path.splitext(summary_path)
    return {
        "trials": f"{root}_repeated_stimulus_trials.csv",
        "summary": f"{root}_repeated_stimulus_summary.json",
        "fig": f"{root}_repeated_stimulus_dynamics.png",
    }


def pre_ts_leakage_paths(summary_path):
    root, _ = os.path.splitext(summary_path)
    return {
        "summary": f"{root}_pre_ts_leakage_summary.json",
        "fig": f"{root}_pre_ts_leakage_plot.png",
    }


def sure_output_dynamics_paths(summary_path):
    root, _ = os.path.splitext(summary_path)
    return {
        "summary": f"{root}_sure_output_dynamics_summary.json",
        "trials": f"{root}_sure_output_dynamics_trials.csv",
        "fig": f"{root}_sure_output_dynamics.png",
    }


def deterministic_repeated_stimulus_paths(summary_path):
    root, _ = os.path.splitext(summary_path)
    return {
        "trials": f"{root}_deterministic_repeated_stimulus_trials.csv",
        "summary": f"{root}_deterministic_repeated_stimulus_summary.json",
        "fig": f"{root}_deterministic_repeated_stimulus_dynamics.png",
    }


def population_dynamics_paths(summary_path):
    root, _ = os.path.splitext(summary_path)
    return {
        "summary": f"{root}_population_dynamics_summary.json",
        "fig": f"{root}_population_dynamics.png",
    }


def model_test_with_fixed_batch(model, task, batch_x):
    batch_x = np.asarray(batch_x, dtype=np.float32)
    target_batch = int(task.N_batch)
    outputs_all = []
    states_all = []
    for start in range(0, batch_x.shape[0], target_batch):
        end = min(start + target_batch, batch_x.shape[0])
        chunk = batch_x[start:end]
        valid = chunk.shape[0]
        if valid < target_batch:
            pad = np.repeat(chunk[-1:, :, :], target_batch - valid, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        outputs, states = model.test(chunk)
        outputs_all.append(outputs[:valid])
        states_all.append(states[:valid])
    return np.concatenate(outputs_all, axis=0), np.concatenate(states_all, axis=0)


def simulator_test(model, batch_x, rec_noise=0.0):
    from psychrnn.backend.simulation import BasicSimulator

    weights = model.get_weights()
    simulator = BasicSimulator(
        params={
            "alpha": float(model.alpha),
            "rec_noise": float(rec_noise),
        },
        weights=weights,
    )
    return simulator.run_trials(np.asarray(batch_x, dtype=np.float32))


def _readout_choice(outputs_trial, delay_end, window, action_only=False):
    start = int(delay_end)
    end = min(outputs_trial.shape[0], start + int(window))
    if end <= start:
        end = min(outputs_trial.shape[0], start + 1)
    avg = outputs_trial[start:end].mean(axis=0)
    if action_only:
        return int(1 + np.argmax(avg[1:4])), avg
    return int(np.argmax(avg)), avg


def softmax_action_outputs(outputs):
    """Return a copy with LEFT/RIGHT/SURE logits normalized per time step."""
    normalized = np.asarray(outputs, dtype=float).copy()
    logits = normalized[..., 1:4]
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    normalized[..., 1:4] = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return normalized


def _confidence_near_ts(outputs_trial, trial_meta, half_window):
    center = int(trial_meta["ts_onset"] if trial_meta.get("sure_available", False) else trial_meta["delay_end"])
    start = max(0, center - int(half_window))
    end = min(outputs_trial.shape[0], center + int(half_window))
    if end <= start:
        end = min(outputs_trial.shape[0], start + 1)
    avg = outputs_trial[start:end].mean(axis=0)
    return float(abs(avg[1] - avg[2]))


def select_repeated_stimuli(task, n_stimuli):
    """Prefer offered, boundary-near stimuli spanning coherence and duration."""
    n_stimuli = min(int(n_stimuli), int(task.x_data.shape[0]))
    candidates = []
    for idx, info in enumerate(task.source_trial_info):
        if not bool(info.get("sure_available", False)):
            continue
        strength = float(info.get("teacher_sure_strength", info.get("sure_strength", 0.5)))
        duration = float(info.get("stimulus_dur", info.get("stimulus_duration", np.nan)))
        candidates.append((idx, float(info.get("coh", np.nan)), duration, abs(strength - 0.5)))
    candidates.sort(key=lambda item: item[3])
    chosen = []
    represented = set()
    for idx, coh, duration, _ in candidates:
        key = (coh, duration)
        if key not in represented:
            chosen.append(idx)
            represented.add(key)
        if len(chosen) >= n_stimuli:
            break
    if len(chosen) < n_stimuli:
        for idx, _, _, _ in candidates:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= n_stimuli:
                break
    if len(chosen) < n_stimuli:
        chosen.extend(idx for idx in range(task.x_data.shape[0]) if idx not in chosen)
    return np.asarray(chosen[:n_stimuli], dtype=int)


def collect_repeated_stimulus_evaluation(
    model,
    task,
    n_stimuli=24,
    n_repeats=5,
    readout_window=10,
    ts_half_window=5,
    deterministic=False,
    rec_noise=0.0,
    action_only=False,
    action_softmax=False,
):
    n_stimuli = min(int(n_stimuli), int(task.x_data.shape[0]))
    n_repeats = int(n_repeats)
    dataset_indices = select_repeated_stimuli(task, n_stimuli)
    repeated_indices = np.repeat(dataset_indices, n_repeats)
    batch_x = task.x_data[repeated_indices]
    if deterministic:
        outputs, states = simulator_test(model, batch_x, rec_noise=rec_noise)
    else:
        outputs, states = model_test_with_fixed_batch(model, task, batch_x)
    if action_softmax:
        outputs = softmax_action_outputs(outputs)

    records = []
    for row_idx, dataset_index in enumerate(repeated_indices):
        info = dict(task.source_trial_info[int(dataset_index)])
        choice, avg = _readout_choice(
            outputs[row_idx],
            info["delay_end"],
            readout_window,
            action_only=action_only,
        )
        pre_ts_step = max(0, int(info.get("ts_onset", info["delay_end"])) - 1)
        pre_ts_state = np.asarray(states[row_idx, pre_ts_step], dtype=float)
        pre_ts_left = float(outputs[row_idx, pre_ts_step, 1])
        pre_ts_right = float(outputs[row_idx, pre_ts_step, 2])
        records.append(
            {
                "row_index": int(row_idx),
                "stimulus_id": int(dataset_index),
                "repeat_index": int(row_idx % n_repeats),
                "coh": float(info.get("coh", np.nan)),
                "stimulus_duration_ms": float(info.get("stimulus_dur", info.get("stimulus_duration", np.nan))),
                "sure_available": bool(info.get("sure_available", False)),
                "deterministic_eval": bool(deterministic),
                "eval_rec_noise": float(rec_noise) if deterministic else np.nan,
                "recurrent_noise_condition": (
                    f"deterministic_{float(rec_noise):g}" if deterministic else "model_stochastic"
                ),
                "dir_choice": int(info.get("dir_choice", -1)),
                "choice": int(choice),
                "chose_sure": bool(choice == 3),
                "confidence": _confidence_near_ts(outputs[row_idx], info, ts_half_window),
                "sure_output": float(avg[3]),
                "left_output": float(avg[1]),
                "right_output": float(avg[2]),
                "pre_ts_step": int(pre_ts_step),
                "pre_ts_hidden_state": json.dumps(pre_ts_state.tolist()),
                "pre_ts_hidden_norm": float(np.linalg.norm(pre_ts_state)),
                "pre_ts_left_output": pre_ts_left,
                "pre_ts_right_output": pre_ts_right,
                "pre_ts_internal_margin": float(abs(pre_ts_left - pre_ts_right)),
            }
        )
    return records, outputs, states


def summarize_repeated_stimulus_trials(records, outputs, states):
    stimulus_ids = sorted({int(record["stimulus_id"]) for record in records})
    choice_consistent = []
    sure_output_std = []
    confidence_std = []
    max_output_abs_diff = 0.0
    max_state_abs_diff = 0.0

    for stimulus_id in stimulus_ids:
        row_indices = [idx for idx, record in enumerate(records) if int(record["stimulus_id"]) == stimulus_id]
        choices = [int(records[idx]["choice"]) for idx in row_indices]
        choice_consistent.append(len(set(choices)) == 1)
        sure_output_std.append(float(np.std([records[idx]["sure_output"] for idx in row_indices])))
        confidence_std.append(float(np.std([records[idx]["confidence"] for idx in row_indices])))
        ref_output = outputs[row_indices[0]]
        ref_state = states[row_indices[0]]
        for idx in row_indices[1:]:
            max_output_abs_diff = max(max_output_abs_diff, float(np.max(np.abs(outputs[idx] - ref_output))))
            max_state_abs_diff = max(max_state_abs_diff, float(np.max(np.abs(states[idx] - ref_state))))

    repeats_per_stimulus = len(records) // max(len(stimulus_ids), 1)
    summary = {
        "n_rows": int(len(records)),
        "n_stimuli": int(len(stimulus_ids)),
        "n_repeats_per_stimulus": int(repeats_per_stimulus),
        "all_repeats_same_choice_fraction": float(np.mean(choice_consistent)) if choice_consistent else np.nan,
        "mean_sure_output_std_within_stimulus": float(np.mean(sure_output_std)) if sure_output_std else np.nan,
        "mean_confidence_std_within_stimulus": float(np.mean(confidence_std)) if confidence_std else np.nan,
        "max_output_abs_diff": float(max_output_abs_diff),
        "max_state_abs_diff": float(max_state_abs_diff),
    }
    usable = [
        record for record in records
        if "pre_ts_internal_margin" in record and np.isfinite(record["pre_ts_internal_margin"])
    ]
    variable_ids = [
        stimulus_id for stimulus_id in stimulus_ids
        if len({bool(r.get("chose_sure", int(r["choice"]) == 3)) for r in usable if int(r["stimulus_id"]) == stimulus_id}) > 1
    ]
    summary["n_stimuli_with_choice_variability"] = int(len(variable_ids))
    if variable_ids:
        selected = [r for r in usable if int(r["stimulus_id"]) in variable_ids]
        stimulus_columns = variable_ids[:-1]
        y = np.asarray([float(r.get("chose_sure", int(r["choice"]) == 3)) for r in selected])
        fixed_effects = np.asarray([
            [float(int(r["stimulus_id"]) == sid) for sid in stimulus_columns]
            for r in selected
        ])

        def fit_predictor(predictor):
            x = np.column_stack([np.asarray(predictor, dtype=float), fixed_effects])
            design = np.column_stack([np.ones(x.shape[0]), x])
            coef = np.zeros(design.shape[1], dtype=float)
            for _ in range(100):
                p = 1.0 / (1.0 + np.exp(-np.clip(design @ coef, -30, 30)))
                w = np.maximum(p * (1.0 - p), 1e-7)
                hessian = design.T @ (design * w[:, None]) + 1e-5 * np.eye(design.shape[1])
                updated = coef + np.linalg.solve(hessian, design.T @ (y - p))
                if np.max(np.abs(updated - coef)) < 1e-8:
                    coef = updated
                    break
                coef = updated
            covariance = np.linalg.pinv(hessian)
            return float(coef[1]), float(np.sqrt(max(covariance[1, 1], 0.0)))

        margin_coef, margin_se = fit_predictor([r["pre_ts_internal_margin"] for r in selected])
        summary["within_stimulus_pre_ts_margin_logit_coefficient"] = margin_coef
        summary["within_stimulus_pre_ts_margin_logit_se"] = margin_se
        if all("pre_ts_hidden_state" in r for r in selected):
            hidden = np.asarray([json.loads(r["pre_ts_hidden_state"]) for r in selected], dtype=float)
            centered = hidden.copy()
            for stimulus_id in variable_ids:
                mask = np.asarray([int(r["stimulus_id"]) == stimulus_id for r in selected])
                centered[mask] -= np.mean(centered[mask], axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            hidden_pc1 = centered @ vt[0]
            margin_values = np.asarray([r["pre_ts_internal_margin"] for r in selected], dtype=float)
            if safe_corrcoef(hidden_pc1, margin_values) < 0:
                hidden_pc1 *= -1.0
            hidden_coef, hidden_se = fit_predictor(hidden_pc1)
            summary["within_stimulus_pre_ts_hidden_pc1_logit_coefficient"] = hidden_coef
            summary["within_stimulus_pre_ts_hidden_pc1_logit_se"] = hidden_se
            summary["pre_ts_hidden_pc1_variance_fraction"] = float(
                np.var(hidden_pc1) / max(np.sum(np.var(centered, axis=0)), 1e-12)
            )
        summary["within_stimulus_analysis_n"] = int(len(selected))
        summary["within_stimulus_analysis"] = (
            "stimulus fixed-effects logistic regression using pre-TS margin and "
            "within-stimulus hidden-state PC1"
        )
    else:
        summary["within_stimulus_pre_ts_margin_logit_coefficient"] = np.nan
        summary["within_stimulus_pre_ts_margin_logit_se"] = np.nan
        summary["within_stimulus_analysis_n"] = 0
        summary["within_stimulus_analysis"] = "not estimable: no repeated stimulus changed sure/direction choice"
    return summary


def summarize_sure_output_dynamics(outputs, trial_info, choices, pre_window=50, post_window=50):
    outputs = np.asarray(outputs, dtype=float)
    choices = np.asarray(choices, dtype=int)
    relative_steps = np.arange(-int(pre_window), int(post_window) + 1)
    aligned = np.full((len(trial_info), relative_steps.size), np.nan, dtype=float)
    trial_rows = []
    for idx, info in enumerate(trial_info):
        ts_onset = int(info.get("ts_onset", info.get("delay_end", 0)))
        delay_end = int(info.get("delay_end", 0))
        for col, rel in enumerate(relative_steps):
            step = ts_onset + int(rel)
            if 0 <= step < outputs.shape[1]:
                aligned[idx, col] = outputs[idx, step, 3]
        pre_slice = outputs[idx, max(0, ts_onset - int(pre_window)):ts_onset, 3]
        immediate_slice = outputs[idx, ts_onset:min(outputs.shape[1], ts_onset + 10), 3]
        pre_go_slice = outputs[idx, max(0, delay_end - 10):delay_end, 3]
        post_go_slice = outputs[idx, delay_end:min(outputs.shape[1], delay_end + 10), 3]
        trial_rows.append({
            "trial_index": int(idx),
            "sure_available": bool(info.get("sure_available", False)),
            "choice": int(choices[idx]),
            "chose_sure": bool(choices[idx] == 3),
            "pre_ts_sure_output": float(np.mean(pre_slice)) if pre_slice.size else np.nan,
            "immediate_post_ts_sure_output": float(np.mean(immediate_slice)) if immediate_slice.size else np.nan,
            "pre_go_sure_output": float(np.mean(pre_go_slice)) if pre_go_slice.size else np.nan,
            "post_go_sure_output": float(np.mean(post_go_slice)) if post_go_slice.size else np.nan,
        })
    offered = np.asarray([bool(info.get("sure_available", False)) for info in trial_info])
    sure = choices == 3
    direction = np.isin(choices, [1, 2])
    def group_mean(field, mask):
        values = np.asarray([row[field] for row in trial_rows], dtype=float)
        return float(np.nanmean(values[mask])) if np.any(mask) else np.nan
    pre_values = np.asarray([row["pre_ts_sure_output"] for row in trial_rows])
    future_sure = offered & sure
    future_waived = offered & direction
    offered_trace = np.nanmean(aligned[offered], axis=0) if np.any(offered) else np.full(relative_steps.size, np.nan)
    unoffered_trace = np.nanmean(aligned[~offered], axis=0) if np.any(~offered) else np.full(relative_steps.size, np.nan)
    if np.any(future_sure) and np.any(future_waived):
        choice_diff_trace = np.nanmean(aligned[future_sure], axis=0) - np.nanmean(aligned[future_waived], axis=0)
        post_choice_trace = choice_diff_trace.copy()
        post_choice_trace[relative_steps < 0] = np.nan
        divergence_step = half_peak_onset(relative_steps, post_choice_trace)
    else:
        divergence_step = np.nan
    pre_cols = relative_steps < 0
    offered_leakage = float(np.nanmean(offered_trace[pre_cols] - unoffered_trace[pre_cols])) if np.any(offered) and np.any(~offered) else np.nan
    summary = {
        "n_trials": int(len(trial_info)),
        "n_offered": int(np.sum(offered)),
        "mean_pre_ts_sure_output": float(np.nanmean(pre_values)),
        "pre_ts_future_sure_minus_waived": group_mean("pre_ts_sure_output", future_sure) - group_mean("pre_ts_sure_output", future_waived),
        "post_ts_future_sure_minus_waived": group_mean("immediate_post_ts_sure_output", future_sure) - group_mean("immediate_post_ts_sure_output", future_waived),
        "pre_ts_offered_minus_unoffered": offered_leakage,
        "pre_go_future_sure_minus_waived": group_mean("pre_go_sure_output", future_sure) - group_mean("pre_go_sure_output", future_waived),
        "post_go_future_sure_minus_waived": group_mean("post_go_sure_output", future_sure) - group_mean("post_go_sure_output", future_waived),
        "post_ts_half_peak_divergence_relative_step": divergence_step,
    }
    plot_data = {
        "relative_steps": relative_steps,
        "aligned": aligned,
        "future_sure": future_sure,
        "future_waived": future_waived,
        "offered": offered,
    }
    return summary, trial_rows, plot_data


def plot_sure_output_dynamics(path, plot_data):
    import matplotlib.pyplot as plt
    rel = plot_data["relative_steps"]
    aligned = plot_data["aligned"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for mask, label, color in [
        (plot_data["future_sure"], "future sure", "#7b4aa0"),
        (plot_data["future_waived"], "future direction / waived", "#2f6f9f"),
    ]:
        if np.any(mask):
            axes[0].plot(rel, np.nanmean(aligned[mask], axis=0), label=label, color=color)
    offered = plot_data["offered"]
    for mask, label, color in [(offered, "offered", "#d17c25"), (~offered, "unoffered", "#555555")]:
        if np.any(mask):
            axes[1].plot(rel, np.nanmean(aligned[mask], axis=0), label=label, color=color)
    for ax in axes:
        ax.axvline(0, color="black", linestyle="--", alpha=0.7)
        ax.set_xlabel("steps from nominal TS onset")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("actual sure output y_sure(t)")
    axes[0].set_title("Choice-conditioned")
    axes[1].set_title("Offer leakage control")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_repeated_stimulus_dynamics(path, records, outputs):
    import matplotlib.pyplot as plt

    stimulus_ids = sorted({int(record["stimulus_id"]) for record in records})[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, stimulus_id in zip(axes, stimulus_ids):
        row_indices = [idx for idx, record in enumerate(records) if int(record["stimulus_id"]) == stimulus_id]
        for idx in row_indices:
            ax.plot(outputs[idx, :, 3], color="#7b4aa0", alpha=0.35, linewidth=1.1)
        ax.set_title(f"stimulus {stimulus_id}")
        ax.grid(alpha=0.2)
    for ax in axes[len(stimulus_ids):]:
        ax.axis("off")
    axes[0].set_ylabel("sure output")
    for ax in axes[-3:]:
        ax.set_xlabel("time step")
    fig.suptitle("Repeated-stimulus sure-output dynamics")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _aligned_sure_axis(states, trial_info, sure_axis, pre_window, post_window):
    sure_axis = np.asarray(sure_axis, dtype=float)
    norm = np.linalg.norm(sure_axis)
    if norm < 1e-12:
        sure_axis = sure_axis * 0.0
    else:
        sure_axis = sure_axis / norm
    traj = np.tensordot(states, sure_axis, axes=([2], [0]))
    relative_steps = np.arange(-int(pre_window), int(post_window) + 1)
    aligned = np.full((len(trial_info), relative_steps.size), np.nan, dtype=float)
    for idx, info in enumerate(trial_info):
        center = int(info.get("ts_onset", info.get("delay_end", 0)))
        for col, rel in enumerate(relative_steps):
            step = center + int(rel)
            if 0 <= step < traj.shape[1]:
                aligned[idx, col] = traj[idx, step]
    return aligned, relative_steps


def summarize_pre_ts_leakage(states, trial_info, choices, axes_data, pre_window=40, post_window=20):
    choices = np.asarray(choices, dtype=int)
    offered = np.array([bool(info.get("sure_available", False)) for info in trial_info], dtype=bool)
    sure_choice = choices == 3
    direction_choice = np.isin(choices, [1, 2])
    aligned, relative_steps = _aligned_sure_axis(
        states,
        trial_info,
        axes_data["sure_axis"],
        pre_window=pre_window,
        post_window=post_window,
    )
    pre_cols = relative_steps < 0
    post_cols = relative_steps >= 0
    pre_mean = np.nanmean(aligned[:, pre_cols], axis=1)
    post_mean = np.nanmean(aligned[:, post_cols], axis=1)

    offered_sure = offered & sure_choice
    offered_dir = offered & direction_choice
    pre_diff = np.nan
    post_diff = np.nan
    if np.any(offered_sure) and np.any(offered_dir):
        pre_diff = float(np.nanmean(pre_mean[offered_sure]) - np.nanmean(pre_mean[offered_dir]))
        post_diff = float(np.nanmean(post_mean[offered_sure]) - np.nanmean(post_mean[offered_dir]))

    summary = {
        "n_trials": int(len(trial_info)),
        "n_offered": int(np.sum(offered)),
        "n_offered_sure_choice": int(np.sum(offered_sure)),
        "n_offered_direction_choice": int(np.sum(offered_dir)),
        "pre_window_steps": int(pre_window),
        "post_window_steps": int(post_window),
        "pre_ts_sure_axis_mean_sure_choice": float(np.nanmean(pre_mean[offered_sure])) if np.any(offered_sure) else np.nan,
        "pre_ts_sure_axis_mean_direction_choice": float(np.nanmean(pre_mean[offered_dir])) if np.any(offered_dir) else np.nan,
        "pre_ts_sure_axis_mean_difference": pre_diff,
        "post_ts_sure_axis_mean_difference": post_diff,
        "corr_pre_ts_sure_axis_with_sure_choice": safe_corrcoef(pre_mean[offered], sure_choice[offered].astype(float)),
        "corr_post_ts_sure_axis_with_sure_choice": safe_corrcoef(post_mean[offered], sure_choice[offered].astype(float)),
    }
    plot_data = {
        "relative_steps": relative_steps,
        "aligned": aligned,
        "offered_sure": offered_sure,
        "offered_direction": offered_dir,
    }
    return summary, plot_data


def plot_pre_ts_leakage(path, plot_data):
    import matplotlib.pyplot as plt

    relative_steps = plot_data["relative_steps"]
    aligned = plot_data["aligned"]
    offered_sure = plot_data["offered_sure"]
    offered_direction = plot_data["offered_direction"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if np.any(offered_sure):
        ax.plot(relative_steps, np.nanmean(aligned[offered_sure], axis=0), color="#7b4aa0", linewidth=2.3, label="offered + sure choice")
    if np.any(offered_direction):
        ax.plot(relative_steps, np.nanmean(aligned[offered_direction], axis=0), color="#2f6f9f", linewidth=2.3, label="offered + direction choice")
    ax.axvline(0, color="black", linestyle="--", alpha=0.7, label="TS onset")
    ax.set_xlabel("steps from TS onset")
    ax.set_ylabel("sure-axis projection")
    ax.set_title("Pre-TS sure-axis leakage diagnostic")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def unit_norm(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec * 0.0
    return vec / norm


def compute_axis_timecourses(states, axes_data):
    states = np.asarray(states, dtype=float)
    return {
        "evidence": np.tensordot(states, unit_norm(axes_data["evidence_axis"]), axes=([2], [0])),
        "sure": np.tensordot(states, unit_norm(axes_data["sure_axis"]), axes=([2], [0])),
        "time": np.tensordot(states, unit_norm(axes_data["time_axis"]), axes=([2], [0])),
    }


def align_timecourses_to_event(timecourses, trial_info, event_key, pre_window, post_window):
    relative_steps = np.arange(-int(pre_window), int(post_window) + 1)
    aligned = {}
    for name, traj in timecourses.items():
        axis_aligned = np.full((len(trial_info), relative_steps.size), np.nan, dtype=float)
        for trial_idx, info in enumerate(trial_info):
            event_step = int(info.get(event_key, info.get("delay_end", 0)))
            for col_idx, rel_step in enumerate(relative_steps):
                step = event_step + int(rel_step)
                if 0 <= step < traj.shape[1]:
                    axis_aligned[trial_idx, col_idx] = traj[trial_idx, step]
        aligned[name] = axis_aligned
    return aligned, relative_steps


def mean_difference_by_window(values, group_a, group_b, columns):
    if not np.any(group_a) or not np.any(group_b) or not np.any(columns):
        return np.nan
    a = np.nanmean(values[group_a][:, columns], axis=1)
    b = np.nanmean(values[group_b][:, columns], axis=1)
    return float(np.nanmean(a) - np.nanmean(b))


def half_peak_onset(relative_steps, diff_trace):
    relative_steps = np.asarray(relative_steps, dtype=int)
    diff_trace = np.asarray(diff_trace, dtype=float)
    post_mask = relative_steps >= 0
    post_diff = diff_trace[post_mask]
    post_steps = relative_steps[post_mask]
    valid = np.isfinite(post_diff)
    if np.sum(valid) == 0:
        return None
    peak = float(np.nanmax(np.abs(post_diff[valid])))
    if peak < 1e-12:
        return None
    crossing = np.where(np.abs(post_diff) >= 0.5 * peak)[0]
    if crossing.size == 0:
        return None
    return int(post_steps[int(crossing[0])])


def summarize_population_dynamics(
    states,
    trial_info,
    choices,
    axes_data,
    align_event="ts_onset",
    pre_window=60,
    post_window=80,
):
    choices = np.asarray(choices, dtype=int)
    offered = np.array([bool(info.get("sure_available", False)) for info in trial_info], dtype=bool)
    sure_choice = choices == 3
    direction_choice = np.isin(choices, [1, 2])
    offered_sure = offered & sure_choice
    offered_direction = offered & direction_choice

    timecourses = compute_axis_timecourses(states, axes_data)
    aligned, relative_steps = align_timecourses_to_event(
        timecourses,
        trial_info,
        event_key=align_event,
        pre_window=pre_window,
        post_window=post_window,
    )
    pre_cols = relative_steps < 0
    post_cols = relative_steps >= 0
    sure_mean = np.nanmean(aligned["sure"][offered_sure], axis=0) if np.any(offered_sure) else np.full(relative_steps.size, np.nan)
    direction_mean = np.nanmean(aligned["sure"][offered_direction], axis=0) if np.any(offered_direction) else np.full(relative_steps.size, np.nan)
    sure_diff_trace = sure_mean - direction_mean

    summary = {
        "align_event": str(align_event),
        "n_trials": int(len(trial_info)),
        "n_offered": int(np.sum(offered)),
        "n_offered_sure_choice": int(np.sum(offered_sure)),
        "n_offered_direction_choice": int(np.sum(offered_direction)),
        "pre_window_steps": int(pre_window),
        "post_window_steps": int(post_window),
        "sure_axis_pre_event_mean_difference": mean_difference_by_window(
            aligned["sure"], offered_sure, offered_direction, pre_cols
        ),
        "sure_axis_post_event_mean_difference": mean_difference_by_window(
            aligned["sure"], offered_sure, offered_direction, post_cols
        ),
        "evidence_axis_abs_pre_event_mean_difference": mean_difference_by_window(
            np.abs(aligned["evidence"]), offered_sure, offered_direction, pre_cols
        ),
        "evidence_axis_abs_post_event_mean_difference": mean_difference_by_window(
            np.abs(aligned["evidence"]), offered_sure, offered_direction, post_cols
        ),
        "time_axis_pre_event_mean_difference": mean_difference_by_window(
            aligned["time"], offered_sure, offered_direction, pre_cols
        ),
        "time_axis_post_event_mean_difference": mean_difference_by_window(
            aligned["time"], offered_sure, offered_direction, post_cols
        ),
        "sure_axis_half_peak_onset_relative_step": half_peak_onset(relative_steps, sure_diff_trace),
        "sure_axis_peak_abs_difference": float(np.nanmax(np.abs(sure_diff_trace)))
        if np.any(np.isfinite(sure_diff_trace))
        else np.nan,
    }
    plot_data = {
        "relative_steps": relative_steps,
        "aligned": aligned,
        "offered_sure": offered_sure,
        "offered_direction": offered_direction,
        "sure_diff_trace": sure_diff_trace,
        "align_event": str(align_event),
    }
    return summary, plot_data


def plot_population_dynamics(path, plot_data):
    import matplotlib.pyplot as plt

    relative_steps = plot_data["relative_steps"]
    aligned = plot_data["aligned"]
    offered_sure = plot_data["offered_sure"]
    offered_direction = plot_data["offered_direction"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axis_specs = [
        ("evidence", "Evidence-axis projection", "#2f8f5b"),
        ("sure", "Sure-axis projection", "#7b4aa0"),
        ("time", "Time-axis projection", "#7a6f2f"),
    ]
    for ax, (key, title, color) in zip(axes, axis_specs):
        if np.any(offered_sure):
            ax.plot(
                relative_steps,
                np.nanmean(aligned[key][offered_sure], axis=0),
                color=color,
                linewidth=2.4,
                label="offered + sure choice",
            )
        if np.any(offered_direction):
            ax.plot(
                relative_steps,
                np.nanmean(aligned[key][offered_direction], axis=0),
                color="#2f6f9f",
                linewidth=2.2,
                linestyle="--",
                label="offered + direction choice",
            )
        ax.axvline(0, color="black", linestyle="--", alpha=0.65)
        ax.set_ylabel("projection")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel(f"steps from {plot_data['align_event']}")
    fig.suptitle("Time-resolved population dynamics")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
