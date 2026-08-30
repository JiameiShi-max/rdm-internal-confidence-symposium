import numpy as np
import matplotlib.pyplot as plt

from sure_target_stage1_targeted_dr import (
    binned_probability_curve,
    evaluate_logistic_curve,
    fit_logistic_curve,
    readout_fd_trial,
    save_figure,
)


class NumpyPCA:
    def __init__(self, n_components=2):
        self.n_components = int(n_components)
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        total_var = max(float(np.sum(s ** 2)), 1e-8)
        self.components_ = vt[: self.n_components]
        self.explained_variance_ratio_ = (s[: self.n_components] ** 2) / total_var
        return self

    def transform(self, x):
        return (x - self.mean_) @ self.components_.T


def model_test_with_fixed_batch(model, task, batch_x):
    target_batch = int(task.N_batch)
    n_trials = int(batch_x.shape[0])
    if n_trials == target_batch:
        return model.test(batch_x)

    outputs_all = []
    states_all = []
    start = 0
    while start < n_trials:
        end = min(start + target_batch, n_trials)
        chunk_x = batch_x[start:end]
        valid_trials = chunk_x.shape[0]
        if valid_trials < target_batch:
            pad_x = np.repeat(chunk_x[-1:, :, :], target_batch - valid_trials, axis=0)
            chunk_x = np.concatenate([chunk_x, pad_x], axis=0)
        chunk_outputs, chunk_states = model.test(chunk_x)
        outputs_all.append(chunk_outputs[:valid_trials])
        states_all.append(chunk_states[:valid_trials])
        start = end
    return np.concatenate(outputs_all, axis=0), np.concatenate(states_all, axis=0)


def mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def sem_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def safe_corrcoef(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def smooth_binary_curve(x, y, num_points=200, bandwidth=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return None, None
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_min, x_max):
        return np.array([x_min]), np.array([np.mean(y)])
    x_grid = np.linspace(x_min, x_max, num_points)
    if bandwidth is None:
        bandwidth = max(float(np.std(x)) * 0.35, 1e-3)
    smooth = np.zeros_like(x_grid)
    for idx, x0 in enumerate(x_grid):
        weights = np.exp(-0.5 * ((x - x0) / bandwidth) ** 2)
        denom = np.sum(weights)
        smooth[idx] = np.sum(weights * y) / denom if denom > 0 else np.nan
    return x_grid, smooth


def get_task_epoch_steps(trial_info):
    fixation_end = int(np.round(np.median([info["fixation_end"] for info in trial_info])))
    stimulus_end = int(np.round(np.median([info["stimulus_end"] for info in trial_info])))
    return fixation_end, stimulus_end


def get_coherence_split_masks(trial_info):
    coh_values = np.array([info["coh"] for info in trial_info], dtype=float)
    unique_cohs = np.unique(coh_values)
    boundary = float(np.median(unique_cohs))
    low_mask = coh_values <= boundary
    high_mask = coh_values > boundary
    return coh_values, boundary, low_mask, high_mask


def run_pca_on_states(states, n_components=2):
    n_trials, n_steps, n_rec = states.shape
    flat = states.reshape(-1, n_rec)
    pca = NumpyPCA(n_components=n_components)
    pca.fit(flat)
    return pca, pca.explained_variance_ratio_


def project_states_to_pcs(states, pca):
    n_trials, n_steps, n_rec = states.shape
    flat = states.reshape(-1, n_rec)
    proj_flat = pca.transform(flat)
    return proj_flat.reshape(n_trials, n_steps, -1)


def align_projected_states_to_event(projected_states, trial_info, event_key="delay_end", window_before=40, window_after=40):
    n_trials, n_steps, n_pcs = projected_states.shape
    aligned_len = int(window_before + window_after + 1)
    aligned = np.full((n_trials, aligned_len, n_pcs), np.nan, dtype=float)
    for trial_idx, info in enumerate(trial_info):
        event_step = int(info[event_key])
        start_src = max(0, event_step - int(window_before))
        end_src = min(n_steps, event_step + int(window_after) + 1)
        start_dst = int(window_before) - (event_step - start_src)
        end_dst = start_dst + (end_src - start_src)
        aligned[trial_idx, start_dst:end_dst, :] = projected_states[trial_idx, start_src:end_src, :]
    relative_steps = np.arange(-int(window_before), int(window_after) + 1)
    return aligned, relative_steps


def compute_group_separation(projected_states, group_a_mask, group_b_mask, pc_dims=(0, 1)):
    group_a_mask = np.asarray(group_a_mask, dtype=bool)
    group_b_mask = np.asarray(group_b_mask, dtype=bool)
    if np.sum(group_a_mask) == 0 or np.sum(group_b_mask) == 0:
        return None
    mean_a = np.nanmean(projected_states[group_a_mask][:, :, pc_dims], axis=0)
    mean_b = np.nanmean(projected_states[group_b_mask][:, :, pc_dims], axis=0)
    separation = np.linalg.norm(mean_a - mean_b, axis=1)
    invalid = np.any(~np.isfinite(mean_a), axis=1) | np.any(~np.isfinite(mean_b), axis=1)
    separation[invalid] = np.nan
    return separation


def compute_choice_separation(projected_states, trial_info, trial_mask=None, pc_dims=(0, 1)):
    left_mask = np.array([info["dir_choice"] == 0 for info in trial_info], dtype=bool)
    right_mask = np.array([info["dir_choice"] == 1 for info in trial_info], dtype=bool)
    if trial_mask is not None:
        trial_mask = np.asarray(trial_mask, dtype=bool)
        left_mask = left_mask & trial_mask
        right_mask = right_mask & trial_mask
    return compute_group_separation(projected_states, left_mask, right_mask, pc_dims=pc_dims)


def estimate_separation_onset(separation, start_step):
    if separation is None:
        return None
    separation = np.asarray(separation, dtype=float)
    post_segment = separation[int(start_step):]
    valid_mask = np.isfinite(post_segment)
    if np.sum(valid_mask) == 0:
        return None
    peak = np.nanmax(post_segment)
    if peak <= 0:
        return None
    threshold = 0.5 * peak
    crossing = np.where(post_segment >= threshold)[0]
    if len(crossing) == 0:
        return None
    return int(start_step + crossing[0])


def plot_pc_trajectories(projected_states, trial_info, output_dir, by="choice"):
    n_steps = projected_states.shape[1]
    time_ax = np.arange(n_steps)
    fixation_end, stimulus_end = get_task_epoch_steps(trial_info)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PC Trajectories - grouped by {by}", fontsize=13)

    if by == "choice":
        groups = {
            "Left (dir=0)": np.array([i for i, info in enumerate(trial_info) if info["dir_choice"] == 0]),
            "Right (dir=1)": np.array([i for i, info in enumerate(trial_info) if info["dir_choice"] == 1]),
        }
        colors = {"Left (dir=0)": "tomato", "Right (dir=1)": "steelblue"}
    elif by == "coherence":
        unique_cohs = sorted(set(info["coh"] for info in trial_info))
        cmap = plt.cm.viridis(np.linspace(0, 1, len(unique_cohs)))
        groups = {f"coh={c:.3f}": np.array([i for i, info in enumerate(trial_info) if info["coh"] == c]) for c in unique_cohs}
        colors = {f"coh={c:.3f}": cmap[j] for j, c in enumerate(unique_cohs)}
    elif by == "delay":
        delay_vals = np.array([info["delay_end"] for info in trial_info])
        median_delay = np.median(delay_vals)
        groups = {
            "Short delay": np.where(delay_vals <= median_delay)[0],
            "Long delay": np.where(delay_vals > median_delay)[0],
        }
        colors = {"Short delay": "darkorange", "Long delay": "purple"}
    else:
        raise ValueError(f"Unknown grouping: {by}")

    sc = None
    for label, idx in groups.items():
        if len(idx) == 0:
            continue
        color = colors[label]
        mean_traj = projected_states[idx].mean(axis=0)
        axes[0].plot(time_ax, mean_traj[:, 0], color=color, label=f"{label} PC1", linewidth=2)
        axes[0].plot(time_ax, mean_traj[:, 1], color=color, linestyle="--", alpha=0.5, label=f"{label} PC2")
        sc = axes[1].scatter(mean_traj[:, 0], mean_traj[:, 1], c=time_ax, cmap="plasma", s=8, alpha=0.7)
        axes[1].plot(mean_traj[:, 0], mean_traj[:, 1], color=color, alpha=0.4, linewidth=1)
        axes[1].plot(mean_traj[0, 0], mean_traj[0, 1], "o", color=color, markersize=7, label=label)

    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("PC Value")
    axes[0].set_title("PC1 & PC2 Over Time")
    axes[0].legend(fontsize=7)
    axes[0].axvline(x=fixation_end, color="k", linestyle=":", alpha=0.5)
    axes[0].axvline(x=stimulus_end, color="k", linestyle="-.", alpha=0.5)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("PC1 vs PC2 Phase Portrait")
    axes[1].legend(fontsize=7)
    if sc is not None:
        plt.colorbar(sc, ax=axes[1], label="Time Step")
    plt.tight_layout()
    save_figure(output_dir, f"pca_trajectories_{by}.png")


def plot_go_cue_aligned_pc_dynamics(projected_states, trial_info, dt, output_dir, by="choice", window_before=40, window_after=40):
    aligned, relative_steps = align_projected_states_to_event(
        projected_states, trial_info, event_key="delay_end", window_before=window_before, window_after=window_after
    )
    relative_time_ms = relative_steps * dt
    if by == "choice":
        groups = [
            ("Left (dir=0)", np.array([info["dir_choice"] == 0 for info in trial_info], dtype=bool), "tomato"),
            ("Right (dir=1)", np.array([info["dir_choice"] == 1 for info in trial_info], dtype=bool), "steelblue"),
        ]
    else:
        _, boundary, low_mask, high_mask = get_coherence_split_masks(trial_info)
        groups = [
            (f"Low coh (<= {boundary:.3f})", low_mask, "darkorange"),
            (f"High coh (> {boundary:.3f})", high_mask, "purple"),
        ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Go-Cue Aligned PCA Dynamics - grouped by {by}", fontsize=13)
    for label, mask, color in groups:
        if np.sum(mask) == 0:
            continue
        mean_pc1 = np.nanmean(aligned[mask, :, 0], axis=0)
        mean_pc2 = np.nanmean(aligned[mask, :, 1], axis=0)
        axes[0].plot(relative_time_ms, mean_pc1, color=color, linewidth=2.5, label=label)
        axes[1].plot(relative_time_ms, mean_pc2, color=color, linewidth=2.5, label=label)
    for ax, title in zip(axes, ["PC1 aligned to go cue", "PC2 aligned to go cue"]):
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.7, label="Go cue")
        ax.set_title(title)
        ax.set_ylabel("PC value")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    axes[1].set_xlabel("Time from go cue (ms)")
    plt.tight_layout()
    save_figure(output_dir, f"go_aligned_pca_{by}.png")


def plot_choice_separation_over_time(projected_states, trial_info, dt, output_dir):
    separation = compute_choice_separation(projected_states, trial_info)
    if separation is None:
        return
    fixation_end, stimulus_end = get_task_epoch_steps(trial_info)
    time_ms = np.arange(projected_states.shape[1]) * dt
    plt.figure(figsize=(8, 4))
    plt.plot(time_ms, separation, color="black", linewidth=2)
    plt.axvline(x=fixation_end * dt, color="gray", linestyle=":", label="Stim On")
    plt.axvline(x=stimulus_end * dt, color="gray", linestyle="-.", label="Stim Off")
    plt.title("Trajectory Separation (L2 distance in PC space) - Left vs Right")
    plt.xlabel("Time (ms)")
    plt.ylabel("PC Space Separation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(output_dir, "pca_choice_separation.png")


def compute_trial_correctness(outputs, trial_info, readout_window=10):
    n_trials = min(len(outputs), len(trial_info))
    correct_mask = np.zeros(n_trials, dtype=bool)
    for idx in range(n_trials):
        pred, _ = readout_fd_trial(outputs[idx], trial_info[idx]["delay_end"], window=readout_window)
        true_choice = 1 if int(trial_info[idx]["dir_choice"]) == 0 else 2
        correct_mask[idx] = int(pred) == true_choice
    return correct_mask


def _preferred_null_mean_traces(outputs, trial_info, trial_indices):
    pref_traces = []
    null_traces = []
    for idx in np.asarray(trial_indices, dtype=int):
        if int(trial_info[idx]["dir_choice"]) == 0:
            pref_traces.append(outputs[idx, :, 1])
            null_traces.append(outputs[idx, :, 2])
        else:
            pref_traces.append(outputs[idx, :, 2])
            null_traces.append(outputs[idx, :, 1])
    if len(pref_traces) == 0:
        return None, None
    return np.mean(pref_traces, axis=0), np.mean(null_traces, axis=0)


def plot_kiani_style_variable_delay_activity(outputs, trial_info, dt, output_dir, readout_window=10):
    if len(trial_info) == 0:
        return
    delays = np.array([info["delay_end"] for info in trial_info], dtype=int)
    coherences = np.array([info["coh"] for info in trial_info], dtype=float)
    correct_mask = compute_trial_correctness(outputs, trial_info, readout_window=readout_window)
    long_delay_mask = delays >= np.percentile(delays, 60)
    coh_boundary = np.median(np.unique(coherences))
    high_coh_mask = coherences > coh_boundary
    low_coh_mask = coherences <= coh_boundary

    high_idx = np.where(correct_mask & long_delay_mask & high_coh_mask)[0]
    low_idx = np.where(correct_mask & long_delay_mask & low_coh_mask)[0]
    high_pref, high_null = _preferred_null_mean_traces(outputs, trial_info, high_idx)
    low_pref, low_null = _preferred_null_mean_traces(outputs, trial_info, low_idx)

    time_ms = np.arange(outputs.shape[1]) * dt
    stim_on = np.median([info["fixation_end"] for info in trial_info]) * dt
    stim_off = np.median([info["stimulus_end"] for info in trial_info]) * dt
    go_cue = np.median(delays[long_delay_mask & correct_mask]) * dt if np.any(long_delay_mask & correct_mask) else np.median(delays) * dt

    plt.figure(figsize=(11, 6))
    if high_pref is not None:
        plt.plot(time_ms, high_pref, lw=2.8, color="tab:blue", label=f"Preferred (correct, long delay, high coh, n={len(high_idx)})")
        plt.plot(time_ms, high_null, lw=2.3, ls="--", color="tab:blue", alpha=0.65, label="Null (correct, long delay, high coh)")
    if low_pref is not None:
        plt.plot(time_ms, low_pref, lw=2.6, color="tab:orange", label=f"Preferred (correct, long delay, low coh, n={len(low_idx)})")
        plt.plot(time_ms, low_null, lw=2.1, ls="--", color="tab:orange", alpha=0.65, label="Null (correct, long delay, low coh)")
    plt.axvline(stim_on, color="k", linestyle=":", alpha=0.5, label="Stimulus on")
    plt.axvline(stim_off, color="k", linestyle="-.", alpha=0.6, label="Stimulus off")
    plt.axvline(go_cue, color="gray", linestyle="--", alpha=0.6, label="Median go cue (correct long delay)")
    plt.title("Neural Activity: Variable Delay Task (Kiani-style analysis)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Activity")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    save_figure(output_dir, "baseline_variable_delay_neural_activity.png")


def summarize_pca_dynamics(projected_states, trial_info, pca, dt, window_before=40, window_after=40):
    stimulus_onset = int(np.round(np.median([info["fixation_end"] for info in trial_info])))
    choice_separation = compute_choice_separation(projected_states, trial_info)
    choice_onset_step = estimate_separation_onset(choice_separation, start_step=stimulus_onset)
    _, coherence_boundary, low_mask, high_mask = get_coherence_split_masks(trial_info)
    coherence_separation = compute_group_separation(projected_states, low_mask, high_mask)
    coherence_onset_step = estimate_separation_onset(coherence_separation, start_step=stimulus_onset)
    explained = pca.explained_variance_ratio_
    return {
        "pc1_explained_variance": float(explained[0]) if len(explained) > 0 else np.nan,
        "pc2_explained_variance": float(explained[1]) if len(explained) > 1 else np.nan,
        "pc12_total_explained_variance": float(np.sum(explained[:2])),
        "choice_separation_onset_ms": float(choice_onset_step * dt) if choice_onset_step is not None else np.nan,
        "coherence_split_boundary": float(coherence_boundary),
        "coherence_separation_onset_ms": float(coherence_onset_step * dt) if coherence_onset_step is not None else np.nan,
    }


def generate_probe_trial(task, left_strength, right_strength, delay_dur=600):
    fixation_dur = 200
    stimulus_dur = 800
    fixation_end = int(fixation_dur / task.dt)
    stimulus_end = fixation_end + int(stimulus_dur / task.dt)
    delay_end = stimulus_end + int(delay_dur / task.dt)
    delay_end = min(delay_end, task.N_steps - 1)
    x = np.zeros((task.N_steps, task.N_in), dtype=np.float32)
    x[:delay_end, 0] = 1.0
    x[fixation_end:stimulus_end, 1] = left_strength
    x[fixation_end:stimulus_end, 2] = right_strength
    meta = {
        "fixation_end": fixation_end,
        "stimulus_end": stimulus_end,
        "delay_end": delay_end,
        "left_strength": float(left_strength),
        "right_strength": float(right_strength),
    }
    return x, meta


def run_competition_probe(model, task, probe_conditions, delay_dur=600, n_repeats=10):
    results = []
    for left_strength, right_strength in probe_conditions:
        trials_x = []
        metas = []
        for _ in range(int(n_repeats)):
            x, meta = generate_probe_trial(task, left_strength, right_strength, delay_dur=delay_dur)
            trials_x.append(x)
            metas.append(meta)
        batch_x = np.stack(trials_x, axis=0)
        outputs, states = model_test_with_fixed_batch(model, task, batch_x)
        results.append(
            {
                "left_strength": left_strength,
                "right_strength": right_strength,
                "outputs": outputs,
                "states": states,
                "meta": metas[0],
            }
        )
    return results


def plot_probe_outputs(results, output_dir):
    n_cond = len(results)
    if n_cond == 0:
        return
    n_steps = results[0]["outputs"].shape[1]
    time_ax = np.arange(n_steps)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Competition / Mutual Inhibition Probe", fontsize=13)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, n_cond))
    left_post_go = []
    right_post_go = []
    labels = []
    for idx, res in enumerate(results):
        ls, rs = res["left_strength"], res["right_strength"]
        label = f"L={ls:.1f}, R={rs:.1f}"
        labels.append(label)
        color = cmap[idx]
        mean_out = res["outputs"].mean(axis=0)
        delay_end = res["meta"]["delay_end"]
        axes[0].plot(time_ax, mean_out[:, 1], color=color, linewidth=2, label=f"{label} (L-out)")
        axes[0].plot(time_ax, mean_out[:, 2], color=color, linestyle="--", alpha=0.5, label=f"{label} (R-out)")
        window = min(10, n_steps - delay_end)
        if window > 0:
            left_post_go.append(mean_out[delay_end:delay_end + window, 1].mean())
            right_post_go.append(mean_out[delay_end:delay_end + window, 2].mean())
        else:
            left_post_go.append(np.nan)
            right_post_go.append(np.nan)
        diff = mean_out[:, 1] - mean_out[:, 2]
        axes[2].plot(time_ax, diff, color=color, linewidth=2, label=label)
    stim_end = results[0]["meta"]["stimulus_end"]
    delay_end = results[0]["meta"]["delay_end"]
    axes[0].axvline(x=results[0]["meta"]["fixation_end"], color="k", linestyle=":", alpha=0.4)
    axes[0].axvline(x=stim_end, color="k", linestyle="-.", alpha=0.4)
    axes[0].axvline(x=delay_end, color="k", linestyle="-", alpha=0.6, label="Go cue")
    axes[0].set_title("Output Traces (solid=Left, dashed=Right)")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Output Activity")
    axes[0].legend(fontsize=6)
    x_pos = np.arange(n_cond)
    width = 0.35
    axes[1].bar(x_pos - width / 2, left_post_go, width, label="Left out", color="tomato", alpha=0.8)
    axes[1].bar(x_pos + width / 2, right_post_go, width, label="Right out", color="steelblue", alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_title("Post-Go-Cue Mean Output")
    axes[1].set_ylabel("Mean Activity")
    axes[1].legend()
    axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.4)
    axes[2].axvline(x=delay_end, color="k", linestyle="-", alpha=0.6, label="Go cue")
    axes[2].set_title("Left - Right Output Difference")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("L - R Output")
    axes[2].legend(fontsize=7)
    plt.tight_layout()
    save_figure(output_dir, "competition_mutual_inhibition_probe.png")


def plot_delay_maintained_neural_activity(outputs, trial_info, dt, output_dir):
    if len(trial_info) == 0:
        return
    delays = np.array([info["delay_end"] for info in trial_info], dtype=int)
    coherences = np.array([info["coh"] for info in trial_info], dtype=float)
    correct_mask = compute_trial_correctness(outputs, trial_info, readout_window=10)
    long_delay_mask = delays >= np.percentile(delays, 60)
    coh_boundary = np.median(np.unique(coherences))
    high_coh_mask = coherences > coh_boundary
    low_coh_mask = coherences <= coh_boundary
    high_mask = correct_mask & long_delay_mask & high_coh_mask
    low_mask = correct_mask & long_delay_mask & low_coh_mask

    high_pref, high_null = _preferred_null_mean_traces(outputs, trial_info, np.where(high_mask)[0])
    low_pref, low_null = _preferred_null_mean_traces(outputs, trial_info, np.where(low_mask)[0])
    time_ms = np.arange(outputs.shape[1]) * dt
    stim_on = np.median([info["fixation_end"] for info in trial_info]) * dt
    stim_off = np.median([info["stimulus_end"] for info in trial_info]) * dt
    go_cue = np.median(delays[correct_mask & long_delay_mask]) * dt if np.any(correct_mask & long_delay_mask) else np.median(delays) * dt
    plt.figure(figsize=(11, 6))
    if high_pref is not None:
        plt.plot(time_ms, high_pref, lw=2.8, color="tab:blue", label=f"Preferred (correct, long delay, high coh, n={int(np.sum(high_mask))})")
        plt.plot(time_ms, high_null, lw=2.4, ls="--", color="tab:blue", alpha=0.65, label="Null (correct, long delay, high coh)")
    if low_pref is not None:
        plt.plot(time_ms, low_pref, lw=2.6, color="tab:orange", label=f"Preferred (correct, long delay, low coh, n={int(np.sum(low_mask))})")
        plt.plot(time_ms, low_null, lw=2.2, ls="--", color="tab:orange", alpha=0.65, label="Null (correct, long delay, low coh)")
    plt.axvline(stim_on, color="k", linestyle=":", alpha=0.5, label="Stimulus on")
    plt.axvline(stim_off, color="k", linestyle="-.", alpha=0.6, label="Stimulus off")
    plt.axvline(go_cue, color="gray", linestyle="--", alpha=0.6, label="Median go cue (correct long delay)")
    plt.title("Delay-maintained choice activity")
    plt.xlabel("Time (ms)")
    plt.ylabel("Activity")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    save_figure(output_dir, "delay_maintained_choice_activity.png")


def get_internal_readout_center(trial_meta):
    if trial_meta["sure_available"]:
        return int(trial_meta["ts_onset"])
    return int(trial_meta["delay_end"])


def avg_output_before_step(outputs_trial, step, window=5):
    end = max(0, int(step))
    start = max(0, end - int(window))
    if end <= start:
        end = min(outputs_trial.shape[0], start + 1)
    return outputs_trial[start:end, :].mean(axis=0)


def get_signed_lr_trace(outputs_trial, trial_meta):
    if trial_meta["dir_choice"] == 0:
        return outputs_trial[:, 1] - outputs_trial[:, 2]
    return outputs_trial[:, 2] - outputs_trial[:, 1]


def get_abs_lr_trace(outputs_trial):
    return np.abs(outputs_trial[:, 1] - outputs_trial[:, 2])


def get_pref_null_traces(outputs_trial, trial_meta):
    if trial_meta["dir_choice"] == 0:
        return outputs_trial[:, 1], outputs_trial[:, 2]
    return outputs_trial[:, 2], outputs_trial[:, 1]


def average_trace(traces):
    if len(traces) == 0:
        return None
    return np.mean(np.asarray(traces, dtype=float), axis=0)


def compute_sure_competition_delta(avg_out):
    sure_score = float(avg_out[3])
    direction_score = float(max(avg_out[1], avg_out[2]))
    return sure_score - direction_score


def analyze_external_vs_internal_confidence(outputs, trial_info, confidence_fn, avg_output_in_window_fn, ts_half_window=5, readout_window=10):
    n_trials = min(len(outputs), len(trial_info))
    coherence = np.zeros(n_trials, dtype=float)
    sure_available = np.zeros(n_trials, dtype=bool)
    sensory_margin = np.zeros(n_trials, dtype=float)
    sure_strength = np.zeros(n_trials, dtype=float)
    internal_margin = np.zeros(n_trials, dtype=float)
    sure_output_near_ts = np.zeros(n_trials, dtype=float)
    final_choice = np.zeros(n_trials, dtype=int)
    hard_sure_choice = np.zeros(n_trials, dtype=bool)
    sure_competition_delta = np.zeros(n_trials, dtype=float)
    pre_onset_sure_output = np.zeros(n_trials, dtype=float)
    for idx in range(n_trials):
        info = trial_info[idx]
        coherence[idx] = float(info["coh"])
        sure_available[idx] = bool(info["sure_available"])
        sensory_margin[idx] = float(info["sensory_margin"])
        sure_strength[idx] = float(info["sure_strength"])
        center_step = get_internal_readout_center(info)
        avg_near_ts = avg_output_in_window_fn(outputs[idx], center_step, half_window=ts_half_window)
        sure_output_near_ts[idx] = float(avg_near_ts[3])
        internal_margin[idx] = float(confidence_fn(outputs[idx], info, half_window=ts_half_window))
        final_choice_i, final_avg_i = readout_fd_trial(outputs[idx], info["delay_end"], window=readout_window)
        final_choice[idx] = int(final_choice_i)
        hard_sure_choice[idx] = bool(final_choice_i == 3)
        sure_competition_delta[idx] = compute_sure_competition_delta(final_avg_i)
        ref_step = min(int(info["ts_onset"]), int(info["delay_end"]))
        pre_onset_avg = avg_output_before_step(outputs[idx], ref_step, window=ts_half_window)
        pre_onset_sure_output[idx] = float(pre_onset_avg[3])
    return {
        "coherence": coherence,
        "sure_available": sure_available,
        "sensory_margin": sensory_margin,
        "sure_strength": sure_strength,
        "internal_margin": internal_margin,
        "sure_output_near_ts": sure_output_near_ts,
        "sure_competition_delta": sure_competition_delta,
        "hard_sure_choice": hard_sure_choice,
        "final_choice": final_choice,
        "pre_onset_sure_output": pre_onset_sure_output,
    }


def summarize_external_vs_internal_confidence(analysis_data):
    internal_margin = analysis_data["internal_margin"]
    sensory_margin = analysis_data["sensory_margin"]
    sure_strength = analysis_data["sure_strength"]
    sure_available = analysis_data["sure_available"].astype(bool)
    sure_output_near_ts = analysis_data["sure_output_near_ts"]
    median_internal_margin = np.median(internal_margin) if internal_margin.size > 0 else np.nan
    low_internal_mask = internal_margin <= median_internal_margin
    high_internal_mask = internal_margin > median_internal_margin
    return {
        "corr_sensory_internal": safe_corrcoef(sensory_margin, internal_margin),
        "corr_sure_strength_internal": safe_corrcoef(sure_strength, internal_margin),
        "mean_internal_margin_sure_available": mean_or_nan(internal_margin[sure_available]),
        "mean_internal_margin_sure_unavailable": mean_or_nan(internal_margin[~sure_available]),
        "mean_sure_output_low_internal": mean_or_nan(sure_output_near_ts[low_internal_mask]),
        "mean_sure_output_high_internal": mean_or_nan(sure_output_near_ts[high_internal_mask]),
        "median_internal_margin": float(median_internal_margin) if np.isfinite(median_internal_margin) else np.nan,
    }


def summarize_soft_vs_hard_readout(analysis_data, near_zero_band=0.05):
    delta = np.asarray(analysis_data["sure_competition_delta"], dtype=float)
    sure_available = np.asarray(analysis_data["sure_available"], dtype=bool)
    offered_delta = delta[sure_available]
    near_zero_mask = np.abs(offered_delta) <= float(near_zero_band)
    return {
        "mean_delta_offered": mean_or_nan(offered_delta),
        "fraction_near_zero_offered": mean_or_nan(near_zero_mask.astype(float)),
        "fraction_near_zero_all": mean_or_nan((np.abs(delta) <= float(near_zero_band)).astype(float)),
        "n_offered": int(np.sum(sure_available)),
        "n_near_zero_offered": int(np.sum(near_zero_mask)),
    }


def summarize_pre_onset_sure_baseline(analysis_data):
    sure_available = analysis_data["sure_available"].astype(bool)
    internal_margin = analysis_data["internal_margin"]
    pre_onset_sure_output = analysis_data["pre_onset_sure_output"]
    median_internal = np.median(internal_margin) if internal_margin.size > 0 else np.nan
    low_internal = internal_margin <= median_internal
    high_internal = internal_margin > median_internal
    return {
        "mean_pre_onset_sure_available": mean_or_nan(pre_onset_sure_output[sure_available]),
        "mean_pre_onset_sure_unavailable": mean_or_nan(pre_onset_sure_output[~sure_available]),
        "mean_pre_onset_low_internal": mean_or_nan(pre_onset_sure_output[low_internal]),
        "mean_pre_onset_high_internal": mean_or_nan(pre_onset_sure_output[high_internal]),
        "median_internal_margin": float(median_internal) if np.isfinite(median_internal) else np.nan,
    }


def plot_external_vs_internal_confidence_relationships(analysis_data, output_dir, stem="external_internal_confidence"):
    coherence = analysis_data["coherence"]
    sure_available = analysis_data["sure_available"]
    sensory_margin = analysis_data["sensory_margin"]
    sure_strength = analysis_data["sure_strength"]
    internal_margin = analysis_data["internal_margin"]
    sure_output_near_ts = analysis_data["sure_output_near_ts"]
    offered_mask = sure_available.astype(bool)
    forced_mask = ~offered_mask
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(sensory_margin[forced_mask], internal_margin[forced_mask], alpha=0.7, label="Sure unavailable")
    plt.scatter(sensory_margin[offered_mask], internal_margin[offered_mask], alpha=0.7, label="Sure available")
    plt.xlabel("External sensory margin")
    plt.ylabel("Internal margin")
    plt.title("Sensory margin vs internal margin")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.scatter(sure_strength[forced_mask], internal_margin[forced_mask], alpha=0.7, label="Sure unavailable")
    plt.scatter(sure_strength[offered_mask], internal_margin[offered_mask], alpha=0.7, label="Sure available")
    plt.xlabel("Teacher sure strength")
    plt.ylabel("Internal margin")
    plt.title("Sure strength vs internal margin")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.scatter(internal_margin[forced_mask], sure_output_near_ts[forced_mask], alpha=0.7, label="Sure unavailable")
    plt.scatter(internal_margin[offered_mask], sure_output_near_ts[offered_mask], alpha=0.7, label="Sure available")
    plt.xlabel("Internal margin")
    plt.ylabel("Sure-channel activity near TS/readout")
    plt.title("Internal margin vs sure-channel activity")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.scatter(coherence[forced_mask], internal_margin[forced_mask], alpha=0.7, label="Sure unavailable")
    plt.scatter(coherence[offered_mask], internal_margin[offered_mask], alpha=0.7, label="Sure available")
    plt.xlabel("Coherence")
    plt.ylabel("Internal margin")
    plt.title("Coherence vs internal margin")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    save_figure(output_dir, f"{stem}.png")


def plot_soft_vs_hard_readout_diagnostics(analysis_data, output_dir, stem="soft_hard_readout_diagnostics"):
    coherence = analysis_data["coherence"]
    sure_available = analysis_data["sure_available"].astype(bool)
    delta = analysis_data["sure_competition_delta"]
    hard_sure_choice = analysis_data["hard_sure_choice"].astype(float)
    offered_mask = sure_available
    unique_coh = np.unique(coherence[offered_mask]) if np.any(offered_mask) else np.array([])
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 2, 1)
    plt.hist(delta, bins=20, alpha=0.8, color="tab:blue")
    plt.axvline(0.0, color="k", linestyle="--", alpha=0.7)
    plt.xlabel("Sure competition delta = sure - max(direction)")
    plt.ylabel("Trial count")
    plt.title("Soft sure competition delta (all trials)")
    plt.grid(True, alpha=0.2)
    plt.subplot(2, 2, 2)
    if np.any(offered_mask):
        for coh in unique_coh:
            idx = offered_mask & np.isclose(coherence, coh)
            if np.any(idx):
                plt.hist(delta[idx], bins=15, alpha=0.45, density=True, label=f"coh={coh:.3f}")
    plt.axvline(0.0, color="k", linestyle="--", alpha=0.7)
    plt.xlabel("Sure competition delta")
    plt.ylabel("Density")
    plt.title("Delta grouped by coherence (offered)")
    plt.grid(True, alpha=0.2)
    if unique_coh.size > 0:
        plt.legend(fontsize=8)
    plt.subplot(2, 2, 3)
    if np.any(offered_mask):
        plt.scatter(delta[offered_mask], hard_sure_choice[offered_mask], alpha=0.6)
    plt.axvline(0.0, color="k", linestyle="--", alpha=0.7)
    plt.yticks([0, 1], ["Direction", "Sure"])
    plt.xlabel("Sure competition delta")
    plt.ylabel("Hard argmax choice")
    plt.title("Hard sure choice vs soft competition delta")
    plt.grid(True, alpha=0.2)
    plt.subplot(2, 2, 4)
    centers, probs, _ = binned_probability_curve(delta[offered_mask], hard_sure_choice[offered_mask].astype(float), n_bins=7)
    fit_result = fit_logistic_curve(delta[offered_mask], hard_sure_choice[offered_mask].astype(float))
    if centers.size > 0:
        plt.plot(centers, probs, "o", color="tab:orange", label="Binned hard P(sure)")
    if fit_result is not None:
        x_grid = np.linspace(np.min(delta[offered_mask]), np.max(delta[offered_mask]), 200)
        plt.plot(x_grid, evaluate_logistic_curve(fit_result, x_grid), linewidth=2.5, label="Logistic fit")
    plt.axvline(0.0, color="k", linestyle="--", alpha=0.7)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sure competition delta")
    plt.ylabel("P(hard sure choice)")
    plt.title("How argmax maps soft competition to choice")
    plt.grid(True, alpha=0.2)
    if centers.size > 0 or fit_result is not None:
        plt.legend()
    plt.tight_layout()
    save_figure(output_dir, f"{stem}.png")


def plot_sure_choice_vs_internal_margin(analysis_data, output_dir, stem="sure_choice_vs_internal_margin"):
    sure_available = analysis_data["sure_available"].astype(bool)
    internal_margin = analysis_data["internal_margin"]
    hard_sure_choice = analysis_data["hard_sure_choice"].astype(float)
    x = internal_margin[sure_available]
    y = hard_sure_choice[sure_available]
    plt.figure(figsize=(9, 6))
    has_handles = False
    if x.size > 0:
        jitter = (np.random.RandomState(0).rand(x.size) - 0.5) * 0.04
        plt.scatter(x, y + jitter, alpha=0.25, s=18, color="tab:blue", label="Trials")
        has_handles = True
        centers, probs, _ = binned_probability_curve(x, y, n_bins=7)
        if centers.size > 0:
            plt.plot(centers, probs, "o-", linewidth=2, color="tab:orange", label="Binned P(sure)")
            has_handles = True
        fit_result = fit_logistic_curve(x, y)
        if fit_result is not None:
            x_grid = np.linspace(np.min(x), np.max(x), 200)
            y_grid = evaluate_logistic_curve(fit_result, x_grid)
            plt.plot(x_grid, y_grid, linewidth=2.5, color="tab:red", label="Logistic fit")
            has_handles = True
        smooth_x, smooth_y = smooth_binary_curve(x, y, num_points=200)
        if smooth_x is not None and smooth_y is not None:
            plt.plot(smooth_x, smooth_y, linewidth=2, linestyle="--", color="tab:green", label="Kernel smooth")
            has_handles = True
    plt.xlabel("Internal margin")
    plt.ylabel("P(sure | offered)")
    plt.title("Sure choice vs internal confidence")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.2)
    if has_handles:
        plt.legend()
    plt.tight_layout()
    save_figure(output_dir, f"{stem}.png")


def plot_pre_onset_sure_baseline(analysis_data, output_dir, stem="pre_onset_sure_baseline"):
    sure_available = analysis_data["sure_available"].astype(bool)
    internal_margin = analysis_data["internal_margin"]
    pre_onset_sure_output = analysis_data["pre_onset_sure_output"]
    median_internal = np.median(internal_margin) if internal_margin.size > 0 else np.nan
    low_internal = internal_margin <= median_internal
    high_internal = internal_margin > median_internal
    plt.figure(figsize=(11, 8))
    plt.subplot(2, 2, 1)
    available_vals = pre_onset_sure_output[sure_available]
    unavailable_vals = pre_onset_sure_output[~sure_available]
    plt.bar([0, 1], [mean_or_nan(available_vals), mean_or_nan(unavailable_vals)],
            yerr=[sem_or_nan(available_vals), sem_or_nan(unavailable_vals)],
            color=["tab:blue", "tab:gray"], alpha=0.8)
    plt.xticks([0, 1], ["Sure available", "Sure unavailable"])
    plt.ylabel("Pre-onset sure activity")
    plt.title("Baseline by sure availability")
    plt.grid(True, axis="y", alpha=0.2)
    plt.subplot(2, 2, 2)
    low_vals = pre_onset_sure_output[low_internal]
    high_vals = pre_onset_sure_output[high_internal]
    plt.bar([0, 1], [mean_or_nan(low_vals), mean_or_nan(high_vals)],
            yerr=[sem_or_nan(low_vals), sem_or_nan(high_vals)],
            color=["tab:orange", "tab:green"], alpha=0.8)
    plt.xticks([0, 1], ["Low internal margin", "High internal margin"])
    plt.ylabel("Pre-onset sure activity")
    plt.title("Baseline by internal margin")
    plt.grid(True, axis="y", alpha=0.2)
    plt.subplot(2, 2, 3)
    has_hist = False
    if np.any(sure_available):
        plt.hist(pre_onset_sure_output[sure_available], bins=18, alpha=0.6, label="Available")
        has_hist = True
    if np.any(~sure_available):
        plt.hist(pre_onset_sure_output[~sure_available], bins=18, alpha=0.6, label="Unavailable")
        has_hist = True
    plt.xlabel("Pre-onset sure activity")
    plt.ylabel("Trial count")
    plt.title("Pre-onset sure distributions")
    plt.grid(True, alpha=0.2)
    if has_hist:
        plt.legend()
    plt.subplot(2, 2, 4)
    plt.scatter(internal_margin, pre_onset_sure_output, alpha=0.5, c=sure_available.astype(float), cmap="coolwarm")
    plt.xlabel("Internal margin")
    plt.ylabel("Pre-onset sure activity")
    plt.title("Pre-onset sure vs internal margin")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    save_figure(output_dir, f"{stem}.png")


def compute_direction_dynamics(outputs, trial_info, internal_margin):
    sure_available = np.array([info["sure_available"] for info in trial_info], dtype=bool)
    median_internal = np.median(internal_margin) if internal_margin.size > 0 else np.nan
    low_internal = internal_margin <= median_internal
    high_internal = internal_margin > median_internal
    groups = {
        "available": sure_available,
        "unavailable": ~sure_available,
        "available_low_internal": sure_available & low_internal,
        "available_high_internal": sure_available & high_internal,
        "unavailable_low_internal": (~sure_available) & low_internal,
        "unavailable_high_internal": (~sure_available) & high_internal,
    }
    dynamics = {}
    for group_name, mask in groups.items():
        signed_traces = []
        abs_traces = []
        pref_traces = []
        null_traces = []
        for idx in np.where(mask)[0]:
            signed_traces.append(get_signed_lr_trace(outputs[idx], trial_info[idx]))
            abs_traces.append(get_abs_lr_trace(outputs[idx]))
            pref_trace, null_trace = get_pref_null_traces(outputs[idx], trial_info[idx])
            pref_traces.append(pref_trace)
            null_traces.append(null_trace)
        dynamics[group_name] = {
            "n": int(np.sum(mask)),
            "signed": average_trace(signed_traces),
            "abs": average_trace(abs_traces),
            "pref": average_trace(pref_traces),
            "null": average_trace(null_traces),
        }
    return dynamics


def plot_direction_dynamics(dynamics, dt, output_dir, stem="direction_dynamics_internal_split"):
    first_trace = next((v["signed"] for v in dynamics.values() if v["signed"] is not None), None)
    if first_trace is None:
        return
    time = np.arange(len(first_trace)) * dt
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    if dynamics["available"]["signed"] is not None:
        plt.plot(time, dynamics["available"]["signed"], linewidth=2.5, label="Sure available")
    if dynamics["unavailable"]["signed"] is not None:
        plt.plot(time, dynamics["unavailable"]["signed"], linewidth=2.5, label="Sure unavailable")
    plt.axvline(1500, linestyle="-.", alpha=0.5, color="k", label="Nominal sure onset")
    plt.xlabel("Time (ms)")
    plt.ylabel("Signed left-right")
    plt.title("Signed direction trajectory")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.subplot(2, 2, 2)
    if dynamics["available"]["abs"] is not None:
        plt.plot(time, dynamics["available"]["abs"], linewidth=2.5, label="Sure available")
    if dynamics["unavailable"]["abs"] is not None:
        plt.plot(time, dynamics["unavailable"]["abs"], linewidth=2.5, label="Sure unavailable")
    plt.axvline(1500, linestyle="-.", alpha=0.5, color="k")
    plt.xlabel("Time (ms)")
    plt.ylabel("|left-right|")
    plt.title("Competition magnitude")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.subplot(2, 2, 3)
    if dynamics["available"]["pref"] is not None:
        plt.plot(time, dynamics["available"]["pref"], linewidth=2.5, label="Preferred available")
        plt.plot(time, dynamics["available"]["null"], linewidth=2.5, label="Null available")
    if dynamics["unavailable"]["pref"] is not None:
        plt.plot(time, dynamics["unavailable"]["pref"], linestyle="--", linewidth=2.0, label="Preferred unavailable")
        plt.plot(time, dynamics["unavailable"]["null"], linestyle="--", linewidth=2.0, label="Null unavailable")
    plt.axvline(1500, linestyle="-.", alpha=0.5, color="k")
    plt.xlabel("Time (ms)")
    plt.ylabel("Activity")
    plt.title("Preferred/null by sure availability")
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=8)
    plt.subplot(2, 2, 4)
    if dynamics["available_low_internal"]["abs"] is not None:
        plt.plot(time, dynamics["available_low_internal"]["abs"], linewidth=2.5, label="Avail low internal")
    if dynamics["available_high_internal"]["abs"] is not None:
        plt.plot(time, dynamics["available_high_internal"]["abs"], linewidth=2.5, label="Avail high internal")
    if dynamics["unavailable_low_internal"]["abs"] is not None:
        plt.plot(time, dynamics["unavailable_low_internal"]["abs"], linestyle="--", linewidth=2.0, label="Unavail low internal")
    if dynamics["unavailable_high_internal"]["abs"] is not None:
        plt.plot(time, dynamics["unavailable_high_internal"]["abs"], linestyle="--", linewidth=2.0, label="Unavail high internal")
    plt.axvline(1500, linestyle="-.", alpha=0.5, color="k")
    plt.xlabel("Time (ms)")
    plt.ylabel("|left-right|")
    plt.title("Competition magnitude with internal-margin split")
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_figure(output_dir, f"{stem}.png")


def sorted_nearest_match_indices(primary_values, candidate_values):
    primary_values = np.asarray(primary_values, dtype=float)
    candidate_values = np.asarray(candidate_values, dtype=float)
    if primary_values.size == 0 or candidate_values.size == 0:
        return []
    primary_order = np.argsort(primary_values)
    candidate_order = np.argsort(candidate_values)
    n_pairs = min(primary_order.size, candidate_order.size)
    return [(int(primary_order[i]), int(candidate_order[i])) for i in range(n_pairs)]


def compute_matched_trial_dynamics(outputs, trial_info, analysis_data, match_key="internal_margin"):
    coherence = np.asarray(analysis_data["coherence"], dtype=float)
    match_values = np.asarray(analysis_data[match_key], dtype=float)
    sure_available = np.asarray(analysis_data["sure_available"], dtype=bool)
    matched_available = []
    matched_unavailable = []
    for coh in np.unique(coherence):
        avail_idx = np.where(sure_available & np.isclose(coherence, coh))[0]
        unavail_idx = np.where((~sure_available) & np.isclose(coherence, coh))[0]
        if avail_idx.size == 0 or unavail_idx.size == 0:
            continue
        pairs_local = sorted_nearest_match_indices(match_values[avail_idx], match_values[unavail_idx])
        for a_local, u_local in pairs_local:
            matched_available.append(int(avail_idx[a_local]))
            matched_unavailable.append(int(unavail_idx[u_local]))
    matched_available = np.asarray(matched_available, dtype=int)
    matched_unavailable = np.asarray(matched_unavailable, dtype=int)
    paired_abs_diff = []
    for a_idx, u_idx in zip(matched_available, matched_unavailable):
        paired_abs_diff.append(get_abs_lr_trace(outputs[a_idx]) - get_abs_lr_trace(outputs[u_idx]))
    return {
        "match_key": match_key,
        "n_pairs": int(len(matched_available)),
        "available_abs": average_trace([get_abs_lr_trace(outputs[i]) for i in matched_available]),
        "unavailable_abs": average_trace([get_abs_lr_trace(outputs[i]) for i in matched_unavailable]),
        "available_pref": average_trace([get_pref_null_traces(outputs[i], trial_info[i])[0] for i in matched_available]),
        "unavailable_pref": average_trace([get_pref_null_traces(outputs[i], trial_info[i])[0] for i in matched_unavailable]),
        "available_null": average_trace([get_pref_null_traces(outputs[i], trial_info[i])[1] for i in matched_available]),
        "unavailable_null": average_trace([get_pref_null_traces(outputs[i], trial_info[i])[1] for i in matched_unavailable]),
        "paired_abs_diff": average_trace(paired_abs_diff),
    }


def plot_matched_trial_dynamics(matched_data, dt, output_dir, stem="matched_trial_dynamics"):
    if matched_data["n_pairs"] == 0 or matched_data["available_abs"] is None:
        return
    time = np.arange(len(matched_data["available_abs"])) * dt
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time, matched_data["available_abs"], linewidth=2.5, label="Matched sure available")
    plt.plot(time, matched_data["unavailable_abs"], linewidth=2.5, label="Matched sure unavailable")
    plt.plot(time, matched_data["paired_abs_diff"], linewidth=2.0, linestyle="--", label="Available - unavailable")
    plt.axvline(1500, linestyle="-.", alpha=0.5, color="k", label="Nominal sure onset")
    plt.xlabel("Time (ms)")
    plt.ylabel("|left-right|")
    plt.title(f"Matched competition dynamics ({matched_data['match_key']})")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time, matched_data["available_pref"], linewidth=2.5, label="Preferred available")
    plt.plot(time, matched_data["unavailable_pref"], linewidth=2.5, label="Preferred unavailable")
    plt.plot(time, matched_data["available_null"], linestyle="--", linewidth=2.0, label="Null available")
    plt.plot(time, matched_data["unavailable_null"], linestyle="--", linewidth=2.0, label="Null unavailable")
    plt.axvline(1500, linestyle="-.", alpha=0.5, color="k")
    plt.xlabel("Time (ms)")
    plt.ylabel("Activity")
    plt.title(f"Matched preferred/null dynamics ({matched_data['n_pairs']} pairs)")
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_figure(output_dir, f"{stem}.png")
