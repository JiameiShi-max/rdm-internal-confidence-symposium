import json
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from psychrnn.tasks.task import Task
from psychrnn.backend.models.basic import Basic

tf.compat.v1.reset_default_graph()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class RDM_SureTarget_Task(Task):
    """
    Variable-delay fixed-duration RDM task with a soft sure-target head.

    Key idea:
    - sure target participates as a 3rd competing action channel
    - supervision is not a hard coherence gate
    - instead, it is based on trial-specific noisy evidence margin

    sure_strength = sigmoid((margin_center - sensory_margin) / margin_temp)

    where
        sensory_margin = |sum(x_left) - sum(x_right)| over stimulus window
    """

    def __init__(
        self,
        dt,
        tau,
        T,
        N_batch,
        margin_center=2.0,
        margin_temp=0.45,
        pre_go_sure_weight=0.35,
    ):
        super(RDM_SureTarget_Task, self).__init__(
            N_in=4,  # fixation, left sensory evidence, right, sure cue input
            N_out=4,
            dt=dt,
            tau=tau,
            T=T,
            N_batch=N_batch,
        )
        self.coherences = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
        self.margin_center = margin_center
        self.margin_temp = margin_temp
        self.pre_go_sure_weight = pre_go_sure_weight
        self.trial_info = []

    def reset_trial_info(self):
        self.trial_info = []

    def generate_trial_params(self, batch, trial_mode):
        if hasattr(batch, "__len__"):
            batch_k = len(batch)
        else:
            batch_k = int(batch)
        return [None] * batch_k

    def trial_function(self, time, params):
        return self.generate_trial(params)

    def generate_trial(self, params):
        # --------------------------------------------------
        # 1) Random trial configuration
        # --------------------------------------------------
        dir_choice = np.random.choice([0, 1])  # 0=left, 1=right
        coh = float(np.random.choice(self.coherences))

        fixation_dur = 200
        stimulus_dur = 800
        min_delay, max_delay = 400, 1000
        delay_dur = np.random.randint(int(min_delay / self.dt), int(max_delay / self.dt)) * self.dt  # variable delay

        fixation_end = int(fixation_dur / self.dt)
        stimulus_end = fixation_end + int(stimulus_dur / self.dt)
        delay_end = stimulus_end + int(delay_dur / self.dt)
        if delay_end > self.N_steps:
            delay_end = self.N_steps - 1

        ts_onset = stimulus_end + int(500 / self.dt)
        sure_available = ts_onset < delay_end

        # --------------------------------------------------
        # 2) Allocate arrays
        # --------------------------------------------------
        x = np.zeros((self.N_steps, self.N_in), dtype=np.float32)
        y = np.zeros((self.N_steps, self.N_out), dtype=np.float32)
        mask = np.zeros((self.N_steps, self.N_out), dtype=np.float32)

        # --------------------------------------------------
        # 3) Inputs
        # --------------------------------------------------
        x[:delay_end, 0] = 1.0  # fixation input

        stim_len = stimulus_end - fixation_end
        scale = 0.5
        if stim_len > 0:
            noise_scale = 0.1
            noise_L = np.random.normal(0, noise_scale, stim_len)
            noise_R = np.random.normal(0, noise_scale, stim_len)

            signal_L = coh if dir_choice == 0 else 0.0
            signal_R = coh if dir_choice == 1 else 0.0

            x[fixation_end:stimulus_end, 1] += (signal_L + noise_L) * scale
            x[fixation_end:stimulus_end, 2] += (signal_R + noise_R) * scale

        if sure_available:
            x[ts_onset:delay_end, 3] = 1.0

        # --------------------------------------------------
        # 4) Trial-specific "confidence" from sensory margin
        # --------------------------------------------------
        # This is NOT just coherence. It depends on the actual noisy input
        # that this trial received.
        sensory_sum_L = np.sum(x[fixation_end:stimulus_end, 1])
        sensory_sum_R = np.sum(x[fixation_end:stimulus_end, 2])
        sensory_margin = float(abs(sensory_sum_L - sensory_sum_R))

        if sure_available:
            sure_strength = float(
                sigmoid((self.margin_center - sensory_margin) / self.margin_temp)
            )
        else:
            sure_strength = 0.0

        # --------------------------------------------------
        # 5) Targets
        # --------------------------------------------------
        y[:delay_end, 0] = 1.0  # fixation output

        # Pre-go sure target: encourage sure head to start ramping when offered,
        # but only in proportion to uncertainty.
        if sure_available:
            y[ts_onset:delay_end, 3] = sure_strength

        # After go cue:
        # - direction output target gets (1 - sure_strength)
        # - sure output target gets sure_strength
        # This makes sure channel participate as a real competitor.
        if dir_choice == 0:
            y[delay_end:, 1] = 1.0 - sure_strength
            y[delay_end:, 2] = 0.0
        else:
            y[delay_end:, 2] = 1.0 - sure_strength
            y[delay_end:, 1] = 0.0

        y[delay_end:, 3] = sure_strength

        # --------------------------------------------------
        # 6) Mask
        # --------------------------------------------------
        mask[:, 0] = 1.0
        mask[:fixation_end, 1:] = 1.0

        # During stimulus+delay:
        # left/right remain unsupervised (network builds memory / competition),
        # but sure channel gets weak supervision after TS onset
        mask[fixation_end:delay_end, 1:] = 0.0
        if sure_available:
            mask[ts_onset:delay_end, 3] = self.pre_go_sure_weight  # weak supervision

        # After go cue all action outputs are supervised
        mask[delay_end:, 1:] = 1.0

        # --------------------------------------------------
        # 7) Save metadata
        # --------------------------------------------------
        self.trial_info.append(
            {
                "dir_choice": int(dir_choice),
                "coh": float(coh),
                "fixation_end": int(fixation_end),
                "stimulus_end": int(stimulus_end),
                "delay_end": int(delay_end),
                "ts_onset": int(ts_onset),
                "sure_available": bool(sure_available),
                "sensory_margin": float(sensory_margin),
                "sure_strength": float(sure_strength),
            }
        )

        return x, y, mask


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def readout_fd_trial(outputs_trial, delay_end, window=10):
    start = int(delay_end)
    end = min(start + int(window), outputs_trial.shape[0])
    if end <= start:
        end = min(start + 1, outputs_trial.shape[0])
    avg = outputs_trial[start:end, :].mean(axis=0)
    choice = int(np.argmax(avg))
    return choice, avg


def avg_output_in_window(outputs_trial, center_step, half_window=5):
    start = max(0, int(center_step) - int(half_window))
    end = min(outputs_trial.shape[0], int(center_step) + int(half_window))
    if end <= start:
        end = min(outputs_trial.shape[0], start + 1)
    return outputs_trial[start:end, :].mean(axis=0)


def get_true_direction_from_metadata(trial_info):
    return np.array([1 if info["dir_choice"] == 0 else 2 for info in trial_info], dtype=int)


def confidence_from_avg(avg_out):
    return float(abs(avg_out[1] - avg_out[2]))


def confidence_near_ts(outputs_trial, trial_meta, half_window=5):
    if trial_meta["sure_available"]:
        avg_ts = avg_output_in_window(outputs_trial, trial_meta["ts_onset"], half_window=half_window)
    else:
        avg_ts = avg_output_in_window(outputs_trial, trial_meta["delay_end"], half_window=half_window)
    return confidence_from_avg(avg_ts)


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


def unit_norm(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec * 0.0
    return vec / norm


def fit_target_axis(states_2d, target, ridge=1e-4):
    x = np.asarray(states_2d, dtype=float)
    y = np.asarray(target, dtype=float)
    x_center = x - x.mean(axis=0, keepdims=True)
    y_center = y - y.mean()
    cov = x_center.T @ x_center + ridge * np.eye(x_center.shape[1])
    rhs = x_center.T @ y_center
    weights = np.linalg.solve(cov, rhs)
    return unit_norm(weights)


def residualize_states_against_axis(states_2d, axis):
    axis = unit_norm(axis)
    centered = states_2d - states_2d.mean(axis=0, keepdims=True)
    proj = centered @ axis
    return centered - np.outer(proj, axis)


def project_states(states_2d, axis):
    axis = unit_norm(axis)
    centered = states_2d - states_2d.mean(axis=0, keepdims=True)
    return centered @ axis


def fit_time_axis(states_3d):
    mean_traj = np.mean(states_3d, axis=0)
    centered = mean_traj - mean_traj.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return unit_norm(vt[0])


def binned_probability_curve(x, y, n_bins=7):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    n_bins = max(1, min(int(n_bins), x.size))
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-8
    edges[-1] += 1e-8
    centers, probs, counts = [], [], []
    for idx in range(n_bins):
        in_bin = (x >= edges[idx]) & (x < edges[idx + 1])
        if np.any(in_bin):
            centers.append(float(np.mean(x[in_bin])))
            probs.append(float(np.mean(y[in_bin])))
            counts.append(int(np.sum(in_bin)))
    return np.array(centers), np.array(probs), np.array(counts, dtype=int)


def _logistic(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def fit_logistic_curve(x, y, max_iter=4000, lr=0.05):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 4 or np.unique(y).size < 2:
        return None
    x_mean = float(np.mean(x))
    x_std = max(float(np.std(x)), 1e-8)
    xz = (x - x_mean) / x_std
    b0 = 0.0
    b1 = 0.0
    n = float(xz.size)
    for _ in range(max_iter):
        pred = _logistic(b0 + b1 * xz)
        b0 -= lr * np.sum(pred - y) / n
        b1 -= lr * np.sum((pred - y) * xz) / n
    return {"b0": float(b0), "b1": float(b1), "x_mean": x_mean, "x_std": x_std}


def evaluate_logistic_curve(fit, x_grid):
    if fit is None:
        return None
    x_grid = np.asarray(x_grid, dtype=float)
    z = fit["b0"] + fit["b1"] * (x_grid - fit["x_mean"]) / fit["x_std"]
    return _logistic(z)


def build_targeted_axes(states, trial_info, choices, sure_flags):
    readout_states = []
    signed_evidence = []

    for idx, info in enumerate(trial_info):
        center_step = info["ts_onset"] if info["sure_available"] else info["delay_end"]
        center_step = min(max(int(center_step), 0), states.shape[1] - 1)
        readout_states.append(states[idx, center_step, :])
        sign = 1.0 if info["dir_choice"] == 0 else -1.0
        signed_evidence.append(sign * float(info["sensory_margin"]))

    readout_states = np.asarray(readout_states, dtype=float)
    signed_evidence = np.asarray(signed_evidence, dtype=float)

    evidence_axis = fit_target_axis(readout_states, signed_evidence)
    evidence_proj = project_states(readout_states, evidence_axis)

    offered = np.array([info["sure_available"] for info in trial_info], dtype=bool)
    sure_float = sure_flags.astype(float)
    sure_residual = sure_float.copy()

    if np.any(offered):
        x = np.abs(evidence_proj[offered])
        design = np.column_stack([np.ones_like(x), x])
        coef, _, _, _ = np.linalg.lstsq(design, sure_float[offered], rcond=None)
        sure_residual[offered] = sure_float[offered] - design @ coef

    residual_states = residualize_states_against_axis(readout_states, evidence_axis)
    if np.sum(offered) >= 4:
        sure_axis = fit_target_axis(residual_states[offered], sure_residual[offered])
    else:
        sure_axis = fit_target_axis(residual_states, sure_residual)
    sure_proj = project_states(readout_states, sure_axis)

    time_axis = fit_time_axis(states)

    return {
        "evidence_axis": evidence_axis,
        "sure_axis": sure_axis,
        "time_axis": time_axis,
        "evidence_proj": evidence_proj,
        "sure_proj": sure_proj,
        "readout_states": readout_states,
    }


def save_figure(output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def summarize_stage1(trial_info, choices, conf_values, axes_data):
    sure_flags = (choices == 3).astype(float)
    offered = np.array([info["sure_available"] for info in trial_info], dtype=bool)
    evidence_proj = np.abs(axes_data["evidence_proj"])
    sure_proj = axes_data["sure_proj"]

    residual_corr = np.nan
    if np.sum(offered) >= 4:
        design = np.column_stack([np.ones(np.sum(offered)), evidence_proj[offered]])
        coef, _, _, _ = np.linalg.lstsq(design, sure_flags[offered], rcond=None)
        residual_sure = sure_flags[offered] - design @ coef
        residual_corr = safe_corrcoef(sure_proj[offered], residual_sure)

    return {
        "n_trials": int(len(trial_info)),
        "n_sure_offered": int(np.sum(offered)),
        "n_sure_choice": int(np.sum(choices == 3)),
        "corr_confidence_vs_evidence_axis": safe_corrcoef(conf_values, evidence_proj),
        "corr_psure_vs_evidence_axis": safe_corrcoef(evidence_proj[offered], sure_flags[offered]),
        "corr_psure_vs_sure_axis": safe_corrcoef(sure_proj[offered], sure_flags[offered]),
        "corr_evidence_axis_vs_sure_axis": safe_corrcoef(evidence_proj[offered], sure_proj[offered]),
        "corr_sure_axis_with_sure_residual_after_evidence": residual_corr,
    }


if __name__ == "__main__":
    dt = 10
    tau = 100
    T = 2000
    N_batch = 50
    N_rec = 50
    readout_window = 10
    ts_half_window = 5
    n_eval_batches = 8

    run_dir = os.path.dirname(os.path.abspath(__file__))
    figure_dir = os.path.join(run_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    rdm_task = RDM_SureTarget_Task(
        dt=dt,
        tau=tau,
        T=T,
        N_batch=N_batch,
        margin_center=2.0,
        margin_temp=0.45,
        pre_go_sure_weight=0.35,
    )

    network_params = rdm_task.get_task_params()
    network_params["name"] = "RDM_SureTarget_TargetedDR_Stage1"
    network_params["N_rec"] = N_rec
    network_params["rec_noise"] = 0.05
    network_params["activation"] = "rectified_linear"
    network_params["alpha"] = 0.01
    network_params["training_iters"] = 50000
    network_params["loss_epoch"] = 1000

    model = Basic(network_params)

    print("Starting training...")
    model.train(rdm_task)
    print("Training complete.")

    print("Collecting evaluation batches...")
    outputs_list, states_list, info_list = [], [], []
    for _ in range(n_eval_batches):
        rdm_task.reset_trial_info()
        batch_x, _, _, _ = rdm_task.get_trial_batch()
        batch_info = list(rdm_task.trial_info)
        batch_outputs, batch_states = model.test(batch_x)
        outputs_list.append(batch_outputs)
        states_list.append(batch_states)
        info_list.extend(batch_info)

    outputs = np.concatenate(outputs_list, axis=0)
    states = np.concatenate(states_list, axis=0)
    trial_info = info_list

    choices = []
    conf_values = []
    for idx, info in enumerate(trial_info):
        choice_idx, _ = readout_fd_trial(outputs[idx], info["delay_end"], window=readout_window)
        choices.append(choice_idx)
        conf_values.append(confidence_near_ts(outputs[idx], info, half_window=ts_half_window))

    choices = np.array(choices, dtype=int)
    conf_values = np.array(conf_values, dtype=float)
    true_dir = get_true_direction_from_metadata(trial_info)
    sure_flags = choices == 3
    offered = np.array([info["sure_available"] for info in trial_info], dtype=bool)

    axes_data = build_targeted_axes(states, trial_info, choices, sure_flags)

    # 1. Neural activity grouped by confidence
    conf_median = np.median(conf_values)
    high_conf = conf_values > conf_median
    low_conf = ~high_conf
    pref_high, null_high, pref_low, null_low, sure_trace = [], [], [], [], []

    for idx in range(len(trial_info)):
        if true_dir[idx] == 1:
            pref_trace = outputs[idx, :, 1]
            null_trace = outputs[idx, :, 2]
        else:
            pref_trace = outputs[idx, :, 2]
            null_trace = outputs[idx, :, 1]

        if high_conf[idx]:
            pref_high.append(pref_trace)
            null_high.append(null_trace)
        else:
            pref_low.append(pref_trace)
            null_low.append(null_trace)

        if sure_flags[idx]:
            sure_trace.append(outputs[idx, :, 3])

    plt.figure(figsize=(10, 6))
    if pref_high:
        plt.plot(np.mean(pref_high, axis=0), linewidth=2.5, label="Preferred (high evidence-axis confidence)")
        plt.plot(np.mean(null_high, axis=0), linewidth=2.5, label="Null (high evidence-axis confidence)")
    if pref_low:
        plt.plot(np.mean(pref_low, axis=0), linestyle="--", linewidth=2, label="Preferred (low evidence-axis confidence)")
        plt.plot(np.mean(null_low, axis=0), linestyle="--", linewidth=2, label="Null (low evidence-axis confidence)")
    if sure_trace:
        plt.plot(np.mean(sure_trace, axis=0), linewidth=3, label="Sure output (sure-choice trials)")
    plt.axvline(x=int(200 / dt), linestyle=":", alpha=0.5, label="Stimulus onset")
    plt.axvline(x=int(1000 / dt), linestyle="--", alpha=0.5, label="Delay start")
    plt.axvline(x=int(1500 / dt), linestyle="-.", alpha=0.5, label="Nominal sure onset")
    plt.title("Neural activity grouped by internal confidence")
    plt.xlabel("Time steps")
    plt.ylabel("Activity")
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(True, alpha=0.15)
    save_figure(figure_dir, "st1_neural_activity_by_internal_confidence.png")

    # 2. Targeted evidence/sure axis analysis
    evidence_abs = np.abs(axes_data["evidence_proj"])
    sure_proj = axes_data["sure_proj"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = np.where(sure_flags, "tab:green", "tab:red")
    axes[0].scatter(evidence_abs, sure_proj, c=colors, alpha=0.65, s=20)
    axes[0].set_xlabel("Evidence-axis |projection|")
    axes[0].set_ylabel("Sure-axis projection")
    axes[0].set_title("Hidden evidence vs sure tendency")
    axes[0].grid(True, alpha=0.2)

    x_ev = evidence_abs[offered]
    y_sure = sure_flags[offered].astype(float)
    centers, probs, _ = binned_probability_curve(x_ev, y_sure, n_bins=7)
    axes[1].scatter(x_ev, y_sure + (np.random.RandomState(0).rand(y_sure.size) - 0.5) * 0.04,
                    alpha=0.2, s=12, color="tab:blue", label="Trials")
    if centers.size:
        axes[1].plot(centers, probs, "o-", linewidth=2, color="tab:orange", label="Binned")
    fit = fit_logistic_curve(x_ev, y_sure)
    if fit is not None:
        x_grid = np.linspace(np.min(x_ev), np.max(x_ev), 200)
        axes[1].plot(x_grid, evaluate_logistic_curve(fit, x_grid), linewidth=2.5, color="tab:red", label="Logistic fit")
    axes[1].set_xlabel("Evidence-axis |projection|")
    axes[1].set_ylabel("P(Sure | offered)")
    axes[1].set_title("P(Sure) vs evidence-axis margin")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    x_sure = sure_proj[offered]
    centers_s, probs_s, _ = binned_probability_curve(x_sure, y_sure, n_bins=7)
    axes[2].scatter(x_sure, y_sure + (np.random.RandomState(1).rand(y_sure.size) - 0.5) * 0.04,
                    alpha=0.2, s=12, color="tab:blue", label="Trials")
    if centers_s.size:
        axes[2].plot(centers_s, probs_s, "o-", linewidth=2, color="tab:orange", label="Binned")
    fit_s = fit_logistic_curve(x_sure, y_sure)
    if fit_s is not None:
        x_grid_s = np.linspace(np.min(x_sure), np.max(x_sure), 200)
        axes[2].plot(x_grid_s, evaluate_logistic_curve(fit_s, x_grid_s), linewidth=2.5, color="tab:red", label="Logistic fit")
    axes[2].set_xlabel("Sure-axis projection")
    axes[2].set_ylabel("P(Sure | offered)")
    axes[2].set_title("P(Sure) vs sure-tendency axis")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2)
    plt.tight_layout()
    save_figure(figure_dir, "st2_targeted_hidden_axes.png")

    # 3. Trajectory-level targeted dynamics
    evidence_traj = np.tensordot(states, axes_data["evidence_axis"], axes=([2], [0]))
    sure_traj = np.tensordot(states, axes_data["sure_axis"], axes=([2], [0]))
    time_traj = np.tensordot(states, axes_data["time_axis"], axes=([2], [0]))

    dir_offered = offered & (~sure_flags)
    sure_offered = offered & sure_flags

    fig, traj_axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    if np.any(sure_offered):
        traj_axes[0].plot(np.mean(evidence_traj[sure_offered], axis=0), linewidth=2.5, label="Offered + sure choice")
        traj_axes[1].plot(np.mean(sure_traj[sure_offered], axis=0), linewidth=2.5, label="Offered + sure choice")
        traj_axes[2].plot(np.mean(time_traj[sure_offered], axis=0), linewidth=2.5, label="Offered + sure choice")
    if np.any(dir_offered):
        traj_axes[0].plot(np.mean(evidence_traj[dir_offered], axis=0), linewidth=2.5, label="Offered + direction choice")
        traj_axes[1].plot(np.mean(sure_traj[dir_offered], axis=0), linewidth=2.5, label="Offered + direction choice")
        traj_axes[2].plot(np.mean(time_traj[dir_offered], axis=0), linewidth=2.5, label="Offered + direction choice")

    traj_axes[0].set_title("Evidence axis")
    traj_axes[1].set_title("Sure-tendency axis")
    traj_axes[2].set_title("Time / urgency axis")
    for ax in traj_axes:
        ax.axvline(x=int(1500 / dt), linestyle="-.", alpha=0.45, color="k")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel("Projection")
    traj_axes[2].set_xlabel("Time steps")
    plt.suptitle("Stage-1 targeted hidden dynamics", y=0.98)
    plt.tight_layout()
    save_figure(figure_dir, "st3_targeted_hidden_dynamics.png")

    # 4. Behavior
    coherences = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
    sure_prob_offered = []
    forced_acc_waived = []
    offer_rate = []
    original_coherences = list(rdm_task.coherences)

    for coh in coherences:
        rdm_task.coherences = [coh]
        local_choices = []
        local_info = []
        for _ in range(4):
            rdm_task.reset_trial_info()
            batch_x, _, _, _ = rdm_task.get_trial_batch()
            batch_info = list(rdm_task.trial_info)
            batch_outputs, _ = model.test(batch_x)
            for idx, info in enumerate(batch_info):
                choice_idx, _ = readout_fd_trial(batch_outputs[idx], info["delay_end"], window=readout_window)
                local_choices.append(choice_idx)
            local_info.extend(batch_info)

        local_choices = np.array(local_choices, dtype=int)
        local_true_dir = get_true_direction_from_metadata(local_info)
        local_offered = np.array([info["sure_available"] for info in local_info], dtype=bool)
        offer_rate.append(float(np.mean(local_offered)))
        sure_prob_offered.append(float(np.mean(local_choices[local_offered] == 3)) if np.any(local_offered) else np.nan)
        waived = local_offered & (local_choices != 3)
        forced_acc_waived.append(float(np.mean(local_choices[waived] == local_true_dir[waived])) if np.any(waived) else np.nan)

    rdm_task.coherences = original_coherences

    plt.figure(figsize=(8, 6))
    plt.plot(coherences, sure_prob_offered, "o-", linewidth=3, label="P(Sure | offered)")
    plt.plot(coherences, forced_acc_waived, "o-", linewidth=2, label="Accuracy (waived sure)")
    plt.plot(coherences, offer_rate, "o--", linewidth=1.5, label="Sure offered rate")
    plt.title("Behavioral performance with competitive sure target")
    plt.xlabel("Motion coherence")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.2)
    save_figure(figure_dir, "st4_behavior.png")

    # 5. Mixed-batch evidence-axis curve
    plt.figure(figsize=(8, 6))
    mixed_centers, mixed_probs, _ = binned_probability_curve(evidence_abs[offered], sure_flags[offered].astype(float), n_bins=7)
    plt.scatter(evidence_abs[offered], sure_flags[offered].astype(float) +
                (np.random.RandomState(2).rand(np.sum(offered)) - 0.5) * 0.04,
                alpha=0.2, s=12, color="tab:blue", label="Trials")
    if mixed_centers.size:
        plt.plot(mixed_centers, mixed_probs, "o-", linewidth=2.5, color="tab:orange", label="Binned P(sure)")
    fit_mixed = fit_logistic_curve(evidence_abs[offered], sure_flags[offered].astype(float))
    if fit_mixed is not None:
        x_grid_mixed = np.linspace(np.min(evidence_abs[offered]), np.max(evidence_abs[offered]), 200)
        plt.plot(x_grid_mixed, evaluate_logistic_curve(fit_mixed, x_grid_mixed),
                 linewidth=2.5, color="tab:red", label="Logistic fit")
    plt.title("Sure choice vs internal confidence (targeted evidence axis)")
    plt.xlabel("Internal margin (targeted evidence axis)")
    plt.ylabel("P(Sure | offered)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.2)
    save_figure(figure_dir, "st5_psure_vs_evidence_axis.png")

    summary = summarize_stage1(trial_info, choices, conf_values, axes_data)
    summary_path = os.path.join(run_dir, "stage1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Stage-1 summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")
