import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from psychrnn.backend.models.basic import Basic
from psychrnn.tasks.task import Task

from internal_confidence_proxy import compute_soft_sure_teacher
from rdm_task_params import task_params_from_task
from stage8_extended_analysis import (
    analyze_external_vs_internal_confidence,
    compute_direction_dynamics,
    compute_matched_trial_dynamics,
    plot_direction_dynamics,
    plot_external_vs_internal_confidence_relationships,
    plot_matched_trial_dynamics,
    plot_pre_onset_sure_baseline,
    plot_soft_vs_hard_readout_diagnostics,
    plot_sure_choice_vs_internal_margin,
    summarize_external_vs_internal_confidence,
    summarize_pre_onset_sure_baseline,
    summarize_soft_vs_hard_readout,
)
from sure_target_stage1_targeted_dr import (
    avg_output_in_window,
    binned_probability_curve,
    build_targeted_axes,
    confidence_near_ts,
    evaluate_logistic_curve,
    fit_logistic_curve,
    get_true_direction_from_metadata,
    readout_fd_trial,
    safe_corrcoef,
    save_figure,
)

tf.compat.v1.reset_default_graph()


def set_global_seed(seed):
    """Keep training and evaluation reproducible across Python, NumPy, and TensorFlow."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    return seed


class RDM_SureTarget_InternalProxy_Task(Task):
    """
    Stable sure-target training with a more internalized confidence/value proxy.

    Compared with stage1:
    - still supervised and stable
    - but sure teacher no longer maps directly from sensory margin to sure strength
    - instead it first estimates a confidence-like probability-correct proxy, then
      converts that into an expected direction value, and finally compares it to
      the sure reward.
    """

    def __init__(
        self,
        dt,
        tau,
        T,
        N_batch,
        stimulus_scale=1.0,
        margin_center=2.0,
        margin_temp=0.70,
        direction_reward=1.0,
        error_reward=0.0,
        sure_reward=0.55,
        sure_value_temp=0.15,
        sure_target_min=0.05,
        sure_target_max=0.95,
        sure_target_blend=0.80,
        pre_go_sure_weight=0.0,
        stimulus_dur_min=800,
        stimulus_dur_max=800,
        delay_min=400,
        delay_max=1000,
        ts_delay=500,
        ts_latency_min=None,
        ts_latency_max=None,
        min_post_ts=0,
        min_response_dur=0,
        sure_offer_prob=None,
        max_timing_resamples=1000,
    ):
        super(RDM_SureTarget_InternalProxy_Task, self).__init__(
            N_in=4,
            N_out=4,
            dt=dt,
            tau=tau,
            T=T,
            N_batch=N_batch,
        )
        self.coherences = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
        self.stimulus_scale = float(stimulus_scale)
        self.margin_center = margin_center
        self.margin_temp = margin_temp
        self.direction_reward = direction_reward
        self.error_reward = error_reward
        self.sure_reward = sure_reward
        self.sure_value_temp = sure_value_temp
        self.sure_target_min = sure_target_min
        self.sure_target_max = sure_target_max
        self.sure_target_blend = sure_target_blend
        self.pre_go_sure_weight = pre_go_sure_weight
        self.stimulus_dur_min = int(stimulus_dur_min)
        self.stimulus_dur_max = int(stimulus_dur_max)
        self.delay_min = int(delay_min)
        self.delay_max = int(delay_max)
        self.ts_delay = int(ts_delay)
        self.ts_latency_min = int(ts_delay if ts_latency_min is None else ts_latency_min)
        self.ts_latency_max = int(ts_delay if ts_latency_max is None else ts_latency_max)
        if self.ts_latency_min > self.ts_latency_max:
            raise ValueError("ts_latency_min must be <= ts_latency_max")
        self.min_post_ts = int(min_post_ts)
        self.min_response_dur = int(min_response_dur)
        self.sure_offer_prob = None if sure_offer_prob is None else float(sure_offer_prob)
        self.max_timing_resamples = int(max_timing_resamples)
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

    def sample_duration_ms(self, min_ms, max_ms):
        min_steps = int(int(min_ms) / self.dt)
        max_steps = int(int(max_ms) / self.dt)
        if max_steps <= min_steps:
            return min_steps * self.dt
        return np.random.randint(min_steps, max_steps) * self.dt

    def sample_inclusive_duration_ms(self, min_ms, max_ms):
        """Sample a dt-compatible duration from an inclusive millisecond range."""
        min_steps = int(np.ceil(float(min_ms) / float(self.dt)))
        max_steps = int(np.floor(float(max_ms) / float(self.dt)))
        if max_steps < min_steps:
            raise ValueError("Timing range contains no dt-compatible value")
        return int(np.random.randint(min_steps, max_steps + 1) * self.dt)

    def sample_trial_timing(self):
        fixation_dur = 200
        fixation_end = int(fixation_dur / self.dt)
        min_response_steps = int(self.min_response_dur / self.dt)
        min_post_ts_steps = int(self.min_post_ts / self.dt)
        effective_delay_min = self.delay_min
        if self.sure_offer_prob is not None:
            effective_delay_min = max(
                effective_delay_min,
                self.ts_latency_max + max(self.min_post_ts, self.dt),
            )
        if effective_delay_min > self.delay_max:
            raise ValueError(
                "Cannot sample valid trial timing: delay_max is shorter than the "
                "maximum TS latency plus required post-TS time"
            )

        for _ in range(max(1, self.max_timing_resamples)):
            stimulus_dur = self.sample_duration_ms(self.stimulus_dur_min, self.stimulus_dur_max)
            delay_dur = self.sample_duration_ms(effective_delay_min, self.delay_max)
            ts_latency = self.sample_inclusive_duration_ms(
                self.ts_latency_min,
                self.ts_latency_max,
            )
            stimulus_end = fixation_end + int(stimulus_dur / self.dt)
            delay_end = stimulus_end + int(delay_dur / self.dt)
            ts_onset = stimulus_end + int(ts_latency / self.dt)
            timing_allows_sure = (
                ts_onset < delay_end
                and ts_onset + min_post_ts_steps <= delay_end
            )
            response_fits = delay_end <= self.N_steps - min_response_steps
            if timing_allows_sure and response_fits:
                return {
                    "fixation_dur": int(fixation_dur),
                    "stimulus_dur": int(stimulus_dur),
                    "delay_dur": int(delay_dur),
                    "fixation_end": int(fixation_end),
                    "stimulus_end": int(stimulus_end),
                    "delay_end": int(delay_end),
                    "ts_onset": int(ts_onset),
                    "ts_latency": int(ts_latency),
                    "timing_clipped": False,
                }

        raise ValueError(
            "Cannot sample valid trial timing with current T, stimulus/delay ranges, "
            "ts_delay, min_post_ts, and min_response_dur"
        )

    def generate_trial(self, params):
        dir_choice = np.random.choice([0, 1])
        coh = float(np.random.choice(self.coherences))

        timing = self.sample_trial_timing()
        fixation_dur = timing["fixation_dur"]
        stimulus_dur = timing["stimulus_dur"]
        delay_dur = timing["delay_dur"]
        fixation_end = timing["fixation_end"]
        stimulus_end = timing["stimulus_end"]
        delay_end = timing["delay_end"]
        ts_onset = timing["ts_onset"]
        timing_allows_sure = ts_onset + int(self.min_post_ts / self.dt) <= delay_end
        if self.sure_offer_prob is None:
            sure_available = ts_onset < delay_end
        else:
            sure_available = bool(np.random.random() < self.sure_offer_prob) and timing_allows_sure

        x = np.zeros((self.N_steps, self.N_in), dtype=np.float32)
        y = np.zeros((self.N_steps, self.N_out), dtype=np.float32)
        mask = np.zeros((self.N_steps, self.N_out), dtype=np.float32)

        x[:delay_end, 0] = 1.0

        stim_len = stimulus_end - fixation_end
        if stim_len > 0:
            noise_scale = 0.1
            noise_l = np.random.normal(0.0, noise_scale, stim_len)
            noise_r = np.random.normal(0.0, noise_scale, stim_len)
            signal_l = coh if dir_choice == 0 else 0.0
            signal_r = coh if dir_choice == 1 else 0.0
            x[fixation_end:stimulus_end, 1] += (signal_l + noise_l) * self.stimulus_scale
            x[fixation_end:stimulus_end, 2] += (signal_r + noise_r) * self.stimulus_scale

        if sure_available:
            x[ts_onset:delay_end, 3] = 1.0

        sensory_sum_l = np.sum(x[fixation_end:stimulus_end, 1])
        sensory_sum_r = np.sum(x[fixation_end:stimulus_end, 2])
        signed_sensory_evidence = float(sensory_sum_r - sensory_sum_l)
        integrated_sensory_margin = float(abs(signed_sensory_evidence))
        mean_sensory_evidence = float((sensory_sum_r - sensory_sum_l) / max(stim_len, 1))
        mean_sensory_margin = float(abs(mean_sensory_evidence))
        sensory_margin = integrated_sensory_margin

        teacher = compute_soft_sure_teacher(
            sensory_margin=sensory_margin,
            sure_available=sure_available,
            margin_center=self.margin_center,
            margin_temp=self.margin_temp,
            direction_reward=self.direction_reward,
            error_reward=self.error_reward,
            sure_reward=self.sure_reward,
            sure_value_temp=self.sure_value_temp,
            sure_target_min=self.sure_target_min,
            sure_target_max=self.sure_target_max,
            sure_target_blend=self.sure_target_blend,
        )
        sure_strength = teacher["sure_strength"]

        y[:delay_end, 0] = 1.0
        if sure_available:
            y[delay_end:, 3] = sure_strength
            direction_target_strength = 1.0 - sure_strength
        else:
            direction_target_strength = 1.0

        if dir_choice == 0:
            y[delay_end:, 1] = direction_target_strength
            y[delay_end:, 2] = 0.0
        else:
            y[delay_end:, 2] = direction_target_strength
            y[delay_end:, 1] = 0.0

        mask[:, 0] = 1.0
        mask[:fixation_end, 1:] = 1.0
        mask[fixation_end:delay_end, 1:] = 0.0
        if sure_available and self.pre_go_sure_weight > 0.0:
            mask[ts_onset:delay_end, 3] = self.pre_go_sure_weight
            y[ts_onset:delay_end, 3] = sure_strength
        mask[delay_end:, 1:] = 1.0

        trial_meta = {
            "dir_choice": int(dir_choice),
            "coh": float(coh),
            "fixation_end": int(fixation_end),
            "stimulus_end": int(stimulus_end),
            "delay_end": int(delay_end),
            "ts_onset": int(ts_onset),
            "fixation_dur": int(fixation_dur),
            "stimulus_dur": int(stimulus_dur),
            "delay_dur": int(delay_dur),
            # ts_delay is retained as a backwards-compatible per-trial alias.
            "ts_delay": int(timing["ts_latency"]),
            "ts_latency_from_motion_offset": int(timing["ts_latency"]),
            "ts_latency_min": int(self.ts_latency_min),
            "ts_latency_max": int(self.ts_latency_max),
            "stimulus_duration": int(stimulus_dur),
            "delay_duration": int(delay_dur),
            "ts_onset_ms": int(ts_onset * self.dt),
            "motion_offset_ms": int(stimulus_end * self.dt),
            "delay_end_ms": int(delay_end * self.dt),
            "go_cue_ms": int(delay_end * self.dt),
            "fixation_end_ms": int(fixation_end * self.dt),
            "dt_ms": int(self.dt),
            "min_post_ts": int(self.min_post_ts),
            "min_response_dur": int(self.min_response_dur),
            "timing_clipped": bool(timing["timing_clipped"]),
            "sure_available": bool(sure_available),
            "sure_offer_prob": None if self.sure_offer_prob is None else float(self.sure_offer_prob),
            "true_direction": int(dir_choice),
            "signed_coherence": float(coh if dir_choice == 1 else -coh),
            "signed_sensory_evidence": float(signed_sensory_evidence),
            "mean_sensory_evidence": float(mean_sensory_evidence),
            "sensory_margin": float(sensory_margin),
            "integrated_sensory_margin": float(integrated_sensory_margin),
            "mean_sensory_margin": float(mean_sensory_margin),
        }
        trial_meta.update(teacher)
        trial_meta["teacher_sure_choice"] = bool(sure_available and sure_strength >= 0.5)
        self.trial_info.append(trial_meta)
        return x, y, mask


def summarize_stage8(trial_info, choices, conf_values, axes_data):
    sure_flags = (choices == 3).astype(float)
    offered = np.array([info["sure_available"] for info in trial_info], dtype=bool)
    evidence_proj = np.abs(axes_data["evidence_proj"])
    sure_proj = axes_data["sure_proj"]
    direction_success_proxy = np.array([info["direction_success_proxy"] for info in trial_info], dtype=float)
    expected_direction_value = np.array([info["expected_direction_value"] for info in trial_info], dtype=float)
    sure_strength = np.array([info["sure_strength"] for info in trial_info], dtype=float)

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
        "corr_teacher_sure_strength_internal": safe_corrcoef(sure_strength, evidence_proj),
        "corr_teacher_success_proxy_internal": safe_corrcoef(direction_success_proxy, evidence_proj),
        "corr_teacher_expected_value_internal": safe_corrcoef(expected_direction_value, evidence_proj),
    }


if __name__ == "__main__":
    seed = set_global_seed(int(os.environ.get("SURE_SEED", "7")))
    dt = 10
    tau = 100
    T = 2000
    N_batch = 50
    N_rec = 50
    readout_window = 10
    ts_half_window = 5
    n_eval_batches = 8

    run_dir = os.path.dirname(os.path.abspath(__file__))
    figure_dir = os.path.join(run_dir, "figures_stage8_internal_proxy")
    os.makedirs(figure_dir, exist_ok=True)

    rdm_task = RDM_SureTarget_InternalProxy_Task(
        dt=dt,
        tau=tau,
        T=T,
        N_batch=N_batch,
        stimulus_scale=float(os.environ.get("SURE_STIMULUS_SCALE", "1.0")),
        margin_center=2.0,
        margin_temp=0.70,
        direction_reward=1.0,
        error_reward=0.0,
        sure_reward=0.55,
        sure_value_temp=0.15,
        sure_target_min=0.05,
        sure_target_max=0.95,
        sure_target_blend=0.80,
        pre_go_sure_weight=0.0,
    )

    network_params = task_params_from_task(rdm_task)
    network_params["name"] = "RDM_SureTarget_InternalProxy_Stage8"
    network_params["N_rec"] = N_rec
    network_params["rec_noise"] = 0.05
    network_params["activation"] = "rectified_linear"
    default_alpha = float(dt) / float(tau)
    network_params["alpha"] = float(os.environ.get("SURE_ALPHA", str(default_alpha)))
    network_params["training_iters"] = int(os.environ.get("SURE_TRAINING_ITERS", "50000"))
    network_params["loss_epoch"] = int(os.environ.get("SURE_LOSS_EPOCH", "1000"))

    model = Basic(network_params)

    print(
        f"Starting stage8 training... "
        f"(seed={seed}, iters={network_params['training_iters']}, "
        f"loss_epoch={network_params['loss_epoch']}, alpha={network_params['alpha']:.4f})",
        flush=True,
    )
    train_start_time = time.time()
    model.train(rdm_task)
    train_elapsed = time.time() - train_start_time
    print(f"Training complete. elapsed={train_elapsed:.1f}s", flush=True)

    print("Collecting evaluation batches...", flush=True)
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

    conf_median = np.median(conf_values)
    high_conf = conf_values > conf_median
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

    evidence_abs = np.abs(axes_data["evidence_proj"])
    sure_proj = axes_data["sure_proj"]
    y_sure = sure_flags[offered].astype(float)
    teacher_expected_value = np.array([info["expected_direction_value"] for info in trial_info], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = np.where(sure_flags, "tab:green", "tab:red")
    axes[0].scatter(evidence_abs, sure_proj, c=colors, alpha=0.65, s=20)
    axes[0].set_xlabel("Evidence-axis |projection|")
    axes[0].set_ylabel("Sure-axis projection")
    axes[0].set_title("Hidden evidence vs sure tendency")
    axes[0].grid(True, alpha=0.2)

    x_ev = evidence_abs[offered]
    centers, probs, _ = binned_probability_curve(x_ev, y_sure, n_bins=7)
    axes[1].scatter(x_ev, y_sure + (np.random.RandomState(0).rand(y_sure.size) - 0.5) * 0.04,
                    alpha=0.2, s=12, color="tab:blue", label="Trials")
    if centers.size > 0:
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

    x_val = teacher_expected_value[offered]
    centers_v, probs_v, _ = binned_probability_curve(x_val, y_sure, n_bins=7)
    axes[2].scatter(x_val, y_sure + (np.random.RandomState(1).rand(y_sure.size) - 0.5) * 0.04,
                    alpha=0.2, s=12, color="tab:blue", label="Trials")
    if centers_v.size > 0:
        axes[2].plot(centers_v, probs_v, "o-", linewidth=2, color="tab:orange", label="Binned")
    fit_v = fit_logistic_curve(x_val, y_sure)
    if fit_v is not None:
        x_grid_v = np.linspace(np.min(x_val), np.max(x_val), 200)
        axes[2].plot(x_grid_v, evaluate_logistic_curve(fit_v, x_grid_v), linewidth=2.5, color="tab:red", label="Logistic fit")
    axes[2].set_xlabel("Teacher expected direction value")
    axes[2].set_ylabel("P(Sure | offered)")
    axes[2].set_title("P(Sure) vs expected direction value")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2)
    plt.tight_layout()
    save_figure(figure_dir, "st2_targeted_hidden_axes.png")

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
    plt.suptitle("Stage-8 targeted hidden dynamics", y=0.98)
    plt.tight_layout()
    save_figure(figure_dir, "st3_targeted_hidden_dynamics.png")

    coherences = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
    sure_prob_offered, forced_acc_waived, offer_rate = [], [], []
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
    plt.title("Behavioral performance with internal-proxy sure target")
    plt.xlabel("Motion coherence")
    plt.ylabel("Probability")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.2)
    save_figure(figure_dir, "st4_behavior.png")

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

    confidence_analysis = analyze_external_vs_internal_confidence(
        outputs,
        trial_info,
        confidence_fn=confidence_near_ts,
        avg_output_in_window_fn=avg_output_in_window,
        ts_half_window=ts_half_window,
        readout_window=readout_window,
    )
    plot_external_vs_internal_confidence_relationships(
        confidence_analysis,
        figure_dir,
        stem="st6_external_vs_internal_confidence",
    )
    plot_soft_vs_hard_readout_diagnostics(
        confidence_analysis,
        figure_dir,
        stem="st7_soft_vs_hard_readout",
    )
    plot_sure_choice_vs_internal_margin(
        confidence_analysis,
        figure_dir,
        stem="st8_sure_choice_vs_internal_margin",
    )
    plot_pre_onset_sure_baseline(
        confidence_analysis,
        figure_dir,
        stem="st9_pre_onset_sure_baseline",
    )
    direction_dynamics = compute_direction_dynamics(
        outputs,
        trial_info,
        confidence_analysis["internal_margin"],
    )
    plot_direction_dynamics(
        direction_dynamics,
        dt,
        figure_dir,
        stem="st10_direction_dynamics_internal_split",
    )
    matched_internal_dynamics = compute_matched_trial_dynamics(
        outputs,
        trial_info,
        confidence_analysis,
        match_key="internal_margin",
    )
    plot_matched_trial_dynamics(
        matched_internal_dynamics,
        dt,
        figure_dir,
        stem="st11_matched_trial_dynamics",
    )

    summary = summarize_stage8(trial_info, choices, conf_values, axes_data)
    summary["confidence_summary"] = summarize_external_vs_internal_confidence(confidence_analysis)
    summary["soft_hard_readout_summary"] = summarize_soft_vs_hard_readout(confidence_analysis)
    summary["pre_onset_summary"] = summarize_pre_onset_sure_baseline(confidence_analysis)
    summary["matched_internal_pairs"] = int(matched_internal_dynamics["n_pairs"])
    summary_path = os.path.join(run_dir, "stage8_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Stage-8 summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")
