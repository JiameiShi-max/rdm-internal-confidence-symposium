import json
import os
import random
import time

import numpy as np

from internal_confidence_proxy import compute_soft_sure_teacher_from_margin
from rdm_task_params import task_params_from_task


def average_output_window(outputs_trial, center_step, half_window=5):
    start = max(0, int(center_step) - int(half_window))
    end = min(outputs_trial.shape[0], int(center_step) + int(half_window))
    if end <= start:
        end = min(outputs_trial.shape[0], start + 1)
    return outputs_trial[start:end, :].mean(axis=0)


def compute_output_internal_margin(outputs_trial, center_step, half_window=5):
    avg_output = average_output_window(outputs_trial, center_step, half_window=half_window)
    return float(abs(float(avg_output[1]) - float(avg_output[2])))


def get_internal_readout_step(trial_meta, readout_anchor="ts_onset", pre_go_offset_steps=5):
    readout_anchor = str(readout_anchor)
    if readout_anchor == "ts_onset":
        if bool(trial_meta.get("sure_available", False)):
            return int(trial_meta["ts_onset"])
        return int(trial_meta["delay_end"])
    if readout_anchor == "pre_go":
        return max(0, int(trial_meta["delay_end"]) - int(pre_go_offset_steps))
    if readout_anchor == "go":
        return int(trial_meta["delay_end"])
    if readout_anchor == "stimulus_end":
        return int(trial_meta["stimulus_end"])
    if readout_anchor == "mid_delay":
        return int((int(trial_meta["stimulus_end"]) + int(trial_meta["delay_end"])) / 2)
    raise ValueError("readout_anchor must be ts_onset, pre_go, go, stimulus_end, or mid_delay")


def estimate_internal_margin_calibration(
    internal_margins,
    sure_available,
    center_quantile=50.0,
    temp_low_quantile=25.0,
    temp_high_quantile=75.0,
    min_temp=1e-3,
):
    internal_margins = np.asarray(internal_margins, dtype=float)
    sure_available = np.asarray(sure_available, dtype=bool)
    usable = sure_available & np.isfinite(internal_margins)
    if np.any(usable):
        calibration_margins = internal_margins[usable]
    else:
        calibration_margins = internal_margins[np.isfinite(internal_margins)]

    if calibration_margins.size == 0:
        margin_center = 0.0
        margin_temp = float(min_temp)
    else:
        margin_center = float(np.percentile(calibration_margins, float(center_quantile)))
        low = float(np.percentile(calibration_margins, float(temp_low_quantile)))
        high = float(np.percentile(calibration_margins, float(temp_high_quantile)))
        margin_temp = max((high - low) / 2.0, float(min_temp))

    return {
        "margin_center": float(margin_center),
        "margin_temp": float(margin_temp),
        "center_quantile": float(center_quantile),
        "temp_low_quantile": float(temp_low_quantile),
        "temp_high_quantile": float(temp_high_quantile),
        "calibration_mode": "auto_quantile",
        "n_calibration_trials": int(calibration_margins.size),
    }


def build_internal_teacher_records(
    outputs,
    trial_info,
    half_window=5,
    margin_center=None,
    margin_temp=None,
    internal_calibration="auto",
    center_quantile=50.0,
    temp_low_quantile=25.0,
    temp_high_quantile=75.0,
    min_margin_temp=1e-3,
    direction_reward=1.0,
    error_reward=0.0,
    sure_reward=0.55,
    sure_value_temp=0.15,
    sure_target_min=0.05,
    sure_target_max=0.95,
    sure_target_blend=0.80,
    readout_anchor="ts_onset",
    pre_go_offset_steps=5,
):
    n_trials = min(int(outputs.shape[0]), len(trial_info))
    teacher_internal_margin = np.zeros(n_trials, dtype=float)
    teacher_sure_strength = np.zeros(n_trials, dtype=float)
    teacher_expected_direction_value = np.zeros(n_trials, dtype=float)
    teacher_direction_success_proxy = np.zeros(n_trials, dtype=float)
    external_sensory_margin = np.zeros(n_trials, dtype=float)
    external_sure_strength = np.zeros(n_trials, dtype=float)
    margin_source = np.full(n_trials, "internal_output", dtype=object)
    sure_available = np.asarray(
        [bool(trial_info[idx].get("sure_available", False)) for idx in range(n_trials)],
        dtype=bool,
    )

    for idx in range(n_trials):
        info = trial_info[idx]
        readout_step = get_internal_readout_step(
            info,
            readout_anchor=readout_anchor,
            pre_go_offset_steps=pre_go_offset_steps,
        )
        teacher_internal_margin[idx] = compute_output_internal_margin(
            outputs[idx],
            center_step=readout_step,
            half_window=half_window,
        )

    if str(internal_calibration) == "auto":
        calibration = estimate_internal_margin_calibration(
            teacher_internal_margin,
            sure_available,
            center_quantile=center_quantile,
            temp_low_quantile=temp_low_quantile,
            temp_high_quantile=temp_high_quantile,
            min_temp=min_margin_temp,
        )
        used_margin_center = calibration["margin_center"] if margin_center is None else float(margin_center)
        used_margin_temp = calibration["margin_temp"] if margin_temp is None else float(margin_temp)
        calibration_mode = calibration["calibration_mode"]
    elif str(internal_calibration) == "manual":
        used_margin_center = 2.0 if margin_center is None else float(margin_center)
        used_margin_temp = 0.70 if margin_temp is None else float(margin_temp)
        calibration = {
            "margin_center": used_margin_center,
            "margin_temp": used_margin_temp,
            "center_quantile": np.nan,
            "temp_low_quantile": np.nan,
            "temp_high_quantile": np.nan,
            "calibration_mode": "manual",
            "n_calibration_trials": int(np.sum(sure_available)),
        }
        calibration_mode = "manual"
    else:
        raise ValueError("internal_calibration must be 'auto' or 'manual'")

    used_margin_temp = max(float(used_margin_temp), float(min_margin_temp))

    for idx in range(n_trials):
        info = trial_info[idx]
        teacher = compute_soft_sure_teacher_from_margin(
            margin=teacher_internal_margin[idx],
            sure_available=bool(info["sure_available"]),
            margin_center=used_margin_center,
            margin_temp=used_margin_temp,
            direction_reward=direction_reward,
            error_reward=error_reward,
            sure_reward=sure_reward,
            sure_value_temp=sure_value_temp,
            sure_target_min=sure_target_min,
            sure_target_max=sure_target_max,
            sure_target_blend=sure_target_blend,
            margin_source="internal_output",
        )

        teacher_sure_strength[idx] = float(teacher["sure_strength"])
        teacher_expected_direction_value[idx] = float(teacher["expected_direction_value"])
        teacher_direction_success_proxy[idx] = float(teacher["direction_success_proxy"])
        external_sensory_margin[idx] = float(info.get("sensory_margin", np.nan))
        external_sure_strength[idx] = float(info.get("sure_strength", np.nan))

    return {
        "teacher_internal_margin": teacher_internal_margin,
        "teacher_sure_strength": teacher_sure_strength,
        "teacher_expected_direction_value": teacher_expected_direction_value,
        "teacher_direction_success_proxy": teacher_direction_success_proxy,
        "external_sensory_margin": external_sensory_margin,
        "external_sure_strength": external_sure_strength,
        "margin_source": margin_source,
        "internal_margin_center_used": np.asarray(float(used_margin_center), dtype=float),
        "internal_margin_temp_used": np.asarray(float(used_margin_temp), dtype=float),
        "internal_margin_calibration_mode": str(calibration_mode),
        "internal_margin_center_quantile": np.asarray(float(calibration["center_quantile"]), dtype=float),
        "internal_margin_temp_low_quantile": np.asarray(float(calibration["temp_low_quantile"]), dtype=float),
        "internal_margin_temp_high_quantile": np.asarray(float(calibration["temp_high_quantile"]), dtype=float),
        "internal_margin_n_calibration_trials": np.asarray(
            int(calibration["n_calibration_trials"]), dtype=int
        ),
        "internal_readout_anchor": str(readout_anchor),
        "internal_pre_go_offset_steps": np.asarray(int(pre_go_offset_steps), dtype=int),
    }


def collect_internal_teacher_dataset(model, task, n_batches=8, half_window=5, **teacher_kwargs):
    x_list = []
    y_list = []
    mask_list = []
    outputs_list = []
    states_list = []
    trial_info = []

    for _ in range(int(n_batches)):
        task.reset_trial_info()
        batch_x, batch_y, batch_mask, _ = task.get_trial_batch()
        batch_info = list(task.trial_info)
        batch_outputs, batch_states = model.test(batch_x)

        x_list.append(batch_x)
        y_list.append(batch_y)
        mask_list.append(batch_mask)
        outputs_list.append(batch_outputs)
        states_list.append(batch_states)
        trial_info.extend(batch_info)

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0)
    states = np.concatenate(states_list, axis=0)
    teacher_records = build_internal_teacher_records(
        outputs,
        trial_info,
        half_window=half_window,
        **teacher_kwargs,
    )
    y_internal, mask_internal = build_stage9_targets_and_masks(
        trial_info,
        teacher_records["teacher_sure_strength"],
        n_steps=y.shape[1],
        n_out=y.shape[2],
    )

    dataset = {
        "x": x,
        "y": y,
        "mask": mask,
        "y_internal": y_internal,
        "mask_internal": mask_internal,
        "outputs": outputs,
        "states": states,
        "trial_info": trial_info,
        "teacher_records": teacher_records,
    }
    dataset.update(teacher_records)
    return dataset


def build_stage9_targets_and_masks(
    trial_info,
    teacher_sure_strength,
    n_steps,
    n_out=4,
    pre_go_sure_weight=0.0,
):
    n_trials = len(trial_info)
    y = np.zeros((n_trials, int(n_steps), int(n_out)), dtype=np.float32)
    mask = np.zeros((n_trials, int(n_steps), int(n_out)), dtype=np.float32)
    teacher_sure_strength = np.asarray(teacher_sure_strength, dtype=float)

    for idx, info in enumerate(trial_info):
        fixation_end = int(info["fixation_end"])
        delay_end = int(info["delay_end"])
        ts_onset = int(info.get("ts_onset", delay_end))
        sure_available = bool(info.get("sure_available", False))
        sure_strength = float(teacher_sure_strength[idx]) if sure_available else 0.0
        direction_target_strength = 1.0 - sure_strength if sure_available else 1.0

        y[idx, :delay_end, 0] = 1.0
        if int(info["dir_choice"]) == 0:
            y[idx, delay_end:, 1] = direction_target_strength
            y[idx, delay_end:, 2] = 0.0
        else:
            y[idx, delay_end:, 1] = 0.0
            y[idx, delay_end:, 2] = direction_target_strength
        y[idx, delay_end:, 3] = sure_strength

        mask[idx, :, 0] = 1.0
        mask[idx, :fixation_end, 1:] = 1.0
        mask[idx, fixation_end:delay_end, 1:] = 0.0
        if sure_available and float(pre_go_sure_weight) > 0.0:
            y[idx, ts_onset:delay_end, 3] = sure_strength
            mask[idx, ts_onset:delay_end, 3] = float(pre_go_sure_weight)
        mask[idx, delay_end:, 1:] = 1.0

    return y, mask


def serialize_trial_info(trial_info):
    return np.asarray([json.dumps(info, sort_keys=True) for info in trial_info], dtype=object)


def save_internal_teacher_dataset(
    path,
    x,
    y,
    mask,
    trial_info,
    teacher_records,
    y_internal=None,
    mask_internal=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    extra_arrays = {}
    if y_internal is not None:
        extra_arrays["y_internal"] = np.asarray(y_internal)
    if mask_internal is not None:
        extra_arrays["mask_internal"] = np.asarray(mask_internal)
    np.savez_compressed(
        path,
        x=np.asarray(x),
        y=np.asarray(y),
        mask=np.asarray(mask),
        trial_info_json=serialize_trial_info(trial_info),
        **extra_arrays,
        **teacher_records,
    )
    return path


def mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def summarize_internal_teacher_dataset(dataset):
    trial_info = dataset["trial_info"]
    teacher_records = dataset["teacher_records"]
    sure_available = np.asarray([bool(info["sure_available"]) for info in trial_info], dtype=bool)
    post_go_sure_targets = []
    for idx, info in enumerate(trial_info):
        if bool(info["sure_available"]):
            delay_end = int(info["delay_end"])
            post_go_sure_targets.append(dataset["y_internal"][idx, delay_end:, 3])
    if len(post_go_sure_targets) > 0:
        post_go_sure_targets = np.concatenate(post_go_sure_targets, axis=0)
    else:
        post_go_sure_targets = np.asarray([], dtype=float)
    return {
        "n_trials": int(len(trial_info)),
        "n_sure_available": int(np.sum(sure_available)),
        "mean_teacher_internal_margin": mean_or_nan(teacher_records["teacher_internal_margin"]),
        "mean_teacher_sure_strength": mean_or_nan(teacher_records["teacher_sure_strength"]),
        "mean_external_sensory_margin": mean_or_nan(teacher_records["external_sensory_margin"]),
        "mean_external_sure_strength": mean_or_nan(teacher_records["external_sure_strength"]),
        "mean_stage9_post_go_sure_target": mean_or_nan(post_go_sure_targets),
        "internal_margin_center_used": float(teacher_records["internal_margin_center_used"]),
        "internal_margin_temp_used": float(teacher_records["internal_margin_temp_used"]),
        "internal_margin_calibration_mode": str(teacher_records["internal_margin_calibration_mode"]),
        "internal_margin_n_calibration_trials": int(
            teacher_records["internal_margin_n_calibration_trials"]
        ),
        "internal_readout_anchor": str(teacher_records["internal_readout_anchor"]),
        "internal_pre_go_offset_steps": int(teacher_records["internal_pre_go_offset_steps"]),
    }


def write_summary_json(path, summary):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def set_global_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)
    except Exception:
        pass
    return seed


def build_stage8_teacher_task(
    dt=10,
    tau=100,
    T=2000,
    n_batch=50,
    stimulus_scale=1.0,
    stimulus_dur_min=800,
    stimulus_dur_max=800,
    delay_min=400,
    delay_max=1000,
    ts_delay=500,
    min_post_ts=0,
    min_response_dur=0,
    sure_offer_prob=None,
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
):
    from sure_target import RDM_SureTarget_InternalProxy_Task

    return RDM_SureTarget_InternalProxy_Task(
        dt=dt,
        tau=tau,
        T=T,
        N_batch=n_batch,
        stimulus_scale=stimulus_scale,
        stimulus_dur_min=stimulus_dur_min,
        stimulus_dur_max=stimulus_dur_max,
        delay_min=delay_min,
        delay_max=delay_max,
        ts_delay=ts_delay,
        min_post_ts=min_post_ts,
        min_response_dur=min_response_dur,
        sure_offer_prob=sure_offer_prob,
        margin_center=margin_center,
        margin_temp=margin_temp,
        direction_reward=direction_reward,
        error_reward=error_reward,
        sure_reward=sure_reward,
        sure_value_temp=sure_value_temp,
        sure_target_min=sure_target_min,
        sure_target_max=sure_target_max,
        sure_target_blend=sure_target_blend,
        pre_go_sure_weight=pre_go_sure_weight,
    )


def build_stage8_teacher_model(task, n_rec=50, training_iters=50000, loss_epoch=1000, alpha=None):
    from psychrnn.backend.models.basic import Basic

    network_params = task_params_from_task(task)
    network_params["name"] = "RDM_SureTarget_InternalProxy_Stage9TeacherExport"
    network_params["N_rec"] = int(n_rec)
    network_params["rec_noise"] = 0.05
    network_params["activation"] = "rectified_linear"
    default_alpha = float(task.dt) / float(task.tau)
    network_params["alpha"] = float(default_alpha if alpha is None else alpha)
    network_params["training_iters"] = int(training_iters)
    network_params["loss_epoch"] = int(loss_epoch)
    return Basic(network_params)


def export_stage9_internal_teacher_dataset(args):
    set_global_seed(args.seed)
    task = build_stage8_teacher_task(
        dt=args.dt,
        tau=args.tau,
        T=args.T,
        n_batch=args.n_batch,
        stimulus_scale=args.stimulus_scale,
        stimulus_dur_min=args.stimulus_dur_min,
        stimulus_dur_max=args.stimulus_dur_max,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        ts_delay=args.ts_delay,
        min_post_ts=args.min_post_ts,
        min_response_dur=args.min_response_dur,
        sure_offer_prob=args.sure_offer_prob,
        margin_center=args.margin_center,
        margin_temp=args.margin_temp,
        direction_reward=args.direction_reward,
        error_reward=args.error_reward,
        sure_reward=args.sure_reward,
        sure_value_temp=args.sure_value_temp,
        sure_target_min=args.sure_target_min,
        sure_target_max=args.sure_target_max,
        sure_target_blend=args.sure_target_blend,
        pre_go_sure_weight=args.pre_go_sure_weight,
    )
    model = build_stage8_teacher_model(
        task,
        n_rec=args.n_rec,
        training_iters=args.training_iters,
        loss_epoch=args.loss_epoch,
        alpha=args.alpha,
    )

    print(
        f"Training Stage8 teacher for Stage9 export... "
        f"seed={args.seed}, iters={args.training_iters}, n_batch={args.n_batch}",
        flush=True,
    )
    start_time = time.time()
    model.train(
        task,
        train_params={
            "training_iters": int(args.training_iters),
            "loss_epoch": int(args.loss_epoch),
        },
    )
    print(f"Teacher training complete. elapsed={time.time() - start_time:.1f}s", flush=True)

    dataset = collect_internal_teacher_dataset(
        model=model,
        task=task,
        n_batches=args.n_batches,
        half_window=args.half_window,
        internal_calibration=args.internal_calibration,
        margin_center=args.internal_margin_center,
        margin_temp=args.internal_margin_temp,
        center_quantile=args.internal_center_quantile,
        temp_low_quantile=args.internal_temp_low_quantile,
        temp_high_quantile=args.internal_temp_high_quantile,
        min_margin_temp=args.internal_min_margin_temp,
        readout_anchor=args.internal_readout_anchor,
        pre_go_offset_steps=args.internal_pre_go_offset_steps,
        direction_reward=args.direction_reward,
        error_reward=args.error_reward,
        sure_reward=args.sure_reward,
        sure_value_temp=args.sure_value_temp,
        sure_target_min=args.sure_target_min,
        sure_target_max=args.sure_target_max,
        sure_target_blend=args.sure_target_blend,
    )
    save_internal_teacher_dataset(
        args.output,
        dataset["x"],
        dataset["y"],
        dataset["mask"],
        dataset["trial_info"],
        dataset["teacher_records"],
        y_internal=dataset["y_internal"],
        mask_internal=dataset["mask_internal"],
    )
    summary = summarize_internal_teacher_dataset(dataset)
    summary["output"] = os.path.abspath(args.output)
    summary["stimulus_dur_min"] = int(args.stimulus_dur_min)
    summary["stimulus_dur_max"] = int(args.stimulus_dur_max)
    summary["delay_min"] = int(args.delay_min)
    summary["delay_max"] = int(args.delay_max)
    summary["ts_delay"] = int(args.ts_delay)
    summary["min_post_ts"] = int(args.min_post_ts)
    summary["min_response_dur"] = int(args.min_response_dur)
    summary["sure_offer_prob"] = None if args.sure_offer_prob is None else float(args.sure_offer_prob)
    summary["internal_readout_anchor"] = str(args.internal_readout_anchor)
    summary["internal_pre_go_offset_steps"] = int(args.internal_pre_go_offset_steps)
    write_summary_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved Stage9 internal teacher dataset: {args.output}", flush=True)
    print(f"Saved summary: {args.summary}", flush=True)
    return dataset, summary


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Stage8 teacher and export a Stage9 internal-readout teacher dataset."
    )
    parser.add_argument("--output", default="stage9_internal_teacher_dataset.npz")
    parser.add_argument("--summary", default="stage9_internal_teacher_dataset_summary.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--tau", type=int, default=100)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--n-batch", type=int, default=50)
    parser.add_argument("--n-rec", type=int, default=50)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--training-iters", type=int, default=int(os.environ.get("STAGE9_TEACHER_ITERS", "50000")))
    parser.add_argument("--loss-epoch", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--stimulus-scale", type=float, default=1.0)
    parser.add_argument("--stimulus-dur-min", type=int, default=800)
    parser.add_argument("--stimulus-dur-max", type=int, default=800)
    parser.add_argument("--delay-min", type=int, default=400)
    parser.add_argument("--delay-max", type=int, default=1000)
    parser.add_argument("--ts-delay", type=int, default=500)
    parser.add_argument("--min-post-ts", type=int, default=0)
    parser.add_argument("--min-response-dur", type=int, default=0)
    parser.add_argument("--sure-offer-prob", type=float, default=None)
    parser.add_argument("--margin-center", type=float, default=2.0)
    parser.add_argument("--margin-temp", type=float, default=0.70)
    parser.add_argument("--internal-calibration", choices=["auto", "manual"], default="auto")
    parser.add_argument("--internal-margin-center", type=float, default=None)
    parser.add_argument("--internal-margin-temp", type=float, default=None)
    parser.add_argument("--internal-center-quantile", type=float, default=50.0)
    parser.add_argument("--internal-temp-low-quantile", type=float, default=25.0)
    parser.add_argument("--internal-temp-high-quantile", type=float, default=75.0)
    parser.add_argument("--internal-min-margin-temp", type=float, default=1e-3)
    parser.add_argument(
        "--internal-readout-anchor",
        choices=["ts_onset", "pre_go", "go", "stimulus_end", "mid_delay"],
        default="ts_onset",
    )
    parser.add_argument("--internal-pre-go-offset-steps", type=int, default=5)
    parser.add_argument("--direction-reward", type=float, default=1.0)
    parser.add_argument("--error-reward", type=float, default=0.0)
    parser.add_argument("--sure-reward", type=float, default=0.55)
    parser.add_argument("--sure-value-temp", type=float, default=0.15)
    parser.add_argument("--sure-target-min", type=float, default=0.05)
    parser.add_argument("--sure-target-max", type=float, default=0.95)
    parser.add_argument("--sure-target-blend", type=float, default=0.80)
    parser.add_argument("--pre-go-sure-weight", type=float, default=0.0)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    export_stage9_internal_teacher_dataset(args)


if __name__ == "__main__":
    main()
