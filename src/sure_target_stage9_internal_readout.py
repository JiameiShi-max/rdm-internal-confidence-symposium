import argparse
import hashlib
import json
import os
import random
import time

import numpy as np

from rdm_behavior_analysis import (
    build_trial_eval_records,
    default_behavior_paths,
    plot_behavior_summary,
    summarize_behavior,
    write_behavior_summary_json,
    write_trial_records_csv,
)
from rdm_population_diagnostics import (
    collect_repeated_stimulus_evaluation,
    deterministic_repeated_stimulus_paths,
    plot_population_dynamics,
    plot_pre_ts_leakage,
    plot_sure_output_dynamics,
    plot_repeated_stimulus_dynamics,
    population_dynamics_paths,
    pre_ts_leakage_paths,
    sure_output_dynamics_paths,
    repeated_stimulus_paths,
    softmax_action_outputs,
    summarize_axis_r2_metrics,
    summarize_population_dynamics,
    summarize_pre_ts_leakage,
    summarize_sure_output_dynamics,
    summarize_repeated_stimulus_trials,
    write_json,
    write_repeated_trials_csv,
)
from rdm_task_params import task_params_from_task
from sure_target_stage1_targeted_dr import (
    build_targeted_axes,
    confidence_near_ts,
    get_true_direction_from_metadata,
    readout_fd_trial,
    safe_corrcoef,
)


def load_stage9_dataset(dataset_path):
    data = np.load(dataset_path, allow_pickle=True)
    required = ["x", "y_internal", "mask_internal", "trial_info_json"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Stage9 dataset is missing required fields: {missing}")

    trial_info = [json.loads(str(item)) for item in data["trial_info_json"]]
    dataset = {
        "x": np.asarray(data["x"], dtype=np.float32),
        "y_internal": np.asarray(data["y_internal"], dtype=np.float32),
        "mask_internal": np.asarray(data["mask_internal"], dtype=np.float32),
        "trial_info": trial_info,
    }
    optional_fields = [
        "teacher_internal_margin",
        "teacher_sure_strength",
        "teacher_expected_direction_value",
        "teacher_direction_success_proxy",
        "teacher_sure_advantage",
        "teacher_direction_choice",
        "teacher_correct",
        "external_sensory_margin",
        "external_sure_strength",
        "confidence_mapping",
        "empirical_confidence_coefficients",
        "internal_margin_center_used",
        "internal_margin_temp_used",
        "internal_margin_calibration_mode",
        "internal_margin_center_quantile",
        "internal_margin_temp_low_quantile",
        "internal_margin_temp_high_quantile",
        "internal_margin_n_calibration_trials",
        "internal_readout_anchor",
        "internal_pre_go_offset_steps",
    ]
    for field in optional_fields:
        if field in data.files:
            dataset[field] = np.asarray(data[field])
    return dataset


class RDM_SureTarget_InternalReadoutDatasetTask:
    """
    Dataset-backed Stage9 task.

    It trains on the same stimuli as the exported Stage8 teacher dataset, but
    uses y_internal/mask_internal targets generated from the teacher RNN's
    pre-TS output confidence readout.
    """

    def __init__(
        self,
        dataset_path,
        dt=10,
        tau=100,
        N_batch=50,
        sample_mode="random",
        seed=7,
        allowed_indices=None,
    ):
        from psychrnn.tasks.task import Task

        dataset = load_stage9_dataset(dataset_path)
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.x_data = dataset["x"]
        self.y_data = dataset["y_internal"]
        self.mask_data = dataset["mask_internal"]
        self.source_trial_info = dataset["trial_info"]
        n_trials = int(self.x_data.shape[0])
        if allowed_indices is None:
            allowed_indices = np.arange(n_trials, dtype=np.int64)
        else:
            allowed_indices = np.asarray(allowed_indices, dtype=np.int64)
            if allowed_indices.ndim != 1 or allowed_indices.size == 0:
                raise ValueError("allowed_indices must be a non-empty one-dimensional array")
            if np.any(allowed_indices < 0) or np.any(allowed_indices >= n_trials):
                raise ValueError("allowed_indices contains an out-of-range dataset index")
            if np.unique(allowed_indices).size != allowed_indices.size:
                raise ValueError("allowed_indices must not contain duplicates")
        self.allowed_indices = allowed_indices.copy()
        self.sample_mode = str(sample_mode)
        self.rng = np.random.RandomState(int(seed))
        self.next_index = 0
        self.trial_info = []

        n_steps = int(self.x_data.shape[1])
        n_in = int(self.x_data.shape[2])
        n_out = int(self.y_data.shape[2])
        T = int(n_steps * dt)

        class _DatasetTaskBase(Task):
            def generate_trial_params(inner_self, batch, trial):
                return self.generate_trial_params(batch, trial)

            def trial_function(inner_self, time, params):
                raise NotImplementedError("Stage9 dataset task overrides generate_trial().")

            def generate_trial(inner_self, params):
                return self.generate_trial(params)

        self._task = _DatasetTaskBase(
            N_in=n_in,
            N_out=n_out,
            dt=dt,
            tau=tau,
            T=T,
            N_batch=N_batch,
        )

    def __getattr__(self, name):
        return getattr(self._task, name)

    def get_task_params(self):
        return task_params_from_task(self._task)

    def get_trial_batch(self):
        return self._task.get_trial_batch()

    def reset_trial_info(self):
        self.trial_info = []

    def generate_trial_params(self, batch, trial):
        if self.sample_mode == "sequential":
            allowed_position = int(self.next_index % self.allowed_indices.size)
            dataset_index = int(self.allowed_indices[allowed_position])
            self.next_index += 1
        elif self.sample_mode == "random":
            allowed_position = int(self.rng.randint(0, self.allowed_indices.size))
            dataset_index = int(self.allowed_indices[allowed_position])
        else:
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}")
        return {"dataset_index": dataset_index}

    def generate_trial(self, params):
        dataset_index = int(params["dataset_index"])
        meta = dict(self.source_trial_info[dataset_index])
        meta["dataset_index"] = dataset_index
        for key in [
            "teacher_internal_margin",
            "teacher_sure_strength",
            "teacher_expected_direction_value",
            "teacher_direction_success_proxy",
            "teacher_sure_advantage",
            "teacher_direction_choice",
            "teacher_correct",
            "external_sensory_margin",
            "external_sure_strength",
        ]:
            if key in self.dataset:
                meta[key] = float(self.dataset[key][dataset_index])
        self.trial_info.append(meta)
        return (
            np.array(self.x_data[dataset_index], dtype=np.float32),
            np.array(self.y_data[dataset_index], dtype=np.float32),
            np.array(self.mask_data[dataset_index], dtype=np.float32),
        )


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


STUDENT_ACTION_OBJECTIVES = ("independent_mse", "categorical_soft")


def categorical_soft_postgo_loss_components(predictions, y, output_mask):
    """Return the independently weightable terms in the categorical loss."""
    import tensorflow as tf

    action_targets = y[:, :, 1:4]
    action_target_sum = tf.reduce_sum(input_tensor=action_targets, axis=2)
    post_go = tf.cast(action_target_sum > 1.0 - 1e-6, predictions.dtype)
    post_go_column = post_go[:, :, None]
    action_selector = tf.concat(
        [tf.zeros_like(post_go_column), tf.repeat(post_go_column, 3, axis=2)],
        axis=2,
    )
    fixation_selector = tf.concat(
        [post_go_column, tf.zeros_like(tf.repeat(post_go_column, 3, axis=2))],
        axis=2,
    )
    legacy_squared_error = tf.square(output_mask * (predictions - y))
    retained_legacy_mse = tf.reduce_mean(
        input_tensor=legacy_squared_error * (1.0 - action_selector)
    )
    postgo_fixation_mse = tf.reduce_mean(
        input_tensor=legacy_squared_error * fixation_selector
    )
    action_ce = tf.nn.softmax_cross_entropy_with_logits(
        labels=action_targets,
        logits=predictions[:, :, 1:4],
    )
    categorical_action_loss = tf.reduce_mean(input_tensor=post_go * action_ce) / 4.0
    return (
        retained_legacy_mse,
        postgo_fixation_mse,
        categorical_action_loss,
    )


def categorical_soft_postgo_loss(
    predictions,
    y,
    output_mask,
    lambda_fix_postgo=1.0,
    lambda_action=1.0,
):
    """Replace only post-go action MSE with soft-label categorical CE.

    The post-go region is identified by the dataset invariant that the three
    action targets sum to one. All other legacy masked-MSE terms, including
    fixation throughout the trial and pre-go action supervision, are retained.
    Both terms use PsychRNN's original outer mean over B x T x four outputs.
    """
    retained_legacy_mse, postgo_fixation_mse, categorical_action_loss = (
        categorical_soft_postgo_loss_components(predictions, y, output_mask)
    )
    loss = retained_legacy_mse
    if float(lambda_fix_postgo) != 1.0:
        loss = loss + (float(lambda_fix_postgo) - 1.0) * postgo_fixation_mse
    if float(lambda_action) == 1.0:
        return loss + categorical_action_loss
    return loss + float(lambda_action) * categorical_action_loss


def configure_student_action_objective(
    network_params,
    objective,
    lambda_fix_postgo=1.0,
    lambda_action=1.0,
):
    objective = str(objective)
    if objective not in STUDENT_ACTION_OBJECTIVES:
        raise ValueError(f"Unknown student action objective: {objective}")
    if objective == "categorical_soft":
        loss_name = "categorical_soft_postgo"
        network_params["loss_function"] = loss_name
        network_params[loss_name] = lambda predictions, y, output_mask: categorical_soft_postgo_loss(
            predictions,
            y,
            output_mask,
            lambda_fix_postgo=lambda_fix_postgo,
            lambda_action=lambda_action,
        )
    return network_params


def readout_post_go(outputs_trial, delay_end, window=10, action_only=False):
    start = int(delay_end)
    end = min(outputs_trial.shape[0], start + int(window))
    if end <= start:
        end = min(outputs_trial.shape[0], start + 1)
    avg = np.asarray(outputs_trial[start:end], dtype=float).mean(axis=0)
    choice = int(1 + np.argmax(avg[1:4])) if action_only else int(np.argmax(avg))
    return choice, avg


def build_post_go_output_records(
    outputs,
    targets,
    trial_info,
    objective,
    readout_window=10,
    raw_outputs=None,
):
    """Build the objective-independent four-channel measurement table."""
    objective = str(objective)
    action_only = objective == "categorical_soft"
    records = []
    raw_outputs = outputs if raw_outputs is None else raw_outputs
    for idx, info in enumerate(trial_info):
        choice, predicted = readout_post_go(
            outputs[idx], info["delay_end"], window=readout_window, action_only=action_only
        )
        _, target = readout_post_go(
            targets[idx], info["delay_end"], window=readout_window, action_only=True
        )
        _, raw_predicted = readout_post_go(
            raw_outputs[idx], info["delay_end"], window=readout_window, action_only=True
        )
        true_direction = int(info.get("true_direction", info.get("dir_choice", -1)))
        correct_channel = 1 if int(info.get("dir_choice", true_direction)) == 0 else 2
        incorrect_channel = 2 if correct_channel == 1 else 1
        target_margin = float(target[3] - target[correct_channel])
        predicted_margin = float(predicted[3] - predicted[correct_channel])
        teacher_preferred_action = int(1 + np.argmax(target[1:4]))
        records.append(
            {
                "trial_index": int(idx),
                "dataset_index": int(info.get("dataset_index", idx)),
                "student_action_objective": objective,
                "true_direction": true_direction,
                "dir_choice": int(info.get("dir_choice", true_direction)),
                "sure_available": bool(info.get("sure_available", False)),
                "teacher_sure_strength": float(info.get("teacher_sure_strength", target[3])),
                "target_fixation": float(target[0]),
                "target_left": float(target[1]),
                "target_right": float(target[2]),
                "target_sure": float(target[3]),
                "teacher_preferred_action": teacher_preferred_action,
                "target_action_margin": target_margin,
                "raw_post_go_fixation": float(raw_predicted[0]),
                "raw_post_go_left": float(raw_predicted[1]),
                "raw_post_go_right": float(raw_predicted[2]),
                "raw_post_go_sure": float(raw_predicted[3]),
                "predicted_fixation": float(predicted[0]),
                "predicted_left": float(predicted[1]),
                "predicted_right": float(predicted[2]),
                "predicted_sure": float(predicted[3]),
                "p_left": float(predicted[1]),
                "p_right": float(predicted[2]),
                "p_sure": float(predicted[3]),
                "predicted_correct_direction": float(predicted[correct_channel]),
                "predicted_incorrect_direction": float(predicted[incorrect_channel]),
                "predicted_action_margin": predicted_margin,
                "final_argmax_choice": int(choice),
                "action_argmax_choice": int(1 + np.argmax(predicted[1:4])),
                "stimulus_coherence": float(info.get("coh", np.nan)),
                "stimulus_duration_ms": float(
                    info.get("stimulus_duration", info.get("stimulus_dur", np.nan))
                ),
                "internal_margin": float(info.get("teacher_internal_margin", np.nan)),
                "empirical_p_correct": float(info.get("teacher_direction_success_proxy", np.nan)),
                "delta_value": float(
                    0.55 - float(info.get("teacher_direction_success_proxy", np.nan))
                ),
            }
        )
    return records


def collect_eval_batches(model, task, n_eval_batches=8):
    inputs_list = []
    outputs_list = []
    states_list = []
    targets_list = []
    trial_info = []
    for _ in range(int(n_eval_batches)):
        task.reset_trial_info()
        batch_x, batch_y, _, _ = task.get_trial_batch()
        batch_outputs, batch_states = model.test(batch_x)
        inputs_list.append(batch_x)
        outputs_list.append(batch_outputs)
        states_list.append(batch_states)
        targets_list.append(batch_y)
        trial_info.extend(task.trial_info)
    return (
        np.concatenate(outputs_list, axis=0),
        np.concatenate(states_list, axis=0),
        np.concatenate(targets_list, axis=0),
        np.concatenate(inputs_list, axis=0),
        trial_info,
    )


def summarize_stage9(trial_info, choices, conf_values, axes_data):
    offered = np.array([info["sure_available"] for info in trial_info], dtype=bool)
    sure_flags = (choices == 3).astype(float)
    evidence_proj = np.abs(axes_data["evidence_proj"])
    sure_proj = axes_data["sure_proj"]
    teacher_internal_margin = np.array([info.get("teacher_internal_margin", np.nan) for info in trial_info], dtype=float)
    teacher_sure_strength = np.array([info.get("teacher_sure_strength", np.nan) for info in trial_info], dtype=float)
    external_sensory_margin = np.array([info.get("external_sensory_margin", np.nan) for info in trial_info], dtype=float)
    external_sure_strength = np.array([info.get("external_sure_strength", np.nan) for info in trial_info], dtype=float)

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
        "corr_teacher_internal_margin_evidence_axis": safe_corrcoef(teacher_internal_margin, evidence_proj),
        "corr_teacher_sure_strength_evidence_axis": safe_corrcoef(teacher_sure_strength, evidence_proj),
        "corr_external_sensory_margin_evidence_axis": safe_corrcoef(external_sensory_margin, evidence_proj),
        "corr_external_sure_strength_evidence_axis": safe_corrcoef(external_sure_strength, evidence_proj),
    }


def train_stage9_student(args):
    import tensorflow as tf
    from psychrnn.backend.models.basic import Basic

    tf.compat.v1.reset_default_graph()
    set_global_seed(args.seed)
    task = RDM_SureTarget_InternalReadoutDatasetTask(
        dataset_path=args.dataset,
        dt=args.dt,
        tau=args.tau,
        N_batch=args.n_batch,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )

    network_params = task_params_from_task(task)
    network_params["name"] = "RDM_SureTarget_Stage9_InternalReadoutStudent"
    network_params["N_rec"] = int(args.n_rec)
    network_params["rec_noise"] = 0.05
    network_params["activation"] = "rectified_linear"
    network_params["alpha"] = float(args.alpha if args.alpha is not None else float(args.dt) / float(args.tau))
    configure_student_action_objective(
        network_params,
        args.student_action_objective,
        lambda_fix_postgo=args.lambda_fix_postgo,
        lambda_action=args.lambda_action,
    )

    model = Basic(network_params)
    print(
        f"Training Stage9 internal-readout student... dataset={args.dataset}, "
        f"iters={args.training_iters}, n_batch={args.n_batch}, "
        f"action_objective={args.student_action_objective}",
        flush=True,
    )
    start_time = time.time()
    losses, training_time, initialization_time = model.train(
        task,
        train_params={
            "training_iters": int(args.training_iters),
            "loss_epoch": int(args.loss_epoch),
            "save_weights_path": args.weights,
        },
    )
    elapsed_seconds = time.time() - start_time
    print(f"Stage9 student training complete. elapsed={elapsed_seconds:.1f}s", flush=True)

    task.sample_mode = "sequential"
    task.next_index = 0
    raw_outputs, states, targets, _, trial_info = collect_eval_batches(
        model, task, n_eval_batches=args.n_eval_batches
    )
    action_only = args.student_action_objective == "categorical_soft"
    outputs = softmax_action_outputs(raw_outputs) if action_only else raw_outputs
    choices = []
    conf_values = []
    for idx, info in enumerate(trial_info):
        choice_idx, _ = readout_post_go(
            outputs[idx], info["delay_end"], window=args.readout_window, action_only=action_only
        )
        choices.append(choice_idx)
        conf_values.append(confidence_near_ts(outputs[idx], info, half_window=args.ts_half_window))
    choices = np.asarray(choices, dtype=int)
    conf_values = np.asarray(conf_values, dtype=float)
    sure_flags = choices == 3
    axes_data = build_targeted_axes(states, trial_info, choices, sure_flags)
    summary = summarize_stage9(trial_info, choices, conf_values, axes_data)
    summary["dataset"] = os.path.abspath(args.dataset)
    with open(args.dataset, "rb") as dataset_file:
        summary["dataset_sha256"] = hashlib.sha256(dataset_file.read()).hexdigest()
    summary["student_action_objective"] = args.student_action_objective
    summary["lambda_fix_postgo"] = float(args.lambda_fix_postgo)
    summary["lambda_action"] = float(args.lambda_action)
    summary["post_go_action_readout"] = (
        "argmax(left,right,sure); fixation excluded"
        if action_only
        else "legacy argmax(fixation,left,right,sure)"
    )
    summary["training_elapsed_seconds"] = float(elapsed_seconds)
    summary["psychrnn_training_seconds"] = float(training_time)
    summary["psychrnn_initialization_seconds"] = float(initialization_time)
    summary["training_losses"] = [float(value) for value in losses]
    summary["weights"] = os.path.abspath(args.weights) if args.weights else None

    output_records = build_post_go_output_records(
        outputs,
        targets,
        trial_info,
        objective=args.student_action_objective,
        readout_window=args.readout_window,
        raw_outputs=raw_outputs,
    )
    post_go_outputs_path = args.post_go_outputs or os.path.splitext(args.summary)[0] + "_post_go_outputs.csv"
    write_repeated_trials_csv(post_go_outputs_path, output_records)
    summary["post_go_outputs"] = os.path.abspath(post_go_outputs_path)

    offered = np.array([info["sure_available"] for info in trial_info], dtype=bool)
    axis_r2_metrics = summarize_axis_r2_metrics(
        evidence_axis=np.abs(axes_data["evidence_proj"]),
        sure_axis=axes_data["sure_proj"],
        p_sure=sure_flags.astype(float),
        confidence=conf_values,
        offered=offered,
    )
    summary.update(axis_r2_metrics)

    behavior_paths = default_behavior_paths(args.summary)
    trial_csv = args.trial_csv or behavior_paths["trial_csv"]
    behavior_summary_path = args.behavior_summary or behavior_paths["behavior_summary"]
    behavior_fig = args.behavior_fig or behavior_paths["behavior_fig"]
    trial_records = build_trial_eval_records(
        trial_info,
        choices,
        conf_values=conf_values,
        axes_data=axes_data,
    )
    behavior_summary = summarize_behavior(trial_records)
    write_trial_records_csv(trial_csv, trial_records)
    write_behavior_summary_json(behavior_summary_path, behavior_summary)
    if not args.focused_action_eval:
        plot_behavior_summary(behavior_fig, behavior_summary, title=os.path.basename(args.summary))
    summary["trial_csv"] = os.path.abspath(trial_csv)
    summary["behavior_summary"] = os.path.abspath(behavior_summary_path)
    summary["behavior_fig"] = None if args.focused_action_eval else os.path.abspath(behavior_fig)

    repeated_paths = repeated_stimulus_paths(args.summary)
    repeated_records, repeated_outputs, repeated_states = collect_repeated_stimulus_evaluation(
        model,
        task,
        n_stimuli=args.repeated_n_stimuli,
        n_repeats=args.repeated_n_repeats,
        readout_window=args.readout_window,
        ts_half_window=args.ts_half_window,
        action_only=action_only,
        action_softmax=action_only,
    )
    repeated_summary = summarize_repeated_stimulus_trials(
        repeated_records,
        repeated_outputs,
        repeated_states,
    )
    write_repeated_trials_csv(repeated_paths["trials"], repeated_records)
    write_json(repeated_paths["summary"], repeated_summary)
    if not args.focused_action_eval:
        plot_repeated_stimulus_dynamics(repeated_paths["fig"], repeated_records, repeated_outputs)
    summary["repeated_stimulus_trials"] = os.path.abspath(repeated_paths["trials"])
    summary["repeated_stimulus_summary"] = os.path.abspath(repeated_paths["summary"])
    summary["repeated_stimulus_fig"] = (
        None if args.focused_action_eval else os.path.abspath(repeated_paths["fig"])
    )

    deterministic_paths = deterministic_repeated_stimulus_paths(args.summary)
    deterministic_records, deterministic_outputs, deterministic_states = collect_repeated_stimulus_evaluation(
        model,
        task,
        n_stimuli=args.repeated_n_stimuli,
        n_repeats=args.repeated_n_repeats,
        readout_window=args.readout_window,
        ts_half_window=args.ts_half_window,
        deterministic=True,
        rec_noise=args.deterministic_eval_rec_noise,
        action_only=action_only,
        action_softmax=action_only,
    )
    deterministic_summary = summarize_repeated_stimulus_trials(
        deterministic_records,
        deterministic_outputs,
        deterministic_states,
    )
    write_repeated_trials_csv(deterministic_paths["trials"], deterministic_records)
    write_json(deterministic_paths["summary"], deterministic_summary)
    if not args.focused_action_eval:
        plot_repeated_stimulus_dynamics(
            deterministic_paths["fig"],
            deterministic_records,
            deterministic_outputs,
        )
    summary["deterministic_repeated_stimulus_trials"] = os.path.abspath(deterministic_paths["trials"])
    summary["deterministic_repeated_stimulus_summary"] = os.path.abspath(deterministic_paths["summary"])
    summary["deterministic_repeated_stimulus_fig"] = (
        None if args.focused_action_eval else os.path.abspath(deterministic_paths["fig"])
    )

    if not args.focused_action_eval:
        leakage_paths = pre_ts_leakage_paths(args.summary)
        leakage_summary, leakage_plot_data = summarize_pre_ts_leakage(
            states,
            trial_info,
            choices,
            axes_data,
            pre_window=args.pre_ts_window,
            post_window=args.post_ts_window,
        )
        write_json(leakage_paths["summary"], leakage_summary)
        plot_pre_ts_leakage(leakage_paths["fig"], leakage_plot_data)
        summary["pre_ts_leakage_summary"] = os.path.abspath(leakage_paths["summary"])
        summary["pre_ts_leakage_fig"] = os.path.abspath(leakage_paths["fig"])

        sure_output_paths = sure_output_dynamics_paths(args.summary)
        sure_output_summary, sure_output_rows, sure_output_plot_data = summarize_sure_output_dynamics(
            outputs,
            trial_info,
            choices,
            pre_window=args.pre_ts_window,
            post_window=args.post_ts_window,
        )
        write_json(sure_output_paths["summary"], sure_output_summary)
        write_repeated_trials_csv(sure_output_paths["trials"], sure_output_rows)
        plot_sure_output_dynamics(sure_output_paths["fig"], sure_output_plot_data)
        summary["sure_output_dynamics_summary"] = os.path.abspath(sure_output_paths["summary"])
        summary["sure_output_dynamics_trials"] = os.path.abspath(sure_output_paths["trials"])
        summary["sure_output_dynamics_fig"] = os.path.abspath(sure_output_paths["fig"])

        population_paths = population_dynamics_paths(args.summary)
        population_summary, population_plot_data = summarize_population_dynamics(
            states,
            trial_info,
            choices,
            axes_data,
            align_event=args.population_align_event,
            pre_window=args.population_pre_window,
            post_window=args.population_post_window,
        )
        write_json(population_paths["summary"], population_summary)
        plot_population_dynamics(population_paths["fig"], population_plot_data)
        summary["population_dynamics_summary"] = os.path.abspath(population_paths["summary"])
        summary["population_dynamics_fig"] = os.path.abspath(population_paths["fig"])

    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved Stage9 summary: {args.summary}", flush=True)
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train Stage9 internal-readout supervised sure-target student RNN.")
    parser.add_argument("--dataset", default="stage9_internal_teacher_dataset.npz")
    parser.add_argument("--summary", default="stage9_internal_readout_summary.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--post-go-outputs", default=None)
    parser.add_argument(
        "--student-action-objective",
        choices=STUDENT_ACTION_OBJECTIVES,
        default="independent_mse",
        help="Keep legacy masked MSE or replace only post-go action MSE with soft-label categorical CE.",
    )
    parser.add_argument("--lambda-fix-postgo", type=float, default=1.0)
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--tau", type=int, default=100)
    parser.add_argument("--n-batch", type=int, default=50)
    parser.add_argument("--n-rec", type=int, default=50)
    parser.add_argument("--training-iters", type=int, default=50000)
    parser.add_argument("--loss-epoch", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--sample-mode", choices=["random", "sequential"], default="random")
    parser.add_argument("--n-eval-batches", type=int, default=8)
    parser.add_argument("--readout-window", type=int, default=10)
    parser.add_argument("--ts-half-window", type=int, default=5)
    parser.add_argument("--trial-csv", default=None)
    parser.add_argument("--behavior-summary", default=None)
    parser.add_argument("--behavior-fig", default=None)
    parser.add_argument(
        "--focused-action-eval",
        action="store_true",
        help="Write action/behavior/repeat tables without unrelated population figures.",
    )
    parser.add_argument("--repeated-n-stimuli", type=int, default=24)
    parser.add_argument("--repeated-n-repeats", type=int, default=5)
    parser.add_argument("--deterministic-eval-rec-noise", type=float, default=0.0)
    parser.add_argument("--pre-ts-window", type=int, default=40)
    parser.add_argument("--post-ts-window", type=int, default=20)
    parser.add_argument("--population-align-event", choices=["ts_onset", "delay_end"], default="ts_onset")
    parser.add_argument("--population-pre-window", type=int, default=60)
    parser.add_argument("--population-post-window", type=int, default=40)
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_stage9_student(args)


if __name__ == "__main__":
    main()
