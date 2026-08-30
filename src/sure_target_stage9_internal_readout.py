import argparse
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
    plot_repeated_stimulus_dynamics,
    population_dynamics_paths,
    pre_ts_leakage_paths,
    repeated_stimulus_paths,
    summarize_axis_r2_metrics,
    summarize_population_dynamics,
    summarize_pre_ts_leakage,
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
        "external_sensory_margin",
        "external_sure_strength",
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

    def __init__(self, dataset_path, dt=10, tau=100, N_batch=50, sample_mode="random", seed=7):
        from psychrnn.tasks.task import Task

        dataset = load_stage9_dataset(dataset_path)
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.x_data = dataset["x"]
        self.y_data = dataset["y_internal"]
        self.mask_data = dataset["mask_internal"]
        self.source_trial_info = dataset["trial_info"]
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
            dataset_index = int(self.next_index % self.x_data.shape[0])
            self.next_index += 1
        elif self.sample_mode == "random":
            dataset_index = int(self.rng.randint(0, self.x_data.shape[0]))
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


def collect_eval_batches(model, task, n_eval_batches=8):
    outputs_list = []
    states_list = []
    trial_info = []
    for _ in range(int(n_eval_batches)):
        task.reset_trial_info()
        batch_x, _, _, _ = task.get_trial_batch()
        batch_outputs, batch_states = model.test(batch_x)
        outputs_list.append(batch_outputs)
        states_list.append(batch_states)
        trial_info.extend(task.trial_info)
    return np.concatenate(outputs_list, axis=0), np.concatenate(states_list, axis=0), trial_info


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

    model = Basic(network_params)
    print(
        f"Training Stage9 internal-readout student... dataset={args.dataset}, "
        f"iters={args.training_iters}, n_batch={args.n_batch}",
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
    print(f"Stage9 student training complete. elapsed={time.time() - start_time:.1f}s", flush=True)

    task.sample_mode = "sequential"
    task.next_index = 0
    outputs, states, trial_info = collect_eval_batches(model, task, n_eval_batches=args.n_eval_batches)
    choices = []
    conf_values = []
    for idx, info in enumerate(trial_info):
        choice_idx, _ = readout_fd_trial(outputs[idx], info["delay_end"], window=args.readout_window)
        choices.append(choice_idx)
        conf_values.append(confidence_near_ts(outputs[idx], info, half_window=args.ts_half_window))
    choices = np.asarray(choices, dtype=int)
    conf_values = np.asarray(conf_values, dtype=float)
    sure_flags = choices == 3
    axes_data = build_targeted_axes(states, trial_info, choices, sure_flags)
    summary = summarize_stage9(trial_info, choices, conf_values, axes_data)
    summary["dataset"] = os.path.abspath(args.dataset)

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
    plot_behavior_summary(behavior_fig, behavior_summary, title=os.path.basename(args.summary))
    summary["trial_csv"] = os.path.abspath(trial_csv)
    summary["behavior_summary"] = os.path.abspath(behavior_summary_path)
    summary["behavior_fig"] = os.path.abspath(behavior_fig)

    repeated_paths = repeated_stimulus_paths(args.summary)
    repeated_records, repeated_outputs, repeated_states = collect_repeated_stimulus_evaluation(
        model,
        task,
        n_stimuli=args.repeated_n_stimuli,
        n_repeats=args.repeated_n_repeats,
        readout_window=args.readout_window,
        ts_half_window=args.ts_half_window,
    )
    repeated_summary = summarize_repeated_stimulus_trials(
        repeated_records,
        repeated_outputs,
        repeated_states,
    )
    write_repeated_trials_csv(repeated_paths["trials"], repeated_records)
    write_json(repeated_paths["summary"], repeated_summary)
    plot_repeated_stimulus_dynamics(repeated_paths["fig"], repeated_records, repeated_outputs)
    summary["repeated_stimulus_trials"] = os.path.abspath(repeated_paths["trials"])
    summary["repeated_stimulus_summary"] = os.path.abspath(repeated_paths["summary"])
    summary["repeated_stimulus_fig"] = os.path.abspath(repeated_paths["fig"])

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
    )
    deterministic_summary = summarize_repeated_stimulus_trials(
        deterministic_records,
        deterministic_outputs,
        deterministic_states,
    )
    write_repeated_trials_csv(deterministic_paths["trials"], deterministic_records)
    write_json(deterministic_paths["summary"], deterministic_summary)
    plot_repeated_stimulus_dynamics(
        deterministic_paths["fig"],
        deterministic_records,
        deterministic_outputs,
    )
    summary["deterministic_repeated_stimulus_trials"] = os.path.abspath(deterministic_paths["trials"])
    summary["deterministic_repeated_stimulus_summary"] = os.path.abspath(deterministic_paths["summary"])
    summary["deterministic_repeated_stimulus_fig"] = os.path.abspath(deterministic_paths["fig"])

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
