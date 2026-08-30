# RDM Internal Confidence Sure-Target RNN

This repository contains the symposium release for a supervised proof-of-concept model of sure-option behavior in a random-dot motion task.

The core idea is that a teacher RNN provides an internal confidence proxy. That proxy is calibrated into an expected value signal and used to supervise a student RNN's sure-option target. This is not a reinforcement-learning model; it tests whether internally generated confidence can support adaptive sure choices under controlled task timing.

## Repository layout

- `src/`: model and task implementation. Some internal files keep legacy `stage9` names because they are part of the tested implementation path.
- `scripts/`: clean entry points for training the original and controlled timed teacher/student models.
- `analysis/`: behavior, population diagnostics, matched controls, teacher calibration, multiseed, and parameter-validation utilities.
- `data/`: compact teacher datasets used for reproduction.
- `results/reports/`: JSON/Markdown summaries.
- `results/tables/`: CSV/JSON exports for symposium figures.
- `results/figures/`: current symposium figure suite.
- `docs/`: content map and legacy-name mapping.
- `tests/`: lightweight tests for analysis and release-critical helpers.

## Main model

The controlled timed model is the main model because it uses variable stimulus duration, an explicit delay period, explicit 50% sure-offer trials, and avoids timing clipping. Use:

`data/timed_internal_teacher_dataset_fixed_50k.npz`

Do not use the old broken timed dataset name:

`stage9b_timed_teacher_dataset_50k_auto.npz`

## Quick analysis checks

```bash
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python analysis/rdm_symposium_outputs.py
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python analysis/rdm_teacher_calibration_report.py \
  --dataset data/timed_internal_teacher_dataset_fixed_50k.npz
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python analysis/rdm_matched_trial_controls.py \
  --trials results/reports/timed_student_summary_trials.csv
```

## Training entry points

These commands require a PsychRNN/TensorFlow 1 compatible environment.

```bash
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python scripts/run_timed_teacher.py
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python scripts/run_timed_student.py
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python scripts/run_original_teacher.py
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python scripts/run_original_student.py
```

## Scientific scope

This release supports a symposium-level claim: internal teacher confidence can supervise adaptive sure-option behavior, and population diagnostics show a sure-related axis beyond sensory evidence under controlled timing.

The repository does not claim biological proof, empirical teacher correctness, or a completed RL account.
