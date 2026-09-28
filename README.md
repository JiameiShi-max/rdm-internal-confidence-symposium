# RDM Internal Confidence Sure-Target RNN

This repository contains the symposium release for a supervised proof-of-concept model of sure-option behavior in a random-dot motion task.

The core idea is that a teacher RNN provides an internal confidence proxy. That proxy is converted into an expected value signal and used to supervise a student RNN's sure-option target. This is not a reinforcement-learning model. Empirical calibration against the teacher's actual directional correctness is a required model-freeze gate; until it passes, the proxy must not be described as internally learned confidence.

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

### Model-freeze retraining

The timed entry point now samples TS latency uniformly over dt-compatible values from 500–750 ms. It preserves the legacy margin-only confidence mapping by default. Run a fresh teacher export, empirical calibration, and student evaluation as follows:

```bash
mkdir -p data/model_freeze results/model_freeze
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python3 scripts/run_timed_teacher.py \
  --seed 7 \
  --output data/model_freeze/teacher_seed7.npz \
  --summary results/model_freeze/teacher_seed7_summary.json
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python3 analysis/rdm_teacher_calibration_report.py \
  --dataset data/model_freeze/teacher_seed7.npz \
  --output-dir results/model_freeze \
  --figure-dir results/model_freeze
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python3 scripts/run_timed_student.py \
  --dataset data/model_freeze/teacher_seed7.npz \
  --seed 7 \
  --summary results/model_freeze/student_seed7_summary.json
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl python3 analysis/rdm_matched_trial_controls.py \
  --trials results/model_freeze/student_seed7_summary_trials.csv \
  --output-dir results/model_freeze \
  --figure-dir results/model_freeze
```

Inspect `results/model_freeze/teacher_calibration_summary.json`. Only if its held-out comparison recommends `empirical_margin_duration`, retrain the teacher with `--confidence-mapping empirical_margin_duration --empirical-confidence-coefficients INTERCEPT MARGIN DURATION_S` using the reported full-fit coefficients. This opt-in step is intentionally not automatic.

After the seed-7 candidate is accepted, repeat teacher and student training for seeds 7, 8, and 9 with seed-specific output paths (do not reuse one teacher dataset). Name student summaries `results/model_freeze/multiseed/timed_seed{seed}_summary.json`, calibration summaries `results/model_freeze/teacher_seed{seed}/teacher_calibration_summary.json`, and matched summaries `results/model_freeze/matched_seed{seed}/matched_trial_controls_summary.json`. Aggregate every seed with:

```bash
PYTHONPATH=src:analysis python3 analysis/rdm_multiseed_validation.py \
  --aggregate-only \
  --seeds 7,8,9 \
  --output-dir results/model_freeze/multiseed \
  --teacher-calibration-template 'results/model_freeze/teacher_seed{seed}/teacher_calibration_summary.json' \
  --matched-template 'results/model_freeze/matched_seed{seed}/matched_trial_controls_summary.json'
PYTHONPATH=src:analysis python3 analysis/model_freeze_report.py \
  --branch feature/model-freeze-validation \
  --baseline-commit 32e5a2a577344b4f3c9170b5512e42a81de50796 \
  --calibration results/model_freeze/teacher_seed7/teacher_calibration_summary.json \
  --student-summary results/model_freeze/multiseed/timed_seed7_summary.json \
  --matched results/model_freeze/matched_seed7/matched_trial_controls_summary.json \
  --multiseed results/model_freeze/multiseed/multiseed_validation_summary.json
```

## Scientific scope

This release supports a symposium-level claim: internal teacher confidence can supervise adaptive sure-option behavior, and population diagnostics show a sure-related axis beyond sensory evidence under controlled timing.

The repository does not claim biological proof, empirical teacher correctness, or a completed RL account.
