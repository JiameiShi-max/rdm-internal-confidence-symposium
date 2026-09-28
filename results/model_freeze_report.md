# Model freeze report

## 1. Exact active pipeline

Supervised timed teacher → exported internal-readout targets → supervised student. The sure target is optional and independently offered; this is not reinforcement learning.

## 2. Git provenance

- Branch: `feature/model-freeze-validation`
- Baseline main commit: `32e5a2a577344b4f3c9170b5512e42a81de50796`
- Seed: `7`
- Runtime packages: `{'python': '3.12.1', 'tensorflow': '2.16.2', 'psychrnn': '1.0.0', 'numpy': '1.26.4', 'scipy': '1.17.1', 'scikit-learn': '1.9.1', 'matplotlib': '3.11.2', 'pytest': '9.1.1'}`

## 3. Final task timing parameters

- `dt = 10 ms`
- stimulus duration: `400–1200 ms` (current timed configuration)
- delay duration: `800–1200 ms`
- TS latency from motion offset: uniformly sampled over dt-compatible values in `500–750 ms`
- minimum post-TS interval: `100 ms`
- minimum post-go response window: `300 ms`
- sure-offer probability: `0.5`, sampled independently after timing generation

## 4. Teacher confidence formula

Selected mapping: `empirical_margin_duration`. The selection run improved held-out log loss by `0.00260594` against a `0.002` threshold. The regenerated targets used coefficients `[0.6477971635444456, 4.913427541959271, -1.2055566867668548]` for intercept, margin, and duration in seconds.

## 5. Empirical P(correct) calibration

- Empirical correctness available: `True`
- Teacher direction accuracy: `0.89` overall; `0.494565` at zero coherence.
- Brier score: `0.0783498`
- Log loss: `0.250233`
- AUC: `0.876997`
- Calibration intercept: `-0.203314` ± `0.182024` SE
- Calibration slope: `1.06082` ± `0.0935713` SE
- Calibrated by the intercept=0/slope=1 95% Wald check: `True`

The target for this section is the correctness of the teacher's left/right preference, including offered trials; proxy-on-proxy comparisons do not satisfy it.

## 6. Does duration add information beyond margin?

- Tested: `True`
- Model A (margin) held-out log loss / Brier / AUC: `0.258583` / `0.0802968` / `0.861798`
- Model B (+ duration) held-out log loss / Brier / AUC: `0.251423` / `0.079435` / `0.876956`
- Model C (+ interaction) held-out log loss / Brier / AUC: `0.252242` / `0.0794347` / `0.87286`
- Duration log-loss improvement: `0.0071595`
- Duration meaningful: `True`
- Selected mapping: `empirical_margin_duration`

## 7. Behavioral validation

- Trials: `400`
- P(sure | offered): `0.0`
- No-sure accuracy: `0.9035532994923858`
- Waived-sure accuracy: `0.8217821782178217`
- Duration analysis available: `True`
- Sure-choice logistic regression: `{'n': 203, 'available': False, 'reason': 'insufficient outcome variation'}`
- Coherence dependence reproduced: `False`
- Duration dependence reproduced: `False`

## 8. Waived-sure versus no-sure controlled comparison

- Waived accuracy: `0.8217821782178217`
- Matched no-sure accuracy: `0.8875739644970414`
- Difference: `-0.011834319526627219`
- Mean stimulus-duration mismatch: `113.19526627218934` ms
- Inference: `{'n': 169, 'mean': -0.011834319526627219, 'bootstrap_ci95_low': -0.047337278106508875, 'bootstrap_ci95_high': 0.029585798816568046, 'permutation_p_two_sided': 0.7762237762237763}`
- Control reuse prevented: `True`
- Waived-sure accuracy exceeds matched no-sure accuracy: `False`

## 9. Repeated-stimulus internal variability

- Stimuli with sure/direction variability: `2`
- Within-stimulus analysis: `stimulus fixed-effects logistic regression using pre-TS margin and within-stimulus hidden-state PC1`
- Repetitions: `48` stimuli × `20` repeats
- Pre-TS margin coefficient: `-0.509856` ± `3.65025` SE
- Pre-TS within-stimulus hidden PC1 coefficient: `-0.890191` ± `1.25229` SE
- Deterministic identical-choice fraction: `1.0`
- Deterministic maximum output/state difference: `0` / `0`

## 10. Sure-output timing

- Mean pre-TS sure output: `-0.0987934`
- Future sure minus waived before TS: `not estimable`
- Offered minus unoffered before TS (leakage control): `0.0156954`
- Future sure minus waived immediately after TS: `not estimable`
- Future sure minus waived immediately before/after go cue: `not estimable` / `not estimable`
- Choice-conditioned timing contrasts are not estimable because the standard evaluation contained zero sure choices.
- Evidence of pre-TS offer leakage: `not established`; the descriptive offered-minus-unoffered difference is `0.0156954` without an inferential interval.

## 11. Multi-seed robustness

- Seeds included: `1`
- Aggregate pass: `False`
- Multi-seed training status: `{'run': False, 'reason': 'The seed-7 candidate failed the single-seed gate because it made zero sure choices in the standard evaluation.'}`
- Seeds 8 and 9 were not trained because seed 7 failed the required single-seed gate; no seeds were cherry-picked.

## 12. Direct answers to freeze questions

1. Empirical teacher P(correct) calibrated? **Yes by the stated Wald check** (Brier `0.0783498`, log loss `0.250233`, intercept `-0.203314`, slope `1.06082`).
2. Duration adds information beyond margin? **True**; held-out log-loss improvement `0.0071595`.
3. Frozen confidence mapping? **`empirical_margin_duration`**.
4. P(sure) dependence on coherence reproduced? **False**; P(sure|offered) was `0` with `0` sure choices.
5. P(sure) dependence on duration reproduced? **False**.
6. Waived-sure accuracy exceeds matched no-sure accuracy? **False**; difference `-0.0118343`.
7. Evidence of pre-TS offer leakage? **Not established**; descriptive negative-control difference `0.0156954` lacks an inferential interval.
8. Identical-input recurrent variability predicts sure choice? **Inconclusive**; margin `-0.509856` ± `3.65025` SE and PC1 `-0.890191` ± `1.25229` SE across `40` observations.
9. Deterministic repeated-input behavior stable? **True**.
10. Reproducible across seeds? **No multi-seed conclusion**; only the required seed-7 gate was run and it failed.

## 13. Configuration and runtime

- Final configuration: `{'dt_ms': 10, 'teacher_training_iterations': 50000, 'student_training_iterations': 50000, 'loss_epoch': 1000, 'batch_size': 50, 'recurrent_units': 50, 'recurrent_noise': 0.05, 'teacher_export_batches': 20, 'student_evaluation_batches': 8, 'stimulus_duration_ms': [400, 1200], 'delay_duration_ms': [800, 1200], 'ts_latency_from_motion_offset_ms': [500, 750], 'minimum_post_ts_ms': 100, 'minimum_response_duration_ms': 300, 'sure_offer_probability': 0.5, 'internal_readout_anchor': 'pre_go', 'internal_pre_go_offset_steps': 5, 'repeated_stimuli': 48, 'repeats_per_stimulus': 20, 'deterministic_evaluation_recurrent_noise': 0.0}`
- Runtime seconds: `{'final_teacher_training': 23.6, 'final_teacher_wall': 27.42, 'final_student_training': 22.4, 'final_student_and_validation_wall': 32.69, 'final_training_total': 46.0, 'final_teacher_student_wall_total': 60.11}`
- Key output paths: `{'teacher_dataset': 'data/model_freeze/teacher_seed7.npz', 'teacher_summary': 'results/model_freeze/teacher_seed7/teacher_seed7_summary.json', 'teacher_calibration': 'results/model_freeze/teacher_seed7/teacher_calibration_summary.json', 'student_summary': 'results/model_freeze/multiseed/timed_seed7_summary.json', 'behavior_summary': 'results/model_freeze/multiseed/timed_seed7_summary_behavior.json', 'trial_level_behavior_csv': 'results/model_freeze/multiseed/timed_seed7_summary_trials.csv', 'repeat_level_csv': 'results/model_freeze/multiseed/timed_seed7_summary_repeated_stimulus_trials.csv', 'deterministic_repeat_level_csv': 'results/model_freeze/multiseed/timed_seed7_summary_deterministic_repeated_stimulus_trials.csv', 'sure_output_trials_csv': 'results/model_freeze/multiseed/timed_seed7_summary_sure_output_dynamics_trials.csv', 'matched_controls': 'results/model_freeze/matched_seed7/matched_trial_controls_summary.json', 'single_seed_aggregate': 'results/model_freeze/multiseed/multiseed_validation_summary.json', 'report': 'results/model_freeze_report.md'}`
- Exact scientific commands:

```text
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl TF_CPP_MIN_LOG_LEVEL=2 /private/tmp/rdm-model-freeze-env/bin/python -m pytest -q
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl TF_CPP_MIN_LOG_LEVEL=2 /private/tmp/rdm-model-freeze-env/bin/python scripts/run_timed_teacher.py --seed 7 --output data/model_freeze/teacher_seed7.npz --summary results/model_freeze/teacher_seed7/teacher_seed7_summary.json
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl /private/tmp/rdm-model-freeze-env/bin/python analysis/rdm_teacher_calibration_report.py --dataset data/model_freeze/teacher_seed7.npz --output-dir results/model_freeze/teacher_seed7 --figure-dir results/model_freeze/teacher_seed7
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl TF_CPP_MIN_LOG_LEVEL=2 /private/tmp/rdm-model-freeze-env/bin/python scripts/run_timed_teacher.py --seed 7 --output data/model_freeze/teacher_seed7.npz --summary results/model_freeze/teacher_seed7/teacher_seed7_summary.json --confidence-mapping empirical_margin_duration --empirical-confidence-coefficients 0.6477971635444456 4.913427541959271 -1.2055566867668548
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl /private/tmp/rdm-model-freeze-env/bin/python analysis/rdm_teacher_calibration_report.py --dataset data/model_freeze/teacher_seed7.npz --output-dir results/model_freeze/teacher_seed7 --figure-dir results/model_freeze/teacher_seed7
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl TF_CPP_MIN_LOG_LEVEL=2 /private/tmp/rdm-model-freeze-env/bin/python scripts/run_timed_student.py --dataset data/model_freeze/teacher_seed7.npz --seed 7 --summary results/model_freeze/multiseed/timed_seed7_summary.json --repeated-n-stimuli 48 --repeated-n-repeats 20
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl /private/tmp/rdm-model-freeze-env/bin/python analysis/rdm_matched_trial_controls.py --trials results/model_freeze/multiseed/timed_seed7_summary_trials.csv --output-dir results/model_freeze/matched_seed7 --figure-dir results/model_freeze/matched_seed7
PYTHONPATH=src:analysis /private/tmp/rdm-model-freeze-env/bin/python analysis/rdm_multiseed_validation.py --aggregate-only --seeds 7 --output-dir results/model_freeze/multiseed --teacher-calibration-template results/model_freeze/teacher_seed{seed}/teacher_calibration_summary.json --matched-template results/model_freeze/matched_seed{seed}/matched_trial_controls_summary.json
```

## 14. Remaining limitations

- Missing or incomplete: non-degenerate sure behavior.
- Missing or incomplete: coherence-conditioned sure behavior.
- Missing or incomplete: duration-conditioned sure behavior.
- Missing or incomplete: positive controlled waived-sure advantage.
- Missing or incomplete: complete actual sure-output timing.
- Missing or incomplete: multi-seed final-candidate validation.
- This remains a supervised model and does not establish that confidence is learned without supervision.
- Population-dynamics expansion is intentionally deferred until these freeze gates pass.

## 15. Recommendation

**NOT READY FOR POPULATION ANALYSIS**
