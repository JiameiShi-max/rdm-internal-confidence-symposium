# Student-head calibration: seed 7

## Decision

**B. STUDENT HEAD STILL NOT CALIBRATED.** Stronger post-go fixation weighting suppresses fixation, but every tested increase violates the predeclared direction-accuracy validity gate. Changing the categorical action weight produces no clear practical fidelity improvement. The nominal Stage-B winner, `lambda_action=4`, improves deterministic action CE by only `0.00030` while worsening fixation, margin correlation, target-sure correlation, incorrect-direction probability, and sure-favoring-trial fidelity.

The teacher policy was not changed or retrained. The frozen dataset hash was identical before and after:

```text
data/model_freeze/teacher_seed7.npz
SHA-256 8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d
```

## Exact objective

Let `z[b,t,c]` be the raw four-channel linear output, `y` the target, and `m` the output mask. Let `g[b,t]=1` exactly at post-go steps, identified by the invariant that the LEFT/RIGHT/SURE targets sum to one. Let `A[b,t,c]=g[b,t]` for the three action channels and zero for fixation. Let `F[b,t,c]=g[b,t]` for fixation and zero for the three actions.

The retained legacy term is

```text
L_retained = (1 / 4BT) Σ_b,t,c (1-A) [m(z-y)]².
```

This contains all pre-go fixation/action MSE and post-go fixation MSE at its original unit weight. The separately measurable post-go fixation contribution is

```text
L_fix_postgo = (1 / 4BT) Σ_b,t,c F [m(z-y)]²,
```

where the post-go fixation target is zero and `z_fixation` is the raw linear output. The action distribution and categorical term are

```text
p_a = softmax(z_LEFT, z_RIGHT, z_SURE)_a
L_action = (1 / 4BT) Σ_b,t g[b,t] [-Σ_a q_a log p_a].
```

The calibrated objective is

```text
L = L_retained
  + (lambda_fix_postgo - 1) L_fix_postgo
  + lambda_action L_action.
```

Thus post-go fixation has total coefficient `lambda_fix_postgo`; pre-go fixation remains exactly unit-weighted; fixation remains outside the action softmax; and `lambda_fix_postgo=1, lambda_action=1` uses the original categorical computation exactly. All PsychRNN L1, L2, firing-rate, and custom regularization coefficients are zero.

## Predeclared selection rules

A candidate was invalid if stochastic no-sure accuracy fell below `0.9140` (two percentage points below the existing categorical value), any output was non-finite, training was unstable, or standard trial identity/timing changed. Every candidate used dataset indices `0–399` in the same order for both stochastic and deterministic evaluation.

Stage A chose minimum deterministic fixation among valid candidates that stayed within the predeclared categorical-control guardrails: CE `+0.02`, margin correlation `-0.05`, and sign agreement `-0.03`. Stage B used the requested lexicographic deterministic ordering: CE, KL, margin correlation, margin sign agreement, sure-favoring fidelity, incorrect-direction probability, then fixation. Hard sure rate was never an optimization criterion.

## Stage A — fixation cleanup

`lambda_action=1` throughout. Fidelity metrics are deterministic noise=0; direction accuracies and hard P(sure) shown here are the standard stochastic behavioral check.

| λ fix | Valid | Fix abs | Fix RMS | CE | KL | MAE | Margin r | Sign | Sure-target sign | Incorrect | Sure r | No-sure acc. | Offered-dir. acc. | Hard P(sure\|offered) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | yes | 0.3775 | 0.3854 | 0.4440 | 0.2002 | 0.1016 | 0.6272 | 0.8079 | 0.6667 | 0.1238 | 0.5875 | 0.9340 | 0.8902 | 0.1478 |
| 2 | **no** | 0.3146 | 0.3244 | 0.4701 | 0.2262 | 0.1045 | 0.6263 | 0.8177 | 0.6667 | 0.1346 | 0.5760 | 0.8832 | 0.8265 | 0.0345 |
| 5 | **no** | 0.1888 | 0.2059 | 0.5017 | 0.2578 | 0.1205 | 0.5882 | 0.8571 | 0.6667 | 0.1511 | 0.4814 | 0.8832 | 0.7929 | 0.0246 |
| 10 | **no** | 0.1375 | 0.1851 | 0.6350 | 0.3912 | 0.1932 | 0.2765 | 0.8621 | 0.2500 | 0.2973 | 0.1819 | 0.8934 | 0.7143 | 0.0000 |

Teacher hard sure preference is `12/203 = 5.91%` for every row. Increasing fixation weight does suppress raw fixation, but the cost is large: all stronger candidates lose 4–5 percentage points of no-sure accuracy and materially degrade policy fidelity. **Stage-A selection: `lambda_fix_postgo=1`.**

## Stage B — categorical action weight

`lambda_fix_postgo=1` throughout.

| λ action | Valid | Fix abs | Fix RMS | CE | KL | MAE | Margin r | Sign | Sure-target sign | Incorrect | Sure r | No-sure acc. | Offered-dir. acc. | Hard P(sure\|offered) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | yes | 0.3599 | 0.3689 | 0.4556 | 0.2117 | 0.1068 | 0.6374 | 0.8571 | 0.5000 | 0.1448 | 0.6265 | 0.9391 | 0.8660 | 0.0443 |
| 1 | yes | 0.3775 | 0.3854 | 0.4440 | 0.2002 | 0.1016 | 0.6272 | 0.8079 | 0.6667 | 0.1238 | 0.5875 | 0.9340 | 0.8902 | 0.1478 |
| 2 | **no** | 0.4119 | 0.4234 | 0.4731 | 0.2292 | 0.1099 | 0.5830 | 0.7931 | 0.5833 | 0.1331 | 0.4991 | 0.9086 | 0.8802 | 0.1773 |
| 4 | yes | 0.5163 | 0.5301 | **0.4438** | **0.1999** | **0.1001** | 0.6135 | 0.8424 | 0.5833 | 0.1342 | 0.5407 | 0.9391 | 0.8750 | 0.1330 |

The predeclared lexicographic rule nominally selects **`lambda_action=4`**, because its deterministic CE is `0.44375` versus `0.44405` for weight 1. This `0.00030` (0.067%) difference is not a material improvement, and the criteria disagree strongly: fixation worsens by 36.8%, margin correlation falls, sure correlation falls, incorrect-direction probability rises, and sure-favoring sign fidelity falls. Therefore the selected numerical finalist is not evidence that action reweighting solved the head.

## Legacy, initial categorical, and nominal finalist

This table uses standard stochastic outputs where available. The legacy action-margin, CE/KL, incorrect-output, and sign values come from its saved 48×20 repeat geometry because legacy standard outputs were not persisted; its CE/KL required clipping and renormalizing independent scores.

| Metric | Legacy MSE | Initial categorical | Final nominal head (1,4) |
|---|---:|---:|---:|
| teacher hard sure preference | 0.0591 | 0.0591 | 0.0591 |
| model hard P(sure\|offered) | 0.0000 | 0.1478 | 0.1330 |
| target/predicted sure r | 0.4510 | 0.5594 | 0.4973 |
| action-margin r | 0.3524* | 0.5978 | 0.5782 |
| action-margin sign agreement | 0.7958* | 0.7980 | 0.8128 |
| action CE | 1.0022* | 0.4675 | 0.4623 |
| action KL | 0.4493* | 0.2236 | 0.2184 |
| incorrect-direction probability | 0.2335* | 0.1326 | 0.1415 |
| post-go fixation | not saved | 0.3758 | 0.5135 |
| no-sure accuracy | 0.9036 | 0.9340 | 0.9391 |
| offered-direction accuracy | 0.8218 | 0.8902 | 0.8750 |

`*` Repeat-set diagnostic, not standard-trial value.

## Teacher sure-favoring versus direction-favoring trials

Standard stochastic nominal-finalist results:

| Teacher subset | N | Target P(sure) | Pred. P(sure) | Pred. P(correct) | Pred. P(incorrect) | Mean pred. margin | Sure > correct | Sure final argmax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sure-favoring | 12 | 0.6120 | 0.3249 | 0.3439 | 0.3311 | -0.0190 | 50.0% | 33.3% |
| direction-favoring | 191 | 0.1909 | 0.1864 | 0.6841 | 0.1296 | -0.4977 | 16.8% | 12.0% |

The finalist underpredicts sure on teacher sure-favoring trials and still assigns roughly one third of probability to the incorrect direction there. It also predicts sure over the correct direction on 16.8% of teacher direction-favoring trials. The mismatch is not a clean threshold shift.

## Distance from the teacher value boundary

`delta_value = 0.55 - empirical_P(correct)`. The following uses deterministic finalist outputs.

| Bin | Δ value | N | Hard disagreement | Mean absolute margin error |
|---|---|---:|---:|---:|
| strongly direction-favoring | < -0.20 | 162 | 10.5% | 0.2197 |
| weakly direction-favoring | [-0.20, -0.05) | 25 | 60.0% | 0.3054 |
| near boundary | [-0.05, 0.05) | 8 | 62.5% | 0.1585 |
| weakly sure-favoring | [0.05, 0.15) | 6 | 33.3% | 0.2646 |
| strongly sure-favoring | ≥ 0.15 | 2 | 100% | 0.4826 |

Disagreement is strongly enriched near the boundary, but it is **not confined** there: 17 of 162 strongly direction-favoring trials disagree, accounting for 17 of 41 total deterministic disagreements. The strongly sure-favoring bin has only two trials and cannot support a broad scientific conclusion.

## Graded confidence structure

Standard stochastic offered-trial correlations:

| Variable | Initial categorical | Nominal finalist |
|---|---:|---:|
| teacher sure_strength | +0.559 | +0.497 |
| empirical P(correct) | -0.660 | -0.585 |
| internal margin | -0.811 | -0.731 |
| coherence | -0.708 | -0.625 |
| duration | -0.246 | -0.272 |

All expected directions are preserved, but four of five relationships weaken. Action reweighting therefore does not improve hard-policy fidelity by preserving or strengthening the full soft structure.

## Finalist repeated-stimulus check

Only the nominal finalist received the 48 stimuli × 20 repeats evaluation:

- stochastic hard P(sure): `0.13125`;
- sure/direction switching: `23/48` stimuli, versus `33/48` initially;
- deterministic all-repeat stability: `1.0`, maximum output/state difference `0`;
- deterministic hard P(sure): `0.08333`;
- within-stimulus pre-TS margin coefficient: `-6.576 ± 0.993`, versus `-16.47 ± 1.57` initially;
- within-stimulus hidden PC1 coefficient: `-1.008 ± 0.147`, versus `-0.718 ± 0.124` initially.

Internal stochastic variability remains measurable, but this secondary result does not rescue the standard policy mismatch.

## Explicit answers

1. **Can fixation be cleanly suppressed without harming behavior?** No. Stronger weights reduce fixation but every increase is invalid due to direction-accuracy loss.
2. **Does changing categorical loss weight materially improve fidelity?** No. The nominal CE improvement at weight 4 is only 0.067% and conflicts with worse correlation, subset fidelity, incorrect probability, and fixation.
3. **Is incorrect-direction residual reduced?** No. It rises from `0.1326` initially to `0.1415` in the nominal finalist (stochastic offered trials).
4. **Are disagreements mostly near the boundary?** They are enriched there but not confined there; substantial errors remain in strongly direction-favoring trials.
5. **Is graded confidence preserved?** Directionally yes, but mostly weakened.
6. **Is direction accuracy preserved?** No-sure accuracy is preserved, but offered-direction accuracy falls from `0.8902` to `0.8750`.
7. **Is head calibration no longer the main blocker?** No. Simple relative weighting does not cleanly solve fixation or teacher-policy fidelity.

The remaining issue is not a reward-policy question yet. Before changing `sure_reward`, the next student-head investigation should address why one shared recurrent linear head cannot simultaneously suppress post-go fixation, eliminate incorrect-direction mass, and match the teacher distribution—potentially through a more explicit head parameterization or optimization design, while keeping the teacher fixed.

## Reproducibility

Valid sweep command:

```bash
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl TF_CPP_MIN_LOG_LEVEL=2 /private/tmp/rdm-model-freeze-env/bin/python analysis/rdm_student_head_calibration.py --output-dir results/model_freeze/student_head_calibration_v2
```

Superseded audit command:

```bash
PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/mpl TF_CPP_MIN_LOG_LEVEL=2 /private/tmp/rdm-model-freeze-env/bin/python analysis/rdm_student_head_calibration.py --output-dir results/model_freeze/student_head_calibration
```

It was preserved but superseded after detecting that the initial parameterization changed floating-point reduction order at `(1,1)`. No result from that first sweep was used for scientific selection.

The valid sweep ran eight 50,000-iteration seed-7 student trainings and took `261.64 s` wall time. The superseded audit took `275.96 s`; total actual sweep wall time was `537.59 s`. Full tests: `62 passed`, with two third-party Python 3.14 deprecation warnings.

Primary artifacts:

- `student_head_calibration_v2/stage_a_sweep.csv`
- `student_head_calibration_v2/stage_b_sweep.csv`
- `student_head_calibration_v2/selection_summary.json`
- `student_head_calibration_v2/value_distance.csv`
- `student_head_calibration_v2/stage_b/fix_1_action_4/standard_stochastic.csv`
- `student_head_calibration_v2/stage_b/fix_1_action_4/standard_deterministic.csv`
- `student_head_calibration_v2/finalist_repeated_summary.json`

Every candidate directory contains its weights, full stochastic/deterministic standard-trial geometry, and summary JSON. No teacher was retrained, no reward or confidence parameter changed, legacy/categorical artifacts were not overwritten, and `main` was untouched.
