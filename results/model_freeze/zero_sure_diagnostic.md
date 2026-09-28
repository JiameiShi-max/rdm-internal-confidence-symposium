# Zero-sure diagnostic: current seed-7 artifacts

## Scope and artifact integrity

No teacher or student training was run. No inference rerun was possible or needed for the saved quantities below, and no model parameter, reward, temperature, blend, clipping value, loss weight, or readout threshold was changed.

Artifacts analyzed:

- `data/model_freeze/teacher_seed7.npz`
- `results/model_freeze/multiseed/timed_seed7_summary.json`
- `results/model_freeze/multiseed/timed_seed7_summary_trials.csv`
- `results/model_freeze/multiseed/timed_seed7_summary_sure_output_dynamics_trials.csv`
- `results/model_freeze/multiseed/timed_seed7_summary_repeated_stimulus_trials.csv`
- `results/model_freeze/teacher_seed7_legacy/teacher_calibration_summary.json`
- `results/model_freeze/teacher_seed7/teacher_calibration_summary.json`

The standard evaluation saved post-go sure output but did **not** save fixation, left, and right post-go outputs or a model checkpoint. Therefore exact standard-trial predicted-correct-direction metrics cannot be reconstructed. The existing repeat CSV does save left/right/sure post-go averages for 48 selected offered inputs × 20 stochastic repeats; those records are used only for the explicitly labeled output-geometry analyses.

## Exact active formulas

The formulas were traced from `src/internal_confidence_proxy.py` and `src/generate_stage9_internal_teacher_dataset.py`, not inferred from field names:

```text
empirical_logit = b0 + b_margin * internal_margin + b_duration * duration_seconds
empirical_P_correct = sigmoid(empirical_logit)
expected_direction_value = error_reward + P_correct * (direction_reward - error_reward)
                         = P_correct          [direction_reward=1, error_reward=0]
sure_advantage = sure_reward - expected_direction_value
               = 0.55 - P_correct
raw_sure_transform = sigmoid(sure_advantage / 0.15)
sure_strength = clip(0.5 + 0.8 * (raw_sure_transform - 0.5), 0.05, 0.95)
target_sure = sure_strength
target_correct_direction = 1 - sure_strength
final_choice = argmax(mean output channels over the first 10 post-go steps)
```

The empirical coefficients saved in the final dataset are `[0.6477971635444456, 4.913427541959271, -1.2055566867668548]`.

## Teacher/target diagnostics

Sure-offered teacher trials: **516**.

| statistic | final sure_strength |
|---|---:|
| minimum | 0.138043 |
| maximum | 0.740912 |
| mean | 0.21387 |
| median | 0.154838 |
| standard deviation | 0.125407 |

| quantile | sure_strength |
|---|---|
| q01 | 0.138172 |
| q05 | 0.138493 |
| q10 | 0.138924 |
| q25 | 0.142574 |
| q50 | 0.154838 |
| q75 | 0.214226 |
| q90 | 0.408317 |
| q95 | 0.50031 |
| q99 | 0.655132 |

| condition | fraction | count / 516 |
|---|---|---|
| > 0.10 | 1.000000 | 516 |
| > 0.20 | 0.275194 | 142 |
| > 0.30 | 0.170543 | 88 |
| > 0.40 | 0.114341 | 59 |
| > 0.45 | 0.077519 | 40 |
| > 0.50 | 0.052326 | 27 |
| > 0.55 | 0.036822 | 19 |
| > 0.60 | 0.029070 | 15 |

Using equality tolerance `1e-08`:

- fraction `target_sure > target_direction`: **0.052326** (27/516)
- fraction approximately equal: **0.000000**
- fraction `target_sure < target_direction`: **0.947674**

Thus the targets are not literally all direction-favoring, but only **5.23%** favor sure. A perfectly trained hard-argmax student would not be expected to make zero sure choices, because 27/516 teacher trials do favor sure; however, categorical sure supervision is sparse.

## Source of the target distribution

Offered-trial distribution summaries:

| variable | mean | SD | minimum | median | maximum |
|---|---:|---:|---:|---:|---:|
| empirical P(correct) | 0.873418 | 0.151567 | 0.340985 | 0.941382 | 0.999577 |
| internal margin | 0.642898 | 0.375777 | 8.46684e-05 | 0.629854 | 1.68457 |
| duration (ms) | 788.488 | 235.748 | 400 | 800 | 1190 |
| expected direction value | 0.873418 | 0.151567 | 0.340985 | 0.941382 | 0.999577 |
| sure reward | 0.55 | 0 | 0.55 | 0.55 | 0.55 |
| sure advantage | -0.323418 | 0.151567 | -0.449577 | -0.391382 | 0.209015 |
| raw sure transform | 0.142337 | 0.156759 | 0.0475535 | 0.0685477 | 0.80114 |
| final sure_strength | 0.21387 | 0.125407 | 0.138043 | 0.154838 | 0.740912 |

Correlations with final sure_strength:

| variable | Pearson r |
|---|---|
| empirical_p_correct | -0.976101 |
| internal_margin | -0.724385 |
| duration_ms | -0.0271631 |
| coherence | -0.457542 |

Figure: [teacher target chain](zero_sure_diagnostic/teacher_target_chain.png).

## Explicit value boundary

With direction reward 1, error reward 0, and sure reward 0.55, equality occurs at:

```text
P(correct) = 0.55
```

- fraction below 0.55: **0.052326** (27/516)
- fraction above 0.55: **0.947674**
- fraction at the boundary within `1e-08`: **0.000000**
- fraction final sure_strength > 0.5: **0.052326**

These fractions are identical because the sigmoid and the blend are strictly monotonic and both preserve the 0.5 crossing; clipping is inactive at that crossing. The later transformation compresses magnitudes toward 0.5 but does not move the categorical boundary.

## Student target versus output

### Standard 203 offered evaluation trials

The saved post-go sure channel is graded:

| statistic | predicted_sure |
|---|---:|
| minimum | 0.0254483 |
| maximum | 0.337096 |
| mean | 0.212593 |
| median | 0.212621 |

| quantile | predicted_sure |
|---|---|
| q01 | 0.0875179 |
| q05 | 0.129519 |
| q10 | 0.14371 |
| q25 | 0.176989 |
| q50 | 0.212621 |
| q75 | 0.252907 |
| q90 | 0.279157 |
| q95 | 0.296686 |
| q99 | 0.313374 |

- MAE predicted_sure vs target_sure: **0.076888**
- RMSE: **0.113715**
- correlation: **0.450953**
- target sure-favoring fraction: **0.059113** (12/203)
- current argmax sure frequency: **0.000000**
- predicted_sure > 0.5 frequency: **0.000000**
- on the 12 sure-favoring targets, mean target_sure is **0.61198**, but mean/max predicted_sure are only **0.258859 / 0.327936**

Predicted-sure correlations:

| variable | Pearson r |
|---|---|
| teacher_sure_strength | 0.450953 |
| empirical_p_correct | -0.533912 |
| coherence | -0.639578 |
| duration_ms | -0.166658 |
| internal_margin | -0.635569 |

Figure: [student soft output](zero_sure_diagnostic/student_soft_output.png).

The standard artifacts do not contain post-go direction or fixation channels, so standard-trial predicted-direction MAE/correlation, predicted choice margin, diagnostic readout B, and fixation-output checks are **not recoverable** without rerunning inference. No inference or training was rerun in this pass.

### Saved repeat-set action geometry

The repeat set supplies the missing left/right/sure action channels for 960 rows (48 offered inputs × 20 repeats):

- predicted_sure MAE / RMSE / correlation: **0.0997245 / 0.131839 / 0.287094**
- predicted_correct_direction MAE / correlation: **0.268217 / 0.32376**
- target/predicted choice-margin correlation: **0.352373**
- mean choice-margin bias `(predicted − target)`: **0.214649**
- fraction target choice margin > 0: **0.083333**
- fraction predicted choice margin > 0: **0.172917**

Average post-go action outputs on this offered repeat set:

| output | target mean | prediction mean |
|---|---:|---:|
| correct direction | 0.718425 | 0.466081 |
| incorrect direction | 0 | 0.233537 |
| sure | 0.281575 | 0.243879 |

The incorrect-direction output is not close to its target of zero (mean **0.233537**), so comparing sure only with the correct-direction channel is not equivalent to the actual four-channel argmax.

For the standard 400-trial evaluation, the one action channel saved for both conditions was sure: mean post-go predicted_sure is **0.212593** on offered trials versus **-0.0310637** on no-sure trials (whose target_sure is zero). This confirms the offer input is learned at the soft-output level. Fixation and direction channel averages by offered condition are not present in the saved standard artifacts.

Figure: [student action geometry](zero_sure_diagnostic/student_action_geometry.png).

## Hard-readout diagnostic

| diagnostic readout | standard 203 | repeat-set 960 |
|---|---:|---:|
| A. current four-channel argmax | 0.000000 | 0.002083 |
| B. predicted_sure > predicted_correct_direction | not recoverable | 0.172917 |
| C. predicted_sure > 0.5 | 0.000000 | 0.000000 |
| sure > max(left,right), diagnostic only | not recoverable | 0.003125 |

On the repeat set, A and B disagree in **164/960** rows. The reason is output geometry: an incorrect-direction output (and potentially the unsaved fixation channel) can exceed sure even when sure exceeds the correct-direction output. The hard readout therefore contributes materially, but it is not the only issue because predicted_sure never exceeds 0.5 in either saved evaluation.

## Coherence and duration at the soft level

Standard offered trials by coherence:

| coherence | N | mean target_sure | mean predicted_sure |
|---|---|---|---|
| 0 | 50 | 0.353737 | 0.253643 |
| 0.032 | 31 | 0.231764 | 0.247398 |
| 0.064 | 28 | 0.181414 | 0.219187 |
| 0.128 | 26 | 0.151882 | 0.201418 |
| 0.256 | 39 | 0.147017 | 0.178374 |
| 0.512 | 29 | 0.144001 | 0.154279 |

Standard offered trials by duration quartile:

| duration ms | N | mean target_sure | mean predicted_sure |
|---|---|---|---|
| 400–560 | 50 | 0.217042 | 0.232319 |
| 560–780 | 49 | 0.186554 | 0.200316 |
| 780–970 | 51 | 0.237443 | 0.221932 |
| 970–1190 | 53 | 0.220886 | 0.196346 |

The intended structure is present at the soft level: predicted_sure decreases with empirical P(correct) (`r=-0.533912`), coherence (`r=-0.639578`), internal margin (`r=-0.635569`), and more weakly with duration (`r=-0.166658`). Zero categorical sure choices therefore does not mean zero learned confidence modulation.

## Empirical mapping versus legacy mapping on the same final teacher trials

| mapping | mean P(correct) | mean sure advantage | mean sure_strength | fraction sure_strength > 0.5 |
|---|---:|---:|---:|---:|
| final empirical margin+duration | 0.873418 | -0.323418 | 0.21387 | 0.052326 |
| legacy margin, same trials | 0.501437 | 0.0485628 | 0.541941 | 0.579457 |

The empirical mapping moved the implied sure-favoring prevalence from **57.95%** to **5.23%** on the exact same final teacher trials. This is the dominant source of sparse categorical sure targets. This comparison is diagnostic only; no legacy student was trained.

## The two duration-improvement numbers

- **0.00260594** comes from `results/model_freeze/teacher_seed7_legacy/teacher_calibration_summary.json`: the preliminary legacy-mapping teacher export, before regeneration. It compares held-out margin-only log loss 0.243621 with margin+duration log loss 0.241015 and was the **selection-stage** metric used to choose the empirical mapping. Its original dataset path was reused and later overwritten, but the calibration summary was preserved.
- **0.0071595** comes from `results/model_freeze/teacher_seed7/teacher_calibration_summary.json`: the regenerated/retrained final teacher export using empirical-derived targets. It compares held-out margin-only log loss 0.258583 with margin+duration log loss 0.251423 and is the **final-validation** metric.

Both numbers are correct for their distinct teacher runs. They are not two estimates from the same frozen teacher outputs. The existing report already labels the first as the selection-run improvement and the second as final validation, so no reporting correction is required.

## Final classification

**D. MIXED**

Two mechanisms contribute materially:

1. **Target degeneracy/sparsity:** only 27/516 (5.23%) final teacher targets favor sure; the final empirical mapping reduced same-trial sure-favoring prevalence from 57.95% under the legacy mapping to 5.23%.
2. **Student/output-geometry failure:** 12/203 standard evaluation targets favor sure, but the student makes 0 sure choices. Predicted sure is graded and correlated with target (`r=0.450953`), yet its maximum is only `0.337096`. In the repeat set, the incorrect-direction output averages `0.233537` despite a zero target, and `sure > correct direction` occurs in 17.29% of rows while actual sure argmax occurs in only 0.21%.

This is not explained by one implementation bug in the traced mapping or target formulas. The mapping, value boundary, transformation, saved targets, and current readout are internally consistent. The failure is a sparse rational teacher policy combined with imperfect student action-output competition and a hard argmax that exposes those residual competing outputs.

## Minimum recommended next change (not implemented)

Keep the current teacher value policy, rewards, and empirical calibration fixed for the next controlled test, and change only the **student post-go action-head objective** from independent channel regression to a normalized categorical competition over left/right/sure using the existing teacher-derived action preferences (with fixation excluded after go). This is the smallest change that directly addresses the observed nonzero incorrect-direction output and the mismatch between soft targets and four-channel argmax, while preserving the interpretation `confidence/value → rational sure-vs-direction competition` and avoiding arbitrary sure-rate tuning.

Before that retraining, extend evaluation persistence to save all four post-go channel averages so the standard-trial direction/fixation diagnostics are auditable. That persistence change is measurement-only; it should not alter model behavior.

## Outputs

- Trial-level chain: `results/model_freeze/zero_sure_diagnostic.csv`
- Teacher figure: `results/model_freeze/zero_sure_diagnostic/teacher_target_chain.png`
- Student soft-output figure: `results/model_freeze/zero_sure_diagnostic/student_soft_output.png`
- Repeat-set geometry: `results/model_freeze/zero_sure_diagnostic/student_action_geometry.png`
