# Frozen-hidden-state readout diagnostic: seed 7

## Final classification

**D. MIXED / INCONCLUSIVE.** The two residuals have different diagnoses.

- **Fixation is readout/optimization limited (A).** Pre-go versus post-go phase is perfectly linearly separable on held-out trials (accuracy and AUC both `1.000`). A held-out linear fixation readout predicts `0.9972` pre-go and `0.0026` post-go, with RMSE `0.0210`, even though the native post-go fixation output remains `0.3743` mean absolute (`0.3834` RMS). The recurrent state contains an almost exact linear phase signal; the native fixation channel is not using it.
- **Action policy is not cleanly readout limited.** A fresh held-out linear softmax head reduces CE by only `6.92%` and raises action-margin correlation by only `0.0277`. It lowers incorrect-direction mass, but worsens sign agreement and direction accuracy. This is a modest calibration gain, not a broad policy rescue.
- **This is not a nonlinear-representation result (C).** The single allowed 16-unit nonlinear probe does not beat the linear probe: CE is `0.4208` versus `0.4204`, and margin correlation is `0.7098` versus `0.7133`.

The action state contains substantial direction/confidence structure, but a new readout—linear or small nonlinear—does not remove the remaining mismatch. The evidence therefore does not support a single pure A, B, or C label for the whole model.

## Frozen artifact and method

The diagnostic used the requested initial categorical checkpoint:

```text
lambda_fix_postgo = 1
lambda_action     = 1
results/model_freeze/student_head_calibration_v2/stage_a/fix_1_action_1/weights.npz
SHA-256 bab443333a2b962a10bee9ed675e13216eb352ca76114f5052a2db133b5a1329
```

It did **not** use the nominal `(1,4)` sweep finalist. The frozen teacher dataset was:

```text
data/model_freeze/teacher_seed7.npz
SHA-256 8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d
```

Inference was deterministic (`rec_noise=0`). Each epoch feature is the mean hidden state over a 10-step window. Five-fold cross-validation was stratified by teacher preferred action, with entire trials assigned to folds. Every reported probe prediction is out-of-fold; standardization and fitting used training folds only. The action probes were trained on the existing soft LEFT/RIGHT/SURE targets. The native head and probe were evaluated on the same 1,000 trials, including 516 sure-offered trials and 27 teacher sure-favoring trials.

The operational rule for invoking the optional nonlinear probe was a linear CE reduction of at least 10% together with an action-margin correlation gain of at least `0.10`. The linear probe met neither threshold, so one fixed nonlinear architecture was run; no architecture grid was searched.

## Native head versus held-out linear readout

| Metric | Native categorical head | Held-out linear probe |
|---|---:|---:|
| action CE | 0.4516 | **0.4204** |
| action KL | 0.2046 | **0.1733** |
| per-action MAE | 0.1011 | **0.0821** |
| action-margin r | 0.6856 | **0.7133** |
| sign agreement | **0.8508** | 0.8256 |
| incorrect-direction probability | 0.1025 | **0.0917** |
| sure-favoring fidelity | 0.6667 | 0.6667 |
| hard P(sure\|offered) | 0.1163 | 0.1395 |
| direction accuracy | **0.9100** | 0.8950 |

The clean linear readout improves absolute CE by `0.0313` (`6.92%`) and KL by `15.28%`; MAE falls by `18.75%`. Those soft-distribution gains coexist with a `2.52` percentage-point drop in margin-sign agreement and a `1.50` percentage-point drop in direction accuracy. Sure-favoring fidelity is unchanged at `18/27 = 66.67%`.

**Readout ceiling answer:** using exactly the same recurrent states, a clean linear readout can recover a somewhat better probability distribution, especially lower MAE and incorrect-direction mass, but it cannot produce a substantial or consistent hard-policy improvement. This is evidence for some native-head optimization loss, not evidence that the action mismatch is primarily a bad head.

## Fixation-phase probe

| Held-out phase/fixation result | Value |
|---|---:|
| pre-go versus post-go accuracy | 1.0000 |
| pre-go versus post-go AUC | 1.0000 |
| fixation-target RMSE | 0.0210 |
| fixation-target correlation | 0.9991 |
| predicted fixation, pre-go target 1 | 0.9972 |
| predicted fixation, post-go target 0 | 0.0026 |
| native post-go fixation, mean absolute | 0.3743 |
| native post-go fixation, RMS | 0.3834 |

Yes: phase information is linearly available even though the native post-go fixation remains near `0.38`. This is strong, direct evidence that the fixation failure is in the native readout/optimization rather than the recurrent phase representation.

## Time-resolved held-out action decoding

| Epoch | Action-margin r | Sure-vs-direction AUC | Direction accuracy |
|---|---:|---:|---:|
| late motion | 0.6962 | 0.8958 | 0.9030 |
| pre-TS | 0.6918 | 0.8878 | 0.9000 |
| immediately before go | 0.7040 | 0.8787 | 0.9000 |
| immediately after go | 0.7133 | 0.8807 | 0.8950 |

Teacher action/confidence structure is already linearly available by late motion. It remains broadly stable through the task and does not appear suddenly at go. Post-go gives only a small margin-correlation increase and no direction-accuracy gain.

## Value-distance analysis

`delta_value = 0.55 - empirical_P(correct)`. Disagreement is the hard predicted class versus the teacher target argmax on sure-offered trials.

| Value-distance bin | N | Native disagreement | Linear-probe disagreement |
|---|---:|---:|---:|
| strongly direction-favoring, `< -0.20` | 417 | 6.95% | 7.43% |
| weakly direction-favoring, `[-0.20, -0.05)` | 54 | 62.96% | 72.22% |
| near boundary, `[-0.05, 0.05)` | 27 | 55.56% | 62.96% |
| weakly sure-favoring, `[0.05, 0.15)` | 15 | 40.00% | 33.33% |
| strongly sure-favoring, `>= 0.15` | 3 | 66.67% | 0.00% |

Errors are strongly enriched near the value boundary, but they are not eliminated on strongly direction-favoring trials: the held-out linear probe disagrees on `31/417` such trials. It also performs worse than the native head in both direction-favoring boundary bins. The apparent improvement in the strongly sure-favoring bin is only `3` trials and is not a stable scientific result.

## Optional nonlinear ceiling

Because the linear improvement was modest, one fixed held-out nonlinear probe was run: one 16-unit ReLU layer, Adam optimization, and `L2 = 1e-4`. The same five trial folds and soft targets were used; no architecture search was performed.

| Metric | Linear probe | Nonlinear probe |
|---|---:|---:|
| action CE | **0.4204** | 0.4208 |
| action KL | **0.1733** | 0.1738 |
| per-action MAE | **0.0821** | 0.0829 |
| action-margin r | **0.7133** | 0.7098 |
| sign agreement | 0.8256 | **0.8295** |
| incorrect-direction probability | **0.0917** | 0.0919 |
| sure-favoring fidelity | **0.6667** | 0.5926 |
| direction accuracy | 0.8950 | 0.8950 |

The nonlinear readout essentially ties or slightly underperforms the linear readout. There is no evidence that a modest nonlinear head unlocks a hidden policy representation that the linear head misses.

## Minimum next model change

The minimum next change should be a **frozen-backbone, staged output-only fit with independently optimized fixation and three-way action readouts**. This directly addresses the decisive fixation result and should capture the modest action calibration gain without changing recurrent weights, rewards, or targets. Expectations should be explicit: it is likely to fix post-go fixation, but the present ceiling says it will not by itself resolve all action-policy errors.

If that controlled head-only step leaves the action mismatch at this ceiling, the next investigation should change the recurrent training objective or representation—not `sure_reward` or the confidence mapping.

## Reproducibility and integrity

Command:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src:analysis \
  /private/tmp/rdm-model-freeze-env/bin/python \
  analysis/rdm_frozen_readout_diagnostic.py
```

Machine-readable artifacts:

- `results/model_freeze/frozen_readout_diagnostic/summary.json`
- `results/model_freeze/frozen_readout_diagnostic/trial_predictions.csv`
- `results/model_freeze/frozen_readout_diagnostic/phase_predictions.csv`
- `results/model_freeze/frozen_readout_diagnostic/time_resolved.csv`
- `results/model_freeze/frozen_readout_diagnostic/value_distance.csv`

Dataset and checkpoint SHA-256 hashes were identical before and after the diagnostic. The run performed inference plus offline held-out probe fitting only and wrote no model checkpoint.

- No teacher retraining.
- No recurrent-student retraining or modification.
- No reward, confidence mapping, loss-weight, or task-parameter changes.
- No hyperparameter sweep.
- No modification of `main`; the worktree remained on `feature/model-freeze-validation`, with both local branch and `main` at `32e5a2a577344b4f3c9170b5512e42a81de50796` before this diagnostic.
