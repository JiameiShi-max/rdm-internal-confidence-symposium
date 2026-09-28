# Frozen-backbone staged-output-head experiment

## Final decision

**B. FIXATION SOLVED, ACTION REPRESENTATION/TRAINING STILL LIMITED.**

The independently trained fixation head realizes the phase information identified in the prior diagnostic: held-out pre-go fixation is `0.9817`, post-go fixation is `0.0201`, classification accuracy is `0.9989`, and AUC is `1.0000`. The native channel remains at `0.3775` post-go on the same trials.

The action result is qualitatively different. The staged head improves native CE by only `4.06%`, KL by `9.02%`, and margin correlation by `0.0336`. It lands almost exactly at the split-matched linear ceiling: CE `0.4260` staged versus `0.4265` ceiling and margin correlation `0.6608` versus `0.6574`. Hard teacher-policy accuracy rises only `0.75` percentage points, while direction decoding falls `0.50` points. This is the stopping-rule case in which fixation is solved but action remains at the frozen-representation ceiling.

Further output-head tuning should stop. The next experiment should modify the recurrent **training objective or representation**, not the reward, confidence mapping, output-head weights, or action thresholds. That recurrent change is not implemented here.

## Stopping rule declared before results

Output-head cleanup would be called successful only if all of the following held:

- staged pre-go fixation at least `0.95`, post-go fixation at most `0.05`, and AUC at least `0.99`;
- action CE reduced at least `10%` and KL at least `15%` from native;
- action-margin correlation improved by at least `0.05`;
- incorrect-direction probability did not increase;
- sure-favoring fidelity did not decrease;
- direction accuracy stayed within `0.005` of native.

Fixation passed. Action failed the CE, KL, and margin-gain gates, so the predeclared classification is B.

## Frozen artifacts and split

The frozen categorical checkpoint was the requested `(lambda_fix_postgo=1, lambda_action=1)` model, not the nominal `(1,4)` finalist:

```text
results/model_freeze/student_head_calibration_v2/stage_a/fix_1_action_1/weights.npz
SHA-256 bab443333a2b962a10bee9ed675e13216eb352ca76114f5052a2db133b5a1329
```

Teacher dataset:

```text
data/model_freeze/teacher_seed7.npz
SHA-256 8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d
```

The established standard evaluation trials, dataset indices `0–399`, were reserved as the final test set. Only indices `400–999` were used to fit heads. There was no validation set because regularization was fixed at `L2=1e-4`, with no tuning or early stopping.

| Split | Trial indices | Trials | Fixation samples | Action samples |
|---|---|---:|---:|---:|
| train | 400–999 | 600 | 12,000 | 6,000 |
| validation | none | 0 | 0 | 0 |
| test | 0–399 | 400 | 8,000 | 400 trial-level evaluations |

Every sample from a trial stayed in one split. The overlap count was zero. Fixation used the 10 steps immediately before and after go; action used the first 10 post-go steps. Recurrent inference was deterministic (`rec_noise=0`).

## Exact staged heads and objectives

The two heads have separate parameters and separate normalization statistics:

```text
p_fix(h) = sigmoid(w_fix^T standardize(h) + b_fix)

p_action(h) = softmax(W_action^T standardize(h) + b_action)
```

The fixation objective was mean binary cross entropy for pre-go target `1` and post-go target `0`, plus `0.5 × 1e-4 × ||w_fix||²`. The action objective was mean soft-label categorical cross entropy against the unchanged LEFT/RIGHT/SURE teacher distribution, plus `0.5 × 1e-4 × ||W_action||²`.

There was no class weighting, balancing, oversampling, hard-sure relabeling, threshold, hyperparameter search, or recurrent backpropagation. Behavioral choice uses only `argmax(P_left, P_right, P_sure)`; fixation never enters the action softmax or argmax.

## Primary comparison

The primary action comparison uses the same 400 untouched test trials for all three columns. “Linear-probe ceiling” is refit on the 600 training trials using averaged post-go hidden states, then evaluated on these 400 trials.

| Metric | Native head | Linear-probe ceiling | Staged head |
|---|---:|---:|---:|
| fixation post-go | 0.3775 | 0.0026* | **0.0201** |
| action CE | 0.4440 | 0.4265 | **0.4260** |
| action KL | 0.2002 | 0.1827 | **0.1821** |
| action MAE | 0.1016 | 0.0835 | **0.0833** |
| action-margin r | 0.6272 | 0.6574 | **0.6608** |
| sign agreement | 0.8079 | 0.7980 | 0.8079 |
| incorrect-direction probability | 0.1238 | **0.1045** | 0.1053 |
| sure-favoring fidelity | 0.6667 | 0.8333 | 0.8333 |
| hard P(sure\|offered) | 0.1429 | 0.1773 | 0.1626 |
| direction accuracy | **0.9000** | 0.8950 | 0.8950 |

`*` The fixation ceiling is the previously published five-fold held-out linear fixation prediction on all 1,000 trials. The staged result is the new independent sigmoid head on the untouched 400-trial test set.

For context, the previous five-fold action probe reported CE `0.4204` across all 1,000 out-of-fold predictions. Restricted to these 400 test trials, its CE is `0.4140`, but its folds used other standard-evaluation trials for training. The split-matched `0.4265` value is therefore the fair ceiling for this experiment. The staged action head does not materially exceed either reference.

## Fixation evaluation

| Metric | Native fixation channel | Staged fixation head |
|---|---:|---:|
| pre-go mean | 0.8791 | **0.9817** |
| post-go mean | 0.3775 | **0.0201** |
| RMSE | 0.3287 | **0.0541** |
| correlation with target | 0.8250 | **0.9945** |
| classification accuracy | 0.8438 | **0.9989** |
| AUC | 0.9830 | **1.0000** |

The staged fixation head does realize the near-perfect phase information. Its residual post-go mean is below the predeclared `0.05` gate and is not an implementation failure.

## Teacher-policy subsets

### Teacher sure-favoring trials

There are only 12 such test trials, so percentage changes are discrete and should not be generalized broadly.

| Metric | Native | Linear ceiling | Staged |
|---|---:|---:|---:|
| target P(sure) | 0.6120 | 0.6120 | 0.6120 |
| predicted P(sure) | 0.3683 | 0.4087 | 0.4052 |
| predicted P(correct) | 0.3184 | 0.3094 | 0.3135 |
| predicted P(incorrect) | 0.3133 | 0.2819 | 0.2813 |
| mean predicted margin | 0.0500 | 0.0993 | 0.0917 |
| sure over correct | 66.67% | 83.33% | 83.33% |
| teacher-argmax fidelity | 50.00% | 75.00% | 75.00% |

### Teacher direction-favoring offered trials

| Metric | Native | Linear ceiling | Staged |
|---|---:|---:|---:|
| N | 191 | 191 | 191 |
| target P(sure) | 0.1909 | 0.1909 | 0.1909 |
| predicted P(sure) | 0.2098 | 0.2205 | 0.2182 |
| predicted P(correct) | 0.6783 | 0.6862 | 0.6876 |
| predicted P(incorrect) | 0.1119 | 0.0933 | 0.0942 |
| mean predicted margin | -0.4684 | -0.4658 | -0.4694 |
| sure over correct | 18.32% | 20.42% | 19.37% |
| teacher-argmax fidelity | 80.10% | 79.58% | 80.63% |

The staged head improves the small sure-favoring subset and reduces incorrect-direction mass, but on the much larger direction-favoring subset hard fidelity improves by only one trial (`153/191` to `154/191`).

## Hard behavior

All choices below are action-only argmaxes; fixation is excluded.

| Metric | Native | Linear ceiling | Staged |
|---|---:|---:|---:|
| P(left) | 0.4625 | 0.4500 | 0.4575 |
| P(right) | 0.4650 | 0.4600 | 0.4600 |
| P(sure) | 0.0725 | 0.0900 | 0.0825 |
| P(sure\|offered) | 0.1429 | 0.1773 | 0.1626 |
| offered teacher-direction accuracy | 0.8010 | 0.7958 | 0.8063 |
| no-sure direction accuracy | **0.9442** | 0.9391 | 0.9391 |
| teacher-policy accuracy | 0.8625 | 0.8650 | 0.8700 |

The higher sure rate is not treated as success. Overall teacher-policy fidelity increases by only `0.75` percentage points, with a `0.51`-point loss on no-sure direction trials.

No-sure direction accuracy by coherence:

| Coherence | N | Native | Linear ceiling | Staged |
|---:|---:|---:|---:|---:|
| 0.000 | 25 | 0.5600 | 0.5200 | 0.5200 |
| 0.032 | 32 | 1.0000 | 1.0000 | 1.0000 |
| 0.064 | 34 | 1.0000 | 1.0000 | 1.0000 |
| 0.128 | 35 | 1.0000 | 1.0000 | 1.0000 |
| 0.256 | 31 | 1.0000 | 1.0000 | 1.0000 |
| 0.512 | 40 | 1.0000 | 1.0000 | 1.0000 |

The entire no-sure accuracy difference is concentrated at zero coherence.

## Value-boundary analysis

`delta_value = 0.55 - empirical_P(correct)`. Disagreement is hard action argmax versus teacher target argmax on sure-offered trials.

| Value bin | N | Native disagreement | Staged disagreement |
|---|---:|---:|---:|
| strongly direction-favoring, `< -0.20` | 162 | 11.11% | 10.49% |
| weakly direction-favoring, `[-0.20, -0.05)` | 25 | 72.00% | 72.00% |
| near boundary, `[-0.05, 0.05)` | 8 | 50.00% | 50.00% |
| weakly sure-favoring, `[0.05, 0.15)` | 6 | 50.00% | 16.67% |
| strongly sure-favoring, `>= 0.15` | 2 | 50.00% | 0.00% |

The staged head removes only one error among 162 strongly direction-favoring trials and none in the weak-direction or near-boundary bins. Its visible gains occur in two sure-favoring bins containing only eight trials total. It does not broadly reduce errors away from the boundary.

## Graded confidence structure

Correlations use the 203 sure-offered held-out trials.

| P(sure) correlation with | Native | Previous OOF probe | Split ceiling | Staged |
|---|---:|---:|---:|---:|
| teacher sure_strength | 0.5875 | 0.6716 | 0.6734 | **0.6837** |
| empirical P(correct) | -0.6900 | -0.7512 | -0.7487 | **-0.7571** |
| internal margin | **-0.8293** | -0.7266 | -0.7207 | -0.7264 |
| coherence | **-0.7258** | -0.6365 | -0.6306 | -0.6291 |
| stimulus duration | **-0.2569** | -0.1323 | -0.1302 | -0.1462 |

The staged head preserves every expected sign and improves direct teacher-target and empirical-value correlations. It attenuates relationships with internal margin, coherence, and duration in approximately the same way as the frozen linear probes. It does not destroy graded structure, but it does not dominate the native head on all of it.

## Time-resolved sanity check

To avoid future-offer leakage, the target at every epoch is a **counterfactual value preference independent of whether sure is offered**. It is computed for every test trial from `0.55 - empirical_P(correct)`, followed by the unchanged sigmoid, blend, and clip mapping. Thus pre-TS AUC means only that confidence/value-related information is present in the hidden state; it does not mean the network knows whether the sure target will appear.

| Epoch | Margin r | Counterfactual sure-vs-direction AUC | Direction accuracy |
|---|---:|---:|---:|
| late motion | 0.5484 | 0.8838 | 0.8750 |
| pre-TS | 0.5862 | 0.8923 | 0.8950 |
| immediately before go | 0.5993 | 0.8792 | 0.9000 |
| immediately after go | 0.6060 | 0.8789 | 0.8950 |

The staged readout sees confidence/value-related structure before TS, and the margin signal strengthens modestly toward go. No claim about advance knowledge of offer status is made.

## Deployable staged-head artifacts

The recurrent checkpoint was not overwritten. The independently reconstructable heads are:

- `results/model_freeze/staged_output_head_experiment/fixation_head_weights.npz`
- `results/model_freeze/staged_output_head_experiment/action_head_weights.npz`
- `results/model_freeze/staged_output_head_experiment/model_metadata.json`

Each head file stores weights, bias, training-feature mean and scale, and `L2`. Reloaded heads reproduce fixation and action predictions with maximum absolute difference `0.0`.

Additional machine-readable outputs:

- `results/model_freeze/staged_output_head_experiment/summary.json`
- `results/model_freeze/staged_output_head_experiment/test_trial_predictions.csv`
- `results/model_freeze/staged_output_head_experiment/fixation_predictions.csv`
- `results/model_freeze/staged_output_head_experiment/hard_behavior.csv`
- `results/model_freeze/staged_output_head_experiment/coherence_accuracy.csv`
- `results/model_freeze/staged_output_head_experiment/value_distance.csv`
- `results/model_freeze/staged_output_head_experiment/time_resolved.csv`

## Tests, commands, runtime, and safety

Branch and initial HEAD:

```text
feature/model-freeze-validation
32e5a2a577344b4f3c9170b5512e42a81de50796
```

The worktree already contained the prior model-freeze implementation and generated artifacts, so switching to `experiment/staged-output-heads` was not safe without carrying an ambiguous dirty state. The experiment remained on `feature/model-freeze-validation`; no reset, clean, discard, merge, or checkout occurred.

Experiment command:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src:analysis \
  /private/tmp/rdm-model-freeze-env/bin/python \
  analysis/rdm_staged_output_heads.py
```

Test command and result:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src:analysis \
  /private/tmp/rdm-model-freeze-env/bin/python -m pytest -q

69 passed, 2 third-party Python 3.14 deprecation warnings
```

Experiment runtime was `9.25 s`. Experiment-specific new files are the analysis script, its test file, this report, and the staged-output-head artifact directory listed above. Pre-existing dirty files were preserved.

Backbone SHA-256 before and after:

```text
bab443333a2b962a10bee9ed675e13216eb352ca76114f5052a2db133b5a1329
```

Teacher dataset SHA-256 before and after:

```text
8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d
```

No teacher retraining occurred. No recurrent weight was trained or modified. Only the two offline staged heads were fitted. No reward, confidence mapping, task timing, teacher target, loss weight, or threshold changed. `main` was not modified or merged.
