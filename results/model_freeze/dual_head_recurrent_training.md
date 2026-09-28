# End-to-end dual-head recurrent student: seed 7

## Final classification

**B. RECURRENT REPRESENTATION STILL LIMITED.**

The controlled objective change was numerically stable and reconstructable, but it did not improve the student. Relative to the staged frozen-head ceiling, the end-to-end model worsens action CE from `0.4260` to `0.5597`, action-margin correlation from `0.6608` to `0.3962`, and teacher-policy accuracy from `0.8700` to `0.8225`. Fixation also fails to remain clean: post-go fixation probability is `0.5026`.

A held-out linear readout on the new recurrent states recovers action CE `0.4311`. That is much better than the trained native dual-head output (`0.5597`) but still does not beat the previous split-matched frozen ceiling (`0.4265`). The recurrent representation therefore did not improve under this objective, and its native action head also uses the available information inefficiently.

This is not classified as C: all logged losses are finite, the implemented TensorFlow loss matches an independent numerical reference, action loss is proven inactive before go, saved-checkpoint reconstruction is exact, and legacy checkpoints remain runnable. The failure is empirical under the specified seed, objective, and 50,000-example training budget rather than an identified implementation fault.

Per the stopping rule, simple supervised-objective tuning should stop. The next scientific discussion should reconsider the teacher-to-student distillation formulation, task architecture, whether this supervised setup can produce the desired policy, or a longer-term reinforcement-learning formulation. It should not automatically move to `sure_reward` tuning.

## Predeclared success rule

Before training, success required all of the following:

- pre-go fixation at least `0.95`, post-go fixation at most `0.05`, and fixation AUC at least `0.99`;
- at least `5%` lower CE and `10%` lower KL than the staged frozen head;
- action-margin correlation at least `0.05` above staged;
- teacher-policy accuracy at least `0.02` above staged;
- incorrect-direction probability no higher than staged;
- sure-favoring fidelity no lower than staged;
- no-sure and offered-direction accuracy within `0.01` of staged.

The model passed none of the broad action-improvement gates and did not pass the fixation gate.

## Architecture

The model has one 50-unit recurrent backbone shared by two logically independent output heads:

```text
input x_t
    ↓
50-unit continuous-time ReLU RNN
    ├── fixation logit → sigmoid → P(fixation)
    └── LEFT / RIGHT / SURE logits → 3-way softmax
```

The recurrent update is the existing PsychRNN `Basic` update with `alpha=0.1`, `rec_noise=0.05`, and ReLU state-to-output activation. The fixation parameters are row 0 of `W_out` and `b_out`; action parameters are disjoint rows 1–3. Fixation is never included in the action normalization or behavioral argmax.

## Exact objective

For fixation-active observations:

```text
L_fix = (1 / N_fix) Σ sigmoid_cross_entropy(fixation_target, fixation_logit)
```

The fixation target is `1` before go and `0` from go onward. `N_fix` is the number of active fixation observations.

For post-go observations only:

```text
L_action = (1 / N_action) Σ [-Σ_a q_a log softmax(action_logits)_a]
```

`q` is the existing soft LEFT/RIGHT/SURE teacher target, including the unchanged no-sure logic. Before go, action loss is exactly inactive; pre-go logits are not driven toward zero.

```text
L_total = L_fix + L_action
```

Both terms have unit weight and are normalized by their own active observations. There is no four-output averaging factor, sure-specific weighting, oversampling, thresholding, auxiliary confidence loss, or direct supervision from margin, coherence, or duration.

## Existing regularization audit

PsychRNN defaults and the existing student configuration have no nonzero recurrent regularization. The run made those zeros explicit:

| Regularizer | Coefficient |
|---|---:|
| L1 input | 0 |
| L1 recurrent | 0 |
| L1 output | 0 |
| L2 input | 0 |
| L2 recurrent | 0 |
| L2 output | 0 |
| L2 firing rate | 0 |

No new regularizer was introduced.

## Frozen teacher proof

Teacher dataset before and after training:

```text
data/model_freeze/teacher_seed7.npz
SHA-256 8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d
```

The hash was identical before and after. The empirical-margin-duration mapping, coefficients, `sure_reward=0.55`, correct/error rewards, temperature, blend, clipping, teacher targets, coherence set, offer probability, and timing were not changed. The teacher was not retrained.

## Training configuration

| Setting | Value |
|---|---:|
| seed | 7 |
| recurrent units | 50 |
| recurrent noise | 0.05 |
| dt / tau | 10 / 100 |
| alpha | 0.1 |
| activation | ReLU |
| batch size | 50 |
| training budget | 50,000 examples |
| optimizer | PsychRNN default Adam |
| learning rate | 0.001 |
| gradient clipping | per-gradient norm 1.0 |
| task dataset | unchanged 1,000-trial teacher dataset |
| evaluation identities | dataset indices 0–399 |

As in the original categorical control, the random training sampler draws from all 1,000 frozen dataset trials. The standard evaluation identities remain exactly indices 0–399; they are a fixed comparison set, not a training-disjoint holdout. The only intended model difference is the task-aligned dual-head loss/parameter interpretation.

PsychRNN completed 999 minibatch updates under its `training_iters=50000`, batch-50 stopping convention. Nine sampled training losses were recorded from 5,000 through 45,000 examples; all were finite. Training wall time was `25.21 s`, and total run time including evaluation and held-out probes was `31.29 s`.

## Primary seed-7 comparison

All metrics use the same deterministic (`rec_noise=0`) standard evaluation trials.

| Metric | Initial categorical | Staged frozen head | End-to-end dual head |
|---|---:|---:|---:|
| fixation pre-go | 0.8791 | **0.9817** | 0.8252 |
| fixation post-go | 0.3775 | **0.0201** | 0.5026 |
| fixation AUC | 0.9830 | **1.0000** | 0.8390 |
| action CE | 0.4440 | **0.4260** | 0.5597 |
| action KL | 0.2002 | **0.1821** | 0.3159 |
| action MAE | 0.1016 | **0.0833** | 0.1445 |
| action-margin r | 0.6272 | **0.6608** | 0.3962 |
| sign agreement | 0.8079 | 0.8079 | **0.8276** |
| incorrect-direction probability | 0.1238 | **0.1053** | 0.1821 |
| sure-favoring fidelity | 0.6667 | **0.8333** | 0.4167 |
| teacher-policy accuracy | 0.8625 | **0.8700** | 0.8225 |
| P(sure\|offered) | 0.1429 | 0.1626 | 0.0000 |
| no-sure direction accuracy | **0.9442** | 0.9391 | 0.8782 |
| offered-direction accuracy | 0.8010 | 0.8063 | **0.8168** |

The isolated improvement in sign agreement or offered-direction accuracy does not offset the broad degradation. The zero sure rate is not itself the classification criterion; it matters because it accompanies worse teacher-policy fidelity, calibration, margin structure, and direction performance.

## Teacher-policy subsets

On 12 teacher sure-favoring trials, the dual-head student predicts mean `P(sure)=0.2385` for target `0.6120`, assigns `0.3329` to the incorrect direction, and gives sure more mass than the correct direction on only `5/12` trials. Its final teacher-argmax fidelity is `0/12` because another direction channel wins whenever sure is not the action argmax.

On 191 teacher direction-favoring offered trials, mean probabilities are:

| Quantity | End-to-end dual head |
|---|---:|
| P(correct direction) | 0.6422 |
| P(incorrect direction) | 0.1726 |
| P(sure) | 0.1852 |
| mean sure-minus-correct margin | -0.4570 |
| teacher-argmax fidelity | 0.8168 |

The larger direction-favoring subset remains usable, but incorrect-direction mass is materially worse than both controls.

## Value-boundary analysis

`delta_value = 0.55 - empirical_P(correct)`. Disagreement is hard action argmax versus teacher target argmax on sure-offered trials.

| Value bin | N | Initial categorical | End-to-end dual head |
|---|---:|---:|---:|
| strongly direction-favoring, `< -0.20` | 162 | 11.11% | 12.96% |
| weakly direction-favoring, `[-0.20, -0.05)` | 25 | 72.00% | 48.00% |
| near boundary, `[-0.05, 0.05)` | 8 | 50.00% | 75.00% |
| weakly sure-favoring, `[0.05, 0.15)` | 6 | 50.00% | 100.00% |
| strongly sure-favoring, `>= 0.15` | 2 | 50.00% | 100.00% |

The objective shifts errors between bins rather than reducing them coherently. Critically, strongly direction-favoring errors increase from 18 to 21 of 162 trials. The weak-direction improvement is accompanied by worse near-boundary and sure-favoring behavior.

## Soft confidence structure

Correlations use the 203 sure-offered standard trials. None of these variables was used as a direct training target.

| P(sure) correlation with | Initial categorical | Staged frozen head | End-to-end dual head |
|---|---:|---:|---:|
| teacher sure_strength | 0.5875 | **0.6837** | 0.2908 |
| empirical P(correct) | -0.6900 | **-0.7571** | -0.3568 |
| internal margin | **-0.8293** | -0.7264 | -0.4989 |
| coherence | **-0.7258** | -0.6291 | -0.6188 |
| stimulus duration | -0.2569 | -0.1462 | **-0.3081** |

The new student preserves the expected signs, but direct teacher-sure and empirical-value relationships become much weaker. Better policy fidelity did not emerge alongside a cleaner confidence representation.

## New-backbone held-out linear readout

A five-fold trial-level held-out linear softmax probe was fitted to the new model's post-go recurrent states.

| Metric | Native dual-head action | Held-out linear probe |
|---|---:|---:|
| action CE | 0.5597 | **0.4311** |
| action KL | 0.3159 | **0.1872** |
| action MAE | 0.1445 | **0.0833** |
| action-margin r | 0.3962 | **0.6273** |
| sign agreement | 0.8276 | **0.8325** |
| incorrect-direction probability | 0.1821 | **0.1197** |
| sure-favoring fidelity | 0.4167 | **0.5000** |
| direction accuracy | 0.8400 | **0.8875** |

The CE gap is `0.1286` in favor of the offline probe, so the native dual-head readout is not using its states efficiently. However, the new-state probe CE `0.4311` is still worse than the old split-matched ceiling `0.4265`, and probe margin correlation `0.6273` remains below the staged model's native `0.6608`. Representation learning under this objective therefore did not materially exceed the old ceiling.

## Time-resolved held-out representation

Each row is a separate five-fold held-out linear decoder. The target is a counterfactual sure preference computed from `0.55 - empirical_P(correct)` using the unchanged mapping and is independent of actual sure-offer status. Pre-TS results therefore indicate confidence/value-related information only, not advance knowledge of whether sure will appear.

| Epoch | Action-margin r | Counterfactual sure-vs-direction AUC | Direction accuracy |
|---|---:|---:|---:|
| late motion | 0.6503 | 0.9297 | 0.8950 |
| pre-TS | 0.6800 | 0.9395 | 0.8925 |
| immediately before go | 0.6622 | 0.9162 | 0.8925 |
| immediately after go | 0.6686 | 0.9199 | 0.8925 |

The new states retain decodable confidence/value and direction structure, but this does not translate into a better native policy or a readout ceiling above the previous model.

## Repeated-stimulus stopping rule

The 48-stimulus × 20-repeat analysis was **not run**. Standard seed-7 behavior failed the predeclared success rule, so the requested conditional gate was not met. No parameters were selected from repeat behavior.

## Saved artifacts

- `results/model_freeze/dual_head_seed7/weights.npz` — complete reconstructable checkpoint, SHA-256 `4a745832d6076888f96a04334e44c394ae75b69bbbf45f9cfd4f71993ed2206e`
- `results/model_freeze/dual_head_seed7/recurrent_backbone_weights.npz`
- `results/model_freeze/dual_head_seed7/fixation_head_weights.npz`
- `results/model_freeze/dual_head_seed7/action_head_weights.npz`
- `results/model_freeze/dual_head_seed7/configuration.json`
- `results/model_freeze/dual_head_seed7/training_history.csv`
- `results/model_freeze/dual_head_seed7/reconstruction_reference.npz`
- `results/model_freeze/dual_head_seed7/summary.json`

Reconstructing the complete checkpoint reproduces saved raw outputs and recurrent states exactly.

## Reproducibility, tests, and git safety

Training command:

```bash
MPLCONFIGDIR=/tmp/mpl TF_CPP_MIN_LOG_LEVEL=2 PYTHONPATH=src:analysis \
  /private/tmp/rdm-model-freeze-env/bin/python \
  analysis/rdm_dual_head_recurrent_training.py
```

Full test command:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src:analysis \
  /private/tmp/rdm-model-freeze-env/bin/python -m pytest -q
```

Result: **76 passed**, with two third-party Python 3.14 deprecation warnings.

Tests confirm independent head parameter slices, sigmoid/BCE fixation, three-way softmax/soft-CE action, zero pre-go action loss, exclusion of fixation from action probabilities, probability normalization, unchanged teacher targets and dataset hash, exact saved-checkpoint reconstruction, and continued execution of the legacy categorical checkpoint.

The run remained on:

```text
feature/model-freeze-validation
HEAD 32e5a2a577344b4f3c9170b5512e42a81de50796
main 32e5a2a577344b4f3c9170b5512e42a81de50796
```

The worktree was already dirty with the prior model-freeze sequence, so creating or switching to `experiment/end-to-end-dual-head-student` was not safe without carrying ambiguous uncommitted work. No reset, clean, discard, checkout, merge, or main-branch modification occurred.

Experiment-specific new files are:

- `src/dual_head_student.py`
- `analysis/rdm_dual_head_recurrent_training.py`
- `tests/test_dual_head_student.py`
- this report;
- the `results/model_freeze/dual_head_seed7/` artifact directory.

The teacher was not retrained. The existing categorical control and staged-head artifacts were not overwritten. Only seed 7 was trained; seeds 8–9 were not run. `main` was untouched.
