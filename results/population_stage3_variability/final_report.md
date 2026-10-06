# Population Representation Stage 3: identical-stimulus recurrent-noise variability

## A. Replay design

For each frozen network, every sure-offered TEST source tensor was replayed exactly 100 times with `rec_noise=0.05`. Trials were never regenerated. The repeat seed formula was `30000000 + model_seed*1,000,000 + source_trial_id*1,000 + repeat_id`. The frozen categorical readout was the mean Left/Right/Sure softmax probability over the first 10 post-go steps followed by argmax.

## B. Identity checks

All repeats within a source stimulus used byte-identical saved input tensors and identical saved timing/metadata. Input equality was checked exactly. Input channel 3 was zero throughout every primary −100…0-ms pre-target window. The only repeat-varying exogenous quantity was the explicitly seeded recurrent-noise tensor.

## C. Behavioral variability

| seed | offered TEST stimuli | stable Direction | variable | stable Sure | overall P(Sure) |
|---:|---:|---:|---:|---:|---:|
| 7 | 75 | 49 | 26 | 0 | 0.139 |
| 8 | 75 | 52 | 23 | 0 | 0.115 |
| 9 | 75 | 50 | 25 | 0 | 0.092 |

All stimuli remain in these descriptive counts. Only variable stimuli contribute outcome information to fixed-effect inference. The stimulus-level P(Sure) distribution was:

| seed | median [IQR] P(Sure) | range |
|---:|---:|---:|
| 7 | 0.000 [0.000, 0.165] | 0.000–0.910 |
| 8 | 0.000 [0.000, 0.065] | 0.000–0.920 |
| 9 | 0.000 [0.000, 0.120] | 0.000–0.560 |

## D. Pre-TS confidence variability

The primary state is the average of the 10 recurrent states immediately before `ts_onset`. The unchanged Stage-1 TRAIN scaler and empirical-P(correct) axis were applied with their saved orientation (higher projection means higher empirical P(correct)). The predictor was centered within each exact source stimulus and then globally standardized over variable-stimulus repeats.

## E. Primary within-stimulus result

| seed | variable stimuli | beta per SD | bootstrap median | bootstrap 95% CI | permutation p (two-sided) | negative-tail p |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 26 | -0.581 | -0.582 | [-0.744, -0.441] | 0.0010 | 0.0010 |
| 8 | 23 | -0.391 | -0.390 | [-0.572, -0.219] | 0.0010 | 0.0010 |
| 9 | 25 | -0.336 | -0.336 | [-0.481, -0.222] | 0.0010 | 0.0010 |

The model profiles a separate intercept for every source stimulus. The cluster bootstrap resampled source identities 2,000 times; the 1,000-permutation test shuffled outcomes independently within each source stimulus.

## F. Transparent stimulus-level summary

| seed | mean Sure−Direction difference | median difference | fraction negative |
|---:|---:|---:|---:|
| 7 | -0.0715 | -0.0704 | 0.962 |
| 8 | -0.0445 | -0.0420 | 0.696 |
| 9 | -0.0794 | -0.0532 | 0.920 |

Negative values mean that, for the same input, Sure repeats occupied a lower-confidence state than Direction repeats.

## G. Specificity control

| seed | confidence beta controlling absolute choice-axis magnitude | absolute choice beta |
|---:|---:|---:|
| 7 | -0.266 | -1.425 |
| 8 | 0.231 | -1.816 |
| 9 | -0.295 | -0.114 |

This secondary model used only the two independently frozen Stage-1 axes and stimulus fixed effects; it did not fit a Sure/Direction neural decoder. The confidence coefficient remained negative for seeds 7 and 9 but reversed for seed 8, so specificity beyond absolute choice-axis magnitude did not replicate uniformly.

## H. Temporal analysis

| seed | operational internal-variability prediction onset |
|---:|---:|
| 7 | -300 ms |
| 8 | -300 ms |
| 9 | -250 ms |

The secondary curve used the unchanged confidence axis, 50-ms trailing windows every 10 ms from −300 to 0 ms, 500 within-stimulus permutations, a maximum negative-association statistic across time, and a prospective five-point (50-ms) run rule. Seeds 7 and 8 already met the rule at the earliest evaluated point, so their −300-ms values are left-boundary-limited rather than exact emergence times. These are operational prediction onsets, not causal or biological latencies.

## I. Deterministic control

| seed | choice consistency | max state difference | max output difference |
|---:|---:|---:|---:|
| 7 | 1.000 | 0 | 0 |
| 8 | 1.000 | 0 | 0 |
| 9 | 1.000 | 0 | 0 |

With `rec_noise=0`, ten replays of every source stimulus were exactly repeatable.

## J. Cross-seed replication

The pipeline, TEST identities, repeat count, readout, windows, fixed-effect models, bootstrap and permutation counts, and onset rule were identical for seeds 7, 8, and 9. No second-level inference over three seeds was performed. Seed 7 remained the predetermined example; its illustrative trajectory uses the lowest source-trial ID among variable stimuli.

## K. Integrity and leakage checks

- Frozen checkpoint hashes were identical before and after replay.
- Stage-1 scaler, confidence-axis, and choice-axis hashes were identical before and after replay.
- Primary inference used sure-offered TEST source identities only.
- No network training, new confidence axis, or Sure/Direction decoder was used.
- Recurrent noise was exactly 0.05 for stochastic replay and exactly 0 for deterministic control.
- The categorical readout was unchanged from the final supervised freeze.
- All 22,500 stochastic repeat seeds were unique, and the saved record/statistical archives passed a complete artifact audit.
- The full repository test suite passed: 100 passed, 0 failed (two unrelated protobuf deprecation warnings).

## L. Supported conclusion

Across all three seeds, under identical external input, lower-than-usual pre-target recurrent states along the independently defined confidence dimension were associated with a higher probability of later Sure choice after controlling source-stimulus identity. The primary association replicated, while the secondary adjustment for absolute choice-axis magnitude did not retain the predicted confidence sign in seed 8. These are observational within-network associations under controlled in-silico input; they do not establish that confidence fluctuations cause Sure choice or that the axis is a causal biological mechanism.
