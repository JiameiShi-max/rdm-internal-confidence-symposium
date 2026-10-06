# Population Representation Stage 2: time-resolved dynamics

## A. Event definitions

| event | metadata field | meaning | timing range from trial start | trials |
|---|---|---|---:|---|
| motion onset | `fixation_end / motion_onset_step` | motion evidence begins after fixation-only epoch | 200–200 ms | all |
| motion offset | `stimulus_end / motion_offset_step` | motion evidence terminates | 600–1390 ms | all |
| sure-target onset | `ts_onset / sure_target_onset_step` | input channel 3 turns on when Sure is offered | 1130–2130 ms | sure_offered only |
| go cue | `delay_end / go_cue_step` | fixation turns off and post-go behavioral readout begins | 1460–2580 ms | all |

The ordering follows the live task: fixation/motion onset, motion offset, optional Sure-target onset, then go cue.

## B. Analysis definitions

Every plotted time `t` is the mean recurrent state during the preceding 50 ms (`hidden[t-5:t]`) evaluated every 10 ms. Analysis A applies the unchanged Stage-1 TRAIN scaler and unchanged Stage-1 choice/confidence axes. Analysis B fits a new neuron-wise scaler and fixed-regularization decoder at each time using TRAIN only, then evaluates TEST. A 200-permutation maximum-statistic null across every event/time point supplies one family-wise threshold per metric and seed. Operational onset is the first point in a run of at least 50 consecutive ms above that threshold.

Common windows were motion onset −100…+400 ms, motion offset −350…+400 ms, Sure-target onset −300…+100 ms, and go cue −500…+300 ms. Motion-offset −400 ms was excluded because its trailing 50-ms window would precede motion onset on the shortest trials. Sure-target curves stop at +100 ms because the earliest go cue is 110 ms after target onset.

## C. Motion-onset dynamics

| seed | choice onset | confidence onset | peak choice AUC (time) | peak confidence r (time) |
|---:|---:|---:|---:|---:|
| 7 | 80 ms | 140 ms | 0.987 (400 ms) | 0.651 (400 ms) |
| 8 | 80 ms | 180 ms | 0.987 (400 ms) | 0.658 (400 ms) |
| 9 | 30 ms | 80 ms | 0.984 (360 ms) | 0.648 (400 ms) |

Peak values and times are descriptive; they did not define any analysis window.

## D. Motion-offset dynamics

The motion-offset curves track the final 350 ms of evidence accumulation and the first 400 ms after evidence termination. This range remains at least 100 ms before Sure-target onset for every trial, so it does not mix wagering-option presentation into the post-evidence curve.

## E. Sure-target dynamics

The fixed Stage-1 confidence axis was evaluated in three predetermined raw-state windows:

| seed | pre-target −100…0 ms | post-target 0…100 ms | post-target 100…200 ms | n in late window |
|---:|---:|---:|---:|---:|
| 7 | 0.590 | 0.646 | 0.735 | 73 |
| 8 | 0.615 | 0.683 | 0.741 | 73 |
| 9 | 0.665 | 0.689 | 0.706 | 73 |

The first two windows include all sure-offered TEST trials. The +100…+200-ms window prospectively includes only trials whose go cue occurs at least 200 ms after Sure-target onset, preventing post-go mixing. Confidence-related structure before target onset supports an internal confidence representation that is already present before wager availability.

## F. Go-cue dynamics

Go-aligned curves cover −500…+300 ms. The fixed-axis figure tracks Left/Right separation and low/high confidence groups defined by the TRAIN empirical-P(correct) median. Continuous TEST correlation remains the confidence inference; the grouping is visualization only.

## G. Operational onset

| seed | choice onset from motion onset | confidence onset from motion onset | family-wise choice threshold | family-wise confidence threshold |
|---:|---:|---:|---:|---:|
| 7 | 80 ms | 140 ms | 0.948 | 0.577 |
| 8 | 80 ms | 180 ms | 0.961 | 0.612 |
| 9 | 30 ms | 80 ms | 0.928 | 0.530 |

These are operational decoding onsets, not biological or causal latencies.

## H. Cross-seed replication

| seed | pre-TS confidence r | Stage-1 pre-go confidence r |
|---:|---:|---:|
| 7 | 0.590 | 0.776 |
| 8 | 0.615 | 0.749 |
| 9 | 0.665 | 0.740 |

All seeds used identical windows, models, regularization, permutation count, maximum-statistic correction, and onset rule. Individual traces remain visible in the replication figure; no second-level inference over three seeds was performed.

## I. Leakage and integrity checks

- Stage-1 trajectory, scaler, choice-axis, and confidence-axis hashes were identical before and after Stage 2.
- Time-specific scaler statistics and decoders were fit on TRAIN only; TEST entered final metrics only.
- The confidence decoder target was empirical P(correct); Sure/Direction labels never entered its fitting.
- No RNN was constructed, replayed, trained, or modified; only saved deterministic Stage-1 trajectories were read.
- No Stage-3 repeated-input or recurrent-noise analysis was run.
- The full repository test suite passed: 93 passed, 0 failed (two unrelated protobuf deprecation warnings).

## J. Supported conclusions

The frozen population trajectories reveal when directional-choice and continuous confidence information become linearly available under a prospective, family-wise-corrected onset definition. Confidence-related alignment is present before Sure-target presentation, persists through the pre-go period, and remains descriptively associated with later wagering. Timing differences indicate information availability under these decoders; they do not establish a causal processing hierarchy or a precise biological latency.
