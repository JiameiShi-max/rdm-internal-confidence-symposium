# Final supervised categorical-model freeze

## A. Repository audit

The active implementation before this change was `scripts/run_timed_student.py` calling `main()` in `src/sure_target_stage9_internal_readout.py`.

- `load_stage9_dataset()` loaded `x`, `y_internal`, `mask_internal`, JSON trial metadata, and teacher diagnostic arrays from the NPZ with `allow_pickle=True`.
- `RDM_SureTarget_InternalReadoutDatasetTask` sampled a uniform random integer over the entire teacher dataset during training. It had no split restriction, so the training sampler could see all 1,000 trials.
- Standard evaluation changed that same task to sequential mode, reset its position to zero, and evaluated `n_eval_batches * n_batch` trials. With the reference command, this was the first 400 source trials rather than a training-disjoint set.
- `--student-action-objective categorical_soft` installed `categorical_soft_postgo_loss`: legacy masked MSE was retained outside the post-go action entries and post-go Left/Right/Sure used soft-label categorical cross-entropy. Both loss multipliers were 1.0.
- PsychRNN `Basic.train(..., save_weights_path=...)` saved a reconstructable NPZ containing recurrent/input/output weights, biases, initial state, connectivity values, and optimizer state.
- Categorical evaluation softmaxed raw Left/Right/Sure logits, averaged the first 10 steps from `delay_end`, and used `argmax(Left, Right, Sure)`. Fixation was excluded from the behavioral choice.

## B. Code changes

- `src/sure_target_stage9_internal_readout.py`: added a validated, backward-compatible `allowed_indices` task argument; random and sequential samplers now draw only from that allow-list. Added loading of existing teacher provenance fields.
- `analysis/final_supervised_freeze.py`: added the fixed split, three-seed training, validation/test-only evaluators, held-out metrics, checkpoint reload verification, hashes, provenance, and multi-seed summary. It contains no population analysis.
- `scripts/run_final_supervised_freeze.py`: added the command-line entry point.
- `tests/test_final_supervised_freeze.py`: added split, sampler containment, evaluator containment, reproducibility, frozen-configuration, and checkpoint reload tests.
- `results/final_supervised_freeze/`: added the new split, checkpoints, per-seed artifacts, multi-seed summary, manifest, and this report. Earlier results were not overwritten.

## C. Fixed split

The teacher dataset has 1,000 source trial identities. A single NumPy `RandomState(2026)` permutation produced:

| split | trials | fraction |
|---|---:|---:|
| train | 700 | 70% |
| validation | 150 | 15% |
| held-out test | 150 | 15% |

Automated assertions and the saved summary confirm zero overlap for every pair, no duplicate identities, in-range indices, and an exact union equal to `0..999`. Whole trial tensors are selected by identity, so all timepoints from a source trial remain together. All three students reused `split_indices.npz`; the training tasks received only its train allow-list, and validation/test tasks received only their respective allow-lists.

## D. Frozen configuration

| setting | value |
|---|---|
| objective | `categorical_soft` |
| teacher mapping | `empirical_margin_duration` |
| empirical coefficients | `[0.6477971635, 4.9134275420, -1.2055566868]` |
| sure reward | `0.55` |
| loss multipliers | fixation `1.0`, action `1.0` |
| learning rate | `0.001` |
| training iterations argument | `50000` |
| batch size | `50` |
| recurrent units | `50` |
| recurrent noise | `0.05` |
| activation | rectified linear |
| dt / tau / alpha | `10 ms` / `100 ms` / `0.1` |
| trial length | 300 steps / 3,000 ms |
| stimulus duration | 400–1,190 ms |
| target latency after motion offset | 500–750 ms |
| sure-offer probability | `0.5` |
| readout | mean first 10 post-go steps; categorical action argmax |

No teacher construction, reward, loss, architecture, noise, timing, training length, or behavioral readout setting was changed.

## E. Held-out test results

Every test set contains 150 trials, including 75 sure-offered trials.

| seed | no-sure direction accuracy | offered direction-choice accuracy | P(Sure \| offered) | Left | Right | Sure |
|---:|---:|---:|---:|---:|---:|---:|
| 7 (example) | 0.880 | 0.968 | 0.160 | 0.433 | 0.487 | 0.080 |
| 8 | 0.840 | 0.912 | 0.093 | 0.500 | 0.453 | 0.047 |
| 9 | 0.827 | 0.970 | 0.107 | 0.507 | 0.440 | 0.053 |

The choice proportions are over all held-out trials. Offered direction-choice accuracy conditions on sure being offered and a direction being chosen.

Predicted soft Sure-probability correlations on sure-offered held-out trials:

| seed | teacher sure strength | empirical P(correct) | internal margin | coherence | stimulus duration |
|---:|---:|---:|---:|---:|---:|
| 7 | +0.534 | −0.612 | −0.802 | −0.790 | −0.239 |
| 8 | +0.587 | −0.656 | −0.726 | −0.787 | −0.222 |
| 9 | +0.457 | −0.548 | −0.736 | −0.679 | −0.113 |

Hard P(Sure | offered) by coherence:

| coherence | seed 7 | seed 8 | seed 9 | mean ± SD |
|---:|---:|---:|---:|---:|
| 0.000 | 0.692 | 0.385 | 0.462 | 0.513 ± 0.160 |
| 0.032 | 0.214 | 0.143 | 0.143 | 0.167 ± 0.041 |
| 0.064 | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 0.128 | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 0.256 | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 0.512 | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |

Per-coherence direction accuracy and soft Sure probabilities are preserved in each `test_summary.json`.

## F. Three-seed replication summary

All three networks show both Left and Right choices, both Sure and Direction choices when Sure is offered, decreasing P(Sure) with coherence, negative relationships of predicted P(Sure) with empirical P(correct) and internal margin, and finite outputs. Direction performance and qualitative confidence/value structure therefore replicate.

The principal variation is quantitative: held-out P(Sure | offered) ranges from 0.093 to 0.160, and correlation magnitudes vary across initializations. Stimulus-duration correlations are consistently negative but weaker than margin/coherence relationships. No seed was selected, rejected, or retrained because of these scientific differences; seed 7 remains the predetermined example.

## G. Reproducibility and provenance

- Branch: `feature/model-freeze-validation`
- Commit at training: `63f2b41c51e0f6da462625387745e421948cb8f1`
- Teacher dataset SHA-256: `8ab578794710ab3fd142107f1ad2d14172742e149525ac894c8305b68fbcf94d`
- Split SHA-256: `67d7547a3dd3d244191fbdba1d249ab1959e9be11c432d9524121ed3be77b607`
- Seed 7 checkpoint SHA-256: `eccc658ab2f496a729d391ca3a123bf094871461229676fad4401bfcfb22b243`
- Seed 8 checkpoint SHA-256: `11dbbab7f95aff8a1c2a15a79cba6df3f60a3e0a128c29284f01411779ac5e5e`
- Seed 9 checkpoint SHA-256: `294d7c3362c6a9322fd6a48225e5f2d591c0dfb47e406dd22d75d1c7871d0815`
- Full suite: `83 passed`, with two third-party Python 3.14 deprecation warnings.

Checkpoint reload was checked deterministically on eight held-out inputs per seed at `rtol=atol=1e-7`; all output and state maximum absolute differences were exactly zero.

The worktree was already dirty with untracked prior model-freeze datasets/results. Those artifacts were preserved. During seed-7 artifact finalization, a scalar provenance serialization error occurred after training, checkpointing, validation, test evaluation, CSV writing, and reload-reference writing. The finalization code was fixed and resumed from the intact seed-7 artifacts without retraining. Consequently, seed 7's wall-time fields are unavailable; its weights, held-out records, finite-output checks, hash, and zero-difference reload check are complete. Both exact invocations are recorded in `manifest.json`.

## H. Freeze decision

All three predetermined technical runs completed successfully. All checkpoints are finite, reconstructable, hashed, and paired with the shared split plus per-seed configuration and held-out records. They are ready for downstream population-representation analysis without retraining.

No PCA, choice/confidence-axis analysis, decoding, repeated-stimulus variability analysis, TDR, dPCA, or fixed-point analysis was run in this stage.
