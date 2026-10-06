# Population Representation Stage 1

## A. State extraction

Each frozen PsychRNN checkpoint was loaded read-only into `BasicSimulator` with `alpha=0.1`, ReLU dynamics, and `rec_noise=0`. The exact 1,000 saved teacher-dataset input tensors were replayed in source-trial order; no task trials were regenerated. Hidden trajectories have 50 units and shape `1000 × 300 × 50` per seed. Raw and categorical outputs, trial metadata, and reconstructable input references were saved. Checkpoint hashes were identical before and after analysis.

## B. Primary analysis window

`delay_end` is the go cue and post-go behavioral-readout onset. The primary state is the mean of recurrent states over `[delay_end-10, delay_end)`, exactly 100 ms at `dt=10 ms`, excluding post-go activity. Motion onset is `fixation_end`, motion offset is `stimulus_end`, and sure-target onset is `ts_onset` when `sure_available` is true.

## C. PCA

PCA was fit separately for each seed using standardized TRAIN pre-go states only. PCA is used for visualization, not as the inferential test.

| seed | variance PC1+PC2 |
|---:|---:|
| 7 | 0.866 |
| 8 | 0.855 |
| 9 | 0.775 |

The seed-7 figures show choice- and confidence-related geometry in the held-out projection, but PCA component separation alone is not treated as evidence for a confidence representation.

## D. Choice representation

The L2 logistic decoder used only TRAIN trials with deterministic Left or Right choices; Sure choices were excluded. Five-fold stratified cross-validation used fixed `C=1`.

| seed | CV accuracy mean ± SD | CV AUC mean ± SD |
|---:|---:|---:|
| 7 | 0.995 ± 0.004 | 1.000 ± 0.000 |
| 8 | 0.997 ± 0.004 | 1.000 ± 0.000 |
| 9 | 0.997 ± 0.004 | 1.000 ± 0.000 |

## E. Confidence representation

The primary confidence axis used Ridge regression (`alpha=1`) from TRAIN pre-go states to empirical P(correct), with no Sure/Direction labels. Internal margin was analyzed with the identical complementary procedure.

| seed | CV Pearson r | CV Spearman rho | CV R² | held-out correlation |
|---:|---:|---:|---:|---:|
| 7 | 0.768 | 0.861 | 0.583 | 0.776 |
| 8 | 0.755 | 0.820 | 0.569 | 0.749 |
| 9 | 0.748 | 0.816 | 0.557 | 0.740 |

The complementary internal-margin decoder was also fixed in advance and used the same folds and Ridge regularization:

| seed | CV Pearson r | CV Spearman rho | CV R² | held-out correlation |
|---:|---:|---:|---:|---:|
| 7 | 0.908 | 0.916 | 0.824 | 0.913 |
| 8 | 0.894 | 0.902 | 0.799 | 0.892 |
| 9 | 0.893 | 0.904 | 0.797 | 0.896 |

## F. Wager prediction

The TRAIN-defined confidence axis was evaluated among sure-offered TEST trials. Higher projection is oriented toward higher TRAIN empirical P(correct); orientation never used test wagering.

| seed | n Sure / Direction | mean Sure | mean Direction | Sure − Direction [bootstrap 95% CI] | logistic coefficient ± SE |
|---:|---:|---:|---:|---:|---:|
| 7 | 9 / 66 | -0.358 | -0.019 | -0.338 [-0.383, -0.290] | -21.000 ± 9.334 |
| 8 | 9 / 66 | -0.479 | -0.034 | -0.445 [-0.516, -0.375] | -7.238 ± 2.381 |
| 9 | 8 / 67 | -0.564 | -0.044 | -0.520 [-0.590, -0.447] | -10.473 ± 4.104 |

Sample sizes are small for later Sure choices, so these estimates are presented as associations with uncertainty, not causal effects.

## G. Geometry

The independently fit choice and empirical-confidence axes were not orthogonalized.

| seed | cosine | absolute cosine | angle |
|---:|---:|---:|---:|
| 7 | -0.133 | 0.133 | 97.6° |
| 8 | -0.040 | 0.040 | 92.3° |
| 9 | -0.090 | 0.090 | 95.1° |

## H. Cross-seed replication

Identical extraction, preprocessing, folds, fixed regularization, permutation count, bootstrap count, and plotting definitions were applied to seeds 7, 8, and 9. Seed 7 was retained as the predetermined example regardless of result strength.

## I. Leakage checks

- Scaler statistics and PCA were fit only on the 700 TRAIN identities.
- Choice and confidence decoders were fit and cross-validated only within TRAIN.
- TEST states and labels entered only final held-out transformations and metrics.
- The confidence target was continuous empirical P(correct); Sure-vs-Direction labels were never used to fit or orient it.
- Checkpoint SHA-256 values were unchanged before versus after replay.
- The analysis imports only the inference simulator and contains no network training call.

## J. Artifact paths

Each `seedN/` directory contains `state_metadata.csv`, full deterministic `state_trajectories.npz`, `pre_go_states.npz`, `scaler.npz`, `pca.npz`, both axes, CV metrics, and held-out metrics. Seed 7 additionally contains the detailed figures. Cross-seed tables and the replication figure are under `multiseed/`; provenance, paths, frozen-checkpoint hashes, and analysis constants are in `analysis_manifest.json`.

## Supported conclusion and next stage

The results support that frozen pre-go population activity contains strongly cross-validated linear information about directional choice and teacher-derived confidence, and that both representations generalize to held-out trials. The independently TRAIN-defined confidence projection is lower for later Sure choices in all three networks, while choice and confidence axes have low absolute cosine similarity (`0.040–0.133`). Thus the population state contains approximately distinct choice- and confidence-related dimensions, and the confidence-related dimension predicts later wagering without having been fit to wagering labels. Decoding and association do not establish causal use.

Time-resolved decoding, recurrent-noise variability, pre-TS variability, TDR, dPCA, and fixed-point analysis were not performed and remain separate future stages.
