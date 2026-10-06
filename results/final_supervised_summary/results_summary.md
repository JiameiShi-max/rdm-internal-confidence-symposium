# Final supervised-RNN results

## 1. Supervised RNN reproduces confidence-dependent wagering behavior

Three categorical recurrent networks were evaluated on the same 150 held-out TEST trials. No-Sure direction accuracy was 0.880, 0.840, and 0.827 for seeds 7–9. When Sure was offered, the networks chose it on 0.160, 0.093, and 0.107 of trials. Hard P(Sure) decreased from 0.385–0.692 at zero coherence to zero at coherence ≥0.064. Predicted Sure probability was negatively correlated with empirical P(correct) (r = −0.548 to −0.656) and internal margin (r = −0.726 to −0.802). Thus wagering was confidence-dependent and non-degenerate, but its quantitative policy did not exactly recover the teacher.

## 2. Recurrent population activity contains separable choice- and confidence-related dimensions

Stage 1 used deterministic pre-go states and TRAIN-only linear models. Choice-decoder cross-validation accuracy was 0.995–0.997. Confidence decoding reached CV Pearson r = 0.748–0.768 and held-out r = 0.740–0.776. Absolute choice–confidence axis cosine was only 0.040–0.133. The independently TRAIN-defined confidence projection was lower for later Sure than Direction choices in every seed. Population activity therefore contained approximately distinct, low-overlap choice- and confidence-related linear dimensions; the axes were not mathematically orthogonalized and the association does not establish causal use.

## 3. Confidence information emerges during evidence accumulation before wager availability

Motion-onset-aligned Stage-2 decoding used a prospective 50-ms sustained-run criterion and a family-wise permutation threshold. Operational choice onsets were 80, 80, and 30 ms; confidence onsets were 140, 180, and 80 ms. Immediately before Sure-target onset, the fixed confidence-axis correlation with empirical P(correct) was 0.590, 0.615, and 0.665. Confidence-related population structure was therefore present during evidence accumulation and before the wagering option appeared. These are operational information-availability estimates, not biological or causal latencies.

## 4. Endogenous confidence-state variability predicts wagering under identical sensory input

Stage 3 replayed each of 75 sure-offered TEST inputs 100 times while changing only the recurrent-noise realization. After within-stimulus centering, lower pre-target confidence projection predicted later Sure choice in all networks: seed 7 β = -0.581, 95% bootstrap CI [-0.744, -0.441]; seed 8 β = -0.391 [-0.572, -0.219]; seed 9 β = -0.336 [-0.481, -0.222]. Every within-stimulus permutation p was 0.001. Accordingly, the same exact external input combined with a different recurrent-noise realization produced different internal confidence states associated with different probabilities of later Sure choice. This is a stimulus-controlled predictive association, not a causal effect.

## 5. Limitations

The student networks captured the qualitative confidence-dependent wagering pattern but did not exactly reproduce the teacher policy. Linear axes and decoding quantify information and association rather than causal implementation. Most importantly, after adding within-stimulus absolute choice-axis magnitude to the Stage-3 model, the confidence coefficient was negative for seeds 7 and 9 (-0.266, -0.295) but positive for seed 8 (0.231). The primary confidence–wager association replicated across all three networks, but its specificity relative to absolute choice-axis magnitude was not uniform across seeds.

## Conclusion

A supervised recurrent neural network trained to perform confidence-dependent post-decision wagering developed approximately distinct population dimensions related to directional choice and graded confidence. Confidence-related activity emerged during evidence accumulation and was already present before the wagering option appeared. Under identical external input, endogenous fluctuations toward lower-confidence states predicted a greater probability of later Sure choice across all three trained networks.

These analyses establish robust representational and predictive associations within the trained RNN, but do not demonstrate that the identified confidence axis is a causal decision variable. Specificity relative to absolute choice-axis magnitude was not uniform across seeds.
