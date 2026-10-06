# Poster-ready supervised-RNN summary

### Question

Do recurrent networks trained for post-decision wagering develop an internal confidence representation that predicts later Sure choices beyond differences in external stimuli?

### Approach

Three frozen 50-unit categorical RNNs were evaluated on a fixed held-out split. TRAIN-defined choice and empirical-P(correct) axes were tracked through time, then each sure-offered TEST input was replayed 100 times with independent recurrent noise to isolate within-stimulus state variability.

### Key findings

- Held-out no-Sure direction accuracy was 0.827–0.880; P(Sure | offered) was non-degenerate (0.093–0.160) and decreased with coherence.
- Population activity contained approximately distinct choice and confidence dimensions: choice CV accuracy was 0.995–0.997, confidence held-out r was 0.740–0.776, and absolute axis cosine was 0.040–0.133.
- Confidence information became operationally decodable 80–180 ms after motion onset and remained correlated with empirical P(correct) before Sure-target onset (r = 0.590–0.665).
- Under identical input, lower-confidence fluctuations predicted later Sure choice in every seed (β = -0.581, -0.391, -0.336; all permutation p = 0.001).

### Conclusion

The networks developed low-overlap linear dimensions related to directional choice and graded confidence. Confidence-related population activity preceded wager availability, and endogenous movement toward lower-confidence states predicted a greater probability of later Sure wagering under identical sensory input.

### Limitation

These results are representational and predictive, not causal. The Stage-3 confidence effect remained negative after controlling absolute choice-axis magnitude in seeds 7 and 9 but not seed 8, so specificity was not uniform across networks.
