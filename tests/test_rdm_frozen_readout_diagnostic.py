import numpy as np
from sklearn.model_selection import StratifiedKFold

from rdm_frozen_readout_diagnostic import (
    cross_validated_action_probe,
    cross_validated_phase_probes,
    softmax,
)


def test_held_out_linear_probe_recovers_synthetic_soft_policy():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(150, 4))
    weights = np.array(
        [
            [1.0, -0.7, 0.1],
            [-0.4, 0.8, -0.2],
            [0.2, -0.3, 0.7],
            [0.5, 0.1, -0.6],
        ]
    )
    targets = softmax(features @ weights)
    preferred = np.argmax(targets, axis=1)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    folds = list(splitter.split(features, preferred))

    predictions = cross_validated_action_probe(features, targets, folds)

    assert np.all(np.isfinite(predictions))
    assert np.allclose(np.sum(predictions, axis=1), 1.0)
    assert np.mean(np.abs(predictions - targets)) < 0.01


def test_phase_and_fixation_probes_use_held_out_trial_folds():
    rng = np.random.default_rng(11)
    n_trials = 100
    shared = rng.normal(size=(n_trials, 5))
    pre = shared.copy()
    post = shared.copy()
    pre[:, 0] += 3.0
    post[:, 0] -= 3.0
    strata = np.tile(np.arange(2), n_trials // 2)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    folds = list(splitter.split(np.arange(n_trials), strata))

    result = cross_validated_phase_probes(pre, post, folds)

    assert result["phase_accuracy"] > 0.99
    assert result["phase_auc"] > 0.99
    assert result["fixation_rmse"] < 0.20
    assert np.all(result["fold_id"] >= 0)
