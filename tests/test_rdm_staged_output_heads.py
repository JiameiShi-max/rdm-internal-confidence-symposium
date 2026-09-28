import numpy as np

from rdm_staged_output_heads import (
    fit_action_head_from_samples,
    fit_fixation_head_from_samples,
    fixed_train_test_split,
    load_head,
    predict_action_head,
    predict_fixation_head,
    save_head,
    sha256,
    softmax,
)


def synthetic_samples():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(120, 5))
    fixation = (features[:, 0] + 0.2 * features[:, 1] > 0).astype(float)
    action_weights = rng.normal(scale=0.5, size=(5, 3))
    actions = softmax(features @ action_weights)
    return features, fixation, actions


def test_head_fits_leave_backbone_checkpoint_unchanged(tmp_path):
    checkpoint = tmp_path / "backbone.npz"
    np.savez(checkpoint, W_rec=np.eye(5), W_in=np.ones((2, 5)))
    before = sha256(checkpoint)
    features, fixation, actions = synthetic_samples()

    fit_fixation_head_from_samples(features, fixation)
    fit_action_head_from_samples(features, actions)

    assert sha256(checkpoint) == before


def test_only_independent_head_parameters_are_returned():
    features, fixation, actions = synthetic_samples()
    fixation_head = fit_fixation_head_from_samples(features, fixation)
    action_head = fit_action_head_from_samples(features, actions)

    assert set(fixation_head) == {"weights", "bias", "mean", "scale", "l2"}
    assert set(action_head) == {"weights", "bias", "mean", "scale", "l2"}
    assert fixation_head["weights"].shape == (features.shape[1],)
    assert action_head["weights"].shape == (features.shape[1], 3)


def test_fixation_is_independent_and_never_enters_action_softmax():
    features, fixation, actions = synthetic_samples()
    fixation_head = fit_fixation_head_from_samples(features, fixation)
    action_head = fit_action_head_from_samples(features, actions)
    action_before = predict_action_head(features, action_head)

    fixation_head["weights"][:] = 1e6
    fixation_head["bias"] = np.asarray(-1e6)
    action_after = predict_action_head(features, action_head)

    assert np.array_equal(action_before, action_after)
    assert np.allclose(np.sum(action_after, axis=1), 1.0)
    assert predict_fixation_head(features, fixation_head).shape == (len(features),)


def test_fixed_split_has_no_trial_overlap():
    train, test = fixed_train_test_split(1000)

    assert np.array_equal(test, np.arange(400))
    assert np.array_equal(train, np.arange(400, 1000))
    assert np.intersect1d(train, test).size == 0


def test_saved_heads_reconstruct_identical_predictions(tmp_path):
    features, fixation, actions = synthetic_samples()
    fixation_head = fit_fixation_head_from_samples(features, fixation)
    action_head = fit_action_head_from_samples(features, actions)
    fixation_path = tmp_path / "fixation.npz"
    action_path = tmp_path / "action.npz"

    save_head(fixation_path, fixation_head)
    save_head(action_path, action_head)

    assert np.array_equal(
        predict_fixation_head(features, fixation_head),
        predict_fixation_head(features, load_head(fixation_path)),
    )
    assert np.array_equal(
        predict_action_head(features, action_head),
        predict_action_head(features, load_head(action_path)),
    )
