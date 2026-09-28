import numpy as np


DUAL_HEAD_LOSS_NAME = "dual_head_task_aligned"
REGULARIZATION_KEYS = (
    "L1_in",
    "L1_rec",
    "L1_out",
    "L2_in",
    "L2_rec",
    "L2_out",
    "L2_firing_rate",
)


def dual_head_loss_components(predictions, y, output_mask):
    """Return independently normalized fixation BCE and post-go action CE."""
    import tensorflow as tf

    dtype = predictions.dtype
    fixation_logits = predictions[:, :, 0]
    fixation_targets = y[:, :, 0]
    fixation_active = tf.cast(output_mask[:, :, 0] > 0, dtype)
    fixation_point_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=fixation_targets,
        logits=fixation_logits,
    )
    fixation_loss = tf.math.divide_no_nan(
        tf.reduce_sum(input_tensor=fixation_active * fixation_point_loss),
        tf.reduce_sum(input_tensor=fixation_active),
    )

    action_targets = y[:, :, 1:4]
    action_active = tf.cast(
        tf.reduce_sum(input_tensor=action_targets, axis=2) > 1.0 - 1e-6,
        dtype,
    )
    action_point_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=action_targets,
        logits=predictions[:, :, 1:4],
    )
    action_loss = tf.math.divide_no_nan(
        tf.reduce_sum(input_tensor=action_active * action_point_loss),
        tf.reduce_sum(input_tensor=action_active),
    )
    return fixation_loss, action_loss


def dual_head_loss(predictions, y, output_mask):
    fixation_loss, action_loss = dual_head_loss_components(predictions, y, output_mask)
    return fixation_loss + action_loss


def configure_dual_head_objective(network_params):
    network_params["loss_function"] = DUAL_HEAD_LOSS_NAME
    network_params[DUAL_HEAD_LOSS_NAME] = dual_head_loss
    return network_params


def regularization_configuration(network_params):
    return {key: float(network_params.get(key, 0.0)) for key in REGULARIZATION_KEYS}


def sigmoid(logits):
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def dual_head_probabilities(raw_outputs):
    """Transform independent raw heads; fixation never enters the action softmax."""
    raw_outputs = np.asarray(raw_outputs, dtype=float)
    return sigmoid(raw_outputs[..., 0]), softmax(raw_outputs[..., 1:4])


def dual_head_loss_numpy(raw_outputs, targets, output_mask):
    """Reference implementation used by tests and audit output."""
    raw_outputs = np.asarray(raw_outputs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    output_mask = np.asarray(output_mask, dtype=float)
    fixation_target = targets[..., 0]
    fixation_active = output_mask[..., 0] > 0
    fixation_probability = sigmoid(raw_outputs[..., 0])
    eps = 1e-12
    fixation_point_loss = -(
        fixation_target * np.log(np.maximum(fixation_probability, eps))
        + (1.0 - fixation_target)
        * np.log(np.maximum(1.0 - fixation_probability, eps))
    )
    fixation_loss = float(np.mean(fixation_point_loss[fixation_active]))

    action_targets = targets[..., 1:4]
    action_active = np.sum(action_targets, axis=-1) > 1.0 - 1e-6
    action_probability = softmax(raw_outputs[..., 1:4])
    action_point_loss = -np.sum(
        action_targets * np.log(np.maximum(action_probability, eps)), axis=-1
    )
    action_loss = float(np.mean(action_point_loss[action_active]))
    return {
        "fixation_bce": fixation_loss,
        "action_soft_ce": action_loss,
        "total": fixation_loss + action_loss,
        "n_fixation_active": int(np.sum(fixation_active)),
        "n_action_active": int(np.sum(action_active)),
    }


def split_output_head_weights(weights):
    """Extract disjoint fixation and LEFT/RIGHT/SURE parameter slices."""
    output_weights = np.asarray(weights["W_out"])
    output_bias = np.asarray(weights["b_out"])
    if output_weights.shape[0] != 4 or output_bias.shape != (4,):
        raise ValueError("Dual-head checkpoint must have 1 fixation + 3 action outputs")
    fixation = {
        "W_fix": output_weights[0].copy(),
        "b_fix": np.asarray(output_bias[0]).copy(),
    }
    action = {
        "W_action": output_weights[1:4].copy(),
        "b_action": output_bias[1:4].copy(),
        "action_order": np.asarray(["LEFT", "RIGHT", "SURE"]),
    }
    return fixation, action
