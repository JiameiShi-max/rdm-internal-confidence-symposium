import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_direction_success_proxy(sensory_margin, margin_center=2.0, margin_temp=0.70):
    sensory_margin = float(max(0.0, sensory_margin))
    margin_temp = max(float(margin_temp), 1e-6)
    return float(sigmoid((sensory_margin - float(margin_center)) / margin_temp))


def compute_soft_sure_teacher_from_margin(
    margin,
    sure_available,
    margin_center=2.0,
    margin_temp=0.70,
    direction_reward=1.0,
    error_reward=0.0,
    sure_reward=0.55,
    sure_value_temp=0.15,
    sure_target_min=0.05,
    sure_target_max=0.95,
    sure_target_blend=0.80,
    margin_source="sensory",
):
    teacher_margin = float(max(0.0, margin))
    direction_success_proxy = compute_direction_success_proxy(
        sensory_margin=teacher_margin,
        margin_center=margin_center,
        margin_temp=margin_temp,
    )
    expected_direction_value = (
        float(direction_reward) * direction_success_proxy
        + float(error_reward) * (1.0 - direction_success_proxy)
    )
    sure_advantage = float(sure_reward) - expected_direction_value

    if sure_available:
        raw_sure_strength = float(sigmoid(sure_advantage / max(float(sure_value_temp), 1e-6)))
        blended_sure_strength = 0.5 + float(sure_target_blend) * (raw_sure_strength - 0.5)
        sure_strength = float(
            np.clip(blended_sure_strength, float(sure_target_min), float(sure_target_max))
        )
    else:
        raw_sure_strength = 0.0
        sure_strength = 0.0

    return {
        "teacher_margin": float(teacher_margin),
        "margin_source": str(margin_source),
        "direction_success_proxy": float(direction_success_proxy),
        "expected_direction_value": float(expected_direction_value),
        "sure_advantage": float(sure_advantage),
        "raw_sure_strength": float(raw_sure_strength),
        "sure_strength": float(sure_strength),
    }


def compute_soft_sure_teacher(
    sensory_margin,
    sure_available,
    margin_center=2.0,
    margin_temp=0.70,
    direction_reward=1.0,
    error_reward=0.0,
    sure_reward=0.55,
    sure_value_temp=0.15,
    sure_target_min=0.05,
    sure_target_max=0.95,
    sure_target_blend=0.80,
):
    return compute_soft_sure_teacher_from_margin(
        margin=sensory_margin,
        sure_available=sure_available,
        margin_center=margin_center,
        margin_temp=margin_temp,
        direction_reward=direction_reward,
        error_reward=error_reward,
        sure_reward=sure_reward,
        sure_value_temp=sure_value_temp,
        sure_target_min=sure_target_min,
        sure_target_max=sure_target_max,
        sure_target_blend=sure_target_blend,
        margin_source="sensory",
    )
