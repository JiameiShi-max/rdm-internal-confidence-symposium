from rdm_parameter_validation_plan import build_command_plan


def test_build_command_plan_uses_fixed_timed_dataset_and_variant_outputs():
    plan = build_command_plan(
        output_root="param_validation_timed",
        seeds=[7],
        teacher_training_iters=50000,
        student_training_iters=50000,
    )

    joined = "\n".join(item["command"] for item in plan["commands"])
    assert "stage9b_timed_teacher_dataset_50k_auto.npz" not in joined
    assert "stage9b_timed_teacher_dataset_50k_fixed_auto.npz" in joined
    assert "param_validation_timed/timed_reward_057/timed_reward_057_teacher_dataset.npz" in joined
    assert "--sure-reward 0.57" in joined
    assert "--sure-value-temp 0.12" in joined
    assert "--sure-target-blend 0.9" in joined


def test_build_command_plan_keeps_each_training_call_as_independent_process():
    plan = build_command_plan(
        output_root="param_validation_timed",
        seeds=[7, 9],
        teacher_training_iters=50000,
        student_training_iters=50000,
    )

    train_commands = [
        item["command"]
        for item in plan["commands"]
        if item["kind"] in {"teacher_export", "student_train"}
    ]
    assert train_commands
    assert all(" && " not in command for command in train_commands)
    assert all("rdm_multiseed_validation.py --model timed --seeds 7,9" not in command for command in train_commands)


def main():
    test_build_command_plan_uses_fixed_timed_dataset_and_variant_outputs()
    test_build_command_plan_keeps_each_training_call_as_independent_process()
    print("rdm parameter validation plan test passed")


if __name__ == "__main__":
    main()
