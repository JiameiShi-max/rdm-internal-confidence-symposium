import argparse
import json
import os
import shlex


PYTHON = "/opt/anaconda3/envs/psychrnn_m1/bin/python"
FIXED_TIMED_DATASET = "stage9b_timed_teacher_dataset_50k_fixed_auto.npz"


def _quote(value):
    return shlex.quote(str(value))


def _command(args):
    return " ".join(_quote(arg) for arg in args)


def default_variants():
    return [
        {
            "name": "timed_reward_057",
            "description": "Raise sure reward from 0.55 to 0.57 to strengthen sure target when internal confidence is low.",
            "teacher_args": ["--sure-reward", "0.57"],
            "dataset_kind": "new_teacher_dataset",
        },
        {
            "name": "timed_contrast_temp012",
            "description": "Lower sure value temperature from 0.15 to 0.12 to sharpen EV-to-sure contrast.",
            "teacher_args": ["--sure-value-temp", "0.12"],
            "dataset_kind": "new_teacher_dataset",
        },
        {
            "name": "timed_contrast_blend090",
            "description": "Increase sure target blend from 0.80 to 0.90 to make the supervised sure target less conservative.",
            "teacher_args": ["--sure-target-blend", "0.9"],
            "dataset_kind": "new_teacher_dataset",
        },
        {
            "name": "timed_student_75k",
            "description": "Keep the fixed timed dataset unchanged and train the student longer.",
            "teacher_args": [],
            "dataset_kind": "existing_fixed_dataset",
            "student_training_iters": 75000,
        },
    ]


def build_command_plan(
    output_root="param_validation_timed",
    seeds=None,
    teacher_training_iters=50000,
    student_training_iters=50000,
):
    seeds = [7] if seeds is None else [int(seed) for seed in seeds]
    commands = []
    variants = []
    for variant in default_variants():
        name = variant["name"]
        variant_dir = os.path.join(output_root, name)
        if variant["dataset_kind"] == "new_teacher_dataset":
            dataset = os.path.join(variant_dir, f"{name}_teacher_dataset.npz")
            teacher_summary = os.path.join(variant_dir, f"{name}_teacher_dataset_summary.json")
            teacher_cmd = [
                "MPLCONFIGDIR=/tmp/mpl",
                PYTHON,
                "run_rdm_timed_internal_teacher.py",
                "--training-iters",
                int(teacher_training_iters),
                "--output",
                dataset,
                "--summary",
                teacher_summary,
            ] + list(variant["teacher_args"])
            commands.append(
                {
                    "variant": name,
                    "kind": "teacher_export",
                    "command": _command(teacher_cmd),
                    "output": dataset,
                }
            )
        else:
            dataset = FIXED_TIMED_DATASET

        variant_seeds = []
        for seed in seeds:
            summary = os.path.join(variant_dir, f"{name}_seed{seed}_summary.json")
            iters = int(variant.get("student_training_iters", student_training_iters))
            student_cmd = [
                "MPLCONFIGDIR=/tmp/mpl",
                PYTHON,
                "sure_target_stage9_internal_readout.py",
                "--dataset",
                dataset,
                "--summary",
                summary,
                "--seed",
                seed,
                "--training-iters",
                iters,
            ]
            commands.append(
                {
                    "variant": name,
                    "kind": "student_train",
                    "seed": int(seed),
                    "command": _command(student_cmd),
                    "output": summary,
                }
            )
            variant_seeds.append({"seed": int(seed), "summary": summary})

        variants.append(
            {
                "name": name,
                "description": variant["description"],
                "dataset": dataset,
                "dataset_kind": variant["dataset_kind"],
                "student_summaries": variant_seeds,
            }
        )

    return {
        "output_root": output_root,
        "seeds": seeds,
        "fixed_timed_dataset": FIXED_TIMED_DATASET,
        "variants": variants,
        "commands": commands,
        "notes": [
            "Run one command at a time in a fresh shell if TensorFlow/Apple Metal appears to hang.",
            "Phase 1 should use a single seed to identify direction; only expand the best variant to 3-5 seeds.",
            "Do not use stage9b_timed_teacher_dataset_50k_auto.npz; it is the old broken timed dataset.",
        ],
    }


def write_command_plan(output_root, plan):
    os.makedirs(output_root, exist_ok=True)
    json_path = os.path.join(output_root, "parameter_validation_command_plan.json")
    md_path = os.path.join(output_root, "parameter_validation_commands.md")
    sh_path = os.path.join(output_root, "parameter_validation_commands.sh")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Timed parameter validation commands\n\n")
        for note in plan["notes"]:
            f.write(f"- {note}\n")
        f.write("\n")
        for item in plan["commands"]:
            f.write(f"## {item['variant']} / {item['kind']}\n\n")
            f.write("```bash\n")
            f.write(item["command"])
            f.write("\n```\n\n")
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        for item in plan["commands"]:
            f.write(f"# {item['variant']} / {item['kind']}\n")
            f.write(item["command"])
            f.write("\n\n")
    return {
        "plan_json": json_path,
        "commands_md": md_path,
        "commands_sh": sh_path,
    }


def _parse_seeds(seed_text):
    return [int(item.strip()) for item in str(seed_text).split(",") if item.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate timed parameter validation commands.")
    parser.add_argument("--output-root", default="param_validation_timed")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--teacher-training-iters", type=int, default=50000)
    parser.add_argument("--student-training-iters", type=int, default=50000)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    plan = build_command_plan(
        output_root=args.output_root,
        seeds=_parse_seeds(args.seeds),
        teacher_training_iters=args.teacher_training_iters,
        student_training_iters=args.student_training_iters,
    )
    paths = write_command_plan(args.output_root, plan)
    print(json.dumps({"outputs": paths, "plan": plan}, indent=2), flush=True)
    return plan


if __name__ == "__main__":
    main()
