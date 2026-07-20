from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
DEFAULT_RUN_DIR = ROOT / "runs" / "minimum_pipeline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimum reproducible experiment-01 pipeline.")
    parser.add_argument(
        "--raw-input",
        default=ROOT / "examples" / "synthetic_reasoning_sample.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_RUN_DIR,
        type=Path,
    )
    parser.add_argument(
        "--run-name",
        default="qwen35_subgoals_pilot",
    )
    parser.add_argument(
        "--student-model",
        default="qwen3.5-student-placeholder",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Allow manifest-only training when PEFT dependencies are unavailable.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "peft", "stub"),
        default="auto",
        help="Training backend passed to train_student.py; use 'stub' for a model-free smoke run.",
    )
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    env = dict(**os.environ)
    pythonpath_entries = [str(VENDOR)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    subprocess.run(cmd, check=True, env=env)


def copy_final_checkpoint(source_dir: Path, target_file: Path) -> None:
    candidate = source_dir / "model_final.pth"
    if candidate.exists():
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_bytes(candidate.read_bytes())


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir
    teacher_output = run_dir / "teacher_traces" / "teacher.jsonl"
    processed_output = run_dir / "processed" / "subgoals.jsonl"
    answer_output = run_dir / "sft" / "answer_only.jsonl"
    brief_output = run_dir / "sft" / "brief_reasoning_then_answer.jsonl"
    filtered_output = run_dir / "sft" / "filtered_cot.jsonl"
    subgoals_output = run_dir / "sft" / "subgoals_then_answer.jsonl"
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    predictions_output = logs_dir / f"{args.run_name}_predictions.jsonl"
    report_output = run_dir / "results" / "experiment_01_eval.md"
    train_log_output = logs_dir / f"train_{args.run_name}.json"
    checkpoint_dir = checkpoints_dir / args.run_name
    final_checkpoint = checkpoints_dir / "model_final.pth"

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts/generate_teacher_traces.py"),
            "--input",
            str(args.raw_input),
            "--output",
            str(teacher_output),
        ]
    )
    run_step(
        [
            sys.executable,
            str(ROOT / "scripts/build_supervision_datasets.py"),
            "--input",
            str(teacher_output),
            "--processed-output",
            str(processed_output),
            "--answer-output",
            str(answer_output),
            "--brief-output",
            str(brief_output),
            "--filtered-output",
            str(filtered_output),
            "--subgoals-output",
            str(subgoals_output),
        ]
    )

    train_cmd = [
        sys.executable,
        str(ROOT / "scripts/train_student.py"),
        "--dataset",
        str(subgoals_output),
        "--run-name",
        args.run_name,
        "--student-model",
        args.student_model,
        "--logs-dir",
        str(logs_dir),
        "--checkpoints-dir",
        str(checkpoints_dir),
        "--backend",
        args.backend,
    ]
    if args.allow_stub:
        train_cmd.append("--allow-stub")
    run_step(train_cmd)

    manifest = load_manifest(train_log_output)
    if manifest.get("training_backend") == "peft_lora":
        run_step(
            [
                sys.executable,
                str(ROOT / "scripts/generate_predictions.py"),
                "--dataset",
                str(subgoals_output),
                "--adapter-path",
                str(checkpoint_dir),
                "--predictions",
                str(predictions_output),
            ]
        )
        copy_final_checkpoint(checkpoint_dir, final_checkpoint)

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_student.py"),
            "--predictions",
            str(predictions_output),
            "--report",
            str(report_output),
            "--run-name",
            args.run_name,
        ]
    )


if __name__ == "__main__":
    main()
