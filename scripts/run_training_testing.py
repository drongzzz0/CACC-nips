from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
VENDOR = ROOT / "vendor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local end-to-end training and testing loop.")
    parser.add_argument(
        "--source-dataset",
        default=PROJECT_ROOT / "Experiment/datasets/sft/gsm8k_subgoals_then_answer.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--dataset",
        default=ROOT / "data/gsm8k_subgoals_then_answer.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--run-name",
        default="inno_experiment_dev_local",
    )
    parser.add_argument(
        "--student-model",
        default=ROOT / "models/qwen2.5_0.5b_instruct",
        type=Path,
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--grad-accum", default=1, type=int)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--max-new-tokens", default=64, type=int)
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_entries = [str(VENDOR)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    return env


def run_step(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_dataset(source: Path, target: Path, refresh: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source dataset: {source}")
    if target.exists() and not refresh:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_final_checkpoint(source_dir: Path, target_file: Path) -> None:
    candidate = source_dir / "model_final.pth"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing final checkpoint in {source_dir}")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, target_file)


def main() -> None:
    args = parse_args()
    env = build_env()
    stage_dataset(args.source_dataset, args.dataset, args.refresh_data)

    checkpoint_dir = ROOT / "checkpoints" / args.run_name
    train_log_output = ROOT / "logs" / f"train_{args.run_name}.json"
    predictions_output = ROOT / "logs" / f"{args.run_name}_predictions.jsonl"
    generation_metrics_output = ROOT / "logs" / f"{args.run_name}_generation_metrics.json"
    report_output = PROJECT_ROOT / "Experiment/analysis/results" / f"{args.run_name}_eval.md"
    summary_output = ROOT / "logs" / f"{args.run_name}_summary.json"
    final_checkpoint = ROOT / "checkpoints/model_final.pth"

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts/train_student.py"),
            "--dataset",
            str(args.dataset),
            "--run-name",
            args.run_name,
            "--student-model",
            str(args.student_model),
            "--backend",
            "peft",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--max-length",
            str(args.max_length),
        ],
        env,
    )

    copy_final_checkpoint(checkpoint_dir, final_checkpoint)

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts/generate_predictions.py"),
            "--dataset",
            str(args.dataset),
            "--adapter-path",
            str(checkpoint_dir),
            "--predictions",
            str(predictions_output),
            "--metrics-output",
            str(generation_metrics_output),
            "--base-model",
            str(args.student_model),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ],
        env,
    )

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
        ],
        env,
    )

    train_manifest = load_json(train_log_output)
    generation_metrics = load_json(generation_metrics_output)
    report_text = report_output.read_text(encoding="utf-8")
    accuracy = 0.0
    for line in report_text.splitlines():
        if line.startswith("- exact-match accuracy:"):
            accuracy = float(line.split(":", maxsplit=1)[1].strip())
            break

    summary = {
        "run_name": args.run_name,
        "dataset": str(args.dataset),
        "source_dataset": str(args.source_dataset),
        "student_model": str(args.student_model),
        "epochs": args.epochs,
        "device": "cuda" if env.get("CUDA_VISIBLE_DEVICES") else "cpu_or_auto",
        "train_manifest": train_manifest,
        "generation_metrics": generation_metrics,
        "evaluation_report": str(report_output),
        "exact_match_accuracy": accuracy,
        "final_checkpoint": str(final_checkpoint),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
