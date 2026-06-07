from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.peft_backend import PeftTrainingConfig, missing_dependencies, run_peft_training
from src.training.train_stub import create_training_manifest, write_training_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PEFT training when dependencies exist, otherwise fail clearly or emit a stub manifest.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--student-model", default="qwen3.5-student-placeholder")
    parser.add_argument(
        "--backend",
        choices=("auto", "peft", "stub"),
        default="auto",
    )
    parser.add_argument(
        "--allow-stub",
        action="store_true",
        help="Permit a manifest-only fallback when PEFT dependencies are unavailable.",
    )
    parser.add_argument(
        "--logs-dir",
        default=ROOT / "logs",
        type=Path,
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=ROOT / "checkpoints",
        type=Path,
    )
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--grad-accum", default=1, type=int)
    parser.add_argument("--max-length", default=512, type=int)
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-device")
    parser.add_argument("--distill-alpha", default=0.0, type=float)
    parser.add_argument("--distill-temperature", default=1.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoints_dir / args.run_name
    log_path = args.logs_dir / f"train_{args.run_name}.json"
    predictions_path = args.logs_dir / f"{args.run_name}_predictions.jsonl"
    requested_backend = args.backend
    missing = missing_dependencies()
    use_stub = requested_backend == "stub" or (requested_backend == "auto" and bool(missing))

    if requested_backend == "peft" and missing:
        raise SystemExit(
            "PEFT backend requested but dependencies are missing: "
            + ", ".join(missing)
        )

    if use_stub and not args.allow_stub:
        raise SystemExit(
            "Training dependencies are missing: "
            + ", ".join(missing)
            + ". Re-run with --allow-stub to emit a manifest-only placeholder, or install the PEFT stack."
        )

    if use_stub:
        manifest = create_training_manifest(
            args.dataset,
            args.run_name,
            args.student_model,
            training_backend="stub_manifest_only",
            status="ready_for_peft_integration",
            metrics=None,
            notes=[
                "No real fine-tuning was run.",
                "Install torch, transformers, datasets, peft, and accelerate to enable the PEFT backend.",
            ],
        )
        write_training_outputs(manifest, log_path, checkpoint_dir, predictions_path)
        return

    result = run_peft_training(
        PeftTrainingConfig(
            dataset_path=args.dataset,
            output_dir=checkpoint_dir,
            run_name=args.run_name,
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            teacher_device=args.teacher_device,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_length=args.max_length,
            distill_alpha=args.distill_alpha,
            distill_temperature=args.distill_temperature,
        )
    )
    manifest = create_training_manifest(
        args.dataset,
        args.run_name,
        args.student_model,
        training_backend=result["training_backend"],
        status=result["status"],
        metrics=result.get("metrics"),
        notes=[
            "Real PEFT fine-tuning completed.",
            *(
                [
                    f"Teacher-logit distillation enabled with teacher_model={args.teacher_model}, distill_alpha={args.distill_alpha}, distill_temperature={args.distill_temperature}."
                ]
                if args.teacher_model and args.distill_alpha > 0.0
                else []
            ),
        ],
    )
    write_training_outputs(manifest, log_path, checkpoint_dir, predictions_path)


if __name__ == "__main__":
    main()
