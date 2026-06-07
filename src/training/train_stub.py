from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path

from src.utils.io_utils import read_jsonl, write_json, write_jsonl


@dataclass
class TrainingRunManifest:
    run_name: str
    dataset_path: str
    num_examples: int
    student_model: str
    training_backend: str
    status: str
    created_at: str
    metrics: dict[str, float | int | str] | None
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def create_training_manifest(
    dataset_path: Path,
    run_name: str,
    student_model: str,
    training_backend: str = "stub_manifest_only",
    status: str = "ready_for_peft_integration",
    metrics: dict[str, float | int | str] | None = None,
    notes: list[str] | None = None,
) -> TrainingRunManifest:
    num_examples = sum(1 for _ in read_jsonl(dataset_path))
    return TrainingRunManifest(
        run_name=run_name,
        dataset_path=str(dataset_path),
        num_examples=num_examples,
        student_model=student_model,
        training_backend=training_backend,
        status=status,
        created_at=datetime.now(UTC).isoformat(),
        metrics=metrics,
        notes=notes or [],
    )


def write_training_outputs(
    manifest: TrainingRunManifest,
    log_path: Path,
    checkpoint_dir: Path,
    predictions_path: Path,
) -> None:
    examples = list(read_jsonl(Path(manifest.dataset_path)))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_path, manifest.to_dict())
    write_json(
        checkpoint_dir / "manifest.json",
        manifest.to_dict(),
    )
    write_jsonl(
        predictions_path,
        [
            {
                "example_id": example["example_id"],
                "prediction": "",
                "gold_answer": example["gold_answer"],
            }
            for example in examples
        ],
    )
