from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

CORE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from src.utils.io_utils import write_jsonl  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a reproducible GPQA repeat/neighboring slice from the in-project raw train split.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Experiment/datasets/raw/gpqa_diamond_train_v1.jsonl"),
        help="Project-local GPQA raw jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Experiment/datasets/raw/gpqa_diamond_train_128_seed11_v1.jsonl"),
        help="Output slice jsonl path.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("Experiment/datasets/raw/gpqa_diamond_train_128_seed11_v1_manifest.json"),
        help="Output manifest json path.",
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tag", default="routec_plus_gpqa_repeat_confirmatory")
    return parser.parse_args()


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    input_path = _project_path(args.input)
    output_path = _project_path(args.output)
    manifest_path = _project_path(args.manifest_output)

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.size > len(rows):
        raise ValueError(f"requested size={args.size} exceeds source size={len(rows)}")

    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(rows)), k=args.size))
    selected_rows = [rows[idx] for idx in selected_indices]
    selected_ids = [str(row["example_id"]) for row in selected_rows]

    write_jsonl(output_path, selected_rows)

    payload = {
        "tag": args.tag,
        "source_path": str(args.input),
        "output_path": str(args.output),
        "seed": args.seed,
        "size": args.size,
        "source_total_examples": len(rows),
        "selected_indices": selected_indices,
        "selected_ids": selected_ids,
        "selected_ids_sha1": hashlib.sha1("\n".join(selected_ids).encode("utf-8")).hexdigest(),
        "source_ids_sha1": hashlib.sha1("\n".join(str(row["example_id"]) for row in rows).encode("utf-8")).hexdigest(),
        "note": "Prepared inside the project repo to support the next Route C+ GPQA second-family repeat/neighboring-slice confirmatory without relying on missing historical slice files.",
    }
    _write_json_atomic(manifest_path, payload)


if __name__ == "__main__":
    main()
