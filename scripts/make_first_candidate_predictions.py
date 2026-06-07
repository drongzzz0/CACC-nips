from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a candidate-set JSONL file into first-candidate prediction JSONL."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = []
    for row in read_jsonl(args.candidates):
        candidates = [str(candidate) for candidate in row["candidates"]]
        prediction = candidates[0] if candidates else ""
        records.append(
            {
                "example_id": str(row["example_id"]),
                "prediction": prediction,
                "gold_answer": str(row["gold_answer"]),
                "answer_mode": str(row.get("answer_mode", "numeric")),
            }
        )

    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
