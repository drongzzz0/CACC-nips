from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert XBai raw outputs into Route C prediction JSONL.")
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--xbai-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = list(read_jsonl(args.source_dataset))
    raw_rows = list(read_jsonl(args.xbai_output))
    if len(source_rows) != len(raw_rows):
        raise ValueError(f"Length mismatch: source={len(source_rows)} raw={len(raw_rows)}")
    rows = []
    for source_row, raw_row in zip(source_rows, raw_rows):
        rows.append(
            {
                "example_id": str(source_row["example_id"]),
                "prediction": str(raw_row.get("output", "")),
                "gold_answer": str(source_row["gold_answer"]),
                "answer_mode": str(source_row.get("answer_mode", "numeric")),
            }
        )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
