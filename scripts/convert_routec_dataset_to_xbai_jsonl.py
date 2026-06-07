from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Route C benchmark JSONL into XBai JSONL format.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for record in read_jsonl(args.input):
        rows.append(
            {
                "prompt": str(record["problem"]),
                "answer": str(record["gold_answer"]),
                "example_id": str(record["example_id"]),
                "answer_mode": str(record.get("answer_mode", "numeric")),
            }
        )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
