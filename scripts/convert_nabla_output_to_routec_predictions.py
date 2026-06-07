from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Nabla responses.json into Route C prediction JSONL.")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--responses-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_rows = list(read_jsonl(args.prompt_file))
    response_rows = json.loads(args.responses_json.read_text(encoding="utf-8"))
    rows = []
    for response_row in response_rows:
        index = int(response_row["global_index"])
        source_row = prompt_rows[index]
        responses = response_row.get("responses") or [""]
        rows.append(
            {
                "example_id": str(source_row["example_id"]),
                "prediction": str(responses[0] if responses else ""),
                "gold_answer": str(source_row["gold_answer"]),
                "answer_mode": str(source_row.get("answer_mode", "numeric")),
            }
        )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
