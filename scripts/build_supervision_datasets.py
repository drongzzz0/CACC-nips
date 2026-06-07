from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transform_supervision import (
    build_answer_only_example,
    build_brief_reasoning_then_answer_example,
    build_filtered_cot_example,
    build_processed_record,
    build_subgoals_then_answer_example,
)
from src.utils.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed traces and SFT datasets.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--processed-output", required=True, type=Path)
    parser.add_argument("--answer-output", type=Path)
    parser.add_argument("--brief-output", type=Path)
    parser.add_argument("--filtered-output", required=True, type=Path)
    parser.add_argument("--subgoals-output", required=True, type=Path)
    parser.add_argument("--valid-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_records = read_jsonl(args.input)
    if args.valid_only:
        input_records = [record for record in input_records if bool(record.get("trace_valid", True))]
    processed_records = [build_processed_record(record) for record in input_records]
    write_jsonl(args.processed_output, [record.to_dict() for record in processed_records])
    if args.answer_output:
        write_jsonl(args.answer_output, [build_answer_only_example(record).to_dict() for record in processed_records])
    if args.brief_output:
        write_jsonl(args.brief_output, [build_brief_reasoning_then_answer_example(record).to_dict() for record in processed_records])
    write_jsonl(args.filtered_output, [build_filtered_cot_example(record).to_dict() for record in processed_records])
    write_jsonl(args.subgoals_output, [build_subgoals_then_answer_example(record).to_dict() for record in processed_records])


if __name__ == "__main__":
    main()
