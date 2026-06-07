from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.teacher_pipeline import TeacherGeneratorConfig, generate_teacher_records
from src.utils.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher traces from a JSONL benchmark shard.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-model", default="qwen3.5-placeholder-teacher")
    parser.add_argument("--backend", choices=("template", "hf"), default="template")
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--do-sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TeacherGeneratorConfig(
        teacher_model=args.teacher_model,
        backend=args.backend,
        reasoning_style="generated" if args.backend == "hf" else "template",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    records = [record.to_dict() for record in generate_teacher_records(read_jsonl(args.input), config)]
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
