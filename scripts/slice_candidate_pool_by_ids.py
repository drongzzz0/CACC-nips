from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice a candidate pool by the ordered example ids from another jsonl file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ids-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _sha1(values: list[str]) -> str:
    return hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    source_rows = list(read_jsonl(args.ids_source))
    ordered_ids = [str(row["example_id"]) for row in source_rows]
    source_lookup = {str(row["example_id"]): row for row in read_jsonl(args.input)}

    missing_ids = [example_id for example_id in ordered_ids if example_id not in source_lookup]
    if missing_ids and args.strict:
        preview = ", ".join(missing_ids[:8])
        raise ValueError(f"Missing {len(missing_ids)} ids while slicing {args.input}: {preview}")

    output_rows = [source_lookup[example_id] for example_id in ordered_ids if example_id in source_lookup]
    write_jsonl(args.output, output_rows)
    write_json(
        args.report_json,
        {
            "input_path": str(args.input),
            "ids_source": str(args.ids_source),
            "output_path": str(args.output),
            "requested_examples": len(ordered_ids),
            "written_examples": len(output_rows),
            "missing_examples": len(missing_ids),
            "missing_example_ids_preview": missing_ids[:16],
            "requested_ids_sha1": _sha1(ordered_ids),
            "written_ids_sha1": _sha1([str(row["example_id"]) for row in output_rows]),
        },
    )


if __name__ == "__main__":
    main()
