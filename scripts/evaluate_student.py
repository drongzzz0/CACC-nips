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

from src.eval.evaluate_predictions import build_markdown_report, compute_exact_match
from src.utils.io_utils import read_jsonl, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a prediction JSONL file.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--run-name", default="unnamed_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.predictions))
    metrics = compute_exact_match(records)
    write_text(args.report, build_markdown_report(args.run_name, metrics))


if __name__ == "__main__":
    main()
