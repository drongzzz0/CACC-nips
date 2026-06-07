from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import build_markdown_report, compute_exact_match
from src.utils.io_utils import read_jsonl, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate verifier candidate-selection predictions.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.predictions))
    metrics = compute_exact_match(records)
    report = build_markdown_report(args.run_name, metrics)
    report += "\nVerifier setting: top-ranked candidate answer selected from a fixed candidate set.\n"
    write_text(args.report, report)


if __name__ == "__main__":
    main()
