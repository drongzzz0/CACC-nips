from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import write_jsonl


TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a reproducible GSM8K train/eval slice.")
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--eval-output", required=True, type=Path)
    parser.add_argument("--train-size", default=32, type=int)
    parser.add_argument("--eval-size", default=16, type=int)
    parser.add_argument("--seed", default=7, type=int)
    return parser.parse_args()


def _load_from_raw_jsonl(url: str) -> list[dict]:
    with urlopen(url, timeout=60) as response:
        payload = response.read().decode("utf-8")
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _extract_gold_answer(answer: str) -> str:
    return answer.split("####")[-1].strip()


def main() -> None:
    args = parse_args()
    try:
        train_dataset = _load_from_raw_jsonl(TRAIN_URL)
        test_dataset = _load_from_raw_jsonl(TEST_URL)
    except Exception:
        from datasets import load_dataset

        train_dataset = list(load_dataset("openai/gsm8k", "main", split="train"))
        test_dataset = list(load_dataset("openai/gsm8k", "main", split="test"))

    rng = random.Random(args.seed)
    train_indices = rng.sample(range(len(train_dataset)), k=args.train_size)
    eval_indices = rng.sample(range(len(test_dataset)), k=args.eval_size)

    train_records = [
        {
            "example_id": f"gsm8k-train-{idx:05d}",
            "dataset": "gsm8k",
            "problem": train_dataset[idx]["question"],
            "gold_answer": _extract_gold_answer(train_dataset[idx]["answer"]),
        }
        for idx in train_indices
    ]
    eval_records = [
        {
            "example_id": f"gsm8k-test-{idx:05d}",
            "dataset": "gsm8k",
            "problem": test_dataset[idx]["question"],
            "gold_answer": _extract_gold_answer(test_dataset[idx]["answer"]),
        }
        for idx in eval_indices
    ]

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.eval_output, eval_records)


if __name__ == "__main__":
    main()
