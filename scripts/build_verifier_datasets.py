from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import SFTExample, VerifierCandidateSet
from src.generation.prompts import build_verifier_prompt
from src.utils.io_utils import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verifier yes/no train data and candidate-set eval data.")
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--eval-input", required=True, type=Path)
    parser.add_argument("--answer-pool", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--eval-output", required=True, type=Path)
    parser.add_argument("--negatives-per-example", default=3, type=int)
    parser.add_argument("--seed", default=7, type=int)
    return parser.parse_args()


def _load_gold_answers(path: Path) -> list[str]:
    answers = []
    for record in read_jsonl(path):
        answer = record.get("gold_answer")
        if answer is None and "answer" in record:
            answer = str(record["answer"]).split("####")[-1].strip()
        if answer is not None:
            answers.append(str(answer))
    return answers


def _sample_negatives(
    rng: random.Random,
    answer_pool: list[str],
    gold_answer: str,
    count: int,
) -> list[str]:
    negatives = []
    seen = {gold_answer}
    while len(negatives) < count:
        candidate = rng.choice(answer_pool)
        if candidate in seen:
            continue
        seen.add(candidate)
        negatives.append(candidate)
    return negatives


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    answer_pool = _load_gold_answers(args.answer_pool)
    train_records = list(read_jsonl(args.train_input))
    eval_records = list(read_jsonl(args.eval_input))

    train_examples: list[dict] = []
    for record in train_records:
        gold_answer = str(record["gold_answer"])
        candidates = [gold_answer, *_sample_negatives(rng, answer_pool, gold_answer, args.negatives_per_example)]
        rng.shuffle(candidates)
        for candidate in candidates:
            label = "yes" if candidate == gold_answer else "no"
            train_examples.append(
                SFTExample(
                    example_id=f"{record['example_id']}::{candidate}",
                    dataset=record["dataset"],
                    prompt=build_verifier_prompt(record["problem"], candidate),
                    response=label,
                    supervision_type="verifier_yes_no",
                    gold_answer=gold_answer,
                ).to_dict()
            )

    eval_candidate_sets: list[dict] = []
    for record in eval_records:
        gold_answer = str(record["gold_answer"])
        candidates = [gold_answer, *_sample_negatives(rng, answer_pool, gold_answer, args.negatives_per_example)]
        rng.shuffle(candidates)
        eval_candidate_sets.append(
            VerifierCandidateSet(
                example_id=record["example_id"],
                dataset=record["dataset"],
                problem=record["problem"],
                gold_answer=gold_answer,
                candidates=candidates,
            ).to_dict()
        )

    write_jsonl(args.train_output, train_examples)
    write_jsonl(args.eval_output, eval_candidate_sets)


if __name__ == "__main__":
    main()
