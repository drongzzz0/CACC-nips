from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import answers_match, extract_choice_answer, extract_numeric_answer, normalize_answer
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two candidate pools into a single heterogeneous pool using an anchor-first replacement policy."
    )
    parser.add_argument("--base-candidates", required=True, type=Path)
    parser.add_argument("--anchor-candidates", required=True, type=Path)
    parser.add_argument("--aux-candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--max-candidates", default=8, type=int)
    parser.add_argument("--protect-prefix-candidates", default=1, type=int)
    parser.add_argument("--max-aux-insertions", default=3, type=int)
    parser.add_argument(
        "--dedupe-mode",
        default="numeric_or_text",
        choices=("text", "numeric_or_text"),
    )
    return parser.parse_args()


def _candidate_key(text: str, answer_mode: str, dedupe_mode: str) -> str:
    stripped = str(text).strip()
    if dedupe_mode == "text":
        return f"text:{normalize_answer(stripped)}"
    if answer_mode == "choice_letter":
        choice = extract_choice_answer(stripped)
        if choice is not None:
            return f"choice:{choice}"
    else:
        numeric = extract_numeric_answer(stripped)
        if numeric is not None:
            return f"num:{numeric}"
    return f"text:{normalize_answer(stripped)}"


def _is_correct(candidate: str, gold_answer: str, answer_mode: str) -> bool:
    return answers_match(candidate, gold_answer, answer_mode=answer_mode)


def main() -> None:
    args = parse_args()
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be at least 1.")
    if args.protect_prefix_candidates < 0:
        raise SystemExit("--protect-prefix-candidates must be non-negative.")
    if args.max_aux_insertions < 0:
        raise SystemExit("--max-aux-insertions must be non-negative.")

    started_at = time.perf_counter()
    output_rows = []

    stats = {
        "num_examples": 0,
        "anchor_first_correct": 0,
        "merged_first_correct": 0,
        "anchor_oracle_correct": 0,
        "aux_oracle_correct": 0,
        "merged_oracle_correct": 0,
        "examples_with_aux_unique": 0,
        "examples_with_insertions": 0,
        "total_anchor_new_candidates": 0,
        "total_aux_unique_candidates": 0,
        "total_inserted_candidates": 0,
        "total_appended_candidates": 0,
        "total_replaced_base_candidates": 0,
    }

    for base_row, anchor_row, aux_row in zip(
        read_jsonl(args.base_candidates),
        read_jsonl(args.anchor_candidates),
        read_jsonl(args.aux_candidates),
    ):
        base_example_id = str(base_row["example_id"])
        anchor_example_id = str(anchor_row["example_id"])
        aux_example_id = str(aux_row["example_id"])
        if base_example_id != anchor_example_id or base_example_id != aux_example_id:
            raise ValueError(
                "Mismatched example ids while merging pools: "
                f"base={base_example_id} anchor={anchor_example_id} aux={aux_example_id}"
            )

        answer_mode = str(anchor_row.get("answer_mode", base_row.get("answer_mode", "numeric")))
        gold_answer = str(anchor_row["gold_answer"])
        base_candidates = [str(candidate) for candidate in base_row["candidates"]]
        anchor_candidates = [str(candidate) for candidate in anchor_row["candidates"][: args.max_candidates]]
        aux_candidates = [str(candidate) for candidate in aux_row["candidates"]]

        base_keys = {_candidate_key(candidate, answer_mode, args.dedupe_mode) for candidate in base_candidates}
        anchor_keys = [_candidate_key(candidate, answer_mode, args.dedupe_mode) for candidate in anchor_candidates]
        anchor_key_set = set(anchor_keys)
        anchor_new_count = sum(key not in base_keys for key in anchor_keys)

        aux_unique_candidates = []
        seen_keys = set(anchor_key_set)
        for candidate in aux_candidates:
            candidate_key = _candidate_key(candidate, answer_mode, args.dedupe_mode)
            if candidate_key in seen_keys:
                continue
            aux_unique_candidates.append(candidate)
            seen_keys.add(candidate_key)

        merged_candidates = list(anchor_candidates)
        inserted = 0
        appended = 0
        replaced = 0
        remaining_slots = max(0, args.max_candidates - len(merged_candidates))
        append_budget = min(args.max_aux_insertions, remaining_slots, len(aux_unique_candidates))
        if append_budget > 0:
            merged_candidates.extend(aux_unique_candidates[:append_budget])
            inserted += append_budget
            appended += append_budget

        replaceable_indices = [
            index
            for index, candidate in enumerate(merged_candidates)
            if index >= args.protect_prefix_candidates
            and _candidate_key(candidate, answer_mode, args.dedupe_mode) in base_keys
        ]
        replaceable_indices.sort(reverse=True)
        remaining_aux = aux_unique_candidates[append_budget : args.max_aux_insertions]
        for index, candidate in zip(replaceable_indices, remaining_aux):
            merged_candidates[index] = candidate
            inserted += 1
            replaced += 1

        stats["num_examples"] += 1
        stats["anchor_first_correct"] += int(_is_correct(anchor_candidates[0], gold_answer, answer_mode))
        stats["merged_first_correct"] += int(_is_correct(merged_candidates[0], gold_answer, answer_mode))
        stats["anchor_oracle_correct"] += int(
            any(_is_correct(candidate, gold_answer, answer_mode) for candidate in anchor_candidates)
        )
        stats["aux_oracle_correct"] += int(
            any(_is_correct(candidate, gold_answer, answer_mode) for candidate in aux_candidates)
        )
        stats["merged_oracle_correct"] += int(
            any(_is_correct(candidate, gold_answer, answer_mode) for candidate in merged_candidates)
        )
        stats["examples_with_aux_unique"] += int(bool(aux_unique_candidates))
        stats["examples_with_insertions"] += int(inserted > 0)
        stats["total_anchor_new_candidates"] += anchor_new_count
        stats["total_aux_unique_candidates"] += len(aux_unique_candidates)
        stats["total_inserted_candidates"] += inserted
        stats["total_appended_candidates"] += appended
        stats["total_replaced_base_candidates"] += replaced

        merged_row = dict(anchor_row)
        merged_row["candidates"] = merged_candidates
        output_rows.append(merged_row)

    write_jsonl(args.output, output_rows)

    if args.metrics_output is not None:
        num_examples = stats["num_examples"]
        stats_output = {
            "base_candidates_path": str(args.base_candidates),
            "anchor_candidates_path": str(args.anchor_candidates),
            "aux_candidates_path": str(args.aux_candidates),
            "output_path": str(args.output),
            "dedupe_mode": args.dedupe_mode,
            "max_candidates": args.max_candidates,
            "protect_prefix_candidates": args.protect_prefix_candidates,
            "max_aux_insertions": args.max_aux_insertions,
            "num_examples": num_examples,
            "anchor_first_correct": stats["anchor_first_correct"],
            "anchor_first_accuracy": (stats["anchor_first_correct"] / num_examples) if num_examples else 0.0,
            "merged_first_correct": stats["merged_first_correct"],
            "merged_first_accuracy": (stats["merged_first_correct"] / num_examples) if num_examples else 0.0,
            "anchor_oracle_correct": stats["anchor_oracle_correct"],
            "anchor_oracle_accuracy": (stats["anchor_oracle_correct"] / num_examples) if num_examples else 0.0,
            "aux_oracle_correct": stats["aux_oracle_correct"],
            "aux_oracle_accuracy": (stats["aux_oracle_correct"] / num_examples) if num_examples else 0.0,
            "merged_oracle_correct": stats["merged_oracle_correct"],
            "merged_oracle_accuracy": (stats["merged_oracle_correct"] / num_examples) if num_examples else 0.0,
            "examples_with_aux_unique": stats["examples_with_aux_unique"],
            "examples_with_insertions": stats["examples_with_insertions"],
            "total_anchor_new_candidates": stats["total_anchor_new_candidates"],
            "total_aux_unique_candidates": stats["total_aux_unique_candidates"],
            "total_inserted_candidates": stats["total_inserted_candidates"],
            "total_appended_candidates": stats["total_appended_candidates"],
            "total_replaced_base_candidates": stats["total_replaced_base_candidates"],
            "avg_inserted_candidates_per_example": (
                stats["total_inserted_candidates"] / num_examples
            ) if num_examples else 0.0,
            "total_seconds": round(time.perf_counter() - started_at, 6),
        }
        write_json(args.metrics_output, stats_output)


if __name__ == "__main__":
    main()
