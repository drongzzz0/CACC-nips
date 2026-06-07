#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import answers_match, extract_choice_answer, extract_numeric_answer, normalize_answer
from src.utils.io_utils import read_jsonl, write_json, write_text


INSTRUCTION_PATTERNS = (
    "use at most",
    "just output the final answer",
    "just output the answer",
    "do not include any explanation",
    "do not include any reasoning",
    "do not produce any other text",
    "the last line must contain",
    "the last line should be",
    "this is a multiple-choice task",
    "you may reason about the options briefly",
    "the correct answer is one of the options listed",
    "the final answer must be",
    "your answer must be",
    "the answer must be",
    "single option letter",
    "single option-letter",
    "do not write anything else",
)

SCAFFOLD_REGEXES = (
    re.compile(r"^(?:here'?s the thought process:\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:let'?s think(?: about this)?(?: step by step)?[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:let me think(?: about this)?(?: step by step)?[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:okay,\s*)?let'?s tackle this problem(?: step by step)?[.:]?\s*", flags=re.IGNORECASE),
    re.compile(r"^(?:okay,\s*)?let'?s break down the question again[.:]?\s*", flags=re.IGNORECASE),
    re.compile(r"^(?:wait,\s*but that'?s not correct[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:wait,\s*that'?s not correct[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:the solution (?:should|must)(?: not)? be [^.!?\n]*(?:[.!?]\s*|$))+", flags=re.IGNORECASE),
    re.compile(r"^(?:the last line should be [^.!?\n]*(?:[.!?]\s*|$))+", flags=re.IGNORECASE),
    re.compile(r"^(?:use the options given(?: above)?[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:do not output the letter of the option until the last line[.!?]?\s*)+", flags=re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paired GSM8K conversion diagnosis between a baseline candidate pool and a new completion pool."
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--baseline-candidates", required=True, type=Path)
    parser.add_argument("--baseline-verifier-predictions", required=True, type=Path)
    parser.add_argument("--completion-candidates", required=True, type=Path)
    parser.add_argument("--completion-first-predictions", required=True, type=Path)
    parser.add_argument("--completion-base-predictions", required=True, type=Path)
    parser.add_argument("--completion-verifier-predictions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--max-examples", default=5, type=int)
    return parser.parse_args()


def contains_instruction_leak(text: str) -> bool:
    lowered = text.strip().lower()
    return any(pattern in lowered for pattern in INSTRUCTION_PATTERNS)



def contains_scaffold_residue(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(regex.search(stripped) for regex in SCAFFOLD_REGEXES):
        return True
    return lowered.startswith(("step ", "step\n", "first,", "second,", "third,"))



def has_valid_final_answer(text: str, answer_mode: str) -> bool:
    if answer_mode == "choice_letter":
        return extract_choice_answer(text) is not None
    if answer_mode == "numeric":
        return extract_numeric_answer(text) is not None
    return bool(normalize_answer(text))



def is_short_numeric_like(text: str, answer_mode: str) -> bool:
    if answer_mode != "numeric":
        return False
    return extract_numeric_answer(text) is not None and len(text.strip().split()) <= 6



def prediction_is_correct(row: dict, answer_mode: str) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    return answers_match(str(row["prediction"]), str(row["gold_answer"]), answer_mode=answer_mode)



def load_candidates(path: Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in read_jsonl(path)}



def load_predictions(path: Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in read_jsonl(path)}



def correct_candidates(row: dict) -> list[str]:
    gold_answer = str(row["gold_answer"])
    answer_mode = str(row.get("answer_mode", "numeric"))
    return [
        str(candidate)
        for candidate in row.get("candidates", [])
        if answers_match(str(candidate), gold_answer, answer_mode=answer_mode)
    ]



def fmt_ratio(value: float) -> str:
    return f"{value:.4f}"



def build_markdown_report(run_label: str, result: dict) -> str:
    summary = result["summary"]
    lines = [
        f"# {run_label}",
        "",
        "## Core Readout",
        "",
        f"- Newly covered oracle-hit examples: `{summary['new_oracle_hit']}`",
        f"- Newly covered examples converted into final verifier wins: `{summary['new_oracle_hit_verifier_correct']}`",
        f"- Newly covered examples not converted: `{summary['new_oracle_hit_verifier_wrong']}`",
        f"- Conversion rate on newly covered examples: `{fmt_ratio(summary['new_oracle_hit_conversion_rate'])}`",
        f"- Shared oracle-hit examples: `{summary['common_oracle_hit']}`",
        f"- Shared oracle-hit regressions: `{summary['common_hit_verifier_regressed']}`",
        f"- Baseline verifier accuracy on shared-hit subset: `{fmt_ratio(summary['common_hit_baseline_verifier_accuracy'])}`",
        f"- Completion verifier accuracy on shared-hit subset: `{fmt_ratio(summary['common_hit_completion_verifier_accuracy'])}`",
        f"- Lost oracle-hit examples: `{summary['lost_oracle_hit']}`",
        "",
        "## Failure Buckets",
        "",
        "### Newly Covered but Not Converted",
        "",
        json.dumps(result["newly_covered_failure_buckets"], ensure_ascii=False, indent=2),
        "",
        "### Shared-Hit Regressions",
        "",
        json.dumps(result["common_hit_regression_buckets"], ensure_ascii=False, indent=2),
        "",
        "### Lost Oracle-Hit Examples",
        "",
        json.dumps(result["lost_oracle_hit_buckets"], ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines) + "\n"



def main() -> None:
    args = parse_args()

    baseline_candidates = load_candidates(args.baseline_candidates)
    baseline_verifier = load_predictions(args.baseline_verifier_predictions)
    completion_candidates = load_candidates(args.completion_candidates)
    completion_first = load_predictions(args.completion_first_predictions)
    completion_base = load_predictions(args.completion_base_predictions)
    completion_verifier = load_predictions(args.completion_verifier_predictions)

    summary = Counter()
    newly_covered_failure_buckets = Counter()
    common_hit_regression_buckets = Counter()
    lost_oracle_hit_buckets = Counter()

    examples = {
        "newly_covered_but_not_converted": [],
        "common_hit_regressions": [],
        "lost_oracle_hits": [],
    }

    example_ids = sorted(set(baseline_candidates) & set(completion_candidates) & set(baseline_verifier) & set(completion_first) & set(completion_base) & set(completion_verifier))

    for example_id in example_ids:
        baseline_row = baseline_candidates[example_id]
        completion_row = completion_candidates[example_id]
        answer_mode = str(baseline_row.get("answer_mode", completion_row.get("answer_mode", "numeric")))

        baseline_correct_list = correct_candidates(baseline_row)
        completion_correct_list = correct_candidates(completion_row)
        baseline_oracle_hit = bool(baseline_correct_list)
        completion_oracle_hit = bool(completion_correct_list)

        baseline_verifier_row = baseline_verifier[example_id]
        completion_first_row = completion_first[example_id]
        completion_base_row = completion_base[example_id]
        completion_verifier_row = completion_verifier[example_id]

        baseline_verifier_correct = prediction_is_correct(baseline_verifier_row, answer_mode)
        completion_first_correct = prediction_is_correct(completion_first_row, answer_mode)
        completion_base_correct = prediction_is_correct(completion_base_row, answer_mode)
        completion_verifier_correct = prediction_is_correct(completion_verifier_row, answer_mode)

        if completion_oracle_hit and not baseline_oracle_hit:
            summary["new_oracle_hit"] += 1
            if completion_verifier_correct:
                summary["new_oracle_hit_verifier_correct"] += 1
            else:
                summary["new_oracle_hit_verifier_wrong"] += 1
                if completion_base_correct:
                    newly_covered_failure_buckets["base_correct_but_verifier_wrong"] += 1
                else:
                    newly_covered_failure_buckets["selector_failed_with_no_prior_correct_pick"] += 1
                verifier_prediction = str(completion_verifier_row["prediction"])
                if contains_scaffold_residue(verifier_prediction):
                    newly_covered_failure_buckets["scaffold_residue"] += 1
                if contains_instruction_leak(verifier_prediction):
                    newly_covered_failure_buckets["instruction_leak"] += 1
                if not has_valid_final_answer(verifier_prediction, answer_mode):
                    newly_covered_failure_buckets["invalid_final_answer"] += 1
                if len(examples["newly_covered_but_not_converted"]) < args.max_examples:
                    examples["newly_covered_but_not_converted"].append(
                        {
                            "example_id": example_id,
                            "gold": str(baseline_row["gold_answer"]),
                            "verifier_prediction": verifier_prediction,
                            "base_prediction": str(completion_base_row["prediction"]),
                            "first_prediction": str(completion_first_row["prediction"]),
                        }
                    )

        if baseline_oracle_hit and completion_oracle_hit:
            summary["common_oracle_hit"] += 1
            if baseline_verifier_correct:
                summary["common_hit_baseline_correct"] += 1
            if completion_verifier_correct:
                summary["common_hit_completion_correct"] += 1
            if baseline_verifier_correct and completion_verifier_correct:
                summary["common_hit_both_correct"] += 1
            elif baseline_verifier_correct and not completion_verifier_correct:
                summary["common_hit_verifier_regressed"] += 1
                if completion_first_correct:
                    common_hit_regression_buckets["first_correct_but_verifier_wrong"] += 1
                if completion_base_correct:
                    common_hit_regression_buckets["base_correct_but_verifier_wrong"] += 1
                verifier_prediction = str(completion_verifier_row["prediction"])
                if contains_instruction_leak(verifier_prediction):
                    common_hit_regression_buckets["instruction_leak"] += 1
                if contains_scaffold_residue(verifier_prediction):
                    common_hit_regression_buckets["scaffold_residue"] += 1
                if len(examples["common_hit_regressions"]) < args.max_examples:
                    examples["common_hit_regressions"].append(
                        {
                            "example_id": example_id,
                            "gold": str(baseline_row["gold_answer"]),
                            "baseline_verifier_prediction": str(baseline_verifier_row["prediction"]),
                            "completion_verifier_prediction": verifier_prediction,
                            "completion_base_prediction": str(completion_base_row["prediction"]),
                            "completion_first_prediction": str(completion_first_row["prediction"]),
                        }
                    )
            elif not baseline_verifier_correct and completion_verifier_correct:
                summary["common_hit_verifier_improved"] += 1
            else:
                summary["common_hit_both_wrong"] += 1

        if baseline_oracle_hit and not completion_oracle_hit:
            summary["lost_oracle_hit"] += 1
            if any(is_short_numeric_like(candidate, answer_mode) for candidate in baseline_correct_list):
                lost_oracle_hit_buckets["short_numeric_correct_candidate_removed"] += 1
            if baseline_verifier_correct:
                lost_oracle_hit_buckets["baseline_verifier_was_correct"] += 1
            if any("\\boxed{" in candidate for candidate in baseline_correct_list):
                lost_oracle_hit_buckets["boxed_correct_candidate_removed"] += 1
            if len(examples["lost_oracle_hits"]) < args.max_examples:
                examples["lost_oracle_hits"].append(
                    {
                        "example_id": example_id,
                        "gold": str(baseline_row["gold_answer"]),
                        "baseline_correct_candidates": baseline_correct_list[:3],
                        "baseline_verifier_prediction": str(baseline_verifier_row["prediction"]),
                        "completion_first_candidates": [str(candidate) for candidate in completion_row.get("candidates", [])[:3]],
                    }
                )

    common_oracle_hit = summary["common_oracle_hit"]
    new_oracle_hit = summary["new_oracle_hit"]
    result = {
        "summary": {
            "new_oracle_hit": summary["new_oracle_hit"],
            "new_oracle_hit_verifier_correct": summary["new_oracle_hit_verifier_correct"],
            "new_oracle_hit_verifier_wrong": summary["new_oracle_hit_verifier_wrong"],
            "new_oracle_hit_conversion_rate": (
                summary["new_oracle_hit_verifier_correct"] / new_oracle_hit if new_oracle_hit else 0.0
            ),
            "common_oracle_hit": summary["common_oracle_hit"],
            "common_hit_verifier_improved": summary["common_hit_verifier_improved"],
            "common_hit_verifier_regressed": summary["common_hit_verifier_regressed"],
            "common_hit_both_correct": summary["common_hit_both_correct"],
            "common_hit_both_wrong": summary["common_hit_both_wrong"],
            "common_hit_baseline_verifier_accuracy": (
                summary["common_hit_baseline_correct"] / common_oracle_hit if common_oracle_hit else 0.0
            ),
            "common_hit_completion_verifier_accuracy": (
                summary["common_hit_completion_correct"] / common_oracle_hit if common_oracle_hit else 0.0
            ),
            "lost_oracle_hit": summary["lost_oracle_hit"],
        },
        "newly_covered_failure_buckets": dict(newly_covered_failure_buckets),
        "common_hit_regression_buckets": dict(common_hit_regression_buckets),
        "lost_oracle_hit_buckets": dict(lost_oracle_hit_buckets),
        "examples": examples,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, result)
    write_text(args.output_md, build_markdown_report(args.run_label, result))


if __name__ == "__main__":
    main()
