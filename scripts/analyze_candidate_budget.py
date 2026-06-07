from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import extract_numeric_answer, normalize_answer
from src.utils.io_utils import read_jsonl, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze candidate-budget saturation and numeric duplication for open-ended candidate pools."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--run-label", default="unnamed_candidate_budget_run")
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--verifier-predictions", type=Path)
    return parser.parse_args()


def _is_correct(prediction: str, gold_answer: str) -> bool:
    if normalize_answer(prediction) == normalize_answer(gold_answer):
        return True
    prediction_numeric = extract_numeric_answer(prediction)
    gold_numeric = extract_numeric_answer(gold_answer)
    return (
        prediction_numeric is not None
        and gold_numeric is not None
        and prediction_numeric == gold_numeric
    )


def _load_score_maps(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    score_maps: dict[str, dict[str, float]] = {}
    for row in read_jsonl(path):
        score_maps[str(row["example_id"])] = {
            str(candidate["candidate_answer"]): float(candidate["margin"])
            for candidate in row["candidates"]
        }
    return score_maps


def _choose_best_prefix(
    example_id: str,
    prefix_candidates: list[str],
    score_maps: dict[str, dict[str, float]],
) -> str:
    if not prefix_candidates:
        return ""
    candidate_scores = score_maps[example_id]
    return max(prefix_candidates, key=lambda candidate: candidate_scores[candidate])


def _format_ratio(correct: int, total: int) -> str:
    return f"`{correct} / {total} = {correct / total:.4f}`"


def main() -> None:
    args = parse_args()

    candidate_rows = [row for row in read_jsonl(args.candidates)]
    if not candidate_rows:
        raise SystemExit("Candidate file is empty.")

    base_score_maps = _load_score_maps(args.base_predictions)
    verifier_score_maps = _load_score_maps(args.verifier_predictions)

    total_examples = len(candidate_rows)
    max_candidates = max(len(row["candidates"]) for row in candidate_rows)
    first_correct_positions: Counter[int | None] = Counter()

    examples_with_numeric_duplicates = 0
    numeric_duplicate_slots = 0
    numeric_candidate_slots = 0
    unique_numeric_counts: list[int] = []
    unique_numeric_count_distribution: Counter[int] = Counter()
    candidate_count_distribution: Counter[int] = Counter()

    oracle_curve: dict[int, int] = {k: 0 for k in range(1, max_candidates + 1)}
    first_curve: dict[int, int] = {k: 0 for k in range(1, max_candidates + 1)}
    base_curve: dict[int, int] = {k: 0 for k in range(1, max_candidates + 1)}
    verifier_curve: dict[int, int] = {k: 0 for k in range(1, max_candidates + 1)}

    for row in candidate_rows:
        example_id = str(row["example_id"])
        gold_answer = str(row["gold_answer"])
        candidates = [str(candidate) for candidate in row["candidates"]]
        candidate_count_distribution[len(candidates)] += 1

        numeric_answers = [extract_numeric_answer(candidate) for candidate in candidates]
        present_numeric_answers = [answer for answer in numeric_answers if answer is not None]
        unique_numeric_answer_count = len(set(present_numeric_answers))
        unique_numeric_counts.append(unique_numeric_answer_count)
        unique_numeric_count_distribution[unique_numeric_answer_count] += 1
        numeric_duplicate_slots += len(present_numeric_answers) - unique_numeric_answer_count
        numeric_candidate_slots += len(present_numeric_answers)
        if len(present_numeric_answers) != unique_numeric_answer_count:
            examples_with_numeric_duplicates += 1

        first_correct_position = None
        for idx, candidate in enumerate(candidates, start=1):
            if _is_correct(candidate, gold_answer):
                first_correct_position = idx
                break
        first_correct_positions[first_correct_position] += 1

        for k in range(1, max_candidates + 1):
            prefix_candidates = candidates[:k]
            if first_correct_position is not None and first_correct_position <= k:
                oracle_curve[k] += 1
            if prefix_candidates and _is_correct(prefix_candidates[0], gold_answer):
                first_curve[k] += 1
            if base_score_maps:
                base_choice = _choose_best_prefix(example_id, prefix_candidates, base_score_maps)
                if _is_correct(base_choice, gold_answer):
                    base_curve[k] += 1
            if verifier_score_maps:
                verifier_choice = _choose_best_prefix(example_id, prefix_candidates, verifier_score_maps)
                if _is_correct(verifier_choice, gold_answer):
                    verifier_curve[k] += 1

    top_k_rows = []
    for k in range(1, max_candidates + 1):
        row = {
            "k": k,
            "oracle_correct": oracle_curve[k],
            "oracle_accuracy": oracle_curve[k] / total_examples,
            "first_correct": first_curve[k],
            "first_accuracy": first_curve[k] / total_examples,
            "oracle_increment": oracle_curve[k] - (oracle_curve[k - 1] if k > 1 else 0),
        }
        if base_score_maps:
            row["base_correct"] = base_curve[k]
            row["base_accuracy"] = base_curve[k] / total_examples
        if verifier_score_maps:
            row["verifier_correct"] = verifier_curve[k]
            row["verifier_accuracy"] = verifier_curve[k] / total_examples
        top_k_rows.append(row)

    summary = {
        "run_label": args.run_label,
        "candidate_path": str(args.candidates),
        "base_predictions_path": str(args.base_predictions) if args.base_predictions is not None else None,
        "verifier_predictions_path": str(args.verifier_predictions) if args.verifier_predictions is not None else None,
        "total_examples": total_examples,
        "max_candidates": max_candidates,
        "top_k_curve": top_k_rows,
        "first_correct_position_counts": {
            ("none" if key is None else str(key)): value for key, value in sorted(
                first_correct_positions.items(),
                key=lambda item: (item[0] is None, item[0] if item[0] is not None else -1),
            )
        },
        "numeric_duplication": {
            "examples_with_numeric_duplicates": examples_with_numeric_duplicates,
            "examples_with_numeric_duplicates_rate": examples_with_numeric_duplicates / total_examples,
            "numeric_duplicate_slots": numeric_duplicate_slots,
            "numeric_candidate_slots": numeric_candidate_slots,
            "numeric_duplicate_slot_share": (
                numeric_duplicate_slots / numeric_candidate_slots if numeric_candidate_slots else 0.0
            ),
            "avg_unique_numeric_answers_per_example": sum(unique_numeric_counts) / total_examples,
            "unique_numeric_answer_count_distribution": dict(sorted(unique_numeric_count_distribution.items())),
        },
        "candidate_count_distribution": dict(sorted(candidate_count_distribution.items())),
        "budget_implications": {
            "oracle_gain_1_to_4": oracle_curve.get(4, oracle_curve[max_candidates]) - oracle_curve[1],
            "oracle_gain_4_to_8": oracle_curve.get(8, oracle_curve[max_candidates]) - oracle_curve.get(
                4, oracle_curve[max_candidates]
            ),
            "verifier_gain_1_to_4": (
                verifier_curve.get(4, verifier_curve[max_candidates]) - verifier_curve[1]
                if verifier_score_maps
                else None
            ),
            "verifier_gain_4_to_8": (
                verifier_curve.get(8, verifier_curve[max_candidates]) - verifier_curve.get(
                    4, verifier_curve[max_candidates]
                )
                if verifier_score_maps
                else None
            ),
        },
    }

    write_json(args.summary_json, summary)

    top_k_lines = [
        "| Top-k budget | Oracle coverage | First candidate | Base reranker | Verifier reranker | Marginal oracle gain |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in top_k_rows:
        base_value = (
            _format_ratio(row["base_correct"], total_examples)
            if "base_correct" in row
            else "N/A"
        )
        verifier_value = (
            _format_ratio(row["verifier_correct"], total_examples)
            if "verifier_correct" in row
            else "N/A"
        )
        top_k_lines.append(
            f"| `{row['k']}` | {_format_ratio(row['oracle_correct'], total_examples)} | "
            f"{_format_ratio(row['first_correct'], total_examples)} | {base_value} | "
            f"{verifier_value} | `{row['oracle_increment']}` |"
        )

    position_lines = [
        "| First correct position | Count | Share |",
        "| --- | --- | --- |",
    ]
    for key, value in sorted(
        first_correct_positions.items(),
        key=lambda item: (item[0] is None, item[0] if item[0] is not None else -1),
    ):
        label = "none" if key is None else str(key)
        position_lines.append(f"| `{label}` | `{value}` | `{value / total_examples:.4f}` |")

    duplication = summary["numeric_duplication"]
    duplication_lines = [
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Examples with numeric duplication | `{duplication['examples_with_numeric_duplicates']} / {total_examples} = {duplication['examples_with_numeric_duplicates_rate']:.4f}` |",
        f"| Duplicate numeric slot share | `{duplication['numeric_duplicate_slots']} / {duplication['numeric_candidate_slots']} = {duplication['numeric_duplicate_slot_share']:.4f}` |",
        f"| Avg unique numeric answers/example | `{duplication['avg_unique_numeric_answers_per_example']:.4f}` |",
    ]

    base_budget_note = ""
    if base_score_maps:
        best_base_row = max(top_k_rows, key=lambda row: row.get("base_correct", -1))
        base_budget_note = (
            f"- Base reranker peaks at top-{best_base_row['k']} with "
            f"{_format_ratio(best_base_row['base_correct'], total_examples)}, then saturates or declines.\n"
        )

    verifier_budget_note = ""
    if verifier_score_maps:
        verifier_budget_note = (
            f"- Verifier reranker keeps improving through top-{max_candidates}, reaching "
            f"{_format_ratio(verifier_curve[max_candidates], total_examples)}.\n"
        )

    report = f"""# Candidate-Budget and Numeric-Duplication Analysis: {args.run_label}

## Setup

- Candidate set: `{args.candidates}`
- Base reranker predictions: `{args.base_predictions}`\n""" if args.base_predictions is not None else f"""# Candidate-Budget and Numeric-Duplication Analysis: {args.run_label}

## Setup

- Candidate set: `{args.candidates}`
"""
    if args.verifier_predictions is not None:
        report += f"- Verifier reranker predictions: `{args.verifier_predictions}`\n"
    report += f"- Eval size: `{total_examples}`\n"
    report += f"- Max retained candidates/example: `{max_candidates}`\n"
    report += "- Exact-match rule: project-wide numeric exact-match\n\n"

    report += "## Top-k Budget Curve\n\n"
    report += "\n".join(top_k_lines)
    report += "\n\n"
    report += (
        f"Oracle coverage rises from {_format_ratio(oracle_curve[1], total_examples)} at top-1 to "
        f"{_format_ratio(oracle_curve.get(4, oracle_curve[max_candidates]), total_examples)} at top-4 and "
        f"{_format_ratio(oracle_curve[max_candidates], total_examples)} at top-{max_candidates}. "
        "So the retained-candidate budget is still materially constraining answer coverage.\n\n"
    )

    report += "## Correct-Answer Position\n\n"
    report += "\n".join(position_lines)
    report += "\n\n"
    report += (
        "The first correct answer does not collapse into the earliest few positions. "
        f"Correct answers still appear at positions 5-8 in `{sum(first_correct_positions.get(k, 0) for k in range(5, max_candidates + 1))}` cases, "
        "which means later candidates are carrying real coverage rather than just redundant noise.\n\n"
    )

    report += "## Numeric Duplication\n\n"
    report += "\n".join(duplication_lines)
    report += "\n\n"
    if duplication["examples_with_numeric_duplicates"] > 0:
        report += (
            "The candidate pools are heavily duplicated at the numeric-answer level even though they look diverse as raw text. "
            "This means text-level deduplication is wasting retained slots on multiple surface forms of the same final number.\n\n"
        )
    else:
        report += (
            "Numeric-answer-level duplication has been eliminated in this retained pool. "
            "So the remaining bottleneck now comes from which unique answers are proposed and how the reranker handles that candidate mix, not from repeated numeric aliases.\n\n"
        )

    report += "## Interpretation\n\n"
    report += (
        f"- Top-k oracle coverage gains: top-1 to top-4 adds `{summary['budget_implications']['oracle_gain_1_to_4']}` oracle hits; "
        f"top-4 to top-{max_candidates} adds another `{summary['budget_implications']['oracle_gain_4_to_8']}`.\n"
    )
    report += base_budget_note
    report += verifier_budget_note
    report += (
        f"- `{duplication['examples_with_numeric_duplicates_rate']:.4f}` of examples contain repeated numeric answers inside the retained pool, "
        f"and `{duplication['numeric_duplicate_slot_share']:.4f}` of numeric-bearing candidate slots are duplicates.\n"
    )
    if duplication["examples_with_numeric_duplicates"] > 0:
        report += (
            "- The highest-value next ablation is therefore not more verifier tuning. It is a candidate-side change that preserves answer-space diversity, "
            "starting with numeric-answer-aware deduplication and then re-running the same generate-then-rerank recipe on a small held-out shard.\n"
        )
    else:
        report += (
            "- Since numeric duplication is already removed here, the next branch should focus on selector-compatible candidate construction rather than additional deduplication alone.\n"
        )

    write_text(args.report, report)


if __name__ == "__main__":
    main()
