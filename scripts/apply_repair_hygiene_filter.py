from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.motif_utils import infer_candidate_tag
from src.eval.evaluate_predictions import extract_numeric_answer, normalize_answer
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


_STRONG_INSTRUCTION_PATTERNS = (
    "use at most",
    "exactly one last line",
    "do not mention",
    "do not use markdown",
    "do not output",
    "the solution should",
    "you should not include any other text",
    "the final answer is the only thing you output",
    "the answer is the final line",
    "answer should be in the box",
    "answer should be in a box",
    "the final answer is the time",
    "the final answer is the amount",
    "the final answer is the total cost",
    "final answer: final answer:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply conservative hygiene filtering to a repair candidate pool using existing completion and repair scoring logs."
    )
    parser.add_argument("--completion-candidates", required=True, type=Path)
    parser.add_argument("--repair-candidates", required=True, type=Path)
    parser.add_argument("--base-completion-predictions", required=True, type=Path)
    parser.add_argument("--base-repair-predictions", required=True, type=Path)
    parser.add_argument("--verifier-completion-predictions", required=True, type=Path)
    parser.add_argument("--verifier-repair-predictions", required=True, type=Path)
    parser.add_argument("--output-candidates", required=True, type=Path)
    parser.add_argument("--first-predictions-output", required=True, type=Path)
    parser.add_argument("--base-predictions-output", required=True, type=Path)
    parser.add_argument("--verifier-predictions-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument(
        "--allow-replace-complete-attempt",
        action="store_true",
        help="Allow repaired candidates to replace source complete-attempt slots.",
    )
    return parser.parse_args()


def _load_candidate_map(path: Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in read_jsonl(path)}


def _load_score_map(path: Path) -> dict[str, dict[str, dict]]:
    score_map: dict[str, dict[str, dict]] = {}
    for row in read_jsonl(path):
        score_map[str(row["example_id"])] = {
            str(candidate["candidate_answer"]): candidate for candidate in row.get("candidates", [])
        }
    return score_map


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


def _repair_rejection_reasons(
    problem: str,
    source_candidate: str,
    repaired_candidate: str,
    *,
    allow_replace_complete_attempt: bool,
) -> list[str]:
    if source_candidate == repaired_candidate:
        return []

    reasons: list[str] = []
    lowered = repaired_candidate.lower()
    if any(pattern in lowered for pattern in _STRONG_INSTRUCTION_PATTERNS):
        reasons.append("instruction_or_prompt_leak")
    if extract_numeric_answer(repaired_candidate) is None:
        reasons.append("missing_numeric_answer")
    if not allow_replace_complete_attempt:
        source_quality = infer_candidate_tag(problem, source_candidate).quality.label
        if source_quality == "complete_attempt":
            reasons.append("would_replace_complete_attempt")
    return sorted(set(reasons))


def _resolve_scored_candidate(
    example_id: str,
    candidate_text: str,
    *,
    repair_scores: dict[str, dict[str, dict]],
    completion_scores: dict[str, dict[str, dict]],
) -> dict:
    example_repair_scores = repair_scores.get(example_id, {})
    if candidate_text in example_repair_scores:
        return dict(example_repair_scores[candidate_text])

    example_completion_scores = completion_scores.get(example_id, {})
    if candidate_text in example_completion_scores:
        return dict(example_completion_scores[candidate_text])

    raise KeyError(
        f"Missing scored candidate for example_id={example_id} candidate={candidate_text[:120]!r}"
    )


def _build_prediction_row(
    example_id: str,
    gold_answer: str,
    candidates: list[str],
    *,
    repair_scores: dict[str, dict[str, dict]],
    completion_scores: dict[str, dict[str, dict]],
) -> dict:
    scored_candidates = [
        _resolve_scored_candidate(
            example_id,
            candidate_text,
            repair_scores=repair_scores,
            completion_scores=completion_scores,
        )
        for candidate_text in candidates
    ]
    scored_candidates.sort(key=lambda item: float(item["margin"]), reverse=True)
    best_candidate = str(scored_candidates[0]["candidate_answer"]) if scored_candidates else ""
    return {
        "example_id": example_id,
        "prediction": best_candidate,
        "gold_answer": gold_answer,
        "correct": _is_correct(best_candidate, gold_answer),
        "candidates": scored_candidates,
    }


def main() -> None:
    args = parse_args()

    completion_rows = _load_candidate_map(args.completion_candidates)
    repair_rows = _load_candidate_map(args.repair_candidates)

    base_completion_scores = _load_score_map(args.base_completion_predictions)
    base_repair_scores = _load_score_map(args.base_repair_predictions)
    verifier_completion_scores = _load_score_map(args.verifier_completion_predictions)
    verifier_repair_scores = _load_score_map(args.verifier_repair_predictions)

    example_ids = sorted(completion_rows)
    output_candidates: list[dict] = []
    first_predictions: list[dict] = []
    base_predictions: list[dict] = []
    verifier_predictions: list[dict] = []

    generation_stats = Counter()
    rejection_reason_counts = Counter()

    for example_id in example_ids:
        completion_row = completion_rows[example_id]
        repair_row = repair_rows[example_id]
        problem = str(completion_row["problem"])
        gold_answer = str(completion_row["gold_answer"])

        completion_candidates = [str(candidate) for candidate in completion_row["candidates"]]
        repair_candidates = [str(candidate) for candidate in repair_row["candidates"]]
        if len(completion_candidates) != len(repair_candidates):
            raise ValueError(
                f"Mismatched candidate counts for example_id={example_id}: "
                f"{len(completion_candidates)} vs {len(repair_candidates)}"
            )

        filtered_candidates: list[str] = []
        example_reverted = False
        example_retained_repairs = False
        for source_candidate, repaired_candidate in zip(completion_candidates, repair_candidates):
            reasons = _repair_rejection_reasons(
                problem,
                source_candidate,
                repaired_candidate,
                allow_replace_complete_attempt=args.allow_replace_complete_attempt,
            )
            if not reasons:
                filtered_candidates.append(repaired_candidate)
                if source_candidate != repaired_candidate:
                    generation_stats["retained_repair_candidates"] += 1
                    example_retained_repairs = True
                continue

            filtered_candidates.append(source_candidate)
            if source_candidate != repaired_candidate:
                generation_stats["reverted_repair_candidates"] += 1
                example_reverted = True
                for reason in reasons:
                    rejection_reason_counts[reason] += 1

        if filtered_candidates != repair_candidates:
            generation_stats["examples_modified_by_hygiene"] += 1
        if example_reverted:
            generation_stats["examples_with_reverted_repairs"] += 1
        if example_retained_repairs:
            generation_stats["examples_with_retained_repairs"] += 1

        output_candidates.append(
            {
                "example_id": example_id,
                "dataset": str(completion_row.get("dataset", "unknown")),
                "problem": problem,
                "gold_answer": gold_answer,
                "candidates": filtered_candidates,
            }
        )
        first_prediction = filtered_candidates[0] if filtered_candidates else ""
        first_predictions.append(
            {
                "example_id": example_id,
                "prediction": first_prediction,
                "gold_answer": gold_answer,
            }
        )
        base_predictions.append(
            _build_prediction_row(
                example_id,
                gold_answer,
                filtered_candidates,
                repair_scores=base_repair_scores,
                completion_scores=base_completion_scores,
            )
        )
        verifier_predictions.append(
            _build_prediction_row(
                example_id,
                gold_answer,
                filtered_candidates,
                repair_scores=verifier_repair_scores,
                completion_scores=verifier_completion_scores,
            )
        )

    write_jsonl(args.output_candidates, output_candidates)
    write_jsonl(args.first_predictions_output, first_predictions)
    write_jsonl(args.base_predictions_output, base_predictions)
    write_jsonl(args.verifier_predictions_output, verifier_predictions)

    metrics = {
        "completion_candidates_path": str(args.completion_candidates),
        "repair_candidates_path": str(args.repair_candidates),
        "output_candidates_path": str(args.output_candidates),
        "total_examples": len(output_candidates),
        "retained_repair_candidates": generation_stats.get("retained_repair_candidates", 0),
        "reverted_repair_candidates": generation_stats.get("reverted_repair_candidates", 0),
        "examples_modified_by_hygiene": generation_stats.get("examples_modified_by_hygiene", 0),
        "examples_with_reverted_repairs": generation_stats.get("examples_with_reverted_repairs", 0),
        "examples_with_retained_repairs": generation_stats.get("examples_with_retained_repairs", 0),
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "allow_replace_complete_attempt": args.allow_replace_complete_attempt,
        "instruction_patterns": list(_STRONG_INSTRUCTION_PATTERNS),
    }
    write_json(args.metrics_output, metrics)


if __name__ == "__main__":
    main()
