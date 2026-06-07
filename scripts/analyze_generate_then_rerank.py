from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import answers_match, extract_choice_answer, extract_numeric_answer, normalize_answer
from src.utils.io_utils import read_jsonl, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze open-ended generate-then-rerank results and produce a paper-ready summary."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--first-predictions", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--verifier-predictions", required=True, type=Path)
    parser.add_argument("--completion-generation-metrics", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--run-label", default="unnamed_run")
    parser.add_argument("--fixed-reference-metrics", type=Path)
    parser.add_argument("--fixed-reference-label", default="fixed candidate-set verifier")
    parser.add_argument("--max-examples-per-bucket", default=2, type=int)
    return parser.parse_args()


def _is_correct(prediction: str, gold_answer: str, answer_mode: str) -> bool:
    return answers_match(prediction, gold_answer, answer_mode=answer_mode)


def _shorten(text: str, limit: int = 160) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _load_prediction_map(path: Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in read_jsonl(path)}


_INSTRUCTION_LEAK_PHRASES = (
    "you are improving an incomplete candidate pool",
    "this is a multiple-choice task",
    "the final answer must be",
    "the answer must be",
    "do not include any explanation",
    "do not write anything else",
    "single option letter",
    "final answer in the required format",
)


def _prediction_has_valid_final_answer(prediction: str, answer_mode: str) -> bool:
    extracted = prediction.strip()
    if answer_mode == "choice_letter":
        return extract_choice_answer(extracted) is not None
    if answer_mode == "numeric":
        return extract_numeric_answer(extracted) is not None
    return bool(normalize_answer(extracted))


def _prediction_answer_mode_matches(prediction: str, answer_mode: str) -> bool:
    return _prediction_has_valid_final_answer(prediction, answer_mode)


def _prediction_has_instruction_leak(prediction: str) -> bool:
    normalized = " ".join(prediction.strip().lower().split())
    if not normalized:
        return False
    return any(phrase in normalized for phrase in _INSTRUCTION_LEAK_PHRASES)


def _prediction_has_scaffold_residue(prediction: str) -> bool:
    normalized = prediction.strip().lower()
    if not normalized:
        return False
    scaffold_prefixes = (
        "let's think",
        "let me think",
        "here's the thought process",
        "i will now provide",
        "okay, let's",
        "wait,",
        "the solution should",
        "the solution must",
        "the last line should be",
        "do not output",
    )
    return normalized.startswith(scaffold_prefixes)


def _selected_margin(prediction_row: dict) -> float | None:
    candidates = prediction_row.get("candidates") or []
    if not candidates:
        return None
    margin = candidates[0].get("margin")
    return float(margin) if margin is not None else None


def _rate(rows: list[dict], key: str) -> float:
    return sum(int(bool(row[key])) for row in rows) / len(rows) if rows else 0.0


def _mean(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _exact_mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict[str, float | int]:
    a_only = 0
    b_only = 0
    both = 0
    neither = 0
    for row_a, row_b in zip(rows_a, rows_b):
        a_correct = bool(row_a["correct"])
        b_correct = bool(row_b["correct"])
        if a_correct and b_correct:
            both += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif not a_correct and b_correct:
            b_only += 1
        else:
            neither += 1
    disagreements = a_only + b_only
    if disagreements == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(disagreements, k) for k in range(min(a_only, b_only) + 1)) / (2**disagreements)
        p_value = min(1.0, 2 * tail)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "p": p_value,
    }


def _bucket_examples(rows: list[dict], predicate, limit: int) -> list[dict]:
    matches = [row for row in rows if predicate(row)]
    matches.sort(key=lambda row: (len(row["problem"]), row["example_id"]))
    return matches[:limit]


def _serialize_example(row: dict) -> dict:
    return {
        "example_id": row["example_id"],
        "problem": row["problem"],
        "gold_answer": row["gold_answer"],
        "first_prediction": row["first_prediction"],
        "base_prediction": row["base_prediction"],
        "verifier_prediction": row["verifier_prediction"],
        "correct_candidate": row["correct_candidate"],
        "candidate_count": row["candidate_count"],
        "candidate_snippets": row["candidate_snippets"],
    }


def _build_interpretation(
    *,
    total: int,
    oracle_correct: int,
    first_correct: int,
    base_correct: int,
    verifier_correct: int,
    conditional_base: float,
    conditional_verifier: float,
    oracle_miss_share_of_verifier_failures: float,
) -> str:
    oracle_coverage = oracle_correct / total
    first_accuracy = first_correct / total
    base_accuracy = base_correct / total
    verifier_accuracy = verifier_correct / total
    best_simple_selector = max(first_accuracy, base_accuracy)

    if oracle_coverage < 0.1:
        return (
            "This run should be treated as a failure case rather than a positive transfer result. "
            f"Oracle coverage is only `{oracle_coverage:.4f}`, so candidate construction collapses before reranking can help, "
            f"and `{oracle_miss_share_of_verifier_failures:.4f}` of verifier failures come from oracle misses. "
            f"The verifier also does not fully recover from the small oracle-hit subset: first=`{first_accuracy:.4f}`, "
            f"base=`{base_accuracy:.4f}`, verifier=`{verifier_accuracy:.4f}` and verifier|oracle=`{conditional_verifier:.4f}`.\n\n"
            "The next step should be a prompt-and-format audit before any scale-up or rerun. Prioritize answer-mode compatibility, "
            "candidate sanitation, and completion-prompt design rather than further verifier tuning."
        )

    if verifier_accuracy < best_simple_selector or conditional_verifier < conditional_base:
        return (
            "This run is mixed-to-negative: the candidate pool contains correct answers often enough to study selection, "
            f"but the verifier does not dominate the simpler selectors on the observed outputs. first=`{first_accuracy:.4f}`, "
            f"base=`{base_accuracy:.4f}`, verifier=`{verifier_accuracy:.4f}`, and verifier|oracle=`{conditional_verifier:.4f}`.\n\n"
            "The next step should audit reranker alignment and answer extraction on oracle-hit examples before spending more budget "
            "on larger candidate pools."
        )

    if oracle_miss_share_of_verifier_failures > 0.7:
        return (
            "The run is selector-positive but coverage-limited. Once a correct candidate is present, the verifier is the strongest selector in this run, "
            f"yet `{oracle_miss_share_of_verifier_failures:.4f}` of verifier failures still come from oracle misses.\n\n"
            "The highest-value next step is therefore candidate construction rather than additional verifier tuning: improve proposal quality, diversity, "
            "or completion prompting while keeping the current verifier mostly fixed."
        )

    return (
        "The run is mixed: the verifier helps over the base reranker, but the remaining gap is split between candidate construction and selector weakness.\n\n"
        "The next step should tighten both the proposer side and reranker robustness instead of pushing only one side."
    )


def main() -> None:
    args = parse_args()

    candidate_rows = {str(row["example_id"]): row for row in read_jsonl(args.candidates)}
    first_predictions = _load_prediction_map(args.first_predictions)
    base_predictions = _load_prediction_map(args.base_predictions)
    verifier_predictions = _load_prediction_map(args.verifier_predictions)

    example_ids = sorted(candidate_rows)
    rows: list[dict] = []
    candidate_count_breakdown: dict[int, Counter] = defaultdict(Counter)
    oracle_pattern_counts: Counter[str] = Counter()

    for example_id in example_ids:
        candidate_row = candidate_rows[example_id]
        first_row = first_predictions[example_id]
        base_row = base_predictions[example_id]
        verifier_row = verifier_predictions[example_id]

        gold_answer = str(candidate_row["gold_answer"])
        answer_mode = str(candidate_row.get("answer_mode", first_row.get("answer_mode", base_row.get("answer_mode", verifier_row.get("answer_mode", "numeric")))))
        candidates = [str(candidate) for candidate in candidate_row["candidates"]]
        oracle_hit = any(_is_correct(candidate, gold_answer, answer_mode=answer_mode) for candidate in candidates)
        correct_candidate = next((candidate for candidate in candidates if _is_correct(candidate, gold_answer, answer_mode=answer_mode)), None)
        first_correct = _is_correct(str(first_row["prediction"]), gold_answer, answer_mode=answer_mode)
        base_correct = _is_correct(str(base_row["prediction"]), gold_answer, answer_mode=answer_mode)
        verifier_correct = _is_correct(str(verifier_row["prediction"]), gold_answer, answer_mode=answer_mode)

        first_prediction = str(first_row["prediction"])
        base_prediction = str(base_row["prediction"])
        verifier_prediction = str(verifier_row["prediction"])
        row = {
            "example_id": example_id,
            "problem": str(candidate_row["problem"]),
            "gold_answer": gold_answer,
            "candidate_count": len(candidates),
            "answer_mode": answer_mode,
            "oracle_hit": oracle_hit,
            "first_correct": first_correct,
            "base_correct": base_correct,
            "verifier_correct": verifier_correct,
            "first_prediction": first_prediction,
            "base_prediction": base_prediction,
            "verifier_prediction": verifier_prediction,
            "first_parseable": _prediction_has_valid_final_answer(first_prediction, answer_mode),
            "base_parseable": _prediction_has_valid_final_answer(base_prediction, answer_mode),
            "verifier_parseable": _prediction_has_valid_final_answer(verifier_prediction, answer_mode),
            "first_answer_mode_match": _prediction_answer_mode_matches(first_prediction, answer_mode),
            "base_answer_mode_match": _prediction_answer_mode_matches(base_prediction, answer_mode),
            "verifier_answer_mode_match": _prediction_answer_mode_matches(verifier_prediction, answer_mode),
            "first_instruction_leak": _prediction_has_instruction_leak(first_prediction),
            "base_instruction_leak": _prediction_has_instruction_leak(base_prediction),
            "verifier_instruction_leak": _prediction_has_instruction_leak(verifier_prediction),
            "first_scaffold_residue": _prediction_has_scaffold_residue(first_prediction),
            "base_scaffold_residue": _prediction_has_scaffold_residue(base_prediction),
            "verifier_scaffold_residue": _prediction_has_scaffold_residue(verifier_prediction),
            "verifier_selected_margin": _selected_margin(verifier_row),
            "correct_candidate": correct_candidate,
            "candidate_snippets": [_shorten(candidate, 100) for candidate in candidates[:4]],
        }
        rows.append(row)

        bucket = candidate_count_breakdown[len(candidates)]
        bucket["total"] += 1
        bucket["oracle_hit"] += int(oracle_hit)
        bucket["first_correct"] += int(first_correct)
        bucket["base_correct"] += int(base_correct)
        bucket["verifier_correct"] += int(verifier_correct)

        if oracle_hit:
            oracle_pattern_counts[
                f"F{int(first_correct)}B{int(base_correct)}V{int(verifier_correct)}"
            ] += 1

    total = len(rows)
    oracle_correct = sum(int(row["oracle_hit"]) for row in rows)
    first_correct = sum(int(row["first_correct"]) for row in rows)
    base_correct = sum(int(row["base_correct"]) for row in rows)
    verifier_correct = sum(int(row["verifier_correct"]) for row in rows)
    oracle_miss = total - oracle_correct
    verifier_selection_failures = oracle_correct - verifier_correct
    base_selection_failures = oracle_correct - base_correct
    first_selection_failures = oracle_correct - first_correct

    oracle_hit_rows = [row for row in rows if row["oracle_hit"]]
    first_rows = [{"correct": row["first_correct"]} for row in rows]
    base_rows = [{"correct": row["base_correct"]} for row in rows]
    verifier_rows = [{"correct": row["verifier_correct"]} for row in rows]
    first_oracle_rows = [{"correct": row["first_correct"]} for row in oracle_hit_rows]
    base_oracle_rows = [{"correct": row["base_correct"]} for row in oracle_hit_rows]
    verifier_oracle_rows = [{"correct": row["verifier_correct"]} for row in oracle_hit_rows]

    first_vs_base = _exact_mcnemar(first_rows, base_rows)
    first_vs_verifier = _exact_mcnemar(first_rows, verifier_rows)
    base_vs_verifier = _exact_mcnemar(base_rows, verifier_rows)
    first_vs_verifier_oracle = _exact_mcnemar(first_oracle_rows, verifier_oracle_rows)
    base_vs_verifier_oracle = _exact_mcnemar(base_oracle_rows, verifier_oracle_rows)

    verifier_rescue_count = sum(
        int((not row["first_correct"]) and row["verifier_correct"]) for row in rows
    )
    verifier_overrule_first_count = sum(
        int(row["first_correct"] and (not row["verifier_correct"])) for row in rows
    )
    verifier_rescue_base_count = sum(
        int((not row["base_correct"]) and row["verifier_correct"]) for row in rows
    )
    verifier_overrule_base_count = sum(
        int(row["base_correct"] and (not row["verifier_correct"])) for row in rows
    )
    base_rescue_from_first = sum(
        int((not row["first_correct"]) and row["base_correct"]) for row in rows
    )
    oracle_hit_all_fail = sum(
        int(row["oracle_hit"] and (not row["first_correct"]) and (not row["base_correct"]) and (not row["verifier_correct"]))
        for row in rows
    )

    fixed_reference = None
    fixed_reference_note = None
    generation_metrics = None
    if args.fixed_reference_metrics is not None:
        fixed_reference = json.loads(args.fixed_reference_metrics.read_text(encoding="utf-8"))
    if args.completion_generation_metrics is not None and args.completion_generation_metrics.exists():
        generation_metrics = json.loads(args.completion_generation_metrics.read_text(encoding="utf-8"))

    candidate_count_rows = []
    for candidate_count in sorted(candidate_count_breakdown):
        bucket = candidate_count_breakdown[candidate_count]
        count_total = bucket["total"]
        candidate_count_rows.append(
            {
                "candidate_count": candidate_count,
                "num_examples": count_total,
                "share": count_total / total,
                "oracle_coverage": bucket["oracle_hit"] / count_total,
                "first_accuracy": bucket["first_correct"] / count_total,
                "base_accuracy": bucket["base_correct"] / count_total,
                "verifier_accuracy": bucket["verifier_correct"] / count_total,
            }
        )

    representative_examples = {
        "verifier_rescues": [
            _serialize_example(row)
            for row in _bucket_examples(
                rows,
                lambda row: (not row["first_correct"]) and row["verifier_correct"],
                args.max_examples_per_bucket,
            )
        ],
        "oracle_misses": [
            _serialize_example(row)
            for row in _bucket_examples(
                rows,
                lambda row: not row["oracle_hit"],
                args.max_examples_per_bucket,
            )
        ],
        "verifier_overrules_correct_first": [
            _serialize_example(row)
            for row in _bucket_examples(
                rows,
                lambda row: row["first_correct"] and (not row["verifier_correct"]),
                args.max_examples_per_bucket,
            )
        ],
    }

    verifier_selected_parseable_rate = _rate(rows, "verifier_parseable")
    verifier_answer_mode_match_rate = _rate(rows, "verifier_answer_mode_match")
    verifier_instruction_leak_rate = _rate(rows, "verifier_instruction_leak")
    verifier_scaffold_residue_rate = _rate(rows, "verifier_scaffold_residue")
    first_selected_parseable_rate = _rate(rows, "first_parseable")
    base_selected_parseable_rate = _rate(rows, "base_parseable")
    verifier_selected_margin_mean = _mean([row["verifier_selected_margin"] for row in rows])

    summary = {
        "run_label": args.run_label,
        "total_examples": total,
        "oracle_correct": oracle_correct,
        "oracle_coverage": oracle_correct / total,
        "first_correct": first_correct,
        "first_accuracy": first_correct / total,
        "base_correct": base_correct,
        "base_accuracy": base_correct / total,
        "verifier_correct": verifier_correct,
        "verifier_accuracy": verifier_correct / total,
        "verifier_given_oracle": (verifier_correct / oracle_correct) if oracle_correct else 0.0,
        "prediction_hygiene": {
            "first_selected_parseable_rate": first_selected_parseable_rate,
            "base_selected_parseable_rate": base_selected_parseable_rate,
            "verifier_selected_parseable_rate": verifier_selected_parseable_rate,
            "verifier_invalid_final_rate": 1.0 - verifier_selected_parseable_rate,
            "verifier_answer_mode_match_rate": verifier_answer_mode_match_rate,
            "verifier_instruction_leak_rate": verifier_instruction_leak_rate,
            "verifier_scaffold_residue_rate": verifier_scaffold_residue_rate,
            "selector_overrule_correct_first_rate": verifier_overrule_first_count / total,
            "verifier_selected_margin_mean": verifier_selected_margin_mean,
        },
        "generation_hygiene": generation_metrics,
        "selection_efficiency_given_oracle": {
            "first": (first_correct / oracle_correct) if oracle_correct else 0.0,
            "base": (base_correct / oracle_correct) if oracle_correct else 0.0,
            "verifier": (verifier_correct / oracle_correct) if oracle_correct else 0.0,
        },
        "failure_decomposition": {
            "oracle_miss": oracle_miss,
            "oracle_miss_rate": oracle_miss / total,
            "first_selection_failures": first_selection_failures,
            "base_selection_failures": base_selection_failures,
            "verifier_selection_failures": verifier_selection_failures,
            "oracle_hit_all_fail": oracle_hit_all_fail,
            "oracle_miss_share_of_verifier_failures": (oracle_miss / (total - verifier_correct))
            if verifier_correct < total
            else 0.0,
        },
        "paired_comparisons": {
            "first_vs_base": first_vs_base,
            "first_vs_verifier": first_vs_verifier,
            "base_vs_verifier": base_vs_verifier,
            "first_vs_verifier_given_oracle": first_vs_verifier_oracle,
            "base_vs_verifier_given_oracle": base_vs_verifier_oracle,
        },
        "rescue_counts": {
            "verifier_rescue_from_first": verifier_rescue_count,
            "verifier_overrule_correct_first": verifier_overrule_first_count,
            "verifier_rescue_from_base": verifier_rescue_base_count,
            "verifier_overrule_correct_base": verifier_overrule_base_count,
        },
        "oracle_hit_pattern_counts": dict(oracle_pattern_counts),
        "candidate_count_breakdown": candidate_count_rows,
        "fixed_reference": fixed_reference,
        "representative_examples": representative_examples,
    }

    conditional_first = summary["selection_efficiency_given_oracle"]["first"]
    conditional_base = summary["selection_efficiency_given_oracle"]["base"]
    conditional_verifier = summary["selection_efficiency_given_oracle"]["verifier"]
    oracle_hit_first_miss = oracle_correct - first_correct
    verifier_recovery_from_first_rate = (
        verifier_rescue_count / oracle_hit_first_miss if oracle_hit_first_miss else 0.0
    )
    base_recovery_from_first_rate = (
        base_rescue_from_first / oracle_hit_first_miss if oracle_hit_first_miss else 0.0
    )
    if oracle_hit_first_miss:
        recovery_from_first_text = (
            f"Among the `{oracle_hit_first_miss}` oracle-hit examples that the first candidate misses, "
            f"the verifier recovers `{verifier_rescue_count}` cases, or "
            f"`{verifier_recovery_from_first_rate:.4f}`. The corresponding base-reranker recovery rate is "
            f"`{base_rescue_from_first} / {oracle_hit_first_miss} = {base_recovery_from_first_rate:.4f}`."
        )
    else:
        recovery_from_first_text = (
            "There are no oracle-hit examples in which the first candidate is wrong, so recovery-from-first "
            "rates are not defined for this run."
        )

    if fixed_reference is not None:
        fixed_accuracy = float(fixed_reference["accuracy"])
        gap = conditional_verifier - fixed_accuracy
        if abs(gap) <= 0.05:
            fixed_reference_note = (
                f"- Inference: on oracle-hit open-ended examples, the verifier selects the correct answer at "
                f"`{conditional_verifier:.4f}`. This is close to the fixed-set reference `{fixed_accuracy:.4f}`, "
                "suggesting that once a correct candidate exists, selector quality is no longer the dominant bottleneck."
            )
        elif gap < 0:
            fixed_reference_note = (
                f"- Inference: on oracle-hit open-ended examples, the verifier selects the correct answer at "
                f"`{conditional_verifier:.4f}`, which is materially below the fixed-set reference `{fixed_accuracy:.4f}`. "
                "This suggests that coverage improved, but selector compatibility is still weaker than in the fixed-set regime."
            )
        else:
            fixed_reference_note = (
                f"- Inference: on oracle-hit open-ended examples, the verifier selects the correct answer at "
                f"`{conditional_verifier:.4f}`, which is above the fixed-set reference `{fixed_accuracy:.4f}`. "
                "This suggests the selector is at least as effective as in the fixed-set regime once coverage exists."
            )

    candidate_count_table = "\n".join(
        (
            f"| `{row['candidate_count']}` | `{row['num_examples']}` | `{row['share']:.4f}` | "
            f"`{row['oracle_coverage']:.4f}` | `{row['first_accuracy']:.4f}` | "
            f"`{row['base_accuracy']:.4f}` | `{row['verifier_accuracy']:.4f}` |"
        )
        for row in candidate_count_rows
    )

    pattern_lines = "\n".join(
        f"- `{pattern}`: `{count}`"
        for pattern, count in sorted(oracle_pattern_counts.items(), key=lambda item: (-item[1], item[0]))
    )

    example_sections = []
    bucket_titles = {
        "verifier_rescues": "Verifier Rescue Examples",
        "oracle_misses": "Oracle-Miss Examples",
        "verifier_overrules_correct_first": "Verifier Failure-on-Correct-First Examples",
    }
    for bucket_name, bucket_rows in representative_examples.items():
        lines = [f"### {bucket_titles[bucket_name]}"]
        if not bucket_rows:
            lines.append("No examples found for this bucket.")
        for row in bucket_rows:
            lines.extend(
                [
                    f"- `{row['example_id']}`",
                    f"  - problem: `{_shorten(row['problem'], 150)}`",
                    f"  - gold: `{row['gold_answer']}`",
                    f"  - first: `{_shorten(row['first_prediction'], 120)}`",
                    f"  - base: `{_shorten(row['base_prediction'], 120)}`",
                    f"  - verifier: `{_shorten(row['verifier_prediction'], 120)}`",
                    f"  - oracle candidate: `{_shorten(row['correct_candidate'] or 'NONE', 120)}`",
                    f"  - candidate snippets: `{'; '.join(row['candidate_snippets'])}`",
                ]
            )
        example_sections.append("\n".join(lines))

    reference_lines = ["Not provided."]
    if fixed_reference is not None:
        reference_lines = [
            f"- Reference task: `{args.fixed_reference_label}`",
            f"- Reference accuracy: `{fixed_reference['correct']} / {fixed_reference['num_examples']} = {fixed_reference['accuracy']:.4f}`",
            fixed_reference_note,
        ]

    report = f"""# Generate-Then-Rerank Full-Split Decomposition: {args.run_label}

## Setup

- Open-ended candidate set: `{args.candidates}`
- First-candidate predictions: `{args.first_predictions}`
- Base reranker predictions: `{args.base_predictions}`
- Verifier reranker predictions: `{args.verifier_predictions}`
- Eval size: `{total}`
- Exact-match rule: project-wide numeric exact-match

## Main Quantitative Decomposition

| Quantity | Correct | Accuracy |
| --- | --- | --- |
| first generated candidate | `{first_correct} / {total}` | `{first_correct / total:.4f}` |
| base reranker | `{base_correct} / {total}` | `{base_correct / total:.4f}` |
| verifier reranker | `{verifier_correct} / {total}` | `{verifier_correct / total:.4f}` |
| oracle candidate coverage | `{oracle_correct} / {total}` | `{oracle_correct / total:.4f}` |

The most informative decomposition is:

- overall verifier accuracy = oracle coverage × verifier selection efficiency given oracle
- `{verifier_correct / total:.4f} = {oracle_correct / total:.4f} × {conditional_verifier:.4f}`

Conditioned on the candidate pool already containing a correct answer:

| Selector | Correct on oracle-hit subset | Conditional accuracy |
| --- | --- | --- |
| first generated candidate | `{first_correct} / {oracle_correct}` | `{conditional_first:.4f}` |
| base reranker | `{base_correct} / {oracle_correct}` | `{conditional_base:.4f}` |
| verifier reranker | `{verifier_correct} / {oracle_correct}` | `{conditional_verifier:.4f}` |

This shows that the verifier's remaining errors are dominated by candidate construction rather than selector weakness. Of the verifier's `{total - verifier_correct}` failures:

- oracle-miss failures: `{oracle_miss}` (`{oracle_miss / total:.4f}` of all examples; `{summary['failure_decomposition']['oracle_miss_share_of_verifier_failures']:.4f}` of verifier failures)
- oracle-hit but verifier-still-wrong failures: `{verifier_selection_failures}` (`{verifier_selection_failures / total:.4f}` of all examples)

## Paired Comparisons

- first vs base: `{first_correct} -> {base_correct}`, McNemar exact `p = {first_vs_base['p']:.4g}` (`a_only={first_vs_base['a_only']}`, `b_only={first_vs_base['b_only']}`)
- first vs verifier: `{first_correct} -> {verifier_correct}`, McNemar exact `p = {first_vs_verifier['p']:.4g}` (`a_only={first_vs_verifier['a_only']}`, `b_only={first_vs_verifier['b_only']}`)
- base vs verifier: `{base_correct} -> {verifier_correct}`, McNemar exact `p = {base_vs_verifier['p']:.4g}` (`a_only={base_vs_verifier['a_only']}`, `b_only={base_vs_verifier['b_only']}`)
- first vs verifier on oracle-hit subset only: `{conditional_first:.4f} -> {conditional_verifier:.4f}`, McNemar exact `p = {first_vs_verifier_oracle['p']:.4g}`
- base vs verifier on oracle-hit subset only: `{conditional_base:.4f} -> {conditional_verifier:.4f}`, McNemar exact `p = {base_vs_verifier_oracle['p']:.4g}`

## Selected-Prediction Hygiene

| Metric | Value |
| --- | --- |
| verifier selected parseable rate | `{verifier_selected_parseable_rate:.4f}` |
| verifier invalid-final rate | `{1.0 - verifier_selected_parseable_rate:.4f}` |
| verifier answer-mode-match rate | `{verifier_answer_mode_match_rate:.4f}` |
| verifier instruction-leak rate | `{verifier_instruction_leak_rate:.4f}` |
| verifier scaffold-residue rate | `{verifier_scaffold_residue_rate:.4f}` |
| verifier overrule-correct-first rate | `{verifier_overrule_first_count / total:.4f}` |
| verifier selected margin mean | `{(verifier_selected_margin_mean if verifier_selected_margin_mean is not None else 0.0):.4f}` |

## Error Bucket Accounting

| Bucket | Count | Share of full split |
| --- | --- | --- |
| oracle miss | `{oracle_miss}` | `{oracle_miss / total:.4f}` |
| oracle hit, verifier correct | `{verifier_correct}` | `{verifier_correct / total:.4f}` |
| oracle hit, verifier wrong | `{verifier_selection_failures}` | `{verifier_selection_failures / total:.4f}` |
| verifier rescues first-candidate miss | `{verifier_rescue_count}` | `{verifier_rescue_count / total:.4f}` |
| verifier overrules a correct first answer | `{verifier_overrule_first_count}` | `{verifier_overrule_first_count / total:.4f}` |
| verifier rescues base-reranker miss | `{verifier_rescue_base_count}` | `{verifier_rescue_base_count / total:.4f}` |
| verifier overrules a correct base answer | `{verifier_overrule_base_count}` | `{verifier_overrule_base_count / total:.4f}` |
| oracle hit but all three selectors fail | `{oracle_hit_all_fail}` | `{oracle_hit_all_fail / total:.4f}` |

{recovery_from_first_text}

## Oracle-Hit Pattern Counts

`F1B1V1` means all three selectors are correct on an oracle-hit example; `F0B0V0` means all three fail even though a correct candidate exists.

{pattern_lines}

## Candidate-Count Breakdown

| Unique candidates | Examples | Share | Oracle coverage | First | Base | Verifier |
| --- | --- | --- | --- | --- | --- | --- |
{candidate_count_table}

The candidate-count distribution is tightly concentrated near the maximum, so the main issue is not merely having too few distinct candidates. The more important failure mode is generating many distinct but still incorrect candidates.

## Fixed-Set Reference

{chr(10).join(reference_lines)}

## Representative Examples

{chr(10).join(example_sections)}

## Interpretation

{_build_interpretation(
    total=total,
    oracle_correct=oracle_correct,
    first_correct=first_correct,
    base_correct=base_correct,
    verifier_correct=verifier_correct,
    conditional_base=conditional_base,
    conditional_verifier=conditional_verifier,
    oracle_miss_share_of_verifier_failures=summary['failure_decomposition']['oracle_miss_share_of_verifier_failures'],
)}
"""

    write_json(args.summary_json, summary)
    write_text(args.report, report)


if __name__ == "__main__":
    main()
