#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import (  # type: ignore
    answers_match,
    canonicalize_numeric_token,
    extract_choice_answer,
    extract_numeric_answer,
    normalize_answer,
)
from src.utils.io_utils import read_jsonl, write_json, write_jsonl, write_text  # type: ignore


EXPECTED_MODES = {
    "gsm8k": "numeric",
    "gsm8k_full_clean": "numeric",
    "competition_math_numeric": "numeric",
    "competition_math": "numeric",
    "mmlu_pro": "choice_letter",
    "gpqa_diamond": "choice_letter",
}

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
        description="Apply an offline selector policy to an existing verifier-scored candidate pool."
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--verifier-predictions", required=True, type=Path)
    parser.add_argument("--candidates-meta", type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--answer-mode", choices=("numeric", "choice_letter", "free_text"))
    parser.add_argument("--output-predictions", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--report-md", required=True, type=Path)
    parser.add_argument("--numeric-validity-mode", choices=("loose", "tail_anchored"), default="loose")
    parser.add_argument("--duplicate-penalty-mode", choices=("original_rank", "answer_group"), default="original_rank")
    parser.add_argument("--first-close-gap-override", type=float)
    parser.add_argument("--legacy-gsm8k-align", action="store_true")
    parser.add_argument("--max-examples", default=5, type=int)
    args = parser.parse_args()
    if args.legacy_gsm8k_align:
        if args.duplicate_penalty_mode == "original_rank":
            args.duplicate_penalty_mode = "answer_group"
        if args.first_close_gap_override is None:
            args.first_close_gap_override = 0.5
    return args


def extract_tail_anchored_numeric_answer(text: str) -> str | None:
    normalized = normalize_answer(text)
    if not normalized:
        return None
    stripped = normalized.strip()
    boxed_match = re.search(r"\\boxed\{([^}]+)\}\s*[.!?]*\s*$", stripped)
    if boxed_match:
        boxed = canonicalize_numeric_token(boxed_match.group(1))
        if boxed is not None:
            return boxed

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    tail = lines[-1] if lines else stripped
    tail_patterns = (
        r"^(?:final answer|answer)\s*[:\-]?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?)\s*[.!?]*$",
        r"^(?:the\s+)?answer\s+is\s+([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?)\s*[.!?]*$",
        r"^\(?([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?)\)?[.!?]*$",
        r"^.*=\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?)\s*[.!?]*$",
    )
    for pattern in tail_patterns:
        match = re.fullmatch(pattern, tail, flags=re.IGNORECASE)
        if match:
            canonical = canonicalize_numeric_token(match.group(1))
            if canonical is not None:
                return canonical
    return None


def parse_answer(text: str, answer_mode: str, numeric_validity_mode: str = "loose") -> str | None:
    if answer_mode == "choice_letter":
        return extract_choice_answer(text)
    if answer_mode == "numeric":
        if numeric_validity_mode == "tail_anchored":
            return extract_tail_anchored_numeric_answer(text)
        return extract_numeric_answer(text)
    normalized = normalize_answer(text)
    return normalized or None


def canonical_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", normalize_answer(text))
    return collapsed.strip()


def dedupe_key(text: str, answer_mode: str, numeric_validity_mode: str) -> str:
    parsed = parse_answer(text, answer_mode, numeric_validity_mode=numeric_validity_mode)
    if parsed is not None:
        return f"parsed::{parsed}"
    return f"text::{canonical_text(text)}"


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
    return lowered.startswith(
        (
            "let me think",
            "let's think",
            "okay, let's",
            "here's the thought process",
            "i will now provide",
            "the sentence structure",
            "use at most",
        )
    )


def is_obviously_malformed(text: str, answer_mode: str, numeric_validity_mode: str = "loose") -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return True
    if lowered in {"correct", "incorrect", "true", "false", "none", "n/a"}:
        return True
    if lowered.count("final answer:") > 1:
        return True
    if contains_instruction_leak(stripped):
        return True
    if stripped.endswith((":", "-", "=", "(", "[", "{", ",", " or", " and", " the")):
        return True
    if re.search(r"\[[^\]]*$|\([^\)]*$|\{[^\}]*$", stripped):
        return True
    if parse_answer(stripped, answer_mode, numeric_validity_mode=numeric_validity_mode) is None and len(stripped.split()) <= 2:
        return True
    return False


def clip(text: str, limit: int = 160) -> str:
    flat = re.sub(r"\s+", " ", text.strip())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def infer_answer_mode(args: argparse.Namespace, row: dict, meta: dict | None) -> str:
    if args.answer_mode:
        return args.answer_mode
    if meta is not None:
        meta_mode = str(meta.get("answer_mode", "")).strip().lower()
        if meta_mode:
            return meta_mode
    row_mode = str(row.get("answer_mode", "")).strip().lower()
    if row_mode:
        return row_mode
    benchmark = (args.benchmark or (meta or {}).get("dataset") or "").strip().lower()
    if benchmark in EXPECTED_MODES:
        return EXPECTED_MODES[benchmark]
    return "numeric"


def infer_benchmark(args: argparse.Namespace, meta: dict | None) -> str:
    if args.benchmark:
        return args.benchmark
    if meta is not None:
        dataset = str(meta.get("dataset", "")).strip()
        if dataset:
            return dataset
    return "unknown"


def load_meta_map(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    return {str(row["example_id"]): row for row in read_jsonl(path)}


def candidate_from_item(item: object, rank_position: int, answer_mode: str, numeric_validity_mode: str) -> dict:
    if isinstance(item, dict):
        answer = str(item.get("candidate_answer", item.get("prediction", "")))
        margin_value = item.get("margin")
        yes_score = item.get("yes_score")
        no_score = item.get("no_score")
        candidate_index = int(item.get("candidate_index", rank_position))
    else:
        answer = str(item)
        margin_value = None
        yes_score = None
        no_score = None
        candidate_index = rank_position
    margin = float(margin_value) if margin_value is not None else None
    parsed = parse_answer(answer, answer_mode, numeric_validity_mode=numeric_validity_mode)
    invalid = parsed is None
    instruction = contains_instruction_leak(answer)
    scaffold = contains_scaffold_residue(answer)
    malformed = is_obviously_malformed(answer, answer_mode, numeric_validity_mode=numeric_validity_mode)
    dirty_count = int(invalid) + int(instruction) + int(scaffold) + int(malformed)
    return {
        "candidate_index": candidate_index,
        "rank_position": rank_position,
        "candidate_answer": answer,
        "margin": margin,
        "yes_score": yes_score,
        "no_score": no_score,
        "parsed_answer": parsed,
        "tail_anchored_numeric_valid": answer_mode == "numeric" and extract_tail_anchored_numeric_answer(answer) is not None,
        "invalid": invalid,
        "instruction_leak": instruction,
        "scaffold_residue": scaffold,
        "malformed": malformed,
        "dirty_count": dirty_count,
        "is_clean": dirty_count == 0,
        "dedupe_key": dedupe_key(answer, answer_mode, numeric_validity_mode),
    }


def sort_key_margin(candidate: dict) -> tuple[float, int]:
    margin = candidate["margin"]
    if margin is None:
        margin = float("-inf")
    return (float(margin), -int(candidate["rank_position"]))


def choose_top_margin(candidates: list[dict]) -> dict:
    return max(candidates, key=sort_key_margin)


def choose_parseable_then_margin(candidates: list[dict]) -> dict:
    parseable = [candidate for candidate in candidates if not candidate["invalid"]]
    if parseable:
        return max(parseable, key=sort_key_margin)
    return choose_top_margin(candidates)


def get_first_generated_candidate(candidates: list[dict]) -> dict:
    return min(candidates, key=lambda candidate: (candidate["candidate_index"], candidate["rank_position"]))


def choose_first_if_close(candidates: list[dict], close_gap: float) -> tuple[dict, dict]:
    top = choose_top_margin(candidates)
    first = get_first_generated_candidate(candidates)
    first_margin = float(first["margin"]) if first["margin"] is not None else float("-inf")
    top_margin = float(top["margin"]) if top["margin"] is not None else float("-inf")
    if not first["invalid"] and (top_margin - first_margin) <= close_gap:
        return first, {"top_margin": top_margin, "first_margin": first_margin, "triggered": True}
    return top, {"top_margin": top_margin, "first_margin": first_margin, "triggered": False}


def apply_clean_dup_adjustment(
    candidates: list[dict], clean_penalty: float, duplicate_penalty: float, duplicate_penalty_mode: str
) -> list[dict]:
    by_original_order = sorted(candidates, key=lambda candidate: (candidate["candidate_index"], candidate["rank_position"]))
    seen: Counter[str] = Counter()
    group_sizes: Counter[str] = Counter(candidate["dedupe_key"] for candidate in candidates)
    duplicate_rank_map: dict[int, int] = {}
    for candidate in by_original_order:
        duplicate_rank_map[id(candidate)] = seen[candidate["dedupe_key"]]
        seen[candidate["dedupe_key"]] += 1

    adjusted: list[dict] = []
    for candidate in candidates:
        duplicate_rank = duplicate_rank_map[id(candidate)]
        duplicate_group_size = group_sizes[candidate["dedupe_key"]]
        if duplicate_penalty_mode == "answer_group":
            duplicate_penalty_units = max(0, duplicate_group_size - 1)
        else:
            duplicate_penalty_units = duplicate_rank
        margin = float(candidate["margin"]) if candidate["margin"] is not None else float("-inf")
        adjusted_score = margin - clean_penalty * candidate["dirty_count"] - duplicate_penalty * duplicate_penalty_units
        enriched = dict(candidate)
        enriched["duplicate_rank"] = duplicate_rank
        enriched["duplicate_group_size"] = duplicate_group_size
        enriched["duplicate_penalty_units"] = duplicate_penalty_units
        enriched["adjusted_score"] = adjusted_score
        adjusted.append(enriched)
    return adjusted


def choose_clean_dup(
    candidates: list[dict],
    clean_penalty: float,
    duplicate_penalty: float,
    duplicate_penalty_mode: str,
    legacy_gsm8k_align: bool,
) -> tuple[dict, list[dict]]:
    adjusted = apply_clean_dup_adjustment(candidates, clean_penalty, duplicate_penalty, duplicate_penalty_mode)
    if legacy_gsm8k_align:
        best = max(
            adjusted,
            key=lambda candidate: (
                candidate["adjusted_score"],
                -candidate["duplicate_group_size"],
                candidate["margin"] if candidate["margin"] is not None else float("-inf"),
                -candidate["rank_position"],
            ),
        )
    else:
        best = max(
            adjusted,
            key=lambda candidate: (
                candidate["adjusted_score"],
                candidate["margin"] if candidate["margin"] is not None else float("-inf"),
                -candidate["rank_position"],
            ),
        )
    return best, adjusted


def choose_clean_dup_firstclose(
    candidates: list[dict],
    clean_penalty: float,
    duplicate_penalty: float,
    close_gap: float,
    duplicate_penalty_mode: str,
    legacy_gsm8k_align: bool,
) -> tuple[dict, dict, list[dict]]:
    best, adjusted = choose_clean_dup(candidates, clean_penalty, duplicate_penalty, duplicate_penalty_mode, legacy_gsm8k_align)
    first = min(adjusted, key=lambda candidate: (candidate["candidate_index"], candidate["rank_position"]))
    best_score = float(best["adjusted_score"])
    first_score = float(first["adjusted_score"])
    first_is_clean = bool(first["is_clean"])
    if legacy_gsm8k_align and first.get("tail_anchored_numeric_valid") is not None:
        best_is_answerlike = bool(best.get("tail_anchored_numeric_valid", False))
        if best_is_answerlike:
            first_is_clean = (
                bool(first.get("tail_anchored_numeric_valid", False))
                and not first["instruction_leak"]
                and not first["scaffold_residue"]
                and not first["malformed"]
            )
    if first_is_clean and (best_score - first_score) <= close_gap:
        return first, {
            "best_adjusted_score": best_score,
            "first_adjusted_score": first_score,
            "triggered": True,
            "first_is_clean_effective": first_is_clean,
        }, adjusted
    return best, {
        "best_adjusted_score": best_score,
        "first_adjusted_score": first_score,
        "triggered": False,
        "first_is_clean_effective": first_is_clean,
    }, adjusted


def apply_rule(candidates: list[dict], rule: str, args: argparse.Namespace) -> tuple[dict, dict]:
    if rule == "baseline_top_margin":
        selected = choose_top_margin(candidates)
        return selected, {"family": rule}
    if rule == "parseable_then_margin":
        selected = choose_parseable_then_margin(candidates)
        return selected, {"family": rule}
    if rule.startswith("first_if_close_"):
        close_gap = float(rule.split("_")[-1])
        effective_close_gap = args.first_close_gap_override if args.first_close_gap_override is not None else close_gap
        selected, details = choose_first_if_close(candidates, effective_close_gap)
        return selected, {"family": "first_if_close", "close_gap": effective_close_gap, "configured_close_gap": close_gap, **details}
    if rule.startswith("clean_dup_firstclose_"):
        _, _, _, clean_penalty_str, duplicate_penalty_str, close_gap_str = rule.split("_", 5)
        clean_penalty = float(clean_penalty_str)
        duplicate_penalty = float(duplicate_penalty_str)
        close_gap = float(close_gap_str)
        effective_close_gap = args.first_close_gap_override if args.first_close_gap_override is not None else close_gap
        selected, details, adjusted = choose_clean_dup_firstclose(
            candidates,
            clean_penalty,
            duplicate_penalty,
            effective_close_gap,
            args.duplicate_penalty_mode,
            args.legacy_gsm8k_align,
        )
        return selected, {
            "family": "clean_dup_firstclose",
            "clean_penalty": clean_penalty,
            "duplicate_penalty": duplicate_penalty,
            "close_gap": effective_close_gap,
            "configured_close_gap": close_gap,
            "duplicate_penalty_mode": args.duplicate_penalty_mode,
            "candidates": adjusted,
            **details,
        }
    if rule.startswith("clean_dup_"):
        _, _, clean_penalty_str, duplicate_penalty_str = rule.split("_", 3)
        clean_penalty = float(clean_penalty_str)
        duplicate_penalty = float(duplicate_penalty_str)
        selected, adjusted = choose_clean_dup(
            candidates,
            clean_penalty,
            duplicate_penalty,
            args.duplicate_penalty_mode,
            args.legacy_gsm8k_align,
        )
        return selected, {
            "family": "clean_dup",
            "clean_penalty": clean_penalty,
            "duplicate_penalty": duplicate_penalty,
            "duplicate_penalty_mode": args.duplicate_penalty_mode,
            "candidates": adjusted,
        }
    raise ValueError(f"Unsupported rule: {rule}")


def base_selected_candidate(row: dict, candidates: list[dict]) -> dict:
    prediction = str(row.get("prediction", ""))
    matches = [candidate for candidate in candidates if candidate["candidate_answer"] == prediction]
    if matches:
        return max(matches, key=sort_key_margin)
    return choose_top_margin(candidates)


def accuracy_rate(values: list[bool]) -> float:
    return sum(int(value) for value in values) / len(values) if values else 0.0


def rate_from_counter(counter: Counter[str], key: str, total: int) -> float:
    if total == 0:
        return 0.0
    return counter[key] / total


def summary_rule_text(rule: str, args: argparse.Namespace) -> list[str]:
    if rule == "baseline_top_margin":
        return ["直接保留 verifier 原始 top-margin 候选，不做任何离线 selector 修正。"]
    if rule == "parseable_then_margin":
        return ["先过滤到可解析候选，再按 verifier margin 选 top；若全不可解析，则退回 top-margin。"]
    if rule.startswith("first_if_close_"):
        close_gap = float(rule.split("_")[-1])
        return [
            "先按 verifier margin 找最优候选。",
            f"如果原始 first generated candidate 可解析，且与最优 margin 差距不超过 `{close_gap}`，则保留 first。",
        ]
    if rule.startswith("clean_dup_firstclose_"):
        parts = rule.split("_")
        clean_penalty = float(parts[3])
        duplicate_penalty = float(parts[4])
        close_gap = float(parts[5])
        return [
            f"每个候选的 adjusted score = verifier margin - `{clean_penalty}` × dirty_count - `{duplicate_penalty}` × duplicate penalty units。",
            "dirty_count 统计 invalid final / instruction leak / scaffold residue / obvious malformed 四类问题。",
            f"duplicate penalty mode = `{args.duplicate_penalty_mode}`。",
            f"然后做 first-close protection：若原始 first candidate 干净且 adjusted score 距最佳不超过 `{args.first_close_gap_override if args.first_close_gap_override is not None else close_gap}`，则保留 first。",
        ]
    if rule.startswith("clean_dup_"):
        parts = rule.split("_")
        clean_penalty = float(parts[2])
        duplicate_penalty = float(parts[3])
        return [
            f"每个候选的 adjusted score = verifier margin - `{clean_penalty}` × dirty_count - `{duplicate_penalty}` × duplicate penalty units。",
            f"duplicate penalty mode = `{args.duplicate_penalty_mode}`。",
            "不做 first-close protection，直接选 adjusted score 最高的候选。",
        ]
    return ["规则说明缺失。"]


def main() -> None:
    args = parse_args()
    meta_map = load_meta_map(args.candidates_meta)
    rows = list(read_jsonl(args.verifier_predictions))

    output_rows: list[dict] = []
    baseline_correct_flags: list[bool] = []
    new_correct_flags: list[bool] = []
    first_correct_flags: list[bool] = []
    switched_examples = 0
    gain_count = 0
    loss_count = 0
    gains: list[dict] = []
    losses: list[dict] = []
    baseline_metrics: Counter[str] = Counter()
    new_metrics: Counter[str] = Counter()
    rule_triggers: Counter[str] = Counter()
    answer_mode_counter: Counter[str] = Counter()
    benchmark_counter: Counter[str] = Counter()

    for row in rows:
        example_id = str(row["example_id"])
        meta = meta_map.get(example_id)
        answer_mode = infer_answer_mode(args, row, meta)
        benchmark = infer_benchmark(args, meta)
        answer_mode_counter[answer_mode] += 1
        benchmark_counter[benchmark] += 1

        raw_candidates = row.get("candidates") or []
        candidates = [
            candidate_from_item(item, rank_position, answer_mode, args.numeric_validity_mode)
            for rank_position, item in enumerate(raw_candidates)
        ]
        if not candidates:
            continue

        gold_answer = str((meta or {}).get("gold_answer", row.get("gold_answer", "")))
        problem = str((meta or {}).get("problem", row.get("problem", "")))

        baseline_candidate = base_selected_candidate(row, candidates)
        selected_candidate, rule_details = apply_rule(candidates, args.rule, args)
        first_candidate = get_first_generated_candidate(candidates)

        baseline_prediction = baseline_candidate["candidate_answer"]
        selected_prediction = selected_candidate["candidate_answer"]
        first_prediction = first_candidate["candidate_answer"]

        first_correct = answers_match(first_prediction, gold_answer, answer_mode=answer_mode)
        baseline_correct = answers_match(baseline_prediction, gold_answer, answer_mode=answer_mode)
        new_correct = answers_match(selected_prediction, gold_answer, answer_mode=answer_mode)

        first_correct_flags.append(first_correct)
        baseline_correct_flags.append(baseline_correct)
        new_correct_flags.append(new_correct)

        for prefix, candidate in (("baseline", baseline_candidate), ("new", selected_candidate)):
            metrics = baseline_metrics if prefix == "baseline" else new_metrics
            metrics["invalid"] += int(candidate["invalid"])
            metrics["instruction_leak"] += int(candidate["instruction_leak"])
            metrics["scaffold_residue"] += int(candidate["scaffold_residue"])
            metrics["malformed"] += int(candidate["malformed"])
            metrics["clean"] += int(candidate["is_clean"])
            metrics["overrule_correct_first"] += int(first_correct and not (baseline_correct if prefix == "baseline" else new_correct))

        if baseline_candidate["candidate_index"] != selected_candidate["candidate_index"]:
            switched_examples += 1

        if (not baseline_correct) and new_correct:
            gain_count += 1
            if len(gains) < args.max_examples:
                gains.append(
                    {
                        "example_id": example_id,
                        "gold": gold_answer,
                        "problem": problem,
                        "base_idx": int(baseline_candidate["candidate_index"]),
                        "base_pred": clip(baseline_prediction),
                        "new_idx": int(selected_candidate["candidate_index"]),
                        "new_pred": clip(selected_prediction),
                    }
                )
        if baseline_correct and (not new_correct):
            loss_count += 1
            if len(losses) < args.max_examples:
                losses.append(
                    {
                        "example_id": example_id,
                        "gold": gold_answer,
                        "problem": problem,
                        "base_idx": int(baseline_candidate["candidate_index"]),
                        "base_pred": clip(baseline_prediction),
                        "new_idx": int(selected_candidate["candidate_index"]),
                        "new_pred": clip(selected_prediction),
                    }
                )

        if rule_details.get("triggered"):
            rule_triggers["triggered"] += 1
        else:
            rule_triggers["not_triggered"] += 1

        output_rows.append(
            {
                "example_id": example_id,
                "prediction": selected_prediction,
                "gold_answer": gold_answer,
                "correct": new_correct,
                "answer_mode": answer_mode,
                "benchmark": benchmark,
                "selector_rule": args.rule,
                "baseline_prediction": baseline_prediction,
                "baseline_correct": baseline_correct,
                "first_prediction": first_prediction,
                "first_correct": first_correct,
                "first_candidate_index": int(first_candidate["candidate_index"]),
                "selected_candidate_index": int(selected_candidate["candidate_index"]),
                "baseline_candidate_index": int(baseline_candidate["candidate_index"]),
                "selection_metadata": {
                    "rule_family": rule_details.get("family"),
                    "rule_triggered": bool(rule_details.get("triggered", False)),
                    "top_margin": choose_top_margin(candidates)["margin"],
                    "selected_margin": selected_candidate["margin"],
                    "selected_adjusted_score": selected_candidate.get("adjusted_score"),
                },
                "candidates": raw_candidates,
            }
        )

    total = len(output_rows)
    baseline_accuracy = accuracy_rate(baseline_correct_flags)
    new_accuracy = accuracy_rate(new_correct_flags)
    first_accuracy = accuracy_rate(first_correct_flags)
    delta_accuracy = new_accuracy - baseline_accuracy

    summary = {
        "run_label": args.run_label,
        "rule": args.rule,
        "rule_description": summary_rule_text(args.rule, args),
        "numeric_validity_mode": args.numeric_validity_mode,
        "duplicate_penalty_mode": args.duplicate_penalty_mode,
        "first_close_gap_override": args.first_close_gap_override,
        "legacy_gsm8k_align": args.legacy_gsm8k_align,
        "verifier_predictions": str(args.verifier_predictions.resolve()),
        "candidates_meta": str(args.candidates_meta.resolve()) if args.candidates_meta is not None else None,
        "benchmark_counts": dict(benchmark_counter),
        "answer_mode_counts": dict(answer_mode_counter),
        "total_examples": total,
        "first_accuracy": first_accuracy,
        "baseline_top_accuracy": baseline_accuracy,
        "selector_accuracy": new_accuracy,
        "delta_accuracy": delta_accuracy,
        "switched_examples": switched_examples,
        "gain_count": gain_count,
        "loss_count": loss_count,
        "baseline": {
            "invalid_rate": rate_from_counter(baseline_metrics, "invalid", total),
            "instruction_leak_rate": rate_from_counter(baseline_metrics, "instruction_leak", total),
            "scaffold_residue_rate": rate_from_counter(baseline_metrics, "scaffold_residue", total),
            "malformed_rate": rate_from_counter(baseline_metrics, "malformed", total),
            "clean_rate": rate_from_counter(baseline_metrics, "clean", total),
            "overrule_correct_first_rate": rate_from_counter(baseline_metrics, "overrule_correct_first", total),
        },
        "selector": {
            "invalid_rate": rate_from_counter(new_metrics, "invalid", total),
            "instruction_leak_rate": rate_from_counter(new_metrics, "instruction_leak", total),
            "scaffold_residue_rate": rate_from_counter(new_metrics, "scaffold_residue", total),
            "malformed_rate": rate_from_counter(new_metrics, "malformed", total),
            "clean_rate": rate_from_counter(new_metrics, "clean", total),
            "overrule_correct_first_rate": rate_from_counter(new_metrics, "overrule_correct_first", total),
        },
        "rule_trigger_counts": dict(rule_triggers),
        "representative_gains": gains,
        "representative_losses": losses,
        "output_predictions": str(args.output_predictions.resolve()),
        "output_report": str(args.report_md.resolve()),
    }

    lines = [
        f"# Fixed-Pool Selector Policy Report: {args.run_label}",
        "",
        "## Config",
        "",
        f"- rule: `{args.rule}`",
        f"- verifier predictions: `{args.verifier_predictions}`",
        f"- candidates meta: `{args.candidates_meta}`" if args.candidates_meta is not None else "- candidates meta: `None`",
        f"- total examples: `{total}`",
        f"- numeric validity mode: `{args.numeric_validity_mode}`",
        f"- duplicate penalty mode: `{args.duplicate_penalty_mode}`",
        f"- first-close gap override: `{args.first_close_gap_override}`",
        f"- legacy GSM8K align: `{args.legacy_gsm8k_align}`",
        "",
        "## Rule Interpretation",
        "",
    ]
    for item in summary_rule_text(args.rule, args):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Main Readout",
            "",
            f"- first accuracy: `{first_accuracy:.4f}`",
            f"- baseline top-margin accuracy: `{baseline_accuracy:.4f}`",
            f"- selector accuracy: `{new_accuracy:.4f}`",
            f"- delta: `{delta_accuracy:+.4f}`",
            f"- switched examples: `{switched_examples}`",
            f"- gains: `{gain_count}`",
            f"- losses: `{loss_count}`",
            "",
            "## Hygiene Delta",
            "",
            f"- invalid final: `{summary['baseline']['invalid_rate']:.4f} -> {summary['selector']['invalid_rate']:.4f}`",
            f"- instruction leak: `{summary['baseline']['instruction_leak_rate']:.4f} -> {summary['selector']['instruction_leak_rate']:.4f}`",
            f"- scaffold residue: `{summary['baseline']['scaffold_residue_rate']:.4f} -> {summary['selector']['scaffold_residue_rate']:.4f}`",
            f"- malformed selected: `{summary['baseline']['malformed_rate']:.4f} -> {summary['selector']['malformed_rate']:.4f}`",
            f"- clean selected: `{summary['baseline']['clean_rate']:.4f} -> {summary['selector']['clean_rate']:.4f}`",
            f"- overrule-correct-first: `{summary['baseline']['overrule_correct_first_rate']:.4f} -> {summary['selector']['overrule_correct_first_rate']:.4f}`",
            "",
            "## Representative Gains",
            "",
        ]
    )
    if gains:
        for item in gains:
            lines.extend(
                [
                    f"- `{item['example_id']}` | gold=`{item['gold']}` | base_idx=`{item['base_idx']}` -> new_idx=`{item['new_idx']}`",
                    f"  - problem: {item['problem']}",
                    f"  - base: `{item['base_pred']}`",
                    f"  - new: `{item['new_pred']}`",
                ]
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Representative Losses", ""])
    if losses:
        for item in losses:
            lines.extend(
                [
                    f"- `{item['example_id']}` | gold=`{item['gold']}` | base_idx=`{item['base_idx']}` -> new_idx=`{item['new_idx']}`",
                    f"  - problem: {item['problem']}",
                    f"  - base: `{item['base_pred']}`",
                    f"  - new: `{item['new_pred']}`",
                ]
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append(f"JSON artifact: `{args.summary_json}`")

    write_jsonl(args.output_predictions, output_rows)
    write_json(args.summary_json, summary)
    write_text(args.report_md, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
