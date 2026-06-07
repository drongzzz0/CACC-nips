from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.motif_utils import infer_candidate_tag
from src.eval.evaluate_predictions import (
    answers_match,
    extract_choice_answer,
    extract_numeric_answer,
    normalize_answer,
)
from src.utils.io_utils import read_jsonl, write_json, write_jsonl

REQUIRED_EVENT_FIELDS = [
    "example_id",
    "candidate_id",
    "origin",
    "method_variant",
    "source_slot_id",
    "source_bucket",
    "before_parseable",
    "after_parseable",
    "before_answer_mode_match",
    "after_answer_mode_match",
    "before_correct",
    "after_correct",
    "selected_by_verifier",
    "selected_correct",
    "verifier_score",
    "repair_applied",
    "repair_type",
    "parser_status_before",
    "parser_status_after",
    "token_count_before",
    "token_count_after",
    "generation_seed",
    "repair_seed",
]

SUMMARY_FIELDS = [
    "split",
    "method",
    "base_model",
    "N",
    "retained_cap",
    "repair_or_fresh_count",
    "parser",
    "verifier",
    "first",
    "base",
    "oracle",
    "final",
    "V_given_O",
    "parseable",
    "unique",
    "duplicate_rate",
    "generated_tokens",
    "repair_tokens",
    "total_tokens",
    "verifier_calls",
    "wall_clock",
    "gpu_sec",
]

HYGIENE_FIELDS = [
    "split",
    "method",
    "parseable",
    "selected_parseable",
    "invalid_final",
    "answer_mode_match",
    "instruction_leak",
    "scaffold_residue",
    "malformed_wrapper",
    "duplicate_rate",
    "unique_candidates",
    "avg_candidate_tokens",
    "selected_candidate_tokens",
]

PAIRED_FIELDS = ["example_id", "gold_answer", "base_pool_correct", "salvage_amc_sch_correct"]

INSTRUCTION_PATTERNS = (
    "use at most",
    "exactly one last line",
    "do not mention",
    "do not output",
    "the solution should",
    "the solution must",
    "you are improving",
    "you are finishing",
    "do not include",
)

SCAFFOLD_PATTERNS = (
    "here's a complete solution",
    "here is the correct solution",
    "here's the candidate solution",
    "let's think",
    "let us think",
    "okay, let's",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build E00 candidate-pool salvage provenance smoke artifacts from existing candidate pools."
    )
    parser.add_argument("--base-pool", required=True, type=Path)
    parser.add_argument("--salvage-pool", required=True, type=Path)
    parser.add_argument("--first-predictions", required=True, type=Path)
    parser.add_argument("--base-rerank-predictions", required=True, type=Path)
    parser.add_argument("--verifier-predictions", required=True, type=Path)
    parser.add_argument("--run-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", default="E00_logging_smoke")
    parser.add_argument("--split", default="gsm8k_clean_eval128")
    parser.add_argument("--base-model", default="Qwen3-1.7B proposer / Qwen3-1.7B verifier family")
    parser.add_argument("--verifier", default="qwen3_17b_verifier512")
    parser.add_argument("--generation-seed", default="unknown_existing_artifact")
    parser.add_argument("--repair-seed", default="unknown_existing_artifact")
    return parser.parse_args()


def load_by_example(path: Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in read_jsonl(path)}


def normalize_for_match(text: str) -> str:
    return normalize_answer(str(text))


def candidate_key(text: str, answer_mode: str) -> str:
    text = str(text)
    if answer_mode == "choice_letter":
        choice = extract_choice_answer(text)
        if choice is not None:
            return f"choice:{choice}"
    numeric = extract_numeric_answer(text)
    if numeric is not None:
        return f"num:{numeric}"
    return f"text:{normalize_for_match(text)}"


def parse_status(text: str, answer_mode: str) -> str:
    if answer_mode == "choice_letter":
        return "parseable_choice" if extract_choice_answer(text) is not None else "unparseable_choice"
    return "parseable_numeric" if extract_numeric_answer(text) is not None else "unparseable_numeric"


def is_parseable(text: str, answer_mode: str) -> bool:
    return parse_status(text, answer_mode).startswith("parseable")


def has_instruction_leak(text: str) -> bool:
    lowered = str(text).lower()
    return any(pattern in lowered for pattern in INSTRUCTION_PATTERNS)


def has_scaffold_residue(text: str) -> bool:
    lowered = str(text).lower()
    return any(pattern in lowered for pattern in SCAFFOLD_PATTERNS)


def malformed_wrapper(text: str) -> bool:
    lowered = str(text).lower()
    return lowered.count("final answer:") > 1 or "final answer: final answer:" in lowered


def source_bucket(problem: str, text: str, answer_mode: str) -> str:
    tag = infer_candidate_tag(problem, text)
    quality = tag.quality.label
    status = parse_status(text, answer_mode)
    flags: list[str] = []
    if has_instruction_leak(text):
        flags.append("instruction_leak")
    if has_scaffold_residue(text):
        flags.append("scaffold_residue")
    if malformed_wrapper(text):
        flags.append("malformed_wrapper")
    suffix = "+".join(flags) if flags else "cleanish"
    return f"{quality}:{status}:{suffix}"


def token_count(text: str) -> int:
    return len(str(text).split())


def load_prediction_correct(path: Path) -> dict[str, bool]:
    out = {}
    for row in read_jsonl(path):
        answer_mode = str(row.get("answer_mode", "numeric"))
        out[str(row["example_id"])] = bool(
            row.get("correct", answers_match(str(row.get("prediction", "")), str(row.get("gold_answer", "")), answer_mode))
        )
    return out


def load_verifier_lookup(path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for row in read_jsonl(path):
        answer_mode = str(row.get("answer_mode", "numeric"))
        selected = str(row.get("prediction", "")).strip()
        selected_key = candidate_key(selected, answer_mode)
        candidate_scores = {}
        for cand in row.get("candidates", []):
            text = str(cand.get("candidate_answer", ""))
            payload = {
                "candidate_answer": text,
                "yes_score": cand.get("yes_score"),
                "no_score": cand.get("no_score"),
                "margin": cand.get("margin"),
            }
            candidate_scores[text.strip()] = payload
            candidate_scores.setdefault(normalize_for_match(text), payload)
            candidate_scores.setdefault(candidate_key(text, answer_mode), payload)
        lookup[str(row["example_id"])] = {
            "prediction": selected,
            "selected_key": selected_key,
            "correct": bool(row.get("correct", answers_match(selected, str(row.get("gold_answer", "")), answer_mode))),
            "candidate_scores": candidate_scores,
        }
    return lookup


def lookup_score(verifier_info: dict | None, text: str, answer_mode: str) -> float | str:
    if not verifier_info:
        return ""
    scores = verifier_info.get("candidate_scores", {})
    for key in (str(text).strip(), normalize_for_match(text), candidate_key(text, answer_mode)):
        if key in scores:
            value = scores[key].get("margin")
            return "" if value is None else value
    return ""


def selected_candidate_index(candidates: list[str], verifier_info: dict | None, answer_mode: str) -> int | None:
    if not verifier_info:
        return None
    selected = str(verifier_info.get("prediction", "")).strip()
    if not selected:
        return None
    for idx, candidate in enumerate(candidates):
        if str(candidate).strip() == selected:
            return idx
    selected_key = verifier_info.get("selected_key")
    for idx, candidate in enumerate(candidates):
        if candidate_key(candidate, answer_mode) == selected_key:
            return idx
    selected_norm = normalize_for_match(selected)
    for idx, candidate in enumerate(candidates):
        if normalize_for_match(candidate) == selected_norm:
            return idx
    return None


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def build_method_summary(
    *,
    split: str,
    method: str,
    pool_rows: list[dict],
    first_correct: dict[str, bool],
    base_correct: dict[str, bool],
    verifier_correct: dict[str, bool],
    base_model: str,
    verifier: str,
    run_summary: dict | None = None,
) -> dict:
    total = len(pool_rows)
    oracle_hits = 0
    parseable_total = 0
    unique_total = 0
    duplicate_total = 0
    token_total = 0
    retained_cap = 0
    for row in pool_rows:
        answer_mode = str(row.get("answer_mode", "numeric"))
        gold = str(row.get("gold_answer", ""))
        candidates = [str(c) for c in row.get("candidates", [])]
        retained_cap = max(retained_cap, len(candidates))
        keys = [candidate_key(c, answer_mode) for c in candidates]
        unique_total += len(set(keys))
        duplicate_total += max(0, len(keys) - len(set(keys)))
        parseable_total += sum(1 for c in candidates if is_parseable(c, answer_mode))
        token_total += sum(token_count(c) for c in candidates)
        if any(answers_match(c, gold, answer_mode) for c in candidates):
            oracle_hits += 1
    n_candidates = sum(len(row.get("candidates", [])) for row in pool_rows)
    first_acc = ratio(
        sum(answers_match(str(row.get("candidates", [""])[0]), str(row.get("gold_answer", "")), str(row.get("answer_mode", "numeric"))) for row in pool_rows if row.get("candidates")),
        total,
    )
    base_acc: float | str = ""
    final_acc: float | str = ""
    oracle = ratio(oracle_hits, total)
    if run_summary and method == "salvage_amc_sch":
        first_acc = float(run_summary.get("first_accuracy", first_acc))
        base_acc = float(run_summary.get("base_accuracy", 0.0))
        final_acc = float(run_summary.get("verifier_accuracy", 0.0))
        oracle = float(run_summary.get("oracle_coverage", oracle))
    return {
        "split": split,
        "method": method,
        "base_model": base_model,
        "N": total,
        "retained_cap": retained_cap,
        "repair_or_fresh_count": "reconstructed_from_pool_diff" if method == "salvage_amc_sch" else 0,
        "parser": "project_numeric_exact_match",
        "verifier": verifier,
        "first": first_acc,
        "base": base_acc,
        "oracle": oracle,
        "final": final_acc,
        "V_given_O": ratio(float(final_acc), oracle) if final_acc != "" else "",
        "parseable": ratio(parseable_total, n_candidates),
        "unique": ratio(unique_total, total),
        "duplicate_rate": ratio(duplicate_total, n_candidates),
        "generated_tokens": "",
        "repair_tokens": "",
        "total_tokens": token_total,
        "verifier_calls": n_candidates,
        "wall_clock": "",
        "gpu_sec": "",
    }


def build_hygiene_row(split: str, method: str, pool_rows: list[dict], verifier_lookup: dict[str, dict]) -> dict:
    total_candidates = 0
    parseable_count = 0
    answer_mode_match = 0
    invalid_final = 0
    instruction_leak = 0
    scaffold = 0
    malformed = 0
    unique_total = 0
    duplicate_total = 0
    token_total = 0
    selected_parseable = 0
    selected_tokens = []
    for row in pool_rows:
        example_id = str(row["example_id"])
        answer_mode = str(row.get("answer_mode", "numeric"))
        candidates = [str(c) for c in row.get("candidates", [])]
        keys = [candidate_key(c, answer_mode) for c in candidates]
        unique_total += len(set(keys))
        duplicate_total += max(0, len(keys) - len(set(keys)))
        verifier_info = verifier_lookup.get(example_id)
        for candidate in candidates:
            total_candidates += 1
            parseable = is_parseable(candidate, answer_mode)
            parseable_count += int(parseable)
            answer_mode_match += int(parseable)
            invalid_final += int(not parseable)
            instruction_leak += int(has_instruction_leak(candidate))
            scaffold += int(has_scaffold_residue(candidate))
            malformed += int(malformed_wrapper(candidate))
            token_total += token_count(candidate)
        selected_idx = selected_candidate_index(candidates, verifier_info, answer_mode)
        for idx, candidate in enumerate(candidates):
            if idx == selected_idx:
                selected_parseable += int(is_parseable(candidate, answer_mode))
                selected_tokens.append(token_count(candidate))
    n_examples = len(pool_rows)
    return {
        "split": split,
        "method": method,
        "parseable": ratio(parseable_count, total_candidates),
        "selected_parseable": ratio(selected_parseable, n_examples),
        "invalid_final": ratio(invalid_final, total_candidates),
        "answer_mode_match": ratio(answer_mode_match, total_candidates),
        "instruction_leak": ratio(instruction_leak, total_candidates),
        "scaffold_residue": ratio(scaffold, total_candidates),
        "malformed_wrapper": ratio(malformed, total_candidates),
        "duplicate_rate": ratio(duplicate_total, total_candidates),
        "unique_candidates": ratio(unique_total, n_examples),
        "avg_candidate_tokens": ratio(token_total, total_candidates),
        "selected_candidate_tokens": ratio(sum(selected_tokens), len(selected_tokens)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = list(read_jsonl(args.base_pool))
    salvage_rows = list(read_jsonl(args.salvage_pool))
    salvage_by_id = {str(row["example_id"]): row for row in salvage_rows}
    first_correct = load_prediction_correct(args.first_predictions)
    base_correct = load_prediction_correct(args.base_rerank_predictions)
    verifier_correct = load_prediction_correct(args.verifier_predictions)
    verifier_lookup = load_verifier_lookup(args.verifier_predictions)
    run_summary = json.loads(args.run_summary.read_text(encoding="utf-8"))

    events = []
    paired_rows = []
    provenance_failures = []
    stats = Counter()

    for base_row in base_rows:
        example_id = str(base_row["example_id"])
        salvage_row = salvage_by_id.get(example_id)
        if salvage_row is None:
            provenance_failures.append({"example_id": example_id, "reason": "missing_salvage_row"})
            continue
        answer_mode = str(base_row.get("answer_mode", salvage_row.get("answer_mode", "numeric")))
        problem = str(base_row.get("problem", salvage_row.get("problem", "")))
        gold = str(base_row.get("gold_answer", salvage_row.get("gold_answer", "")))
        base_candidates = [str(c) for c in base_row.get("candidates", [])]
        salvage_candidates = [str(c) for c in salvage_row.get("candidates", [])]
        verifier_info = verifier_lookup.get(example_id)
        max_len = max(len(base_candidates), len(salvage_candidates))
        selected_traced = False

        paired_rows.append(
            {
                "example_id": example_id,
                "gold_answer": gold,
                "base_pool_correct": any(answers_match(c, gold, answer_mode) for c in base_candidates),
                "salvage_amc_sch_correct": verifier_correct.get(example_id, False),
            }
        )

        for idx in range(max_len):
            before = base_candidates[idx] if idx < len(base_candidates) else ""
            after = salvage_candidates[idx] if idx < len(salvage_candidates) else ""
            if not before and not after:
                continue
            repair_applied = bool(before != after and after)
            selected_idx = selected_candidate_index(salvage_candidates, verifier_info, answer_mode)
            selected = idx == selected_idx
            if selected:
                selected_traced = True
            before_parseable = is_parseable(before, answer_mode) if before else False
            after_parseable = is_parseable(after, answer_mode) if after else False
            before_correct = answers_match(before, gold, answer_mode) if before else False
            after_correct = answers_match(after, gold, answer_mode) if after else False
            bucket = source_bucket(problem, before or after, answer_mode)
            event = {
                "example_id": example_id,
                "candidate_id": f"{example_id}::slot{idx}",
                "origin": "repaired" if repair_applied else "original",
                "method_variant": "salvage_amc_sch",
                "source_slot_id": idx,
                "source_bucket": bucket,
                "before_parseable": before_parseable,
                "after_parseable": after_parseable,
                "before_answer_mode_match": before_parseable,
                "after_answer_mode_match": after_parseable,
                "before_correct": before_correct,
                "after_correct": after_correct,
                "selected_by_verifier": selected,
                "selected_correct": bool(selected and verifier_info and verifier_info.get("correct")),
                "verifier_score": lookup_score(verifier_info, after, answer_mode),
                "repair_applied": repair_applied,
                "repair_type": "slot_replacement" if repair_applied else "none",
                "parser_status_before": parse_status(before, answer_mode) if before else "missing_before_candidate",
                "parser_status_after": parse_status(after, answer_mode) if after else "missing_after_candidate",
                "token_count_before": token_count(before) if before else 0,
                "token_count_after": token_count(after) if after else 0,
                "generation_seed": args.generation_seed,
                "repair_seed": args.repair_seed,
            }
            events.append(event)
            stats["events_total"] += 1
            stats["repair_applied"] += int(repair_applied)
            stats["selected_events"] += int(selected)
            stats["after_correct"] += int(after_correct)
            stats["after_parseable"] += int(after_parseable)
        if not selected_traced:
            provenance_failures.append({"example_id": example_id, "reason": "selected_verifier_candidate_not_traced"})

    missing_required = 0
    for event in events:
        for field in REQUIRED_EVENT_FIELDS:
            if field not in event or event[field] is None:
                missing_required += 1

    total_examples = len(base_rows)
    provenance_coverage = ratio(total_examples - len(provenance_failures), total_examples)
    passed = provenance_coverage == 1.0 and missing_required == 0

    summary_rows = [
        build_method_summary(
            split=args.split,
            method="base_pool",
            pool_rows=base_rows,
            first_correct=first_correct,
            base_correct=base_correct,
            verifier_correct=verifier_correct,
            base_model=args.base_model,
            verifier=args.verifier,
        ),
        build_method_summary(
            split=args.split,
            method="salvage_amc_sch",
            pool_rows=salvage_rows,
            first_correct=first_correct,
            base_correct=base_correct,
            verifier_correct=verifier_correct,
            base_model=args.base_model,
            verifier=args.verifier,
            run_summary=run_summary,
        ),
    ]
    hygiene_rows = [
        build_hygiene_row(args.split, "base_pool", base_rows, verifier_lookup),
        build_hygiene_row(args.split, "salvage_amc_sch", salvage_rows, verifier_lookup),
    ]

    candidate_events_path = args.output_dir / "candidate_events_E00_logging_smoke.jsonl"
    summary_path = args.output_dir / "summary_metrics_E00_logging_smoke.csv"
    hygiene_path = args.output_dir / "hygiene_E00_logging_smoke.csv"
    paired_path = args.output_dir / "paired_predictions_E00_logging_smoke.csv"
    validation_path = args.output_dir / "validation_E00_logging_smoke.json"
    readme_path = args.output_dir / "README_E00_logging_smoke.md"

    write_jsonl(candidate_events_path, events)
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(hygiene_path, HYGIENE_FIELDS, hygiene_rows)
    write_csv(paired_path, PAIRED_FIELDS, paired_rows)
    validation = {
        "experiment_id": args.experiment_id,
        "split": args.split,
        "status": "passed" if passed else "failed",
        "total_examples": total_examples,
        "candidate_events": len(events),
        "provenance_coverage": provenance_coverage,
        "missing_required_fields": missing_required,
        "provenance_failures": provenance_failures[:20],
        "num_provenance_failures": len(provenance_failures),
        "repair_applied_events": stats["repair_applied"],
        "selected_events": stats["selected_events"],
        "candidate_events_path": str(candidate_events_path),
        "summary_metrics_path": str(summary_path),
        "hygiene_path": str(hygiene_path),
        "paired_predictions_path": str(paired_path),
        "inputs": {
            "base_pool": str(args.base_pool),
            "salvage_pool": str(args.salvage_pool),
            "first_predictions": str(args.first_predictions),
            "base_rerank_predictions": str(args.base_rerank_predictions),
            "verifier_predictions": str(args.verifier_predictions),
            "run_summary": str(args.run_summary),
        },
    }
    write_json(validation_path, validation)

    readme = f"""# E00 Logging/Provenance Smoke

- Status: `{validation['status']}`
- Split: `{args.split}`
- Method variants: `base_pool`, `salvage_amc_sch`
- Provenance coverage: `{provenance_coverage:.6f}`
- Missing required fields: `{missing_required}`
- Candidate events: `{len(events)}`
- Repaired slot events: `{stats['repair_applied']}`

## Inputs

- Base pool: `{args.base_pool}`
- Salvage/repaired pool: `{args.salvage_pool}`
- First predictions: `{args.first_predictions}`
- Base rerank predictions: `{args.base_rerank_predictions}`
- Verifier predictions: `{args.verifier_predictions}`
- Run summary: `{args.run_summary}`

## Outputs

- Candidate events: `{candidate_events_path}`
- Summary metrics: `{summary_path}`
- Hygiene diagnostics: `{hygiene_path}`
- Paired predictions: `{paired_path}`
- Validation JSON: `{validation_path}`

## Scope Note

This is an E00 logging/provenance smoke built from existing eval128 artifacts. It validates that selected verifier outputs can be traced to candidate slots and that repair-before/after fields are populated. It is not a new E01/E02 attribution or matched-budget result.
"""
    readme_path.write_text(readme, encoding="utf-8")
    print(json.dumps(validation, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
