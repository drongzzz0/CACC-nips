from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalize_candidate_pool import _infer_answer_mode, _normalize_candidate, _sanitize_text
from src.eval.evaluate_predictions import extract_choice_answer
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply family-aware hygiene to an existing candidate pool."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument(
        "--rewrite-mode",
        choices=("canonical_answerline", "preserve_surface"),
        default="canonical_answerline",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=("answer_token", "surface_only", "none"),
        default="answer_token",
    )
    parser.add_argument("--max-examples", type=int)
    return parser.parse_args()


def _preserve_surface_candidate(text: str) -> tuple[str | None, str | None]:
    original = text.strip()
    sanitized = _sanitize_text(original)
    working = sanitized or original
    token = extract_choice_answer(working)
    if token is None:
        return None, None
    return working, token


def _rewrite_choice_candidate(text: str, rewrite_mode: str) -> tuple[str | None, str | None]:
    if rewrite_mode == "preserve_surface":
        return _preserve_surface_candidate(text)

    normalized, _ = _normalize_candidate(text, "choice_letter", "canonical_answerline")
    token = extract_choice_answer(normalized)
    if token is None:
        return None, None
    return f"Final answer: {token}", token


def _dedupe_key(candidate: str, token: str | None, dedupe_mode: str) -> str:
    if dedupe_mode == "none":
        return f"surface::{candidate}"
    if dedupe_mode == "surface_only":
        return f"surface::{candidate}"
    return f"token::{token or candidate}"


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    output_rows: list[dict] = []
    stats = Counter()

    for row in rows:
        answer_mode = _infer_answer_mode(row)
        candidates = [str(candidate) for candidate in row.get("candidates", [])]
        stats["total_examples"] += 1
        stats["total_candidates"] += len(candidates)
        stats[f"answer_mode_{answer_mode}"] += 1

        if answer_mode != "choice_letter":
            updated = dict(row)
            updated["answer_mode"] = answer_mode
            updated["candidates"] = candidates
            output_rows.append(updated)
            continue

        stats["choice_letter_examples"] += 1
        retained_candidates: list[str] = []
        seen_keys: set[str] = set()
        invalid_drops = 0
        duplicate_drops = 0

        for candidate in candidates:
            rewritten_candidate, token = _rewrite_choice_candidate(candidate, args.rewrite_mode)
            if rewritten_candidate is None:
                invalid_drops += 1
                continue
            dedupe_key = _dedupe_key(rewritten_candidate, token, args.dedupe_mode)
            if dedupe_key in seen_keys:
                duplicate_drops += 1
                continue
            seen_keys.add(dedupe_key)
            retained_candidates.append(rewritten_candidate)
            if rewritten_candidate != candidate.strip():
                stats["changed_candidates"] += 1

        if invalid_drops:
            stats["examples_with_invalid_drop"] += 1
            stats["dropped_invalid_candidates"] += invalid_drops
        if duplicate_drops:
            stats["examples_with_duplicate_drop"] += 1
            stats["dropped_duplicate_candidates"] += duplicate_drops

        if not retained_candidates:
            stats["fallback_examples"] += 1
            fallback = candidates[0].strip() if candidates else ""
            retained_candidates = [fallback] if fallback else []

        if retained_candidates != candidates:
            stats["examples_changed"] += 1

        stats["retained_candidates"] += len(retained_candidates)

        updated = dict(row)
        updated["answer_mode"] = answer_mode
        updated["candidates"] = retained_candidates
        output_rows.append(updated)

    write_jsonl(args.output, output_rows)
    write_json(
        args.report_json,
        {
            "input_path": str(args.input),
            "output_path": str(args.output),
            "rewrite_mode": args.rewrite_mode,
            "dedupe_mode": args.dedupe_mode,
            "num_examples": len(output_rows),
            "stats": dict(stats),
        },
    )


if __name__ == "__main__":
    main()
