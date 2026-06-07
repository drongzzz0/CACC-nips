from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import extract_choice_answer, extract_numeric_answer
from src.inference.peft_generation import _extract_prediction
from src.utils.io_utils import read_jsonl, write_json, write_jsonl

EXPECTED_MODES = {
    "gsm8k": "numeric",
    "gsm8k_full_clean": "numeric",
    "competition_math_numeric": "numeric",
    "competition_math": "numeric",
    "mmlu_pro": "choice_letter",
    "gpqa_diamond": "choice_letter",
}

STRONG_INSTRUCTION_PATTERNS = (
    "use at most",
    "exactly one last line",
    "do not mention",
    "do not critique",
    "do not stop mid-sentence",
    "do not output",
    "do not use markdown",
    "reply with yes or no",
    "single option letter only",
    "the last line must contain",
    "the final answer must be",
    "your answer must be",
    "the answer must be",
)

META_LINE_PREFIXES = (
    "human:",
    "user:",
    "assistant:",
    "problem:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize / style-normalize an existing candidate pool.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument(
        "--style",
        choices=("canonical_answerline", "preserve_prefix_answerline"),
        default="canonical_answerline",
    )
    parser.add_argument("--max-examples", type=int)
    return parser.parse_args()


def _infer_answer_mode(row: dict) -> str:
    answer_mode = str(row.get("answer_mode", "")).strip().lower()
    if answer_mode:
        return answer_mode
    dataset = str(row.get("dataset", "")).strip().lower()
    return EXPECTED_MODES.get(dataset, "numeric")


def _strip_meta_scaffold_prefixes(text: str) -> str:
    cleaned = text.strip()
    patterns = (
        re.compile(r"^(?:here'?s the thought process:\s*)+", flags=re.IGNORECASE),
        re.compile(r"^(?:let'?s think(?: about this)?(?: step by step)?[.!]?\s*)+", flags=re.IGNORECASE),
        re.compile(r"^(?:let me think(?: about this)?(?: step by step)?[.!]?\s*)+", flags=re.IGNORECASE),
        re.compile(r"^(?:okay,\s*)?let'?s tackle this problem(?: step by step)?[.:]?\s*", flags=re.IGNORECASE),
        re.compile(r"^(?:okay,\s*)?let'?s break down the question again[.:]?\s*", flags=re.IGNORECASE),
        re.compile(r"^(?:wait,\s*but that'?s not correct[.!]?\s*)+", flags=re.IGNORECASE),
        re.compile(r"^(?:wait,\s*that'?s not correct[.!]?\s*)+", flags=re.IGNORECASE),
    )
    while True:
        updated = cleaned
        for pattern in patterns:
            updated = pattern.sub("", updated).strip()
        if updated == cleaned:
            return updated
        cleaned = updated


def _looks_like_meta_instruction(text: str) -> bool:
    normalized = " ".join(_strip_meta_scaffold_prefixes(text).strip().lower().split())
    if not normalized:
        return False
    if normalized.startswith(META_LINE_PREFIXES):
        return True
    return any(pattern in normalized for pattern in STRONG_INSTRUCTION_PATTERNS)


def _strip_meta_instruction_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    cleaned_lines: list[str] = []
    for line in lines:
        line = _strip_meta_scaffold_prefixes(line)
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(META_LINE_PREFIXES):
            break
        if lowered.startswith(("final answer:", "answer:", "option:", "choice:")):
            cleaned_lines.append(line)
            continue
        if _looks_like_meta_instruction(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _sanitize_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    multi_turn_match = re.search(r"\n(?:human|user|assistant|problem)\b[,:]?", stripped, flags=re.IGNORECASE)
    if multi_turn_match is not None:
        stripped = stripped[: multi_turn_match.start()].strip()
    stripped = _strip_meta_instruction_lines(stripped)
    if not stripped:
        return ""
    final_answer_position = stripped.lower().find("final answer:")
    if final_answer_position != -1:
        prefix = _strip_meta_instruction_lines(stripped[:final_answer_position])
        suffix = stripped[final_answer_position + len("final answer:") :].strip()
        suffix = suffix.splitlines()[0].strip() if suffix else ""
        suffix = re.sub(r"^(?:final answer\s*:\s*)+", "", suffix, flags=re.IGNORECASE)
        if _looks_like_meta_instruction(prefix):
            if suffix:
                return f"Final answer: {suffix}"
            return ""
        if suffix:
            if prefix:
                return f"{prefix}\nFinal answer: {suffix}"
            return f"Final answer: {suffix}"
    return stripped


def _extract_answer_token(text: str, answer_mode: str) -> str | None:
    prediction = _extract_prediction(text).strip() or text.strip()
    if answer_mode == "choice_letter":
        return extract_choice_answer(prediction) or extract_choice_answer(text)
    if answer_mode == "numeric":
        return extract_numeric_answer(prediction) or extract_numeric_answer(text)
    return prediction or None


def _prefix_without_answerline(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prefix_lines: list[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith(("final answer:", "answer:", "option:", "choice:")):
            break
        if _looks_like_meta_instruction(line):
            continue
        prefix_lines.append(line)
    return "\n".join(prefix_lines).strip()


def _normalize_candidate(text: str, answer_mode: str, style: str) -> tuple[str, str]:
    original = text.strip()
    sanitized = _sanitize_text(original)
    working_text = sanitized or original
    token = _extract_answer_token(working_text, answer_mode)
    if token is None:
        if sanitized and sanitized != original:
            return sanitized, "sanitized_only"
        return original, "unchanged"

    if style == "canonical_answerline":
        normalized = f"Final answer: {token}"
    else:
        prefix = _prefix_without_answerline(working_text)
        if prefix:
            normalized = f"{prefix}\nFinal answer: {token}"
        else:
            normalized = f"Final answer: {token}"

    if normalized != original:
        return normalized, "canonicalized"
    return normalized, "unchanged"


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    normalized_rows: list[dict] = []
    stats = Counter()
    answer_mode_counts = Counter()

    for row in rows:
        answer_mode = _infer_answer_mode(row)
        answer_mode_counts[answer_mode] += 1
        candidates = [str(candidate) for candidate in row.get("candidates", [])]
        normalized_candidates: list[str] = []
        for candidate in candidates:
            normalized_candidate, action = _normalize_candidate(candidate, answer_mode, args.style)
            normalized_candidates.append(normalized_candidate)
            stats["total_candidates"] += 1
            stats[f"action_{action}"] += 1
            if normalized_candidate != candidate.strip():
                stats["changed_candidates"] += 1
            if answer_mode == "numeric" and extract_numeric_answer(normalized_candidate) is not None:
                stats["numeric_parseable_after"] += 1
            if answer_mode == "choice_letter" and extract_choice_answer(normalized_candidate) is not None:
                stats["choice_parseable_after"] += 1
        updated = dict(row)
        updated["answer_mode"] = answer_mode
        updated["candidates"] = normalized_candidates
        normalized_rows.append(updated)
        stats["total_examples"] += 1
        if normalized_candidates != candidates:
            stats["examples_changed"] += 1

    write_jsonl(args.output, normalized_rows)
    write_json(
        args.report_json,
        {
            "input_path": str(args.input),
            "output_path": str(args.output),
            "style": args.style,
            "num_examples": len(normalized_rows),
            "answer_mode_counts": dict(answer_mode_counts),
            "stats": dict(stats),
        },
    )


if __name__ == "__main__":
    main()
