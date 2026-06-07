from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import VerifierCandidateSet
from src.eval.evaluate_predictions import extract_choice_answer, extract_numeric_answer, normalize_answer
from src.generation.prompts import build_completion_prompt
from src.inference.peft_generation import _extract_prediction, _resolve_base_model, load_tokenizer_for_inference, missing_dependencies
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


_STRONG_INSTRUCTION_PATTERNS = (
    "you are improving an incomplete candidate pool",
    "you are finishing the strongest incomplete attempt",
    "you are writing one concise complete solution",
    "use any useful partial steps",
    "pick one useful path from the incomplete attempts",
    "use the likely reasoning motif as a planning cue",
    "produce one complete alternative solution",
    "choose the most promising path",
    "the final solution must be self-contained and decisive",
    "do not mention the attempts",
    "do not critique them",
    "do not stop mid-sentence",
    "do not output general advice",
    "this is a multiple-choice task",
    "you may reason about the options briefly",
    "the last line must contain the single option letter only",
    "on the last line, write",
    "you may use any of the given options",
    "you may use any of the options",
    "you may use any mathematical or logical steps",
    "you may use any of the following mathematical operations",
    "just output the final answer",
    "do not include any explanation",
    "do not include any explanation or reasoning",
    "do not produce any other text besides the final answer",
    "the correct answer is one of the options listed",
)

_SOFT_META_INSTRUCTION_PHRASES = (
    "the final answer must be",
    "your answer must be",
    "the answer must be",
    "the final answer is one of the options",
    "the answer is one of the options",
    "one of the options listed above",
    "the option text",
    "single option letter",
    "single option-letter",
    "the solution should be in the form",
    "the solution must be in the form",
    "the solution should be",
    "the solution must be",
    "the solution should end with",
    "the solution must end with",
    "the last line should be",
    "end with the final answer in the required format",
    "final answer in the required format",
)

_META_SCAFFOLD_PREFIX_PATTERNS = (
    re.compile(r"^(?:here'?s the thought process:\s*)+", flags=re.IGNORECASE),
    re.compile(
        r"^(?:i will now provide (?:the )?(?:complete )?solution(?: for the multiple-choice reasoning problem)?[.:]?\s*)+",
        flags=re.IGNORECASE,
    ),
    re.compile(r"^(?:okay,\s*)?let'?s tackle this problem(?: step by step)?[.:]?\s*", flags=re.IGNORECASE),
    re.compile(r"^(?:okay,\s*)?let'?s break down the question again[.:]?\s*", flags=re.IGNORECASE),
    re.compile(r"^(?:let'?s think(?: about this)?(?: step by step)?[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:let me think(?: about this)?(?: step by step)?[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:let'?s see[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:wait,\s*but that'?s not correct[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:wait,\s*that'?s not correct[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:let me re-?examine[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(
        r"^(?:the solution (?:should|must)(?: not)? be [^.!?\n]*(?:[.!?]\s*|$))+",
        flags=re.IGNORECASE,
    ),
    re.compile(r"^(?:the solution (?:should|must) end with [^.!?\n]*(?:[.!?]\s*|$))+", flags=re.IGNORECASE),
    re.compile(r"^(?:the last line should be [^.!?\n]*(?:[.!?]\s*|$))+", flags=re.IGNORECASE),
    re.compile(r"^(?:use the options given(?: above)?[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:do not output the letter of the option until the last line[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:do not write anything else[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:do not use any markdown[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:do not write in a question-and-answer format[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:do not write in a conversational tone[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:do not write explanations of the reasoning steps[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:just write the solution[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:you may use the observed motifs, but do not mention them[.!?]?\s*)+", flags=re.IGNORECASE),
)


_MERGE_POLICY_CHOICES = (
    "append_if_room",
    "replace_fragments_first",
    "replace_random_nonprefix",
    "replace_partials_first",
    "replace_random_fragment_only",
    "replace_random_partial_only",
    "replace_invalid_first",
    "replace_compatibility_risk_first",
    "replace_hybrid_salvageability",
    "replace_closure_score_first",
    "replace_verifier_uncertainty_first",
    "replace_margin_risk_hybrid",
    "replace_margin_risk_no_salvage",
    "replace_margin_salvage_no_risk",
    "replace_stratified_risk_preserve",
    "replace_margin_stratified_risk_preserve",
    "replace_margin_stratified_numeric_preserve",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate completion-oriented candidates from an existing candidate pool using motif sidecar tags."
    )
    parser.add_argument("--base-candidates", required=True, type=Path)
    parser.add_argument("--motif-tags", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--prompt-preview-output", type=Path)
    parser.add_argument("--verifier-score-sidecar", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--samples-per-example", default=4, type=int)
    parser.add_argument("--prompt-variants", default="default")
    parser.add_argument("--max-candidates", default=8, type=int)
    parser.add_argument("--max-new-tokens", default=160, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--max-context-candidates", default=3, type=int)
    parser.add_argument("--protect-prefix-candidates", default=1, type=int)
    parser.add_argument("--dedupe-mode", default="numeric_or_text", choices=("text", "numeric_or_text"))
    parser.add_argument(
        "--merge-policy",
        default="replace_fragments_first",
        choices=_MERGE_POLICY_CHOICES,
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_prompt_variants(value: str) -> list[str]:
    variants: list[str] = []
    seen = set()
    for part in value.split(","):
        variant = part.strip().lower().replace("-", "_")
        if not variant or variant in seen:
            continue
        seen.add(variant)
        variants.append(variant)
    if not variants:
        raise SystemExit("--prompt-variants must contain at least one variant name.")
    return variants


def _allocate_variant_sample_counts(total_samples: int, prompt_variants: list[str]) -> list[tuple[str, int]]:
    if total_samples < 1:
        raise ValueError("total_samples must be at least 1")
    if not prompt_variants:
        raise ValueError("prompt_variants must not be empty")

    base = total_samples // len(prompt_variants)
    remainder = total_samples % len(prompt_variants)
    allocation: list[tuple[str, int]] = []
    for index, variant in enumerate(prompt_variants):
        count = base + (1 if index < remainder else 0)
        if count > 0:
            allocation.append((variant, count))
    return allocation


def _candidate_dedupe_key(text: str, dedupe_mode: str, answer_mode: str) -> str:
    extracted = _extract_prediction(text).strip() or text.strip()
    if answer_mode == "choice_letter":
        choice = extract_choice_answer(extracted) or extract_choice_answer(text)
        if choice is not None:
            return f"choice:{choice}"
    normalized = normalize_answer(extracted)
    if dedupe_mode == "numeric_or_text":
        numeric = extract_numeric_answer(extracted)
        if numeric is not None:
            return f"num:{numeric}"
    return f"text:{normalized}"


def _format_completion_candidate(generated_text: str, answer_mode: str) -> str:
    stripped = _sanitize_generated_text(generated_text)
    if not stripped:
        return ""
    if answer_mode == "choice_letter":
        return _format_choice_completion_candidate(stripped)
    if "final answer:" in stripped.lower() or "\n" in stripped:
        return stripped
    return _extract_prediction(stripped).strip()


def _load_tags_by_example(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        rows[str(row["example_id"])].append(row)
    for example_rows in rows.values():
        example_rows.sort(key=lambda row: row["candidate_index"])
    return rows


def _load_verifier_score_sidecar(path: Path | None) -> dict[str, dict[int, dict[str, float]]]:
    if path is None:
        return {}
    rows: dict[str, dict[int, dict[str, float]]] = {}
    for row in read_jsonl(path):
        example_id = str(row["example_id"])
        candidate_scores: dict[int, dict[str, float]] = {}
        for item in row.get("candidates", []):
            candidate_index = item.get("candidate_index")
            if candidate_index is None:
                continue
            candidate_scores[int(candidate_index)] = {
                "yes_score": float(item.get("yes_score", 0.0)),
                "no_score": float(item.get("no_score", 0.0)),
                "margin": float(item.get("margin", 0.0)),
            }
        rows[example_id] = candidate_scores
    return rows


def _score_context_candidate(tag_row: dict) -> tuple:
    quality_priority = {
        "partial_solution": 2,
        "fragment": 1,
        "complete_attempt": 0,
    }
    quality_label = str(tag_row["quality_label"])
    motif_matches_problem = int(str(tag_row["motif_label"]) == str(tag_row["problem_motif_label"]))
    length_score = min(len(str(tag_row["candidate_text"]).split()), 40)
    # Deliberately ignore candidate_is_correct to avoid leaking evaluation labels into generation.
    return (
        quality_priority.get(quality_label, 0),
        motif_matches_problem,
        length_score,
        -int(tag_row["candidate_index"]),
    )


def _select_context_attempts(tag_rows: list[dict], max_context_candidates: int) -> list[dict]:
    scored = sorted(tag_rows, key=_score_context_candidate, reverse=True)
    selected = []
    seen_text = set()
    for row in scored:
        text = str(row["candidate_text"]).strip()
        if not text:
            continue
        dedupe_key = normalize_answer(text)
        if dedupe_key in seen_text:
            continue
        seen_text.add(dedupe_key)
        selected.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "candidate_text": text,
                "motif_label": str(row["motif_label"]),
                "quality_label": str(row["quality_label"]),
            }
        )
        if len(selected) >= max_context_candidates:
            break
    return selected


def _build_prompt_preview_record(
    example_id: str,
    problem: str,
    answer_mode: str,
    observed_non_fragment_motifs: list[str],
    attempts: list[dict],
    prompt_variant: str,
    num_return_sequences: int,
    prompt: str,
) -> dict:
    return {
        "example_id": example_id,
        "problem": problem,
        "answer_mode": answer_mode,
        "observed_non_fragment_motifs": observed_non_fragment_motifs,
        "selected_attempts": attempts,
        "prompt_variant": prompt_variant,
        "num_return_sequences": num_return_sequences,
        "prompt": prompt,
    }


def _write_jsonl_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")


def _expand_prompt_instances(prompts_to_run: list[tuple[str, int, str]]) -> list[tuple[str, str]]:
    prompt_instances: list[tuple[str, str]] = []
    for prompt_variant, num_return_sequences, prompt in prompts_to_run:
        prompt_instances.extend((prompt_variant, prompt) for _ in range(num_return_sequences))
    return prompt_instances


def _generate_batched_prompt_mixture(
    model,
    tokenizer,
    device,
    prompt_instances: list[tuple[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[tuple[str, str]]:
    import torch

    if not prompt_instances:
        return []

    encoded = tokenizer(
        [prompt for _, prompt in prompt_instances],
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return [
        (
            prompt_variant,
            tokenizer.decode(sequence[input_length:], skip_special_tokens=True).strip(),
        )
        for (prompt_variant, _), sequence in zip(prompt_instances, generated)
    ]


def _sanitize_generated_text(text: str) -> str:
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

    return _strip_meta_instruction_lines(stripped)


def _strip_meta_scaffold_prefixes(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    while True:
        updated = cleaned
        for pattern in _META_SCAFFOLD_PREFIX_PATTERNS:
            updated = pattern.sub("", updated).strip()
        if updated == cleaned:
            return updated
        cleaned = updated


def _strip_meta_instruction_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    cleaned_lines = []
    for line in lines:
        line = _strip_meta_scaffold_prefixes(line)
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("final answer:", "answer:", "option:", "choice:")):
            cleaned_lines.append(line)
            continue
        if _looks_like_meta_instruction(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _looks_like_meta_instruction(text: str) -> bool:
    normalized = " ".join(_strip_meta_scaffold_prefixes(text).strip().lower().split())
    if not normalized:
        return False
    meta_phrases = (
        "aim to produce",
        "remember to",
        "use step-by-step",
        "ensure it",
        "ensure that",
        "provide the detailed steps",
        "logical and detailed",
        "maintains clarity",
        "incorporates the provided information",
        "aim for clarity",
        "answer:",
    )
    return (
        any(phrase in normalized for phrase in meta_phrases)
        or any(phrase in normalized for phrase in _STRONG_INSTRUCTION_PATTERNS)
        or any(phrase in normalized for phrase in _SOFT_META_INSTRUCTION_PHRASES)
    )


def _contains_instruction_or_prompt_leak(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return False
    return any(phrase in normalized for phrase in _STRONG_INSTRUCTION_PATTERNS)


def _format_choice_completion_candidate(text: str) -> str:
    choice = extract_choice_answer(text)
    if choice is None:
        extracted = _extract_prediction(text).strip()
        choice = extract_choice_answer(extracted)
    if choice is None:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prefix_lines = []
    for line in lines:
        line = _strip_meta_scaffold_prefixes(line)
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("final answer:", "answer:", "option:", "choice:")):
            break
        if _looks_like_meta_instruction(line):
            continue
        prefix_lines.append(line)

    prefix = "\n".join(prefix_lines).strip()
    normalized_prefix = normalize_answer(prefix)
    if normalized_prefix in {
        choice.lower(),
        f"option {choice.lower()}",
        f"choice {choice.lower()}",
        f"answer {choice.lower()}",
    }:
        prefix = ""
    if prefix:
        return f"{prefix}\nFinal answer: {choice}"
    return f"Final answer: {choice}"


def _has_scaffold_residue(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return _strip_meta_instruction_lines(stripped) != stripped or _strip_meta_scaffold_prefixes(stripped) != stripped


def _has_valid_final_answer(text: str, answer_mode: str) -> bool:
    extracted = _extract_prediction(text).strip() or text.strip()
    if answer_mode == "choice_letter":
        return extract_choice_answer(extracted) is not None or extract_choice_answer(text) is not None
    if answer_mode == "numeric":
        return extract_numeric_answer(extracted) is not None or extract_numeric_answer(text) is not None
    if "final answer:" in text.lower():
        return bool(normalize_answer(extracted))
    return bool(normalize_answer(extracted))


def _build_removable_records(
    existing_candidates: list[str],
    tag_rows: list[dict],
    protect_prefix_candidates: int,
    answer_mode: str,
    verifier_scores_by_index: dict[int, dict[str, float]] | None = None,
) -> list[dict]:
    tag_by_index = {int(row["candidate_index"]): row for row in tag_rows}
    verifier_scores_by_index = verifier_scores_by_index or {}
    records: list[dict] = []
    for index, candidate in enumerate(existing_candidates):
        if index < protect_prefix_candidates:
            continue
        tag_row = tag_by_index.get(index, {})
        quality_label = str(tag_row.get("quality_label", "unknown"))
        motif_label = str(tag_row.get("motif_label", ""))
        problem_motif_label = str(tag_row.get("problem_motif_label", ""))
        verifier_score = verifier_scores_by_index.get(index, {})
        verifier_margin = verifier_score.get("margin")
        support_score = float(verifier_margin) if verifier_margin is not None else 0.0
        candidate_text = str(candidate)
        token_count = min(len(candidate_text.split()), 40)
        invalid_final = int(not _has_valid_final_answer(candidate_text, answer_mode))
        instruction_leak = int(_contains_instruction_or_prompt_leak(candidate_text))
        scaffold_residue = int(_has_scaffold_residue(candidate_text))
        has_final_wrapper = int("final answer:" in candidate_text.lower())
        numeric_answer = extract_numeric_answer(candidate_text) if answer_mode == "numeric" else None
        short_numeric_like = int(
            answer_mode == "numeric"
            and numeric_answer is not None
            and token_count <= 6
            and not instruction_leak
        )
        compatibility_risk_score = (
            4.0 * invalid_final
            + 3.0 * instruction_leak
            + 2.0 * scaffold_residue
            + 1.5 * (1 - has_final_wrapper)
        )
        quality_salvage = {
            "complete_attempt": 3.0,
            "partial_solution": 2.5,
            "fragment": 1.5,
        }.get(quality_label, 1.5)
        salvageability_score = (
            quality_salvage
            + 0.75 * int(bool(motif_label) and motif_label == problem_motif_label)
            + 0.50 * has_final_wrapper
            + 0.50 * max(support_score, 0.0)
        )
        closure_score = (
            4 * (1 - invalid_final)
            + 2 * has_final_wrapper
            + min(token_count, 12) / 12.0
            - 2 * instruction_leak
            - 1.5 * scaffold_residue
        )
        records.append(
            {
                "index": index,
                "candidate_text": candidate,
                "quality_label": quality_label,
                "motif_matches_problem": int(bool(motif_label) and motif_label == problem_motif_label),
                "token_count": token_count,
                "invalid_final": invalid_final,
                "instruction_leak": instruction_leak,
                "scaffold_residue": scaffold_residue,
                "has_final_wrapper": has_final_wrapper,
                "short_numeric_like": short_numeric_like,
                "closure_score": closure_score,
                "compatibility_risk_score": compatibility_risk_score,
                "salvageability_score": salvageability_score,
                "support_score": support_score,
                "verifier_margin": float(verifier_margin) if verifier_margin is not None else None,
                "verifier_margin_missing": int(verifier_margin is None),
            }
        )
    return records


def _quality_priority(merge_policy: str, quality_label: str) -> int:
    if merge_policy == "replace_partials_first":
        quality_rank = {
            "partial_solution": 0,
            "fragment": 1,
            "complete_attempt": 2,
        }
    else:
        quality_rank = {
            "fragment": 0,
            "partial_solution": 1,
            "complete_attempt": 2,
        }
    return quality_rank.get(quality_label, 1)


def _removable_sort_key(record: dict, merge_policy: str) -> tuple:
    invalid_priority = 0 if record["invalid_final"] else 1
    leak_priority = 0 if record["instruction_leak"] else 1
    scaffold_priority = 0 if record["scaffold_residue"] else 1
    wrapper_priority = 0 if not record["has_final_wrapper"] else 1
    motif_priority = 0 if not record["motif_matches_problem"] else 1
    quality_priority = _quality_priority(merge_policy, record["quality_label"])
    token_count = record["token_count"]
    index = record["index"]
    closure_priority = record["closure_score"]
    risk_priority = -record["compatibility_risk_score"]
    salvage_priority = record["salvageability_score"]
    support_priority = record["support_score"]
    verifier_margin_missing = record["verifier_margin_missing"]
    verifier_margin = record["verifier_margin"] if record["verifier_margin"] is not None else 1e9
    short_numeric_priority = 1 if record.get("short_numeric_like") else 0

    if merge_policy == "replace_invalid_first":
        return (
            invalid_priority,
            leak_priority,
            scaffold_priority,
            quality_priority,
            motif_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_compatibility_risk_first":
        return (
            invalid_priority,
            leak_priority,
            scaffold_priority,
            wrapper_priority,
            quality_priority,
            motif_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_hybrid_salvageability":
        return (
            invalid_priority,
            motif_priority,
            quality_priority,
            leak_priority,
            scaffold_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_closure_score_first":
        return (
            closure_priority,
            invalid_priority,
            wrapper_priority,
            leak_priority,
            scaffold_priority,
            quality_priority,
            motif_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_verifier_uncertainty_first":
        return (
            verifier_margin_missing,
            verifier_margin,
            invalid_priority,
            leak_priority,
            scaffold_priority,
            quality_priority,
            motif_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_margin_risk_hybrid":
        return (
            verifier_margin_missing,
            verifier_margin,
            invalid_priority,
            leak_priority,
            scaffold_priority,
            wrapper_priority,
            motif_priority,
            quality_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_margin_risk_no_salvage":
        return (
            verifier_margin_missing,
            verifier_margin,
            risk_priority,
            invalid_priority,
            leak_priority,
            scaffold_priority,
            wrapper_priority,
            quality_priority,
            motif_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_margin_salvage_no_risk":
        return (
            verifier_margin_missing,
            verifier_margin,
            salvage_priority,
            quality_priority,
            motif_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_stratified_risk_preserve":
        return (
            risk_priority,
            salvage_priority,
            quality_priority,
            motif_priority,
            support_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_margin_stratified_risk_preserve":
        return (
            risk_priority,
            support_priority,
            salvage_priority,
            quality_priority,
            motif_priority,
            token_count,
            -index,
        )
    if merge_policy == "replace_margin_stratified_numeric_preserve":
        return (
            risk_priority,
            support_priority,
            short_numeric_priority,
            salvage_priority,
            quality_priority,
            motif_priority,
            token_count,
            -index,
        )
    return (
        quality_priority,
        invalid_priority,
        leak_priority,
        scaffold_priority,
        motif_priority,
        token_count,
        -index,
    )


def _select_removal_records(
    removable_records: list[dict],
    merge_policy: str,
    max_to_remove: int,
) -> list[dict]:
    if max_to_remove <= 0 or not removable_records:
        return []

    if merge_policy == "replace_random_nonprefix":
        shuffled = list(removable_records)
        random.shuffle(shuffled)
        return shuffled[:max_to_remove]

    if merge_policy in {"replace_random_fragment_only", "replace_random_partial_only"}:
        preferred_quality = "fragment" if merge_policy == "replace_random_fragment_only" else "partial_solution"
        preferred = [record for record in removable_records if record["quality_label"] == preferred_quality]
        fallback = [record for record in removable_records if record["quality_label"] != preferred_quality]
        random.shuffle(preferred)
        random.shuffle(fallback)
        return (preferred + fallback)[:max_to_remove]

    candidate_pool = list(removable_records)
    if merge_policy in {
        "replace_stratified_risk_preserve",
        "replace_margin_stratified_risk_preserve",
        "replace_margin_salvage_no_risk",
        "replace_margin_stratified_numeric_preserve",
    }:
        protected_indices = set()
        protected_quality_labels = ("partial_solution", "fragment")
        if merge_policy == "replace_margin_stratified_numeric_preserve":
            protected_quality_labels = ("complete_attempt", "partial_solution", "fragment")
        for quality_label in protected_quality_labels:
            group = [record for record in removable_records if record["quality_label"] == quality_label]
            if not group:
                continue
            anchor = max(
                group,
                key=lambda record: (
                    record["salvageability_score"],
                    record["closure_score"],
                    record["motif_matches_problem"],
                    record["support_score"],
                    -record["index"],
                ),
            )
            protected_indices.add(anchor["index"])
        if merge_policy == "replace_margin_stratified_numeric_preserve":
            short_numeric_group = [record for record in removable_records if record.get("short_numeric_like")]
            if short_numeric_group:
                short_numeric_anchor = max(
                    short_numeric_group,
                    key=lambda record: (
                        record["support_score"],
                        record["closure_score"],
                        record["has_final_wrapper"],
                        record["salvageability_score"],
                        record["motif_matches_problem"],
                        -record["token_count"],
                        -record["index"],
                    ),
                )
                protected_indices.add(short_numeric_anchor["index"])
        unprotected = [record for record in removable_records if record["index"] not in protected_indices]
        if unprotected:
            candidate_pool = unprotected

    ordered = sorted(candidate_pool, key=lambda record: _removable_sort_key(record, merge_policy))
    return ordered[:max_to_remove]


def _merge_completion_candidates(
    existing_candidates: list[str],
    tag_rows: list[dict],
    new_candidates: list[str],
    max_candidates: int,
    protect_prefix_candidates: int,
    merge_policy: str,
    answer_mode: str,
    verifier_scores_by_index: dict[int, dict[str, float]] | None = None,
) -> tuple[list[str], Counter]:
    stats = Counter()
    if not new_candidates:
        return existing_candidates[:max_candidates], stats

    merged_candidates = list(existing_candidates[:max_candidates])
    if len(merged_candidates) < max_candidates or merge_policy == "append_if_room":
        for candidate in new_candidates:
            if len(merged_candidates) >= max_candidates:
                break
            merged_candidates.append(candidate)
            stats["kept_completion_candidates"] += 1
        return merged_candidates, stats

    removable_records = _build_removable_records(
        merged_candidates,
        tag_rows,
        protect_prefix_candidates=protect_prefix_candidates,
        answer_mode=answer_mode,
        verifier_scores_by_index=verifier_scores_by_index,
    )
    selected_records = _select_removal_records(
        removable_records,
        merge_policy=merge_policy,
        max_to_remove=min(len(removable_records), len(new_candidates)),
    )
    removed_index_set = {record["index"] for record in selected_records}

    retained_candidates = [
        candidate for index, candidate in enumerate(merged_candidates) if index not in removed_index_set
    ]
    for record in selected_records:
        stats[f"replaced_{record['quality_label']}"] += 1
        if record["invalid_final"]:
            stats["replaced_invalid_final"] += 1
        if record["instruction_leak"]:
            stats["replaced_instruction_leak"] += 1
        if record["scaffold_residue"]:
            stats["replaced_scaffold_residue"] += 1
        if record.get("short_numeric_like"):
            stats["replaced_short_numeric_like"] += 1
    appended_candidates = new_candidates[: len(selected_records)]
    stats["kept_completion_candidates"] += len(appended_candidates)
    return retained_candidates + appended_candidates, stats


def main() -> None:
    args = parse_args()
    if args.samples_per_example < 1:
        raise SystemExit("--samples-per-example must be at least 1.")
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be at least 1.")
    if args.max_context_candidates < 1:
        raise SystemExit("--max-context-candidates must be at least 1.")

    prompt_variants = _parse_prompt_variants(args.prompt_variants)
    variant_sample_counts = _allocate_variant_sample_counts(args.samples_per_example, prompt_variants)

    if not args.dry_run and args.adapter_path is None and not args.model_path:
        raise SystemExit("Pass either --adapter-path or --model-path unless --dry-run is enabled.")

    base_rows = list(read_jsonl(args.base_candidates))
    if args.max_examples is not None:
        base_rows = base_rows[: args.max_examples]
    tags_by_example = _load_tags_by_example(args.motif_tags)
    verifier_scores_by_example = _load_verifier_score_sidecar(args.verifier_score_sidecar)

    output_rows: list[dict] = []
    generation_stats = Counter()
    started_at = time.perf_counter()
    total_examples = len(base_rows)
    prompt_preview_handle = None

    model = None
    tokenizer = None
    device = None
    if not args.dry_run:
        require_peft = args.adapter_path is not None
        missing = missing_dependencies(require_peft=require_peft)
        if missing:
            raise RuntimeError("Missing inference dependencies: " + ", ".join(missing))

        import torch
        from transformers import AutoModelForCausalLM
        if require_peft:
            from peft import PeftModel

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        base_model_name_or_path = _resolve_base_model(args.adapter_path, args.base_model, args.model_path)
        tokenizer = load_tokenizer_for_inference(
            adapter_path=args.adapter_path,
            base_model_name_or_path=base_model_name_or_path,
        )
        tokenizer.padding_side = "left"

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, dtype=dtype)
        model = PeftModel.from_pretrained(base_model, args.adapter_path) if require_peft else base_model
        model.to(device)
        model.eval()
        generation_stats["model_loaded"] = 1

    if args.prompt_preview_output is not None:
        args.prompt_preview_output.parent.mkdir(parents=True, exist_ok=True)
        prompt_preview_handle = args.prompt_preview_output.open("w", encoding="utf-8")

    for example_index, base_row in enumerate(base_rows, start=1):
        example_id = str(base_row["example_id"])
        problem = str(base_row["problem"])
        existing_candidates = [str(candidate) for candidate in base_row["candidates"]]
        answer_mode = str(base_row.get("answer_mode", "numeric"))
        tag_rows = tags_by_example.get(example_id)
        if not tag_rows:
            raise ValueError(f"Missing motif tags for {example_id}")

        problem_motif_label = str(tag_rows[0]["problem_motif_label"])
        observed_non_fragment_motifs = sorted(
            {
                str(row["motif_label"])
                for row in tag_rows
                if str(row["quality_label"]) != "fragment"
            }
        )
        attempts = _select_context_attempts(tag_rows, max_context_candidates=args.max_context_candidates)

        prompts_to_run: list[tuple[str, int, str]] = []
        for prompt_variant, num_return_sequences in variant_sample_counts:
            prompt = build_completion_prompt(
                problem=problem,
                problem_motif_label=problem_motif_label,
                observed_non_fragment_motifs=observed_non_fragment_motifs,
                attempts=attempts,
                answer_mode=answer_mode,
                variant=prompt_variant,
            )
            prompts_to_run.append((prompt_variant, num_return_sequences, prompt))
            if prompt_preview_handle is not None:
                _write_jsonl_record(
                    prompt_preview_handle,
                    _build_prompt_preview_record(
                        example_id=example_id,
                        problem=problem,
                        answer_mode=answer_mode,
                        observed_non_fragment_motifs=observed_non_fragment_motifs,
                        attempts=attempts,
                        prompt_variant=prompt_variant,
                        num_return_sequences=num_return_sequences,
                        prompt=prompt,
                    ),
                )

        generated_candidates: list[str] = []
        if not args.dry_run:
            existing_dedupe = {
                _candidate_dedupe_key(candidate, args.dedupe_mode, answer_mode)
                for candidate in existing_candidates
            }
            seen_completion = set()
            for prompt_variant, num_return_sequences, _ in prompts_to_run:
                generation_stats[f"prompt_variant_{prompt_variant}_requested_samples"] += num_return_sequences
            prompt_instances = _expand_prompt_instances(prompts_to_run)
            generation_stats["prompt_instances_total"] += len(prompt_instances)
            if prompt_instances:
                generation_stats["generate_calls"] += 1
                for prompt_variant, generated_text in _generate_batched_prompt_mixture(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_instances=prompt_instances,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ):
                    formatted = _format_completion_candidate(generated_text, answer_mode)
                    if not formatted:
                        continue
                    if _contains_instruction_or_prompt_leak(formatted):
                        generation_stats["rejected_instruction_or_prompt_leak"] += 1
                        continue
                    dedupe_key = _candidate_dedupe_key(formatted, args.dedupe_mode, answer_mode)
                    if dedupe_key in existing_dedupe or dedupe_key in seen_completion:
                        continue
                    seen_completion.add(dedupe_key)
                    generated_candidates.append(formatted)
                    generation_stats[f"prompt_variant_{prompt_variant}_kept_candidates"] += 1

        merged_candidates, merge_stats = _merge_completion_candidates(
            existing_candidates=existing_candidates,
            tag_rows=tag_rows,
            new_candidates=generated_candidates,
            max_candidates=args.max_candidates,
            protect_prefix_candidates=args.protect_prefix_candidates,
            merge_policy=args.merge_policy,
            answer_mode=answer_mode,
            verifier_scores_by_index=verifier_scores_by_example.get(example_id, {}),
        )
        generation_stats.update(merge_stats)
        if generated_candidates:
            generation_stats["examples_with_new_completion_candidates"] += 1
        if merge_stats["kept_completion_candidates"] > 0:
            generation_stats["examples_with_retained_completion_candidates"] += 1
        if merged_candidates != existing_candidates[: args.max_candidates]:
            generation_stats["examples_modified"] += 1
        if all(str(row["quality_label"]) != "complete_attempt" for row in tag_rows):
            generation_stats["examples_without_complete_attempt_in_base_pool"] += 1

        output_rows.append(
            VerifierCandidateSet(
                example_id=example_id,
                dataset=str(base_row.get("dataset", "unknown")),
                problem=problem,
                gold_answer=str(base_row["gold_answer"]),
                candidates=merged_candidates,
                answer_mode=answer_mode,
                choices=[str(choice) for choice in base_row.get("choices", [])],
                metadata=dict(base_row.get("metadata", {})),
            ).to_dict()
        )

        if args.log_every > 0 and (example_index % args.log_every == 0 or example_index == total_examples):
            elapsed = time.perf_counter() - started_at
            avg_seconds = elapsed / example_index if example_index else 0.0
            remaining_seconds = avg_seconds * max(total_examples - example_index, 0)
            print(
                f"[completion-gen] processed {example_index}/{total_examples} examples "
                f"({example_index / total_examples:.1%}); elapsed={elapsed / 60:.1f}m; "
                f"avg={avg_seconds:.2f}s/ex; eta={remaining_seconds / 3600:.2f}h; "
                f"generate_calls={generation_stats.get('generate_calls', 0)}",
                flush=True,
            )
            if prompt_preview_handle is not None:
                prompt_preview_handle.flush()

    if prompt_preview_handle is not None:
        prompt_preview_handle.close()

    write_jsonl(args.output, output_rows)

    if args.metrics_output is not None:
        total_examples = len(output_rows)
        elapsed = time.perf_counter() - started_at
        metrics = {
            "base_candidates_path": str(args.base_candidates),
            "motif_tags_path": str(args.motif_tags),
            "verifier_score_sidecar": str(args.verifier_score_sidecar) if args.verifier_score_sidecar is not None else None,
            "output_path": str(args.output),
            "prompt_preview_output": str(args.prompt_preview_output) if args.prompt_preview_output is not None else None,
            "dry_run": args.dry_run,
            "prompt_variants": prompt_variants,
            "variant_sample_counts": {variant: count for variant, count in variant_sample_counts},
            "total_examples": total_examples,
            "merge_policy": args.merge_policy,
            "protect_prefix_candidates": args.protect_prefix_candidates,
            "samples_per_example": args.samples_per_example,
            "max_candidates": args.max_candidates,
            "examples_modified": generation_stats.get("examples_modified", 0),
            "examples_with_new_completion_candidates": generation_stats.get("examples_with_new_completion_candidates", 0),
            "examples_with_retained_completion_candidates": generation_stats.get(
                "examples_with_retained_completion_candidates", 0
            ),
            "kept_completion_candidates": generation_stats.get("kept_completion_candidates", 0),
            "rejected_instruction_or_prompt_leak": generation_stats.get("rejected_instruction_or_prompt_leak", 0),
            "examples_without_complete_attempt_in_base_pool": generation_stats.get(
                "examples_without_complete_attempt_in_base_pool", 0
            ),
            "replaced_fragment": generation_stats.get("replaced_fragment", 0),
            "replaced_partial_solution": generation_stats.get("replaced_partial_solution", 0),
            "replaced_complete_attempt": generation_stats.get("replaced_complete_attempt", 0),
            "replaced_invalid_final": generation_stats.get("replaced_invalid_final", 0),
            "replaced_instruction_leak": generation_stats.get("replaced_instruction_leak", 0),
            "replaced_scaffold_residue": generation_stats.get("replaced_scaffold_residue", 0),
            "replaced_short_numeric_like": generation_stats.get("replaced_short_numeric_like", 0),
            "prompt_variant_requested_samples_total": {
                variant: generation_stats.get(f"prompt_variant_{variant}_requested_samples", 0)
                for variant in prompt_variants
            },
            "prompt_variant_kept_candidates_total": {
                variant: generation_stats.get(f"prompt_variant_{variant}_kept_candidates", 0)
                for variant in prompt_variants
            },
            "prompt_instances_total": generation_stats.get("prompt_instances_total", 0),
            "generate_calls": generation_stats.get("generate_calls", 0),
            "total_seconds": round(elapsed, 6),
            "avg_seconds_per_example": round(elapsed / total_examples, 6) if total_examples else 0.0,
        }
        write_json(args.metrics_output, metrics)


if __name__ == "__main__":
    main()
