from __future__ import annotations

import argparse
import hashlib
import random
import re
import time
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.motif_utils import infer_candidate_tag, normalize_freeform
from src.data.schema import VerifierCandidateSet
from src.eval.evaluate_predictions import extract_choice_answer, extract_numeric_answer, normalize_answer
from src.generation.prompts import build_repair_prompt
from src.inference.peft_generation import _extract_prediction, _resolve_base_model, load_tokenizer_for_inference, missing_dependencies
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
        description="Generate conservative residual-repair candidates from an existing verifier candidate pool."
    )
    parser.add_argument("--base-candidates", required=True, type=Path)
    parser.add_argument("--reranker-predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--prompt-preview-output", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--samples-per-target", default=1, type=int)
    parser.add_argument("--max-repair-targets", default=2, type=int)
    parser.add_argument("--max-candidates", default=8, type=int)
    parser.add_argument("--max-new-tokens", default=160, type=int)
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--protect-prefix-candidates", default=1, type=int)
    parser.add_argument("--dedupe-mode", default="numeric_or_text", choices=("text", "numeric_or_text"))
    parser.add_argument(
        "--strict-hygiene",
        action="store_true",
        help="Explicitly enforce strict repair hygiene defaults and record them in metrics/logs.",
    )
    parser.add_argument(
        "--allow-replace-complete-attempt",
        action="store_true",
        help="Allow repaired outputs to replace source candidates tagged as complete attempts.",
    )
    parser.add_argument(
        "--allow-non-numeric-repairs",
        action="store_true",
        help="Allow retained repairs that do not contain any numeric answer span.",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def _sanitize_generated_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    multi_turn_match = re.search(r"\n(?:human|user|assistant|problem)\b[,:]?", stripped, flags=re.IGNORECASE)
    if multi_turn_match is not None:
        stripped = stripped[: multi_turn_match.start()].strip()

    final_answer_position = stripped.lower().find("final answer:")
    if final_answer_position != -1:
        prefix = stripped[:final_answer_position].strip()
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


def _looks_like_meta_instruction(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return False
    meta_phrases = (
        "aim to produce",
        "remember to",
        "use at most",
        "use step-by-step",
        "ensure it",
        "ensure that",
        "exactly one last line",
        "provide the detailed steps",
        "logical and detailed",
        "maintains clarity",
        "incorporates the provided information",
        "aim for clarity",
        "answer:",
    )
    return any(phrase in normalized for phrase in meta_phrases) or any(
        phrase in normalized for phrase in _STRONG_INSTRUCTION_PATTERNS
    )


def _repair_rejection_reasons(
    repair_text: str,
    *,
    answer_mode: str,
    source_quality_label: str,
    allow_replace_complete_attempt: bool,
    allow_non_numeric_repairs: bool,
) -> list[str]:
    lowered = repair_text.lower()
    reasons: list[str] = []
    if any(pattern in lowered for pattern in _STRONG_INSTRUCTION_PATTERNS):
        reasons.append("instruction_or_prompt_leak")
    if answer_mode != "choice_letter" and not allow_non_numeric_repairs and extract_numeric_answer(repair_text) is None:
        reasons.append("missing_numeric_answer")
    if source_quality_label == "complete_attempt" and not allow_replace_complete_attempt:
        reasons.append("would_replace_complete_attempt")
    return sorted(set(reasons))


def _format_repair_candidate(generated_text: str, answer_mode: str) -> str:
    stripped = _sanitize_generated_text(generated_text)
    if not stripped:
        return ""
    if answer_mode == "choice_letter":
        return _format_choice_repair_candidate(stripped)
    if "final answer:" in stripped.lower() or "\n" in stripped:
        return stripped
    return _extract_prediction(stripped).strip()


def _format_choice_repair_candidate(text: str) -> str:
    choice = extract_choice_answer(text)
    if choice is None:
        extracted = _extract_prediction(text).strip()
        choice = extract_choice_answer(extracted)
    if choice is None:
        return _extract_prediction(text).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    prefix_lines = []
    for line in lines:
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


def _load_reranker_margins(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}

    margins_by_example: dict[str, dict[str, object]] = {}
    for row in read_jsonl(path):
        exact_lookup: dict[str, float] = {}
        normalized_lookup: dict[str, float] = {}
        for candidate_row in row.get("candidates", []):
            candidate_text = str(candidate_row.get("candidate_answer", "")).strip()
            margin = float(candidate_row.get("margin", 0.0))
            exact_lookup[candidate_text] = margin
            normalized_lookup.setdefault(normalize_answer(candidate_text), margin)
        winner_text = str(row.get("prediction", "")).strip()
        margins_by_example[str(row["example_id"])] = {
            "exact": exact_lookup,
            "normalized": normalized_lookup,
            "winner_text": winner_text,
            "winner_normalized": normalize_answer(winner_text) if winner_text else "",
        }
    return margins_by_example


def _lookup_margin(example_margins: dict[str, object] | None, candidate_text: str) -> float | None:
    if not example_margins:
        return None
    stripped = candidate_text.strip()
    exact_lookup = example_margins["exact"]
    normalized_lookup = example_margins["normalized"]
    if stripped in exact_lookup:
        return exact_lookup[stripped]
    normalized = normalize_answer(stripped)
    return normalized_lookup.get(normalized)


def _is_current_winner(example_margins: dict[str, object] | None, candidate_text: str) -> bool:
    if not example_margins:
        return False
    stripped = candidate_text.strip()
    winner_text = str(example_margins.get("winner_text", "")).strip()
    if winner_text and stripped == winner_text:
        return True
    winner_normalized = str(example_margins.get("winner_normalized", "")).strip()
    return bool(winner_normalized) and normalize_answer(stripped) == winner_normalized


def _infer_repair_error(problem: str, candidate_text: str, motif_label: str, quality_label: str) -> tuple[str, list[str]]:
    normalized_problem = normalize_freeform(problem)
    normalized_candidate = normalize_freeform(candidate_text)
    cues: list[str] = []

    if quality_label == "complete_attempt" and (
        "final answer:" in normalized_candidate or re.search(r"[+\-*/=]", candidate_text)
    ):
        cues.append("complete_with_final_or_ops")
        return "arithmetic_finish_error", cues

    if motif_label == "temporal_or_age_shift" or re.search(
        r"\bage\b|\bolder\b|\byounger\b|\bbefore\b|\bafter\b|\bremaining\b",
        normalized_problem,
    ):
        cues.append("temporal_keywords")
        return "temporal_shift_error", cues

    if motif_label == "ratio_or_proportion" or re.search(
        r"\bpercent(?:age)?\b|\bhalf\b|\btwice\b|\bdouble\b|\btriple\b|\bper\b|\beach\b",
        f"{normalized_problem}\n{normalized_candidate}",
    ):
        cues.append("ratio_or_base_keywords")
        return "wrong_base_or_reference_value", cues

    if motif_label == "equation_setup" or re.search(r"\blet\b|\bequation\b|=", normalized_candidate):
        cues.append("equation_keywords")
        return "equation_setup_error", cues

    if quality_label == "partial_solution":
        cues.append("partial_solution")
        return "arithmetic_finish_error", cues

    return "other", cues or ["fallback_other"]


def _target_priority(target: dict) -> tuple:
    quality_priority = {
        "complete_attempt": 2,
        "partial_solution": 1,
        "fragment": 0,
    }
    margin = target["verifier_margin"]
    return (
        quality_priority.get(str(target["quality_label"]), 0),
        int(margin is not None),
        float(margin) if margin is not None else float("-inf"),
        int(str(target["motif_label"]) == str(target["problem_motif_label"])),
        min(int(target["word_count"]), 48),
        -int(target["candidate_index"]),
    )


def _select_repair_targets(
    problem: str,
    candidates: list[str],
    example_margins: dict[str, object] | None,
    protect_prefix_candidates: int,
    max_repair_targets: int,
    allow_replace_complete_attempt: bool,
) -> list[dict]:
    target_rows: list[dict] = []
    problem_tag = infer_candidate_tag(problem, "")
    for candidate_index, candidate_text in enumerate(candidates):
        if candidate_index < protect_prefix_candidates:
            continue
        candidate_text = str(candidate_text)
        if _is_current_winner(example_margins, candidate_text):
            continue
        if _looks_like_meta_instruction(candidate_text):
            continue
        tag = infer_candidate_tag(problem, candidate_text)
        if tag.quality.label == "fragment":
            continue
        if tag.quality.label == "complete_attempt" and not allow_replace_complete_attempt:
            continue
        repair_error_label, repair_error_cues = _infer_repair_error(
            problem=problem,
            candidate_text=candidate_text,
            motif_label=tag.motif.label,
            quality_label=tag.quality.label,
        )
        target_rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_text": candidate_text,
                "problem_motif_label": problem_tag.motif.label,
                "motif_label": tag.motif.label,
                "quality_label": tag.quality.label,
                "repair_error_label": repair_error_label,
                "repair_error_cues": repair_error_cues,
                "verifier_margin": _lookup_margin(example_margins, candidate_text),
                "word_count": len(candidate_text.split()),
            }
        )
    target_rows.sort(key=_target_priority, reverse=True)
    return target_rows[:max_repair_targets]


def _build_prompt_preview_record(example_id: str, problem: str, target: dict, prompt: str) -> dict:
    return {
        "example_id": example_id,
        "problem": problem,
        "candidate_index": int(target["candidate_index"]),
        "candidate_text": str(target["candidate_text"]),
        "motif_label": str(target["motif_label"]),
        "quality_label": str(target["quality_label"]),
        "repair_error_label": str(target["repair_error_label"]),
        "repair_error_cues": list(target["repair_error_cues"]),
        "verifier_margin": target["verifier_margin"],
        "prompt": prompt,
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_target < 1:
        raise SystemExit("--samples-per-target must be at least 1.")
    if args.max_repair_targets < 1:
        raise SystemExit("--max-repair-targets must be at least 1.")
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be at least 1.")
    if args.strict_hygiene and (args.allow_replace_complete_attempt or args.allow_non_numeric_repairs):
        raise SystemExit("--strict-hygiene cannot be combined with permissive repair retention flags.")

    if not args.dry_run and args.adapter_path is None and not args.model_path:
        raise SystemExit("Pass either --adapter-path or --model-path unless --dry-run is enabled.")

    base_rows = list(read_jsonl(args.base_candidates))
    if args.max_examples is not None:
        base_rows = base_rows[: args.max_examples]
    margins_by_example = _load_reranker_margins(args.reranker_predictions)

    prompt_preview_records: list[dict] = []
    output_rows: list[dict] = []
    generation_stats = Counter()
    selected_error_counts = Counter()
    selected_quality_counts = Counter()
    started_at = time.perf_counter()

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

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, dtype=dtype)
        model = PeftModel.from_pretrained(base_model, args.adapter_path) if require_peft else base_model
        model.to(device)
        model.eval()
        generation_stats["model_loaded"] = 1

    for base_row in base_rows:
        example_id = str(base_row["example_id"])
        problem = str(base_row["problem"])
        existing_candidates = [str(candidate) for candidate in base_row["candidates"]][: args.max_candidates]
        answer_mode = str(base_row.get("answer_mode", "numeric"))
        example_margins = margins_by_example.get(example_id)

        repair_targets = _select_repair_targets(
            problem=problem,
            candidates=existing_candidates,
            example_margins=example_margins,
            protect_prefix_candidates=args.protect_prefix_candidates,
            max_repair_targets=args.max_repair_targets,
            allow_replace_complete_attempt=args.allow_replace_complete_attempt,
        )

        generation_stats["selected_repair_targets"] += len(repair_targets)
        if repair_targets:
            generation_stats["examples_with_selected_repair_targets"] += 1
        if example_margins:
            generation_stats["examples_with_reranker_margins"] += 1

        for target in repair_targets:
            selected_error_counts[str(target["repair_error_label"])] += 1
            selected_quality_counts[str(target["quality_label"])] += 1

        retained_repairs: list[tuple[dict, str]] = []
        for target in repair_targets:
            prompt = build_repair_prompt(
                problem=problem,
                candidate_text=str(target["candidate_text"]),
                motif_label=str(target["motif_label"]),
                quality_label=str(target["quality_label"]),
                repair_error_label=str(target["repair_error_label"]),
                answer_mode=answer_mode,
            )
            prompt_preview_records.append(_build_prompt_preview_record(example_id, problem, target, prompt))

            if args.dry_run:
                continue

            import torch

            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_length = encoded["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.samples_per_target,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            existing_dedupe = {
                _candidate_dedupe_key(candidate, args.dedupe_mode, answer_mode)
                for index, candidate in enumerate(existing_candidates)
                if index != int(target["candidate_index"])
            }
            repair_text = ""
            for sequence in generated:
                generated_tokens = sequence[prompt_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                formatted = _format_repair_candidate(generated_text, answer_mode)
                if not formatted:
                    generation_stats["rejected_empty_after_sanitize"] += 1
                    continue
                dedupe_key = _candidate_dedupe_key(formatted, args.dedupe_mode, answer_mode)
                if dedupe_key in existing_dedupe:
                    generation_stats["rejected_duplicate_repairs"] += 1
                    continue
                rejection_reasons = _repair_rejection_reasons(
                    formatted,
                    answer_mode=answer_mode,
                    source_quality_label=str(target["quality_label"]),
                    allow_replace_complete_attempt=args.allow_replace_complete_attempt,
                    allow_non_numeric_repairs=args.allow_non_numeric_repairs,
                )
                if rejection_reasons:
                    for reason in rejection_reasons:
                        generation_stats[f"rejected_{reason}"] += 1
                    continue
                repair_text = formatted
                break

            if repair_text:
                retained_repairs.append((target, repair_text))

        merged_candidates = list(existing_candidates)
        for target, repair_text in retained_repairs:
            target_index = int(target["candidate_index"])
            if target_index >= len(merged_candidates):
                continue
            merged_candidates[target_index] = repair_text
            generation_stats["retained_repair_candidates"] += 1
            generation_stats[f"replaced_{target['quality_label']}"] += 1

        if retained_repairs:
            generation_stats["examples_with_retained_repairs"] += 1
        if merged_candidates != existing_candidates:
            generation_stats["examples_modified"] += 1

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

    write_jsonl(args.output, output_rows)
    if args.prompt_preview_output is not None:
        write_jsonl(args.prompt_preview_output, prompt_preview_records)

    if args.metrics_output is not None:
        total_examples = len(output_rows)
        elapsed = time.perf_counter() - started_at
        metrics = {
            "base_candidates_path": str(args.base_candidates),
            "reranker_predictions_path": str(args.reranker_predictions) if args.reranker_predictions is not None else None,
            "output_path": str(args.output),
            "prompt_preview_output": str(args.prompt_preview_output) if args.prompt_preview_output is not None else None,
            "dry_run": args.dry_run,
            "total_examples": total_examples,
            "samples_per_target": args.samples_per_target,
            "max_repair_targets": args.max_repair_targets,
            "max_candidates": args.max_candidates,
            "protect_prefix_candidates": args.protect_prefix_candidates,
            "examples_with_selected_repair_targets": generation_stats.get("examples_with_selected_repair_targets", 0),
            "selected_repair_targets": generation_stats.get("selected_repair_targets", 0),
            "examples_with_retained_repairs": generation_stats.get("examples_with_retained_repairs", 0),
            "retained_repair_candidates": generation_stats.get("retained_repair_candidates", 0),
            "examples_modified": generation_stats.get("examples_modified", 0),
            "examples_with_reranker_margins": generation_stats.get("examples_with_reranker_margins", 0),
            "replaced_fragment": generation_stats.get("replaced_fragment", 0),
            "replaced_partial_solution": generation_stats.get("replaced_partial_solution", 0),
            "replaced_complete_attempt": generation_stats.get("replaced_complete_attempt", 0),
            "rejected_empty_after_sanitize": generation_stats.get("rejected_empty_after_sanitize", 0),
            "rejected_duplicate_repairs": generation_stats.get("rejected_duplicate_repairs", 0),
            "rejected_instruction_or_prompt_leak": generation_stats.get("rejected_instruction_or_prompt_leak", 0),
            "rejected_missing_numeric_answer": generation_stats.get("rejected_missing_numeric_answer", 0),
            "rejected_would_replace_complete_attempt": generation_stats.get("rejected_would_replace_complete_attempt", 0),
            "selected_error_type_counts": dict(sorted(selected_error_counts.items())),
            "selected_quality_counts": dict(sorted(selected_quality_counts.items())),
            "total_seconds": round(elapsed, 6),
            "avg_seconds_per_example": round(elapsed / total_examples, 6) if total_examples else 0.0,
            "strict_hygiene": args.strict_hygiene,
            "allow_replace_complete_attempt": args.allow_replace_complete_attempt,
            "allow_non_numeric_repairs": args.allow_non_numeric_repairs,
            "script_path": str(Path(__file__).resolve()),
            "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        }
        write_json(args.metrics_output, metrics)


if __name__ == "__main__":
    main()
