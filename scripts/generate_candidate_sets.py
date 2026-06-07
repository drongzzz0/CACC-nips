from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import VerifierCandidateSet
from src.eval.evaluate_predictions import answers_match, extract_numeric_answer, normalize_answer
from src.generation.prompts import build_inference_prompt
from src.inference.peft_generation import _extract_prediction, _resolve_base_model, load_tokenizer_for_inference, missing_dependencies
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate open-ended answer candidates and save them as verifier candidate sets."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--supervision-type", default="answer_only")
    parser.add_argument("--samples-per-example", default=8, type=int)
    parser.add_argument("--max-candidates", default=4, type=int)
    parser.add_argument("--max-new-tokens", default=64, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument(
        "--dedupe-mode",
        default="text",
        choices=("text", "numeric_or_text"),
        help="How to deduplicate candidate answers before truncation.",
    )
    parser.add_argument(
        "--selection-strategy",
        default="dedupe_only",
        choices=("dedupe_only", "text_prefix_numeric_fill"),
        help="How to select the retained candidate pool from sampled generations.",
    )
    parser.add_argument(
        "--text-prefix-candidates",
        default=0,
        type=int,
        help="For text_prefix_numeric_fill, keep this many leading text-unique candidates before adding numeric-diverse extras.",
    )
    parser.add_argument("--seed", default=7, type=int)
    return parser.parse_args()


def _build_problem_prompt(example: dict) -> str:
    if "prompt" in example:
        return str(example["prompt"])
    return f"Problem: {example['problem']}\nAnswer with the requested format."


def _is_correct(prediction: str, gold_answer: str, answer_mode: str) -> bool:
    return answers_match(prediction, gold_answer, answer_mode=answer_mode)


def _candidate_dedupe_key(prediction: str, dedupe_mode: str) -> str:
    normalized = normalize_answer(prediction)
    if dedupe_mode == "text":
        return f"text:{normalized}"
    numeric = extract_numeric_answer(prediction)
    if numeric is not None:
        return f"num:{numeric}"
    return f"text:{normalized}"


def _dedupe_candidates(predictions: list[str], dedupe_mode: str, max_candidates: int) -> list[str]:
    unique_candidates: list[str] = []
    seen = set()
    for prediction in predictions:
        dedupe_key = _candidate_dedupe_key(prediction, dedupe_mode)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique_candidates.append(prediction)
        if len(unique_candidates) >= max_candidates:
            break
    return unique_candidates


def _text_prefix_numeric_fill_candidates(
    predictions: list[str],
    max_candidates: int,
    text_prefix_candidates: int,
) -> list[str]:
    text_unique: list[str] = []
    seen_text = set()
    for prediction in predictions:
        normalized = normalize_answer(prediction)
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        text_unique.append(prediction)

    prefix_count = min(max(text_prefix_candidates, 0), max_candidates, len(text_unique))
    selected = list(text_unique[:prefix_count])
    selected_text = {normalize_answer(candidate) for candidate in selected}
    selected_numeric = {
        numeric_answer
        for candidate in selected
        for numeric_answer in [extract_numeric_answer(candidate)]
        if numeric_answer is not None
    }

    for candidate in text_unique[prefix_count:]:
        if len(selected) >= max_candidates:
            break
        normalized = normalize_answer(candidate)
        if normalized in selected_text:
            continue
        numeric_answer = extract_numeric_answer(candidate)
        if numeric_answer is None or numeric_answer in selected_numeric:
            continue
        selected.append(candidate)
        selected_text.add(normalized)
        selected_numeric.add(numeric_answer)

    for candidate in text_unique[prefix_count:]:
        if len(selected) >= max_candidates:
            break
        normalized = normalize_answer(candidate)
        if normalized in selected_text:
            continue
        selected.append(candidate)
        selected_text.add(normalized)

    return selected


def main() -> None:
    args = parse_args()
    if args.adapter_path is None and not args.model_path:
        raise SystemExit("Pass either --adapter-path for PEFT generation or --model-path for base-model generation.")
    if args.samples_per_example < 1:
        raise SystemExit("--samples-per-example must be at least 1.")
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be at least 1.")
    if args.selection_strategy == "text_prefix_numeric_fill" and args.text_prefix_candidates < 1:
        raise SystemExit("--text-prefix-candidates must be at least 1 for text_prefix_numeric_fill.")

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

    candidate_rows = []
    total_candidates = 0
    first_candidate_correct = 0
    oracle_correct = 0
    started_at = time.perf_counter()

    for example in read_jsonl(args.dataset):
        answer_mode = str(example.get("answer_mode", "numeric"))
        prompt = build_inference_prompt(
            _build_problem_prompt(example),
            args.supervision_type,
            answer_mode=answer_mode,
        )
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
                num_return_sequences=args.samples_per_example,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        raw_predictions: list[str] = []
        for sequence in generated:
            generated_tokens = sequence[prompt_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            prediction = _extract_prediction(generated_text).strip()
            if not prediction:
                continue
            raw_predictions.append(prediction)

        if args.selection_strategy == "dedupe_only":
            unique_candidates = _dedupe_candidates(
                raw_predictions,
                dedupe_mode=args.dedupe_mode,
                max_candidates=args.max_candidates,
            )
        else:
            unique_candidates = _text_prefix_numeric_fill_candidates(
                raw_predictions,
                max_candidates=args.max_candidates,
                text_prefix_candidates=args.text_prefix_candidates,
            )

        if not unique_candidates:
            unique_candidates = [""]

        gold_answer = str(example["gold_answer"])
        total_candidates += len(unique_candidates)
        if _is_correct(unique_candidates[0], gold_answer, answer_mode=answer_mode):
            first_candidate_correct += 1
        if any(_is_correct(candidate, gold_answer, answer_mode=answer_mode) for candidate in unique_candidates):
            oracle_correct += 1

        candidate_rows.append(
            VerifierCandidateSet(
                example_id=str(example["example_id"]),
                dataset=str(example.get("dataset", "unknown")),
                problem=str(example.get("problem", example.get("prompt", ""))),
                gold_answer=gold_answer,
                candidates=unique_candidates,
                answer_mode=answer_mode,
                choices=[str(choice) for choice in example.get("choices", [])],
                metadata=dict(example.get("metadata", {})),
            ).to_dict()
        )

    write_jsonl(args.output, candidate_rows)
    if args.metrics_output is not None:
        total_examples = len(candidate_rows)
        total_seconds = time.perf_counter() - started_at
        write_json(
            args.metrics_output,
            {
                "dataset_path": str(args.dataset),
                "output_path": str(args.output),
                "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
                "base_model": base_model_name_or_path,
                "supervision_type": args.supervision_type,
                "samples_per_example": args.samples_per_example,
                "max_candidates": args.max_candidates,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "dedupe_mode": args.dedupe_mode,
                "selection_strategy": args.selection_strategy,
                "text_prefix_candidates": args.text_prefix_candidates,
                "num_examples": total_examples,
                "total_candidates": total_candidates,
                "avg_candidates_per_example": (total_candidates / total_examples) if total_examples else 0.0,
                "first_candidate_correct": first_candidate_correct,
                "first_candidate_accuracy": (first_candidate_correct / total_examples) if total_examples else 0.0,
                "oracle_correct": oracle_correct,
                "oracle_accuracy": (oracle_correct / total_examples) if total_examples else 0.0,
                "total_generation_seconds": round(total_seconds, 6),
                "avg_generation_seconds": round(total_seconds / total_examples, 6) if total_examples else 0.0,
            },
        )


if __name__ == "__main__":
    main()
