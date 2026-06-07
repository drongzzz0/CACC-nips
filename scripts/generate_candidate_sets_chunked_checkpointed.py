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
from src.utils.io_utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate open-ended answer candidates with resumable JSONL checkpoints."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--supervision-type", default="answer_only")
    parser.add_argument("--samples-per-example", default=8, type=int)
    parser.add_argument("--generation-batch-size", default=None, type=int)
    parser.add_argument("--max-candidates", default=4, type=int)
    parser.add_argument("--max-new-tokens", default=64, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--dedupe-mode", default="text", choices=("text", "numeric_or_text"))
    parser.add_argument(
        "--selection-strategy",
        default="dedupe_only",
        choices=("dedupe_only", "text_prefix_numeric_fill"),
    )
    parser.add_argument("--text-prefix-candidates", default=0, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--checkpoint-every", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
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


def _existing_stats(output: Path) -> tuple[int, int, int, int]:
    processed = 0
    total_candidates = 0
    first_candidate_correct = 0
    oracle_correct = 0
    if not output.exists():
        return processed, total_candidates, first_candidate_correct, oracle_correct

    for row in read_jsonl(output):
        processed += 1
        candidates = [str(candidate) for candidate in row.get("candidates", [])]
        gold_answer = str(row["gold_answer"])
        answer_mode = str(row.get("answer_mode", "numeric"))
        total_candidates += len(candidates)
        if candidates and _is_correct(candidates[0], gold_answer, answer_mode):
            first_candidate_correct += 1
        if any(_is_correct(candidate, gold_answer, answer_mode) for candidate in candidates):
            oracle_correct += 1
    return processed, total_candidates, first_candidate_correct, oracle_correct


def _write_progress(
    path: Path | None,
    *,
    dataset: Path,
    output: Path,
    processed: int,
    total_candidates: int,
    first_candidate_correct: int,
    oracle_correct: int,
    started_at: float,
    resumed_from: int,
) -> None:
    if path is None:
        return
    elapsed = time.perf_counter() - started_at
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "dataset_path": str(dataset),
            "output_path": str(output),
            "processed_examples": processed,
            "resumed_from_examples": resumed_from,
            "total_candidates": total_candidates,
            "first_candidate_correct": first_candidate_correct,
            "oracle_correct": oracle_correct,
            "elapsed_seconds_current_process": round(elapsed, 6),
            "avg_seconds_current_process": round(elapsed / max(processed - resumed_from, 1), 6),
        },
    )


def main() -> None:
    args = parse_args()
    if args.adapter_path is None and not args.model_path:
        raise SystemExit("Pass either --adapter-path for PEFT generation or --model-path for base-model generation.")
    if args.samples_per_example < 1:
        raise SystemExit("--samples-per-example must be at least 1.")
    if args.generation_batch_size is None:
        args.generation_batch_size = args.samples_per_example
    if args.generation_batch_size < 1:
        raise SystemExit("--generation-batch-size must be at least 1.")
    args.generation_batch_size = min(args.generation_batch_size, args.samples_per_example)
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be at least 1.")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1.")
    if args.selection_strategy == "text_prefix_numeric_fill" and args.text_prefix_candidates < 1:
        raise SystemExit("--text-prefix-candidates must be at least 1 for text_prefix_numeric_fill.")
    if args.output.exists() and args.output.stat().st_size > 0 and not args.resume:
        raise SystemExit(f"{args.output} already exists; pass --resume to continue from it.")

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

    resumed_from, total_candidates, first_candidate_correct, oracle_correct = _existing_stats(args.output)
    started_at = time.perf_counter()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    append_mode = "a" if args.resume else "w"
    processed_examples = resumed_from

    with args.output.open(append_mode, encoding="utf-8") as output_handle:
        for example_index, example in enumerate(read_jsonl(args.dataset), start=1):
            if example_index <= resumed_from:
                continue

            answer_mode = str(example.get("answer_mode", "numeric"))
            prompt = build_inference_prompt(
                _build_problem_prompt(example),
                args.supervision_type,
                answer_mode=answer_mode,
            )
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_length = encoded["input_ids"].shape[1]

            raw_predictions: list[str] = []
            remaining_samples = args.samples_per_example
            while remaining_samples > 0:
                current_batch = min(args.generation_batch_size, remaining_samples)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_return_sequences=current_batch,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                for sequence in generated:
                    generated_tokens = sequence[prompt_length:]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    prediction = _extract_prediction(generated_text).strip()
                    if not prediction:
                        continue
                    raw_predictions.append(prediction)
                remaining_samples -= current_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

            row = VerifierCandidateSet(
                example_id=str(example["example_id"]),
                dataset=str(example.get("dataset", "unknown")),
                problem=str(example.get("problem", example.get("prompt", ""))),
                gold_answer=gold_answer,
                candidates=unique_candidates,
                answer_mode=answer_mode,
                choices=[str(choice) for choice in example.get("choices", [])],
                metadata=dict(example.get("metadata", {})),
            ).to_dict()
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed_examples = example_index

            if example_index == 1 or example_index % 10 == 0:
                elapsed = time.perf_counter() - started_at
                current_count = max(example_index - resumed_from, 1)
                avg_seconds = elapsed / current_count
                print(
                    f"[checkpointed-gen] processed {example_index} examples; "
                    f"resumed_from={resumed_from}; elapsed={elapsed / 60:.1f}m; avg={avg_seconds:.2f}s/ex",
                    flush=True,
                )

            if example_index % args.checkpoint_every == 0:
                output_handle.flush()
                _write_progress(
                    args.progress_output,
                    dataset=args.dataset,
                    output=args.output,
                    processed=processed_examples,
                    total_candidates=total_candidates,
                    first_candidate_correct=first_candidate_correct,
                    oracle_correct=oracle_correct,
                    started_at=started_at,
                    resumed_from=resumed_from,
                )

    _write_progress(
        args.progress_output,
        dataset=args.dataset,
        output=args.output,
        processed=processed_examples,
        total_candidates=total_candidates,
        first_candidate_correct=first_candidate_correct,
        oracle_correct=oracle_correct,
        started_at=started_at,
        resumed_from=resumed_from,
    )

    if args.metrics_output is not None:
        total_examples = processed_examples
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
                "generation_batch_size": args.generation_batch_size,
                "max_candidates": args.max_candidates,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "dedupe_mode": args.dedupe_mode,
                "selection_strategy": args.selection_strategy,
                "text_prefix_candidates": args.text_prefix_candidates,
                "checkpoint_every": args.checkpoint_every,
                "resumed_from_examples": resumed_from,
                "num_examples": total_examples,
                "total_candidates": total_candidates,
                "avg_candidates_per_example": (total_candidates / total_examples) if total_examples else 0.0,
                "first_candidate_correct": first_candidate_correct,
                "first_candidate_accuracy": (first_candidate_correct / total_examples) if total_examples else 0.0,
                "oracle_correct": oracle_correct,
                "oracle_accuracy": (oracle_correct / total_examples) if total_examples else 0.0,
                "total_generation_seconds": round(total_seconds, 6),
                "avg_generation_seconds": round(total_seconds / max(total_examples - resumed_from, 1), 6),
            },
        )


if __name__ == "__main__":
    main()
