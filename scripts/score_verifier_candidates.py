from __future__ import annotations

import argparse
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import answers_match
from src.generation.prompts import build_verifier_prompt
from src.inference.peft_generation import _resolve_base_model, load_tokenizer_for_inference, missing_dependencies
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score verifier candidate answers with a PEFT adapter or base model.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    return parser.parse_args()


def _is_correct(prediction: str, gold_answer: str, answer_mode: str) -> bool:
    return answers_match(prediction, gold_answer, answer_mode=answer_mode)


def _encode_prompt_continuation_batch(tokenizer, prompts: list[str], continuations: list[str], device):
    import torch

    prompt_id_rows = tokenizer(prompts, add_special_tokens=True)["input_ids"]
    continuation_id_rows = tokenizer(continuations, add_special_tokens=False)["input_ids"]
    merged_rows = [prompt_ids + continuation_ids for prompt_ids, continuation_ids in zip(prompt_id_rows, continuation_id_rows)]
    max_length = max(len(row) for row in merged_rows)

    input_ids = torch.full(
        (len(merged_rows), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    target_mask = torch.zeros((len(merged_rows), max_length - 1), dtype=torch.bool, device=device)

    for row_index, (prompt_ids, continuation_ids, merged_ids) in enumerate(
        zip(prompt_id_rows, continuation_id_rows, merged_rows)
    ):
        row_length = len(merged_ids)
        prompt_length = len(prompt_ids)
        continuation_length = len(continuation_ids)
        input_ids[row_index, :row_length] = torch.tensor(merged_ids, dtype=torch.long, device=device)
        attention_mask[row_index, :row_length] = 1
        if continuation_length:
            target_mask[row_index, prompt_length - 1 : prompt_length - 1 + continuation_length] = True

    return input_ids, attention_mask, target_mask


def _score_continuations_batched(model, tokenizer, device, prompts: list[str], continuations: list[str]) -> list[float]:
    import torch

    input_ids, attention_mask, target_mask = _encode_prompt_continuation_batch(
        tokenizer,
        prompts=prompts,
        continuations=continuations,
        device=device,
    )

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    selected_token_logits = shifted_logits.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    normalization = torch.logsumexp(shifted_logits, dim=-1)
    token_log_probs = (selected_token_logits - normalization).masked_fill(~target_mask, 0.0)
    return token_log_probs.sum(dim=1).tolist()


def _score_continuation(model, tokenizer, device, prompt: str, continuation: str) -> float:
    return _score_continuations_batched(
        model,
        tokenizer,
        device,
        prompts=[prompt],
        continuations=[continuation],
    )[0]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards.")

    require_peft = args.adapter_path is not None
    missing = missing_dependencies(require_peft=require_peft)
    if missing:
        raise RuntimeError("Missing inference dependencies: " + ", ".join(missing))

    import torch
    from transformers import AutoModelForCausalLM
    if require_peft:
        from peft import PeftModel

    base_model_name_or_path = _resolve_base_model(args.adapter_path, args.base_model, args.model_path)
    tokenizer = load_tokenizer_for_inference(
        adapter_path=args.adapter_path,
        base_model_name_or_path=base_model_name_or_path,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, dtype=dtype)
    model = PeftModel.from_pretrained(base_model, args.adapter_path) if args.adapter_path is not None else base_model
    model.to(device)
    model.eval()

    prediction_rows = []
    started_at = time.perf_counter()
    total_batches = 0
    processed_examples = 0
    for record_index, record in enumerate(read_jsonl(args.dataset)):
        if args.max_examples is not None and record_index >= args.max_examples:
            break
        if record_index % args.num_shards != args.shard_index:
            continue
        processed_examples += 1
        answer_mode = str(record.get("answer_mode", "numeric"))
        score_requests = []
        for candidate_index, candidate in enumerate(record["candidates"]):
            prompt = build_verifier_prompt(record["problem"], candidate, answer_mode=answer_mode)
            score_requests.append(
                {
                    "candidate_index": candidate_index,
                    "continuation_label": "yes",
                    "prompt": prompt,
                    "continuation": " yes",
                }
            )
            score_requests.append(
                {
                    "candidate_index": candidate_index,
                    "continuation_label": "no",
                    "prompt": prompt,
                    "continuation": " no",
                }
            )

        score_lookup = {}
        for start_index in range(0, len(score_requests), args.batch_size):
            batch_requests = score_requests[start_index : start_index + args.batch_size]
            batch_scores = _score_continuations_batched(
                model,
                tokenizer,
                device,
                prompts=[request["prompt"] for request in batch_requests],
                continuations=[request["continuation"] for request in batch_requests],
            )
            total_batches += 1
            for request, score in zip(batch_requests, batch_scores):
                score_lookup[(request["candidate_index"], request["continuation_label"])] = score

        scored_candidates = []
        for candidate_index, candidate in enumerate(record["candidates"]):
            yes_score = score_lookup[(candidate_index, "yes")]
            no_score = score_lookup[(candidate_index, "no")]
            scored_candidates.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_answer": candidate,
                    "yes_score": yes_score,
                    "no_score": no_score,
                    "margin": yes_score - no_score,
                }
            )
        scored_candidates.sort(key=lambda item: item["margin"], reverse=True)
        best_candidate = scored_candidates[0]["candidate_answer"]
        prediction_rows.append(
            {
                "example_id": record["example_id"],
                "prediction": best_candidate,
                "gold_answer": record["gold_answer"],
                "answer_mode": answer_mode,
                "correct": _is_correct(best_candidate, record["gold_answer"], answer_mode=answer_mode),
                "candidates": scored_candidates,
            }
        )

    write_jsonl(args.predictions, prediction_rows)
    if args.metrics_output is not None:
        example_count = len(prediction_rows)
        correct = sum(1 for row in prediction_rows if row["correct"])
        total_seconds = time.perf_counter() - started_at
        write_json(
            args.metrics_output,
            {
                "dataset_path": str(args.dataset),
                "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
                "base_model": base_model_name_or_path,
                "batch_size": args.batch_size,
                "max_examples": args.max_examples,
                "num_examples": example_count,
                "correct": correct,
                "accuracy": (correct / example_count) if example_count else 0.0,
                "total_batches": total_batches,
                "total_scoring_seconds": round(total_seconds, 6),
                "avg_scoring_seconds": round(total_seconds / example_count, 6) if example_count else 0.0,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "processed_examples": processed_examples,
            },
        )


if __name__ == "__main__":
    main()
