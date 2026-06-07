from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.common import (
    baseline_inference_prompt,
    dataset_records,
    dedupe_preserve_order,
    finalize_metrics,
    generate_completions,
    load_causal_lm,
    seed_everything,
    write_metrics,
    write_predictions,
    write_trace_jsonl,
)
from src.baselines.prompts import (
    build_ccqa_question_generation_prompt,
    build_ccqa_similarity_prompt,
    parse_ccqa_selected_index,
)
from src.data.schema import VerifierCandidateSet
from src.utils.io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CCQA-style baseline on a reasoning dataset.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidates-output", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--qgen-model", default="google/flan-t5-base")
    parser.add_argument("--supervision-type", default="filtered_cot")
    parser.add_argument("--best-of-n", default=5, type=int)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--qgen-max-new-tokens", default=96, type=int)
    parser.add_argument("--qgen-temperature", default=0.0, type=float)
    parser.add_argument("--qgen-top-p", default=1.0, type=float)
    parser.add_argument("--selector-max-new-tokens", default=16, type=int)
    parser.add_argument("--seed", default=7, type=int)
    return parser.parse_args()


def load_t5_generator(model_name_or_path: str) -> tuple[object, object, object]:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def run_t5_generation(
    tokenizer: object,
    model: object,
    device: object,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output = model.generate(**generation_kwargs)
    # T5 is encoder-decoder, so generated ids do not contain the encoder prompt tokens.
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    lm = load_causal_lm(args.adapter_path, args.model_path, args.base_model)
    qgen_tokenizer, qgen_model, qgen_device = load_t5_generator(args.qgen_model)
    rows = dataset_records(args.dataset)

    candidate_rows = []
    prediction_rows = []
    trace_rows = []
    started_at = time.perf_counter()
    total_generation_calls = 0
    total_qgen_calls = 0
    total_selector_calls = 0

    for example in rows:
        answer_mode = str(example.get("answer_mode", "numeric"))
        problem = str(example.get("problem", example.get("prompt", "")))
        problem_prompt = baseline_inference_prompt(example, args.supervision_type)

        sampled = generate_completions(
            lm,
            problem_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            num_return_sequences=args.best_of_n,
        )
        total_generation_calls += 1

        generated_questions = []
        trace_candidates = []
        for row in sampled:
            candidate_text = row["generated_text"].strip()
            qgen_prompt = build_ccqa_question_generation_prompt(candidate_text, answer_mode)
            generated_question = run_t5_generation(
                qgen_tokenizer,
                qgen_model,
                qgen_device,
                qgen_prompt,
                max_new_tokens=args.qgen_max_new_tokens,
                temperature=args.qgen_temperature,
                top_p=args.qgen_top_p,
                do_sample=args.qgen_temperature > 0.0,
            )
            total_qgen_calls += 1
            generated_questions.append(generated_question)
            trace_candidates.append(
                {
                    "candidate_text": candidate_text,
                    "prediction": row["prediction"].strip() or candidate_text,
                    "generated_question": generated_question,
                    "qgen_prompt": qgen_prompt,
                }
            )

        if trace_candidates:
            similarity_prompt = build_ccqa_similarity_prompt(problem, generated_questions)
            selector_text = run_t5_generation(
                qgen_tokenizer,
                qgen_model,
                qgen_device,
                similarity_prompt,
                max_new_tokens=args.selector_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            total_selector_calls += 1
            selected_index = parse_ccqa_selected_index(selector_text, len(trace_candidates))
        else:
            similarity_prompt = ""
            selector_text = ""
            selected_index = 0

        retained_candidates = dedupe_preserve_order(
            [candidate["prediction"] for candidate in trace_candidates]
        )
        if not retained_candidates:
            retained_candidates = [""]

        best_prediction = trace_candidates[selected_index]["prediction"] if trace_candidates else retained_candidates[0]

        candidate_rows.append(
            VerifierCandidateSet(
                example_id=str(example["example_id"]),
                dataset=str(example.get("dataset", "unknown")),
                problem=problem,
                gold_answer=str(example["gold_answer"]),
                candidates=retained_candidates,
                answer_mode=answer_mode,
                choices=[str(choice) for choice in example.get("choices", [])],
                metadata={
                    **dict(example.get("metadata", {})),
                    "ccqa_best_of_n": args.best_of_n,
                    "ccqa_selected_index": selected_index,
                    "ccqa_generated_questions": generated_questions,
                },
            ).to_dict()
        )
        prediction_rows.append(
            {
                "example_id": str(example["example_id"]),
                "prediction": best_prediction,
                "gold_answer": str(example["gold_answer"]),
                "answer_mode": answer_mode,
                "dataset": str(example.get("dataset", "unknown")),
                "problem": problem,
                "metadata": {
                    **dict(example.get("metadata", {})),
                    "ccqa_best_of_n": args.best_of_n,
                    "ccqa_selected_index": selected_index,
                },
            }
        )
        trace_rows.append(
            {
                "example_id": str(example["example_id"]),
                "dataset": str(example.get("dataset", "unknown")),
                "gold_answer": str(example["gold_answer"]),
                "answer_mode": answer_mode,
                "problem_prompt": problem_prompt,
                "selector_prompt": similarity_prompt,
                "selector_output": selector_text,
                "selected_index": selected_index,
                "candidates": trace_candidates,
            }
        )

    write_jsonl(args.candidates_output, candidate_rows)
    write_predictions(args.predictions, prediction_rows)
    write_trace_jsonl(args.trace_output, trace_rows)
    write_metrics(
        args.metrics_output,
        finalize_metrics(
            dataset_path=args.dataset,
            adapter_path=args.adapter_path,
            base_model_name_or_path=lm.base_model_name_or_path,
            started_at=started_at,
            example_count=len(prediction_rows),
            extra={
                "baseline": "ccqa",
                "qgen_model": args.qgen_model,
                "best_of_n": args.best_of_n,
                "supervision_type": args.supervision_type,
                "total_generation_calls": total_generation_calls,
                "total_qgen_calls": total_qgen_calls,
                "total_selector_calls": total_selector_calls,
            },
        ),
    )


if __name__ == "__main__":
    main()
