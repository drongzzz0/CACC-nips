from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.common import (
    dataset_records,
    finalize_metrics,
    load_causal_lm,
    run_batch_generation,
    run_generation,
    seed_everything,
    write_metrics,
    write_predictions,
    write_trace_jsonl,
)
from src.baselines.prompts import (
    build_self_refine_feedback_prompt,
    build_self_refine_init_prompt,
    build_self_refine_refine_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Self-Refine baseline on a reasoning dataset.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--feedback-max-new-tokens", default=96, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--max-refine-rounds", default=2, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--seed", default=7, type=int)
    return parser.parse_args()


def _run_serial(args: argparse.Namespace, rows: list[dict], lm) -> tuple[list[dict], list[dict], int, float]:
    prediction_rows = []
    trace_rows = []
    started_at = time.perf_counter()
    total_model_calls = 0

    for example in rows:
        problem = str(example.get("problem", example.get("prompt", "")))
        answer_mode = str(example.get("answer_mode", "numeric"))

        init_prompt = build_self_refine_init_prompt(str(example.get("prompt", f"Problem: {problem}")), answer_mode)
        init_result = run_generation(
            lm,
            init_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )
        total_model_calls += 1
        current_text = init_result["generated_text"]
        current_prediction = init_result["prediction"]
        history = [
            {
                "stage": "init",
                "prompt": init_prompt,
                "generated_text": current_text,
                "prediction": current_prediction,
            }
        ]

        for round_index in range(args.max_refine_rounds):
            feedback_prompt = build_self_refine_feedback_prompt(problem, current_text, answer_mode)
            feedback_result = run_generation(
                lm,
                feedback_prompt,
                max_new_tokens=args.feedback_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            total_model_calls += 1

            refine_prompt = build_self_refine_refine_prompt(
                problem,
                current_text,
                feedback_result["generated_text"],
                answer_mode,
            )
            refine_result = run_generation(
                lm,
                refine_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
            )
            total_model_calls += 1

            current_text = refine_result["generated_text"] or current_text
            current_prediction = refine_result["prediction"] or current_prediction
            history.append(
                {
                    "stage": f"feedback_{round_index + 1}",
                    "prompt": feedback_prompt,
                    "generated_text": feedback_result["generated_text"],
                    "prediction": feedback_result["prediction"],
                }
            )
            history.append(
                {
                    "stage": f"refine_{round_index + 1}",
                    "prompt": refine_prompt,
                    "generated_text": current_text,
                    "prediction": current_prediction,
                }
            )

        prediction_rows.append(
            {
                "example_id": str(example["example_id"]),
                "prediction": current_prediction,
                "gold_answer": str(example["gold_answer"]),
                "answer_mode": answer_mode,
                "dataset": str(example.get("dataset", "unknown")),
                "problem": problem,
                "metadata": dict(example.get("metadata", {})),
            }
        )
        trace_rows.append(
            {
                "example_id": str(example["example_id"]),
                "dataset": str(example.get("dataset", "unknown")),
                "gold_answer": str(example["gold_answer"]),
                "answer_mode": answer_mode,
                "history": history,
            }
        )

    return prediction_rows, trace_rows, total_model_calls, started_at


def _batched_rows(rows: list[dict], batch_size: int):
    for start_index in range(0, len(rows), batch_size):
        yield rows[start_index : start_index + batch_size]


def _run_batched(args: argparse.Namespace, rows: list[dict], lm) -> tuple[list[dict], list[dict], int, float]:
    prediction_rows = []
    trace_rows = []
    started_at = time.perf_counter()
    total_model_calls = 0

    for example_batch in _batched_rows(rows, args.batch_size):
        problems = [str(example.get("problem", example.get("prompt", ""))) for example in example_batch]
        answer_modes = [str(example.get("answer_mode", "numeric")) for example in example_batch]
        init_prompts = [
            build_self_refine_init_prompt(str(example.get("prompt", f"Problem: {problem}")), answer_mode)
            for example, problem, answer_mode in zip(example_batch, problems, answer_modes)
        ]
        init_results = run_batch_generation(
            lm,
            init_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )
        total_model_calls += len(example_batch)

        current_texts = [row["generated_text"] for row in init_results]
        current_predictions = [row["prediction"] for row in init_results]
        histories = [
            [
                {
                    "stage": "init",
                    "prompt": prompt,
                    "generated_text": init_result["generated_text"],
                    "prediction": init_result["prediction"],
                }
            ]
            for prompt, init_result in zip(init_prompts, init_results)
        ]

        for round_index in range(args.max_refine_rounds):
            feedback_prompts = [
                build_self_refine_feedback_prompt(problem, current_text, answer_mode)
                for problem, current_text, answer_mode in zip(problems, current_texts, answer_modes)
            ]
            feedback_results = run_batch_generation(
                lm,
                feedback_prompts,
                max_new_tokens=args.feedback_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            total_model_calls += len(example_batch)

            refine_prompts = [
                build_self_refine_refine_prompt(problem, current_text, feedback_result["generated_text"], answer_mode)
                for problem, current_text, feedback_result, answer_mode in zip(
                    problems,
                    current_texts,
                    feedback_results,
                    answer_modes,
                )
            ]
            refine_results = run_batch_generation(
                lm,
                refine_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
            )
            total_model_calls += len(example_batch)

            for item_index, refine_result in enumerate(refine_results):
                current_texts[item_index] = refine_result["generated_text"] or current_texts[item_index]
                current_predictions[item_index] = refine_result["prediction"] or current_predictions[item_index]
                histories[item_index].append(
                    {
                        "stage": f"feedback_{round_index + 1}",
                        "prompt": feedback_prompts[item_index],
                        "generated_text": feedback_results[item_index]["generated_text"],
                        "prediction": feedback_results[item_index]["prediction"],
                    }
                )
                histories[item_index].append(
                    {
                        "stage": f"refine_{round_index + 1}",
                        "prompt": refine_prompts[item_index],
                        "generated_text": current_texts[item_index],
                        "prediction": current_predictions[item_index],
                    }
                )

        for example, problem, answer_mode, current_prediction, history in zip(
            example_batch,
            problems,
            answer_modes,
            current_predictions,
            histories,
        ):
            prediction_rows.append(
                {
                    "example_id": str(example["example_id"]),
                    "prediction": current_prediction,
                    "gold_answer": str(example["gold_answer"]),
                    "answer_mode": answer_mode,
                    "dataset": str(example.get("dataset", "unknown")),
                    "problem": problem,
                    "metadata": dict(example.get("metadata", {})),
                }
            )
            trace_rows.append(
                {
                    "example_id": str(example["example_id"]),
                    "dataset": str(example.get("dataset", "unknown")),
                    "gold_answer": str(example["gold_answer"]),
                    "answer_mode": answer_mode,
                    "history": history,
                }
            )

    return prediction_rows, trace_rows, total_model_calls, started_at


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    lm = load_causal_lm(args.adapter_path, args.model_path, args.base_model)
    rows = dataset_records(args.dataset)

    if args.batch_size == 1:
        prediction_rows, trace_rows, total_model_calls, started_at = _run_serial(args, rows, lm)
    else:
        prediction_rows, trace_rows, total_model_calls, started_at = _run_batched(args, rows, lm)

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
                "baseline": "self_refine",
                "max_refine_rounds": args.max_refine_rounds,
                "total_model_calls": total_model_calls,
                "batch_size": args.batch_size,
                "avg_model_calls_per_example": (total_model_calls / len(prediction_rows)) if prediction_rows else 0.0,
            },
        ),
    )


if __name__ == "__main__":
    main()
