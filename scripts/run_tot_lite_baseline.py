from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.common import (
    build_problem_prompt,
    dataset_records,
    dedupe_preserve_order,
    finalize_metrics,
    generate_completions,
    load_causal_lm,
    run_generation,
    seed_everything,
    write_metrics,
    write_predictions,
)
from src.baselines.prompts import build_tot_expand_prompt, build_tot_value_prompt, parse_tot_score
from src.data.schema import VerifierCandidateSet
from src.utils.io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight Tree-of-Thoughts baseline.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidates-output", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--search-depth", default=2, type=int)
    parser.add_argument("--branch-factor", default=4, type=int)
    parser.add_argument("--beam-size", default=2, type=int)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--value-max-new-tokens", default=16, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--seed", default=7, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    lm = load_causal_lm(args.adapter_path, args.model_path, args.base_model)
    rows = dataset_records(args.dataset)

    candidate_rows = []
    prediction_rows = []
    started_at = time.perf_counter()
    total_generation_calls = 0
    total_value_calls = 0

    for example in rows:
        answer_mode = str(example.get("answer_mode", "numeric"))
        problem = str(example.get("problem", example.get("prompt", "")))
        problem_prompt = build_problem_prompt(example)

        beam = [{"candidate": "", "score": 0, "depth": 0}]
        all_candidates = []

        for depth in range(1, args.search_depth + 1):
            expanded = []
            for state in beam:
                expand_prompt = build_tot_expand_prompt(
                    problem_prompt,
                    state["candidate"],
                    answer_mode,
                    depth,
                    args.search_depth,
                )
                expansions = generate_completions(
                    lm,
                    expand_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    num_return_sequences=args.branch_factor,
                )
                total_generation_calls += 1
                for row in expansions:
                    candidate = row["generated_text"].strip()
                    prediction = row["prediction"].strip() or candidate
                    value_prompt = build_tot_value_prompt(problem, candidate, answer_mode)
                    value_result = run_generation(
                        lm,
                        value_prompt,
                        max_new_tokens=args.value_max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        do_sample=False,
                    )
                    total_value_calls += 1
                    score = parse_tot_score(value_result["generated_text"])
                    expanded.append(
                        {
                            "candidate": candidate,
                            "prediction": prediction,
                            "score": score,
                            "depth": depth,
                            "value_text": value_result["generated_text"],
                        }
                    )
                    all_candidates.append(prediction)

            seen = set()
            deduped_expanded = []
            for row in sorted(expanded, key=lambda item: (-item["score"], item["candidate"])):
                dedupe_key = " ".join(row["prediction"].split())
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                deduped_expanded.append(row)
            beam = deduped_expanded[: max(args.beam_size, 1)] or beam

        retained_candidates = dedupe_preserve_order(all_candidates)[: max(args.beam_size * args.search_depth, 1)]
        if not retained_candidates:
            retained_candidates = [beam[0]["prediction"] if beam and beam[0].get("prediction") else ""]
        best_candidate = beam[0]["prediction"] if beam and beam[0].get("prediction") else retained_candidates[0]

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
                    "tot_lite_search_depth": args.search_depth,
                    "tot_lite_branch_factor": args.branch_factor,
                    "tot_lite_beam": beam,
                },
            ).to_dict()
        )
        prediction_rows.append(
            {
                "example_id": str(example["example_id"]),
                "prediction": best_candidate,
                "gold_answer": str(example["gold_answer"]),
                "answer_mode": answer_mode,
                "dataset": str(example.get("dataset", "unknown")),
                "problem": problem,
                "metadata": {
                    **dict(example.get("metadata", {})),
                    "tot_lite_best_score": int(beam[0]["score"]) if beam else 0,
                    "tot_lite_retained_candidates": len(retained_candidates),
                },
            }
        )

    write_jsonl(args.candidates_output, candidate_rows)
    write_predictions(args.predictions, prediction_rows)
    write_metrics(
        args.metrics_output,
        finalize_metrics(
            dataset_path=args.dataset,
            adapter_path=args.adapter_path,
            base_model_name_or_path=lm.base_model_name_or_path,
            started_at=started_at,
            example_count=len(prediction_rows),
            extra={
                "baseline": "tot_lite",
                "search_depth": args.search_depth,
                "branch_factor": args.branch_factor,
                "beam_size": args.beam_size,
                "total_generation_calls": total_generation_calls,
                "total_value_calls": total_value_calls,
            },
        ),
    )


if __name__ == "__main__":
    main()
