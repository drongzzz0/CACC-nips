from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

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
)
from src.data.schema import VerifierCandidateSet
from src.utils.io_utils import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PairRM best-of-n reranking on sampled candidates.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidates-output", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--supervision-type", default="filtered_cot")
    parser.add_argument("--best-of-n", default=8, type=int)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--pairrm-model", default="llm-blender/PairRM-hf")
    parser.add_argument("--source-max-length", default=1224, type=int)
    parser.add_argument("--candidate-max-length", default=412, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--log-every", default=25, type=int)
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _load_pairrm_model(model_name: str):
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("PairRM requires torch and transformers to be installed.") from exc

    pairrm_path = ROOT.parents[1] / "Experiment" / "code_references" / "LLM-Blender" / "llm_blender" / "pair_ranker" / "pairrm.py"
    spec = importlib.util.spec_from_file_location("pairrm_module", pairrm_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PairRM module from {pairrm_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    DebertaV2PairRM = module.DebertaV2PairRM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    if getattr(config, "sep_token_id", None) is None:
        config.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 2

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaV2PairRM.from_pretrained(model_name, config=config, torch_dtype=dtype).to(device).eval()
    return model, tokenizer, device


def _tokenize_pair(tokenizer, sources: list[str], candidate1s: list[str], candidate2s: list[str], source_max_length: int, candidate_max_length: int):
    encodings = []
    max_length = source_max_length + 2 * candidate_max_length
    for source, cand1, cand2 in zip(sources, candidate1s, candidate2s):
        source_ids = tokenizer.encode("<|source|>" + source, max_length=source_max_length, truncation=True)
        pair_candidate_max = max(8, (max_length - len(source_ids)) // 2)
        cand1_ids = tokenizer.encode("<|candidate1|>" + cand1, max_length=pair_candidate_max, truncation=True)
        cand2_ids = tokenizer.encode("<|candidate2|>" + cand2, max_length=pair_candidate_max, truncation=True)
        encodings.append(source_ids + cand1_ids + cand2_ids)
    return tokenizer.pad(
        {"input_ids": encodings},
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )


def _pairwise_logits(model, tokenizer, device, source: str, candidate_a: str, candidate_b: str, source_max_length: int, candidate_max_length: int) -> float:
    import torch

    encodings = _tokenize_pair(
        tokenizer,
        [source],
        [candidate_a],
        [candidate_b],
        source_max_length=source_max_length,
        candidate_max_length=candidate_max_length,
    )
    encodings = {key: value.to(device) for key, value in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
    return float(outputs.logits[0].item())


def _rank_candidates(model, tokenizer, device, source: str, candidates: list[str], source_max_length: int, candidate_max_length: int) -> list[dict]:
    if len(candidates) == 1:
        return [{"candidate": candidates[0], "wins": 0, "score_sum": 0.0, "rank": 1}]

    wins = [0 for _ in candidates]
    score_sums = [0.0 for _ in candidates]
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            logit = _pairwise_logits(
                model,
                tokenizer,
                device,
                source,
                candidates[left_index],
                candidates[right_index],
                source_max_length=source_max_length,
                candidate_max_length=candidate_max_length,
            )
            score_sums[left_index] += logit
            score_sums[right_index] -= logit
            if logit >= 0:
                wins[left_index] += 1
            else:
                wins[right_index] += 1

    ranked = sorted(
        [
            {
                "candidate": candidate,
                "wins": wins[index],
                "score_sum": round(score_sums[index], 6),
            }
            for index, candidate in enumerate(candidates)
        ],
        key=lambda row: (-row["wins"], -row["score_sum"], row["candidate"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    lm = load_causal_lm(args.adapter_path, args.model_path, args.base_model)
    pairrm_model, pairrm_tokenizer, pairrm_device = _load_pairrm_model(args.pairrm_model)
    rows = dataset_records(args.dataset)
    total_examples = len(rows)

    candidate_rows = []
    prediction_rows = []
    started_at = time.perf_counter()
    total_generation_calls = 0

    print(
        f"[start] pairrm dataset={args.dataset} total_examples={total_examples} best_of_n={args.best_of_n} "
        f"max_new_tokens={args.max_new_tokens} log_every={args.log_every}",
        flush=True,
    )

    for example_index, example in enumerate(rows, start=1):
        prompt = baseline_inference_prompt(example, args.supervision_type)
        generations = generate_completions(
            lm,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            num_return_sequences=args.best_of_n,
        )
        total_generation_calls += 1

        candidates = dedupe_preserve_order([row["prediction"] for row in generations if row["prediction"].strip()])
        if not candidates:
            candidates = [""]

        ranked = _rank_candidates(
            pairrm_model,
            pairrm_tokenizer,
            pairrm_device,
            str(example.get("problem", example.get("prompt", ""))),
            candidates,
            source_max_length=args.source_max_length,
            candidate_max_length=args.candidate_max_length,
        )
        best_candidate = ranked[0]["candidate"]

        candidate_rows.append(
            VerifierCandidateSet(
                example_id=str(example["example_id"]),
                dataset=str(example.get("dataset", "unknown")),
                problem=str(example.get("problem", example.get("prompt", ""))),
                gold_answer=str(example["gold_answer"]),
                candidates=candidates,
                answer_mode=str(example.get("answer_mode", "numeric")),
                choices=[str(choice) for choice in example.get("choices", [])],
                metadata={
                    **dict(example.get("metadata", {})),
                    "pairrm_ranking": ranked,
                },
            ).to_dict()
        )
        prediction_rows.append(
            {
                "example_id": str(example["example_id"]),
                "prediction": best_candidate,
                "gold_answer": str(example["gold_answer"]),
                "answer_mode": str(example.get("answer_mode", "numeric")),
                "dataset": str(example.get("dataset", "unknown")),
                "problem": str(example.get("problem", example.get("prompt", ""))),
                "metadata": {
                    **dict(example.get("metadata", {})),
                    "pairrm_best_rank": int(ranked[0]["rank"]),
                    "pairrm_num_candidates": len(candidates),
                },
            }
        )

        if args.log_every > 0 and (example_index % args.log_every == 0 or example_index == total_examples):
            elapsed_seconds = time.perf_counter() - started_at
            average_seconds = elapsed_seconds / example_index
            remaining_examples = total_examples - example_index
            eta_seconds = average_seconds * remaining_examples
            print(
                f"[progress] pairrm processed={example_index}/{total_examples} "
                f"ratio={example_index / total_examples:.2%} elapsed={_format_duration(elapsed_seconds)} "
                f"avg_per_example={average_seconds:.2f}s eta={_format_duration(eta_seconds)} "
                f"last_example_id={example.get('example_id')}",
                flush=True,
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
                "baseline": "pairrm_best_of_n",
                "supervision_type": args.supervision_type,
                "best_of_n": args.best_of_n,
                "total_generation_calls": total_generation_calls,
                "pairrm_model": args.pairrm_model,
                "pairrm_source_max_length": args.source_max_length,
                "pairrm_candidate_max_length": args.candidate_max_length,
                "log_every": args.log_every,
            },
        ),
    )
    print(
        f"[done] pairrm dataset={args.dataset} total_examples={total_examples} "
        f"elapsed={_format_duration(time.perf_counter() - started_at)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
