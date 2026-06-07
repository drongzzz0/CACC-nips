from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Mapping
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import VerifierCandidateSet
from src.eval.evaluate_predictions import answers_match
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PairRM reranking on an existing fixed candidate pool.")
    parser.add_argument("--candidate-pool", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--ranked-candidates-output", type=Path)
    parser.add_argument("--pairrm-model", default="llm-blender/PairRM-hf")
    parser.add_argument("--source-max-length", default=1224, type=int)
    parser.add_argument("--candidate-max-length", default=412, type=int)
    parser.add_argument("--log-every", default=25, type=int)
    parser.add_argument("--max-examples", type=int)
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


def _candidate_to_text(candidate: object) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, Mapping):
        for key in ("candidate", "candidate_text", "text", "prediction", "response", "answer"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if "content" in candidate:
            return str(candidate["content"])
    return str(candidate)


def _dedupe_preserve_order(candidates: list[object]) -> list[str]:
    seen = set()
    deduped = []
    for raw_candidate in candidates:
        candidate = _candidate_to_text(raw_candidate).strip()
        normalized = " ".join(candidate.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.candidate_pool))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    total_examples = len(rows)
    pairrm_model, pairrm_tokenizer, pairrm_device = _load_pairrm_model(args.pairrm_model)

    prediction_rows = []
    ranked_candidate_rows = []
    correct = 0
    total_pairwise_comparisons = 0
    empty_candidate_examples = 0
    total_candidates_after_dedupe = 0
    started_at = time.perf_counter()

    print(
        f"[start] pairrm-fixed-pool pool={args.candidate_pool} total_examples={total_examples} log_every={args.log_every}",
        flush=True,
    )

    for example_index, example in enumerate(rows, start=1):
        candidates = _dedupe_preserve_order(list(example.get("candidates", [])))
        if not candidates:
            candidates = [""]
            empty_candidate_examples += 1
        total_candidates_after_dedupe += len(candidates)
        total_pairwise_comparisons += len(candidates) * max(0, len(candidates) - 1) // 2

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
        answer_mode = str(example.get("answer_mode", "numeric"))
        is_correct = answers_match(best_candidate, str(example.get("gold_answer", "")), answer_mode=answer_mode)
        correct += int(is_correct)

        metadata = {**dict(example.get("metadata", {})), "pairrm_best_rank": int(ranked[0]["rank"]), "pairrm_num_candidates": len(candidates)}
        prediction_rows.append(
            {
                "example_id": str(example["example_id"]),
                "prediction": best_candidate,
                "gold_answer": str(example.get("gold_answer", "")),
                "answer_mode": answer_mode,
                "dataset": str(example.get("dataset", "unknown")),
                "problem": str(example.get("problem", example.get("prompt", ""))),
                "choices": [str(choice) for choice in example.get("choices", [])],
                "metadata": metadata,
            }
        )

        if args.ranked_candidates_output is not None:
            ranked_candidate_rows.append(
                VerifierCandidateSet(
                    example_id=str(example["example_id"]),
                    dataset=str(example.get("dataset", "unknown")),
                    problem=str(example.get("problem", example.get("prompt", ""))),
                    gold_answer=str(example.get("gold_answer", "")),
                    candidates=candidates,
                    answer_mode=answer_mode,
                    choices=[str(choice) for choice in example.get("choices", [])],
                    metadata={**dict(example.get("metadata", {})), "pairrm_ranking": ranked},
                ).to_dict()
            )

        if args.log_every > 0 and (example_index % args.log_every == 0 or example_index == total_examples):
            elapsed_seconds = time.perf_counter() - started_at
            average_seconds = elapsed_seconds / example_index
            remaining_examples = total_examples - example_index
            eta_seconds = average_seconds * remaining_examples
            print(
                f"[progress] pairrm-fixed-pool processed={example_index}/{total_examples} ratio={example_index / total_examples:.2%} elapsed={_format_duration(elapsed_seconds)} avg_per_example={average_seconds:.2f}s eta={_format_duration(eta_seconds)} last_example_id={example.get('example_id')}",
                flush=True,
            )

    write_jsonl(args.predictions, prediction_rows)
    if args.ranked_candidates_output is not None:
        write_jsonl(args.ranked_candidates_output, ranked_candidate_rows)
    if args.metrics_output is not None:
        elapsed = time.perf_counter() - started_at
        write_json(
            args.metrics_output,
            {
                "candidate_pool": str(args.candidate_pool),
                "pairrm_model": args.pairrm_model,
                "num_examples": total_examples,
                "correct": correct,
                "accuracy": (correct / total_examples) if total_examples else 0.0,
                "empty_candidate_examples": empty_candidate_examples,
                "total_candidates_after_dedupe": total_candidates_after_dedupe,
                "avg_candidates_after_dedupe": (total_candidates_after_dedupe / total_examples) if total_examples else 0.0,
                "total_pairwise_comparisons": total_pairwise_comparisons,
                "source_max_length": args.source_max_length,
                "candidate_max_length": args.candidate_max_length,
                "total_runtime_seconds": round(elapsed, 6),
                "avg_runtime_seconds": round(elapsed / total_examples, 6) if total_examples else 0.0,
                "selector": "pairrm_fixed_pool",
            },
        )
    print(
        f"[done] pairrm-fixed-pool pool={args.candidate_pool} total_examples={total_examples} accuracy={(correct / total_examples) if total_examples else 0.0:.4f} elapsed={_format_duration(time.perf_counter() - started_at)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
