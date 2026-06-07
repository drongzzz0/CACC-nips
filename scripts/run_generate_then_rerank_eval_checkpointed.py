from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run checkpointed candidate generation, reranking, and paired analysis as one pipeline."
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidate-output", required=True, type=Path)
    parser.add_argument("--candidate-metrics-output", required=True, type=Path)
    parser.add_argument("--candidate-progress-output", type=Path)

    parser.add_argument("--generator-adapter-path", type=Path)
    parser.add_argument("--generator-model-path")
    parser.add_argument("--generator-base-model")
    parser.add_argument("--supervision-type", default="filtered_cot")
    parser.add_argument("--samples-per-example", default=16, type=int)
    parser.add_argument("--generation-batch-size", default=None, type=int)
    parser.add_argument("--max-candidates", default=8, type=int)
    parser.add_argument("--max-new-tokens", default=128, type=int)
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

    parser.add_argument("--first-predictions-output", required=True, type=Path)
    parser.add_argument("--base-predictions-output", required=True, type=Path)
    parser.add_argument("--base-metrics-output", required=True, type=Path)
    parser.add_argument("--base-reranker-model-path")
    parser.add_argument("--base-reranker-adapter-path", type=Path)
    parser.add_argument("--base-reranker-base-model")

    parser.add_argument("--verifier-predictions-output", required=True, type=Path)
    parser.add_argument("--verifier-metrics-output", required=True, type=Path)
    parser.add_argument("--verifier-model-path")
    parser.add_argument("--verifier-adapter-path", type=Path)
    parser.add_argument("--verifier-base-model")

    parser.add_argument("--analysis-report", required=True, type=Path)
    parser.add_argument("--analysis-summary-json", required=True, type=Path)
    parser.add_argument("--fixed-reference-metrics", type=Path)
    parser.add_argument("--fixed-reference-label", default="fixed candidate-set verifier")
    parser.add_argument("--max-examples-per-bucket", default=2, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(printable, flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python_bin = sys.executable

    progress_output = args.candidate_progress_output
    if progress_output is None:
        progress_output = args.candidate_output.with_suffix(".progress.json")

    generation_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "generate_candidate_sets_chunked_checkpointed.py"),
        "--dataset",
        str(args.dataset),
        "--output",
        str(args.candidate_output),
        "--metrics-output",
        str(args.candidate_metrics_output),
        "--progress-output",
        str(progress_output),
        "--supervision-type",
        args.supervision_type,
        "--samples-per-example",
        str(args.samples_per_example),
        "--max-candidates",
        str(args.max_candidates),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--dedupe-mode",
        args.dedupe_mode,
        "--selection-strategy",
        args.selection_strategy,
        "--text-prefix-candidates",
        str(args.text_prefix_candidates),
        "--seed",
        str(args.seed),
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    if args.resume:
        generation_cmd.append("--resume")
    if args.generation_batch_size is not None:
        generation_cmd.extend(["--generation-batch-size", str(args.generation_batch_size)])
    if args.generator_adapter_path is not None:
        generation_cmd.extend(["--adapter-path", str(args.generator_adapter_path)])
    if args.generator_model_path:
        generation_cmd.extend(["--model-path", args.generator_model_path])
    if args.generator_base_model:
        generation_cmd.extend(["--base-model", args.generator_base_model])

    first_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "make_first_candidate_predictions.py"),
        "--candidates",
        str(args.candidate_output),
        "--output",
        str(args.first_predictions_output),
    ]

    base_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "score_verifier_candidates.py"),
        "--dataset",
        str(args.candidate_output),
        "--predictions",
        str(args.base_predictions_output),
        "--metrics-output",
        str(args.base_metrics_output),
    ]
    if args.base_reranker_adapter_path is not None:
        base_cmd.extend(["--adapter-path", str(args.base_reranker_adapter_path)])
    if args.base_reranker_model_path:
        base_cmd.extend(["--model-path", args.base_reranker_model_path])
    if args.base_reranker_base_model:
        base_cmd.extend(["--base-model", args.base_reranker_base_model])

    verifier_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "score_verifier_candidates.py"),
        "--dataset",
        str(args.candidate_output),
        "--predictions",
        str(args.verifier_predictions_output),
        "--metrics-output",
        str(args.verifier_metrics_output),
    ]
    if args.verifier_adapter_path is not None:
        verifier_cmd.extend(["--adapter-path", str(args.verifier_adapter_path)])
    if args.verifier_model_path:
        verifier_cmd.extend(["--model-path", args.verifier_model_path])
    if args.verifier_base_model:
        verifier_cmd.extend(["--base-model", args.verifier_base_model])

    analysis_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "analyze_generate_then_rerank.py"),
        "--candidates",
        str(args.candidate_output),
        "--first-predictions",
        str(args.first_predictions_output),
        "--base-predictions",
        str(args.base_predictions_output),
        "--verifier-predictions",
        str(args.verifier_predictions_output),
        "--report",
        str(args.analysis_report),
        "--summary-json",
        str(args.analysis_summary_json),
        "--run-label",
        args.run_label,
        "--fixed-reference-label",
        args.fixed_reference_label,
        "--max-examples-per-bucket",
        str(args.max_examples_per_bucket),
    ]
    if args.fixed_reference_metrics is not None:
        analysis_cmd.extend(["--fixed-reference-metrics", str(args.fixed_reference_metrics)])

    for command in (generation_cmd, first_cmd, base_cmd, verifier_cmd, analysis_cmd):
        _run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
