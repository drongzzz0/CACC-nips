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
        description="Run completion-oriented candidate generation, reranking, and paired analysis as one pipeline."
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--base-candidates", required=True, type=Path)
    parser.add_argument("--motif-tags", required=True, type=Path)
    parser.add_argument("--completion-candidates-output", required=True, type=Path)
    parser.add_argument("--completion-metrics-output", required=True, type=Path)
    parser.add_argument("--completion-prompt-preview-output", type=Path)
    parser.add_argument("--completion-verifier-score-sidecar", type=Path)

    parser.add_argument("--generator-adapter-path", type=Path)
    parser.add_argument("--generator-model-path")
    parser.add_argument("--generator-base-model")
    parser.add_argument("--samples-per-example", default=4, type=int)
    parser.add_argument("--prompt-variants", default="default")
    parser.add_argument("--max-candidates", default=8, type=int)
    parser.add_argument("--max-new-tokens", default=160, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--max-context-candidates", default=3, type=int)
    parser.add_argument("--protect-prefix-candidates", default=1, type=int)
    parser.add_argument("--completion-dedupe-mode", default="numeric_or_text", choices=("text", "numeric_or_text"))
    parser.add_argument("--merge-policy", default="replace_fragments_first", choices=("append_if_room", "replace_fragments_first", "replace_random_nonprefix", "replace_partials_first", "replace_random_fragment_only", "replace_random_partial_only", "replace_invalid_first", "replace_compatibility_risk_first", "replace_hybrid_salvageability", "replace_closure_score_first", "replace_verifier_uncertainty_first", "replace_margin_risk_hybrid", "replace_margin_risk_no_salvage", "replace_margin_salvage_no_risk", "replace_stratified_risk_preserve", "replace_margin_stratified_risk_preserve", "replace_margin_stratified_numeric_preserve"))
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", default=7, type=int)

    parser.add_argument("--first-predictions-output", required=True, type=Path)
    parser.add_argument("--base-predictions-output", required=True, type=Path)
    parser.add_argument("--base-metrics-output", required=True, type=Path)
    parser.add_argument("--base-reranker-model-path")
    parser.add_argument("--base-reranker-adapter-path", type=Path)
    parser.add_argument("--base-reranker-base-model")
    parser.add_argument("--base-reranker-batch-size", default=32, type=int)

    parser.add_argument("--verifier-predictions-output", required=True, type=Path)
    parser.add_argument("--verifier-metrics-output", required=True, type=Path)
    parser.add_argument("--verifier-model-path")
    parser.add_argument("--verifier-adapter-path", type=Path)
    parser.add_argument("--verifier-base-model")
    parser.add_argument("--verifier-batch-size", default=32, type=int)

    parser.add_argument("--analysis-report", required=True, type=Path)
    parser.add_argument("--analysis-summary-json", required=True, type=Path)
    parser.add_argument("--fixed-reference-metrics", type=Path)
    parser.add_argument("--fixed-reference-label", default="fixed candidate-set verifier")
    parser.add_argument("--max-examples-per-bucket", default=2, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_command(command: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(printable)
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python_bin = sys.executable

    completion_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "generate_motif_completion_candidates.py"),
        "--base-candidates",
        str(args.base_candidates),
        "--motif-tags",
        str(args.motif_tags),
        "--output",
        str(args.completion_candidates_output),
        "--metrics-output",
        str(args.completion_metrics_output),
        "--samples-per-example",
        str(args.samples_per_example),
        "--prompt-variants",
        args.prompt_variants,
        "--max-candidates",
        str(args.max_candidates),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-context-candidates",
        str(args.max_context_candidates),
        "--protect-prefix-candidates",
        str(args.protect_prefix_candidates),
        "--dedupe-mode",
        args.completion_dedupe_mode,
        "--merge-policy",
        args.merge_policy,
        "--seed",
        str(args.seed),
    ]
    if args.completion_prompt_preview_output is not None:
        completion_cmd.extend(["--prompt-preview-output", str(args.completion_prompt_preview_output)])
    if args.completion_verifier_score_sidecar is not None:
        completion_cmd.extend(["--verifier-score-sidecar", str(args.completion_verifier_score_sidecar)])
    if args.generator_adapter_path is not None:
        completion_cmd.extend(["--adapter-path", str(args.generator_adapter_path)])
    if args.generator_model_path:
        completion_cmd.extend(["--model-path", args.generator_model_path])
    if args.generator_base_model:
        completion_cmd.extend(["--base-model", args.generator_base_model])
    if args.max_examples is not None:
        completion_cmd.extend(["--max-examples", str(args.max_examples)])

    first_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "make_first_candidate_predictions.py"),
        "--candidates",
        str(args.completion_candidates_output),
        "--output",
        str(args.first_predictions_output),
    ]

    base_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "score_verifier_candidates.py"),
        "--dataset",
        str(args.completion_candidates_output),
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
    if args.base_reranker_batch_size != 32:
        base_cmd.extend(["--batch-size", str(args.base_reranker_batch_size)])

    verifier_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "score_verifier_candidates.py"),
        "--dataset",
        str(args.completion_candidates_output),
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
    if args.verifier_batch_size != 32:
        verifier_cmd.extend(["--batch-size", str(args.verifier_batch_size)])

    analysis_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "analyze_generate_then_rerank.py"),
        "--candidates",
        str(args.completion_candidates_output),
        "--first-predictions",
        str(args.first_predictions_output),
        "--base-predictions",
        str(args.base_predictions_output),
        "--verifier-predictions",
        str(args.verifier_predictions_output),
        "--completion-generation-metrics",
        str(args.completion_metrics_output),
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

    for command in (completion_cmd, first_cmd, base_cmd, verifier_cmd, analysis_cmd):
        _run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
