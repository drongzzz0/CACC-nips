from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run residual-repair candidate generation, reranking, and paired analysis as one pipeline."
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--base-candidates", required=True, type=Path)
    parser.add_argument("--reranker-predictions", type=Path)
    parser.add_argument("--repair-candidates-output", required=True, type=Path)
    parser.add_argument("--repair-metrics-output", required=True, type=Path)
    parser.add_argument("--repair-prompt-preview-output", type=Path)

    parser.add_argument("--generator-adapter-path", type=Path)
    parser.add_argument("--generator-model-path")
    parser.add_argument("--generator-base-model")
    parser.add_argument("--samples-per-target", default=1, type=int)
    parser.add_argument("--max-repair-targets", default=2, type=int)
    parser.add_argument("--max-candidates", default=8, type=int)
    parser.add_argument("--max-new-tokens", default=160, type=int)
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--protect-prefix-candidates", default=1, type=int)
    parser.add_argument("--repair-dedupe-mode", default="numeric_or_text", choices=("text", "numeric_or_text"))
    parser.add_argument("--strict-hygiene", action="store_true")
    parser.add_argument("--allow-replace-complete-attempt", action="store_true")
    parser.add_argument("--allow-non-numeric-repairs", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", default=7, type=int)

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
    print(printable)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _print_script_provenance(path: Path) -> None:
    resolved = path.resolve()
    print(
        f"[script] path={resolved} md5={hashlib.md5(resolved.read_bytes()).hexdigest()}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    python_bin = sys.executable

    repair_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "repair_motif_candidates.py"),
        "--base-candidates",
        str(args.base_candidates),
        "--output",
        str(args.repair_candidates_output),
        "--metrics-output",
        str(args.repair_metrics_output),
        "--samples-per-target",
        str(args.samples_per_target),
        "--max-repair-targets",
        str(args.max_repair_targets),
        "--max-candidates",
        str(args.max_candidates),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--protect-prefix-candidates",
        str(args.protect_prefix_candidates),
        "--dedupe-mode",
        args.repair_dedupe_mode,
        "--seed",
        str(args.seed),
    ]
    if args.strict_hygiene:
        repair_cmd.append("--strict-hygiene")
    if args.allow_replace_complete_attempt:
        repair_cmd.append("--allow-replace-complete-attempt")
    if args.allow_non_numeric_repairs:
        repair_cmd.append("--allow-non-numeric-repairs")
    if args.reranker_predictions is not None:
        repair_cmd.extend(["--reranker-predictions", str(args.reranker_predictions)])
    if args.repair_prompt_preview_output is not None:
        repair_cmd.extend(["--prompt-preview-output", str(args.repair_prompt_preview_output)])
    if args.generator_adapter_path is not None:
        repair_cmd.extend(["--adapter-path", str(args.generator_adapter_path)])
    if args.generator_model_path:
        repair_cmd.extend(["--model-path", args.generator_model_path])
    if args.generator_base_model:
        repair_cmd.extend(["--base-model", args.generator_base_model])
    if args.max_examples is not None:
        repair_cmd.extend(["--max-examples", str(args.max_examples)])

    first_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "make_first_candidate_predictions.py"),
        "--candidates",
        str(args.repair_candidates_output),
        "--output",
        str(args.first_predictions_output),
    ]

    base_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "score_verifier_candidates.py"),
        "--dataset",
        str(args.repair_candidates_output),
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
        str(args.repair_candidates_output),
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
        str(args.repair_candidates_output),
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

    _print_script_provenance(SCRIPTS_DIR / "repair_motif_candidates.py")
    _print_script_provenance(SCRIPTS_DIR / "analyze_generate_then_rerank.py")

    for command in (repair_cmd, first_cmd, base_cmd, verifier_cmd, analysis_cmd):
        _run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
