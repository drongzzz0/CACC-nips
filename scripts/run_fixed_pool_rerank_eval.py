from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reranking and paired analysis on an existing candidate pool."
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--candidates", required=True, type=Path)

    parser.add_argument("--first-predictions-output", required=True, type=Path)
    parser.add_argument("--base-predictions-output", required=True, type=Path)
    parser.add_argument("--base-metrics-output", required=True, type=Path)
    parser.add_argument("--base-reranker-model-path")
    parser.add_argument("--base-reranker-adapter-path", type=Path)
    parser.add_argument("--base-reranker-base-model")
    parser.add_argument("--base-batch-size", default=32, type=int)
    parser.add_argument("--base-gpus", default="")

    parser.add_argument("--verifier-predictions-output", required=True, type=Path)
    parser.add_argument("--verifier-metrics-output", required=True, type=Path)
    parser.add_argument("--verifier-model-path")
    parser.add_argument("--verifier-adapter-path", type=Path)
    parser.add_argument("--verifier-base-model")
    parser.add_argument("--verifier-batch-size", default=32, type=int)
    parser.add_argument("--verifier-gpus", default="")

    parser.add_argument("--analysis-report", required=True, type=Path)
    parser.add_argument("--analysis-summary-json", required=True, type=Path)
    parser.add_argument("--fixed-reference-metrics", type=Path)
    parser.add_argument("--fixed-reference-label", default="fixed candidate-set verifier")
    parser.add_argument("--max-examples-per-bucket", default=2, type=int)
    parser.add_argument("--cuda-alloc-conf", default="expandable_segments:True")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_gpu_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _format_command(command: list[str], env_overrides: dict[str, str] | None = None) -> str:
    prefix = ""
    if env_overrides:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())) + " "
    return prefix + " ".join(shlex.quote(part) for part in command)


def _run_command(command: list[str], dry_run: bool, env_overrides: dict[str, str] | None = None) -> None:
    print(_format_command(command, env_overrides), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(command, check=True, env=env)


def _run_parallel_commands(commands: list[tuple[list[str], dict[str, str] | None]], dry_run: bool) -> None:
    for command, env_overrides in commands:
        print(_format_command(command, env_overrides), flush=True)
    if dry_run:
        return

    processes: list[tuple[subprocess.Popen, list[str]]] = []
    try:
        for command, env_overrides in commands:
            env = os.environ.copy()
            if env_overrides:
                env.update(env_overrides)
            process = subprocess.Popen(command, env=env)
            processes.append((process, command))

        failures: list[tuple[int, list[str]]] = []
        for process, command in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((return_code, command))

        if failures:
            return_code, command = failures[0]
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _ in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def _print_script_provenance(path: Path) -> None:
    resolved = path.resolve()
    print(
        f"[script] path={resolved} md5={hashlib.md5(resolved.read_bytes()).hexdigest()}",
        flush=True,
    )


def _derive_shard_path(path: Path, shard_index: int, num_shards: int) -> Path:
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}.shard{shard_index}of{num_shards}{suffix}")


def _build_score_command(
    *,
    python_bin: str,
    dataset: Path,
    predictions_output: Path,
    metrics_output: Path,
    batch_size: int,
    model_path: str | None,
    adapter_path: Path | None,
    base_model: str | None,
    num_shards: int | None = None,
    shard_index: int | None = None,
) -> list[str]:
    command = [
        python_bin,
        str(SCRIPTS_DIR / "score_verifier_candidates.py"),
        "--dataset",
        str(dataset),
        "--predictions",
        str(predictions_output),
        "--metrics-output",
        str(metrics_output),
        "--batch-size",
        str(batch_size),
    ]
    if adapter_path is not None:
        command.extend(["--adapter-path", str(adapter_path)])
    if model_path:
        command.extend(["--model-path", model_path])
    if base_model:
        command.extend(["--base-model", base_model])
    if num_shards is not None and shard_index is not None:
        command.extend(["--num-shards", str(num_shards), "--shard-index", str(shard_index)])
    return command


def _merge_prediction_shards(
    *,
    dataset: Path,
    shard_prediction_paths: list[Path],
    shard_metrics_paths: list[Path],
    output_predictions: Path,
    output_metrics: Path,
    stage_name: str,
    batch_size: int,
    gpu_ids: list[str],
) -> None:
    dataset_rows = list(read_jsonl(dataset))
    example_ids = [str(row["example_id"]) for row in dataset_rows]
    prediction_lookup: dict[str, dict] = {}
    for path in shard_prediction_paths:
        for row in read_jsonl(path):
            example_id = str(row["example_id"])
            if example_id in prediction_lookup:
                raise ValueError(f"Duplicate prediction for example_id={example_id} while merging {stage_name} shards.")
            prediction_lookup[example_id] = row

    missing = [example_id for example_id in example_ids if example_id not in prediction_lookup]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Missing {len(missing)} predictions while merging {stage_name} shards: {preview}")

    merged_rows = [prediction_lookup[example_id] for example_id in example_ids]
    write_jsonl(output_predictions, merged_rows)

    shard_metrics = [json.loads(path.read_text(encoding="utf-8")) for path in shard_metrics_paths if path.exists()]
    correct = sum(int(bool(row.get("correct"))) for row in merged_rows)
    total = len(merged_rows)
    payload = {
        "dataset_path": str(dataset),
        "stage": stage_name,
        "batch_size": batch_size,
        "num_examples": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "num_shards": len(shard_prediction_paths),
        "gpu_ids": gpu_ids,
        "shard_prediction_paths": [str(path) for path in shard_prediction_paths],
        "shard_metrics_paths": [str(path) for path in shard_metrics_paths],
    }
    if shard_metrics:
        payload.update(
            {
                "adapter_path": shard_metrics[0].get("adapter_path"),
                "base_model": shard_metrics[0].get("base_model"),
                "total_batches": sum(int(metric.get("total_batches", 0)) for metric in shard_metrics),
                "total_scoring_seconds_sum": round(sum(float(metric.get("total_scoring_seconds", 0.0)) for metric in shard_metrics), 6),
                "parallel_wall_clock_seconds_estimate": round(max(float(metric.get("total_scoring_seconds", 0.0)) for metric in shard_metrics), 6),
                "shards": shard_metrics,
            }
        )
    write_json(output_metrics, payload)


def _score_stage(
    *,
    stage_name: str,
    python_bin: str,
    dataset: Path,
    predictions_output: Path,
    metrics_output: Path,
    batch_size: int,
    model_path: str | None,
    adapter_path: Path | None,
    base_model: str | None,
    gpu_ids: list[str],
    cuda_alloc_conf: str,
    dry_run: bool,
) -> None:
    env_overrides = {"PYTORCH_CUDA_ALLOC_CONF": cuda_alloc_conf} if cuda_alloc_conf else None
    if len(gpu_ids) <= 1:
        if gpu_ids:
            env_overrides = dict(env_overrides or {})
            env_overrides["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        command = _build_score_command(
            python_bin=python_bin,
            dataset=dataset,
            predictions_output=predictions_output,
            metrics_output=metrics_output,
            batch_size=batch_size,
            model_path=model_path,
            adapter_path=adapter_path,
            base_model=base_model,
        )
        _run_command(command, dry_run=dry_run, env_overrides=env_overrides)
        return

    num_shards = len(gpu_ids)
    shard_prediction_paths = [_derive_shard_path(predictions_output, shard_index, num_shards) for shard_index in range(num_shards)]
    shard_metrics_paths = [_derive_shard_path(metrics_output, shard_index, num_shards) for shard_index in range(num_shards)]
    commands: list[tuple[list[str], dict[str, str]]] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        shard_env = {"CUDA_VISIBLE_DEVICES": gpu_id}
        if cuda_alloc_conf:
            shard_env["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
        command = _build_score_command(
            python_bin=python_bin,
            dataset=dataset,
            predictions_output=shard_prediction_paths[shard_index],
            metrics_output=shard_metrics_paths[shard_index],
            batch_size=batch_size,
            model_path=model_path,
            adapter_path=adapter_path,
            base_model=base_model,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        commands.append((command, shard_env))

    print(f"[stage] {stage_name}: launching {num_shards} shards across GPUs {','.join(gpu_ids)}", flush=True)
    _run_parallel_commands(commands, dry_run=dry_run)
    if dry_run:
        return
    _merge_prediction_shards(
        dataset=dataset,
        shard_prediction_paths=shard_prediction_paths,
        shard_metrics_paths=shard_metrics_paths,
        output_predictions=predictions_output,
        output_metrics=metrics_output,
        stage_name=stage_name,
        batch_size=batch_size,
        gpu_ids=gpu_ids,
    )


def main() -> None:
    args = parse_args()
    python_bin = sys.executable
    base_gpus = _parse_gpu_list(args.base_gpus)
    verifier_gpus = _parse_gpu_list(args.verifier_gpus) or base_gpus

    first_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "make_first_candidate_predictions.py"),
        "--candidates",
        str(args.candidates),
        "--output",
        str(args.first_predictions_output),
    ]

    analysis_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "analyze_generate_then_rerank.py"),
        "--candidates",
        str(args.candidates),
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

    _print_script_provenance(SCRIPTS_DIR / "make_first_candidate_predictions.py")
    _print_script_provenance(SCRIPTS_DIR / "score_verifier_candidates.py")
    _print_script_provenance(SCRIPTS_DIR / "analyze_generate_then_rerank.py")

    _run_command(first_cmd, dry_run=args.dry_run)
    _score_stage(
        stage_name="base rerank",
        python_bin=python_bin,
        dataset=args.candidates,
        predictions_output=args.base_predictions_output,
        metrics_output=args.base_metrics_output,
        batch_size=args.base_batch_size,
        model_path=args.base_reranker_model_path,
        adapter_path=args.base_reranker_adapter_path,
        base_model=args.base_reranker_base_model,
        gpu_ids=base_gpus,
        cuda_alloc_conf=args.cuda_alloc_conf,
        dry_run=args.dry_run,
    )
    _score_stage(
        stage_name="verifier rerank",
        python_bin=python_bin,
        dataset=args.candidates,
        predictions_output=args.verifier_predictions_output,
        metrics_output=args.verifier_metrics_output,
        batch_size=args.verifier_batch_size,
        model_path=args.verifier_model_path,
        adapter_path=args.verifier_adapter_path,
        base_model=args.verifier_base_model,
        gpu_ids=verifier_gpus,
        cuda_alloc_conf=args.cuda_alloc_conf,
        dry_run=args.dry_run,
    )
    _run_command(analysis_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
