from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark runs into JSON and Markdown artifacts.")
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--logs-dir", default=ROOT / "logs", type=Path)
    parser.add_argument("--results-dir", default=ROOT.parents[1] / "Experiment/analysis/results", type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_accuracy(report_path: Path) -> float:
    text = report_path.read_text(encoding="utf-8")
    match = re.search(r"exact-match accuracy:\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"Could not parse accuracy from {report_path}")
    return float(match.group(1))


def _variant_label(run_name: str) -> str:
    for candidate in ("answer_only", "filtered_cot", "subgoals"):
        if candidate in run_name:
            return candidate
    return run_name


def main() -> None:
    args = parse_args()
    rows = []
    for run_name in args.run_names:
        train_manifest = _load_json(args.logs_dir / f"train_{run_name}.json")
        generation_metrics_path = args.logs_dir / f"{run_name}_generation_metrics.json"
        generation_metrics = _load_json(generation_metrics_path) if generation_metrics_path.exists() else {}
        eval_report = args.results_dir / f"{run_name}_eval.md"
        accuracy = _read_accuracy(eval_report)
        train_metrics = train_manifest.get("metrics") or {}
        row = {
            "run_name": run_name,
            "variant": _variant_label(run_name),
            "train_examples": train_manifest["num_examples"],
            "accuracy": accuracy,
            "train_runtime_seconds": train_metrics.get("train_runtime"),
            "train_samples_per_second": train_metrics.get("train_samples_per_second"),
            "train_steps_per_second": train_metrics.get("train_steps_per_second"),
            "train_loss": train_metrics.get("train_loss"),
            "generation_total_seconds": generation_metrics.get("total_generation_seconds"),
            "generation_avg_seconds": generation_metrics.get("avg_generation_seconds"),
            "generation_examples_per_second": generation_metrics.get("examples_per_second"),
            "generation_avg_tokens": generation_metrics.get("avg_generated_tokens"),
            "generation_tokens_per_second": generation_metrics.get("generated_tokens_per_second"),
        }
        if row["train_runtime_seconds"] and row["accuracy"] is not None:
            row["accuracy_per_train_second"] = row["accuracy"] / row["train_runtime_seconds"]
        else:
            row["accuracy_per_train_second"] = None
        rows.append(row)

    rows.sort(key=lambda row: (-row["accuracy"], row["generation_avg_seconds"] or float("inf")))
    write_json(args.summary_json, {"runs": rows})

    lines = [
        "# Benchmark Summary",
        "",
        "| variant | accuracy | train sec | train ex/s | gen sec/ex | gen tok/ex | gen tok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {accuracy:.4f} | {train_runtime} | {train_eps} | {gen_sec} | {gen_tok} | {gen_tps} |".format(
                variant=row["variant"],
                accuracy=row["accuracy"],
                train_runtime=_fmt(row["train_runtime_seconds"]),
                train_eps=_fmt(row["train_samples_per_second"]),
                gen_sec=_fmt(row["generation_avg_seconds"]),
                gen_tok=_fmt(row["generation_avg_tokens"]),
                gen_tps=_fmt(row["generation_tokens_per_second"]),
            )
        )

    best_accuracy = max(rows, key=lambda row: row["accuracy"]) if rows else None
    fastest_generation = min(
        (row for row in rows if row["generation_avg_seconds"] is not None),
        key=lambda row: row["generation_avg_seconds"],
        default=None,
    )
    lines.extend(["", "## Key Findings", ""])
    if best_accuracy is not None:
        best_accuracy_rows = [row for row in rows if row["accuracy"] == best_accuracy["accuracy"]]
        if len(best_accuracy_rows) == len(rows):
            lines.append(
                f"- Accuracy tie: all variants achieved `{best_accuracy['accuracy']:.4f}` exact-match."
            )
        else:
            lines.append(
                f"- Highest accuracy: `{best_accuracy['variant']}` at `{best_accuracy['accuracy']:.4f}` exact-match."
            )
    if fastest_generation is not None:
        lines.append(
            f"- Fastest generation: `{fastest_generation['variant']}` at `{fastest_generation['generation_avg_seconds']:.4f}` seconds/example."
        )
    if len(rows) >= 2:
        if len(best_accuracy_rows) == len(rows):
            lines.append(
                "- Accuracy-efficiency tradeoff: no meaningful frontier emerged because accuracy was identical across all variants."
            )
        else:
            baseline = min(rows, key=lambda row: row["generation_avg_seconds"] or float("inf"))
            strongest = max(rows, key=lambda row: row["accuracy"])
            lines.append(
                f"- Accuracy-efficiency tradeoff: `{strongest['variant']}` is the strongest accuracy run, while `{baseline['variant']}` is the fastest by generation latency."
            )

    write_text(args.summary_md, "\n".join(lines) + "\n")


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()
