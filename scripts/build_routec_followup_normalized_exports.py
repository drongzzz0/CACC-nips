from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import extract_choice_answer, extract_numeric_answer
from src.utils.io_utils import read_jsonl, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized follow-up exports for Route C paper-facing backfill.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Experiment/analysis/results/routec_followup_2026_04_27/normalized"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar_vgo(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        verifier_value = value.get("verifier")
        if isinstance(verifier_value, (int, float)):
            return float(verifier_value)
    return None


def _selected_parseable_rate(predictions_path: Path, answer_mode: str) -> tuple[float | None, float | None, int]:
    rows = list(read_jsonl(predictions_path))
    if not rows:
        return None, None, 0

    parseable = 0
    invalid_final = 0
    for row in rows:
        prediction = str(row.get("prediction", ""))
        if answer_mode == "choice_letter":
            parsed = extract_choice_answer(prediction)
        else:
            parsed = extract_numeric_answer(prediction)
        if parsed is not None:
            parseable += 1
        else:
            invalid_final += 1
    total = len(rows)
    return parseable / total, invalid_final / total, total


def _build_gpqa_rows() -> list[dict[str, object]]:
    row_specs = [
        {
            "paper_variant": "Diverse-Base",
            "variant_name": "routec_p2_gpqa_diamond_diverse_base_v1",
            "summary_json": Path("Experiment/analysis/results/routec_p2_gpqa_diamond_diverse_base_v1.json"),
            "verifier_predictions": Path("Experiment/core_code/logs/routec_p2_gpqa_diamond_diverse_base_v1_verifier_predictions.jsonl"),
            "role": "supporting_gpqa_anchor",
        },
        {
            "paper_variant": "CACC",
            "variant_name": "routec_p2_gpqa_diamond_cacc_v1",
            "summary_json": Path("Experiment/analysis/results/routec_p2_gpqa_diamond_cacc_v1.json"),
            "verifier_predictions": Path("Experiment/core_code/logs/routec_p2_gpqa_diamond_cacc_v1_verifier_predictions.jsonl"),
            "role": "supporting_gpqa_core",
        },
        {
            "paper_variant": "CACC+SP",
            "variant_name": "routec_p2_gpqa_diamond_cacc_sp_v1",
            "summary_json": Path("Experiment/analysis/results/routec_p2_gpqa_diamond_cacc_sp_v1.json"),
            "verifier_predictions": Path("Experiment/core_code/logs/routec_p2_gpqa_diamond_cacc_sp_v1_verifier_predictions.jsonl"),
            "role": "supporting_gpqa_strengthened",
        },
    ]

    rows: list[dict[str, object]] = []
    for spec in row_specs:
        summary = _load_json(spec["summary_json"])
        selected_parseable, invalid_final, prediction_count = _selected_parseable_rate(
            spec["verifier_predictions"],
            answer_mode="choice_letter",
        )
        oracle_accuracy = summary.get("oracle_coverage")
        verifier_accuracy = summary.get("verifier_accuracy")
        vgo = _scalar_vgo(summary.get("selection_efficiency_given_oracle"))
        if vgo is None and oracle_accuracy not in (None, 0) and verifier_accuracy is not None:
            vgo = float(verifier_accuracy) / float(oracle_accuracy)
        rows.append(
            {
                "benchmark": "gpqa_diamond",
                "split": "train_198",
                "variant_name": spec["variant_name"],
                "paper_facing_variant_name": spec["paper_variant"],
                "role": spec["role"],
                "first_accuracy": summary.get("first_accuracy"),
                "oracle_accuracy": oracle_accuracy,
                "verifier_accuracy": verifier_accuracy,
                "verifier_given_oracle_accuracy": vgo,
                "selected_parseable_rate": selected_parseable,
                "invalid_final_answer_rate": invalid_final,
                "num_examples": summary.get("total_examples", prediction_count),
                "summary_json": str(spec["summary_json"]),
                "verifier_predictions": str(spec["verifier_predictions"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _build_gpqa_rows()

    csv_path = args.output_dir / "offline_metric_backfill.csv"
    json_path = args.output_dir / "offline_metric_backfill.json"
    md_path = args.output_dir / "offline_metric_backfill.md"

    fieldnames = [
        "benchmark",
        "split",
        "variant_name",
        "paper_facing_variant_name",
        "role",
        "first_accuracy",
        "oracle_accuracy",
        "verifier_accuracy",
        "verifier_given_oracle_accuracy",
        "selected_parseable_rate",
        "invalid_final_answer_rate",
        "num_examples",
        "summary_json",
        "verifier_predictions",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(json_path, {"rows": rows})

    md_lines = [
        "# Route C Follow-up Offline Metric Backfill",
        "",
        "| Variant | First | Oracle | Final | V|Oracle | Parseable | Invalid | N |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {paper_facing_variant_name} | {first_accuracy:.4f} | {oracle_accuracy:.4f} | {verifier_accuracy:.4f} | {verifier_given_oracle_accuracy:.4f} | {selected_parseable_rate:.4f} | {invalid_final_answer_rate:.4f} | {num_examples} |".format(**row)
        )
    md_lines.extend(
        [
            "",
            "- Scope: `GPQA canonical export backfill`.",
            f"- CSV: `{csv_path}`",
            f"- JSON: `{json_path}`",
        ]
    )
    write_text(md_path, "\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()
