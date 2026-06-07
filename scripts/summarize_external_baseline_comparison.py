from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize external baseline comparisons against internal references.")
    parser.add_argument("--external-spec", action="append", default=[], help="label::report_path::metrics_path::protocol")
    parser.add_argument("--reference-spec", action="append", default=[], help="label::summary_json_path::accuracy_field::protocol")
    parser.add_argument("--first-reference", required=True, help="summary_json_path::accuracy_field")
    parser.add_argument("--main-reference", required=True, help="summary_json_path::accuracy_field")
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    return parser.parse_args()


def _read_accuracy_from_report(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"exact-match accuracy:\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"Could not parse exact-match accuracy from {path}")
    return float(match.group(1))


def _parse_external_spec(spec: str) -> dict:
    parts = spec.split("::")
    if len(parts) != 4:
        raise ValueError(f"Invalid external spec: {spec}")
    label, report_path, metrics_path, protocol = parts
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    accuracy = _read_accuracy_from_report(Path(report_path))
    return {
        "label": label,
        "kind": "external",
        "protocol": protocol,
        "accuracy": accuracy,
        "runtime_seconds": metrics.get("total_runtime_seconds"),
        "num_examples": metrics.get("num_examples"),
        "metrics_path": metrics_path,
        "report_path": report_path,
    }


def _parse_reference_spec(spec: str) -> dict:
    parts = spec.split("::")
    if len(parts) != 4:
        raise ValueError(f"Invalid reference spec: {spec}")
    label, json_path, accuracy_field, protocol = parts
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return {
        "label": label,
        "kind": "reference",
        "protocol": protocol,
        "accuracy": float(payload[accuracy_field]),
        "runtime_seconds": None,
        "num_examples": payload.get("total_examples") or payload.get("num_examples"),
        "summary_json_path": json_path,
        "accuracy_field": accuracy_field,
    }


def _parse_reference_value(spec: str) -> float:
    parts = spec.split("::")
    if len(parts) != 2:
        raise ValueError(f"Invalid reference value spec: {spec}")
    json_path, accuracy_field = parts
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return float(payload[accuracy_field])


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    rows = [_parse_reference_spec(spec) for spec in args.reference_spec]
    rows.extend(_parse_external_spec(spec) for spec in args.external_spec)

    internal_first_accuracy = _parse_reference_value(args.first_reference)
    internal_main_accuracy = _parse_reference_value(args.main_reference)

    for row in rows:
        row["delta_vs_internal_first"] = row["accuracy"] - internal_first_accuracy
        row["delta_vs_internal_main"] = row["accuracy"] - internal_main_accuracy

    rows.sort(key=lambda row: (-row["accuracy"], row["label"]))
    summary = {
        "internal_first_accuracy": internal_first_accuracy,
        "internal_main_accuracy": internal_main_accuracy,
        "rows": rows,
    }
    write_json(args.summary_json, summary)

    lines = [
        "# External Baseline Comparison",
        "",
        f"- Internal first-answer reference accuracy: `{internal_first_accuracy:.4f}`",
        f"- Internal main-branch reference accuracy: `{internal_main_accuracy:.4f}`",
        "",
        "| method | kind | protocol | accuracy | delta vs first | delta vs main | runtime sec | examples |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {kind} | {protocol} | {accuracy} | {delta_first} | {delta_main} | {runtime} | {examples} |".format(
                label=row["label"],
                kind=row["kind"],
                protocol=row["protocol"],
                accuracy=_fmt(row["accuracy"]),
                delta_first=_fmt(row["delta_vs_internal_first"]),
                delta_main=_fmt(row["delta_vs_internal_main"]),
                runtime=_fmt(row["runtime_seconds"]),
                examples=_fmt(row["num_examples"]),
            )
        )

    best_external = max((row for row in rows if row["kind"] == "external"), key=lambda row: row["accuracy"], default=None)
    lines.extend(["", "## Notes", ""])
    if best_external is not None:
        lines.append(
            f"- Best external baseline so far: `{best_external['label']}` at `{best_external['accuracy']:.4f}` exact-match."
        )
        lines.append(
            f"- Gap to internal main branch: `{best_external['delta_vs_internal_main']:.4f}`."
        )
    else:
        lines.append("- No external baseline rows were provided.")

    write_text(args.summary_md, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
