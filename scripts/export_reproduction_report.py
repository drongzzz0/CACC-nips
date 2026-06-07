from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "reproduction_targets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact CACC Table 1 reproduction-status report from the target manifest."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="Path to configs/reproduction_targets.json.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional markdown report path. If omitted, markdown is printed to stdout.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Optional CSV report path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def final_delta(reference: dict[str, Any] | None, paper: dict[str, Any]) -> float | None:
    if not reference:
        return None
    return float(reference["final"]) - float(paper["final"])


def risk_bucket(row: dict[str, Any]) -> str:
    status = row["release_status"]
    release_delta = final_delta(row.get("release_reference"), row["paper"])
    fallback_delta = final_delta(row.get("fallback_reference"), row["paper"])
    if status in {"artifact_reproduced", "artifact_reproduced_display_ratio"}:
        return "artifact"
    if status.startswith("partial"):
        return "partial"
    if "pending" in status:
        return "pending"
    if release_delta is None and fallback_delta is None:
        return "unknown"

    if fallback_delta is not None and release_delta is not None:
        release_magnitude = abs(release_delta)
        fallback_magnitude = abs(fallback_delta)
        if release_magnitude > 0.05:
            if fallback_magnitude <= 0.02:
                return "fallback_close"
            if fallback_magnitude <= 0.05:
                return "fallback_watch"

    primary_delta = release_delta if release_delta is not None else fallback_delta
    magnitude = abs(primary_delta)
    if magnitude <= 0.02:
        return "close"
    if magnitude <= 0.05:
        return "watch"
    return "large_gap"


def row_to_record(row: dict[str, Any]) -> dict[str, str]:
    paper = row["paper"]
    release_reference = row.get("release_reference")
    fallback_reference = row.get("fallback_reference")
    release_delta = final_delta(release_reference, paper)
    fallback_delta = final_delta(fallback_reference, paper)
    return {
        "row_id": row["row_id"],
        "dataset": row["dataset"],
        "variant": row["variant"],
        "paper_ovf": f"{fmt(paper['oracle'])} / {fmt(paper['v_given_o'])} / {fmt(paper['final'])}",
        "release_reference_ovf": (
            f"{fmt(release_reference['oracle'])} / {fmt(release_reference['v_given_o'])} / {fmt(release_reference['final'])}"
            if release_reference
            else ""
        ),
        "release_final_delta": fmt_delta(release_delta),
        "fallback_reference_ovf": (
            f"{fmt(fallback_reference['oracle'])} / {fmt(fallback_reference['v_given_o'])} / {fmt(fallback_reference['final'])}"
            if fallback_reference
            else ""
        ),
        "fallback_final_delta": fmt_delta(fallback_delta),
        "release_status": row["release_status"],
        "risk_bucket": risk_bucket(row),
        "evidence": row.get("evidence", ""),
        "fallback_evidence": row.get("fallback_evidence", ""),
    }


def build_records(manifest: dict[str, Any]) -> list[dict[str, str]]:
    return [row_to_record(row) for row in manifest.get("rows", [])]


def markdown_report(records: list[dict[str, str]], manifest: dict[str, Any]) -> str:
    lines = [
        "# CACC Table 1 Reproduction Report",
        "",
        f"Source table: {manifest.get('table', 'unknown')}",
        "",
        "| Row | Paper O/V/F | Release ref O/V/F | Release delta | Fallback ref O/V/F | Fallback delta | Bucket | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| {row_id} | {paper} | {release} | {release_delta} | {fallback} | {fallback_delta} | {bucket} | {status} |".format(
                row_id=record["row_id"],
                paper=record["paper_ovf"],
                release=record["release_reference_ovf"] or "-",
                release_delta=record["release_final_delta"] or "-",
                fallback=record["fallback_reference_ovf"] or "-",
                fallback_delta=record["fallback_final_delta"] or "-",
                bucket=record["risk_bucket"],
                status=record["release_status"],
            )
        )
    lines.extend(
        [
            "",
            "Buckets are derived from final accuracy only: artifact, partial, pending, close (`<=0.02`), watch (`<=0.05`), or large_gap.",
            "Rows labeled fallback_close or fallback_watch have a large fresh-rerun gap but a closer documented fallback reference.",
            "Report oracle coverage, verifier efficiency given oracle, and final accuracy together when interpreting any rerun.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "dataset",
        "variant",
        "paper_ovf",
        "release_reference_ovf",
        "release_final_delta",
        "fallback_reference_ovf",
        "fallback_final_delta",
        "release_status",
        "risk_bucket",
        "evidence",
        "fallback_evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    records = build_records(manifest)
    report = markdown_report(records, manifest)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    if args.csv_output:
        write_csv(args.csv_output, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
