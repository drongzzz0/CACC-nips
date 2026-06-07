from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "reproduction_targets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize release readiness from the CACC Table 1 reproduction manifest."
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
        help="Optional markdown output path. If omitted, markdown is printed to stdout.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional machine-readable output path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero for pending rows or caveats, not only large-gap blockers.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def severity_for_bucket(bucket: str) -> str:
    if bucket in {"large_gap", "unknown"}:
        return "blocker"
    if bucket == "pending":
        return "pending"
    if bucket in {"partial", "watch", "fallback_watch", "fallback_close"}:
        return "caveat"
    return "ok"


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def build_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in manifest.get("rows", []):
        bucket = risk_bucket(row)
        release_delta = final_delta(row.get("release_reference"), row["paper"])
        fallback_delta = final_delta(row.get("fallback_reference"), row["paper"])
        records.append(
            {
                "row_id": row["row_id"],
                "dataset": row["dataset"],
                "variant": row["variant"],
                "release_status": row["release_status"],
                "risk_bucket": bucket,
                "severity": severity_for_bucket(bucket),
                "release_final_delta": release_delta,
                "fallback_final_delta": fallback_delta,
                "evidence": row.get("evidence", ""),
                "fallback_evidence": row.get("fallback_evidence", ""),
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"ok": 0, "caveat": 0, "pending": 0, "blocker": 0}
    for record in records:
        counts[record["severity"]] += 1
    return {
        "total_rows": len(records),
        "counts": counts,
        "practical_ready": counts["blocker"] == 0,
        "strict_ready": counts["blocker"] == counts["pending"] == counts["caveat"] == 0,
    }


def markdown_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# CACC Release Readiness",
        "",
        "Practical readiness checks for large final-accuracy gaps in the Table 1 release manifest.",
        "",
        f"- Practical ready: `{str(summary['practical_ready']).lower()}`",
        f"- Strict ready: `{str(summary['strict_ready']).lower()}`",
        f"- Rows: `{summary['total_rows']}` total, `{counts['blocker']}` blocker, `{counts['pending']}` pending, `{counts['caveat']}` caveat, `{counts['ok']}` ok.",
        "",
        "| Severity | Row | Bucket | Release delta | Fallback delta | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    severity_order = {"blocker": 0, "pending": 1, "caveat": 2, "ok": 3}
    for record in sorted(records, key=lambda item: (severity_order[item["severity"]], item["row_id"])):
        lines.append(
            "| {severity} | {row_id} | {bucket} | {release_delta} | {fallback_delta} | {status} |".format(
                severity=record["severity"],
                row_id=record["row_id"],
                bucket=record["risk_bucket"],
                release_delta=fmt(record["release_final_delta"]),
                fallback_delta=fmt(record["fallback_final_delta"]),
                status=record["release_status"],
            )
        )
    lines.extend(
        [
            "",
            "Default practical readiness passes when there are no `large_gap` or `unknown` rows.",
            "`pending`, `partial`, `watch`, and fallback rows still need release notes or follow-up evidence.",
            "Use `--strict` to fail on any pending row or caveat.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    records = build_records(manifest)
    summary = summarize(records)
    report = markdown_report(records, summary)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "records": records}
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    failed = not summary["strict_ready"] if args.strict else not summary["practical_ready"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
