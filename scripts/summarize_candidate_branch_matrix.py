from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize candidate-side branch results into a compact benchmark / ablation matrix."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary-row", action="append", default=[])
    parser.add_argument("--fixed-row", action="append", default=[])
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_row_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Row spec must be LABEL=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def _avg_candidates(summary: dict) -> float | None:
    breakdown = summary.get("candidate_count_breakdown")
    if not breakdown:
        return None
    total_examples = summary.get("total_examples") or 0
    if not total_examples:
        return None
    weighted = sum(row["candidate_count"] * row["num_examples"] for row in breakdown)
    return weighted / total_examples


def _build_summary_row(label: str, summary_path: Path) -> dict:
    summary = _load_json(summary_path)
    return {
        "label": label,
        "type": "generate_then_rerank",
        "path": str(summary_path),
        "num_examples": summary.get("total_examples"),
        "first_accuracy": summary.get("first_accuracy"),
        "base_accuracy": summary.get("base_accuracy"),
        "verifier_accuracy": summary.get("verifier_accuracy"),
        "oracle_coverage": summary.get("oracle_coverage"),
        "verifier_given_oracle": (summary.get("selection_efficiency_given_oracle") or {}).get("verifier"),
        "avg_candidates": _avg_candidates(summary),
        "oracle_correct": summary.get("oracle_correct"),
        "verifier_correct": summary.get("verifier_correct"),
    }


def _build_fixed_row(label: str, metrics_path: Path) -> dict:
    metrics = _load_json(metrics_path)
    return {
        "label": label,
        "type": "fixed_reference",
        "path": str(metrics_path),
        "num_examples": metrics.get("num_examples"),
        "first_accuracy": None,
        "base_accuracy": None,
        "verifier_accuracy": metrics.get("accuracy"),
        "oracle_coverage": None,
        "verifier_given_oracle": None,
        "avg_candidates": None,
        "oracle_correct": None,
        "verifier_correct": metrics.get("correct"),
    }


def _fmt_metric(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    rows = []
    for spec in args.summary_row:
        label, path = _parse_row_spec(spec)
        rows.append(_build_summary_row(label, path))
    for spec in args.fixed_row:
        label, path = _parse_row_spec(spec)
        rows.append(_build_fixed_row(label, path))

    write_json(args.output_json, {"title": args.title, "rows": rows})

    lines = [
        f"# {args.title}",
        "",
        "| Label | Type | N | First | Base | Verifier | Oracle | Verifier|Oracle | Avg cands |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {type} | {n} | {first} | {base} | {verifier} | {oracle} | {vgo} | {avg} |".format(
                label=row["label"],
                type=row["type"],
                n=_fmt_metric(row["num_examples"], digits=0),
                first=_fmt_metric(row["first_accuracy"]),
                base=_fmt_metric(row["base_accuracy"]),
                verifier=_fmt_metric(row["verifier_accuracy"]),
                oracle=_fmt_metric(row["oracle_coverage"]),
                vgo=_fmt_metric(row["verifier_given_oracle"]),
                avg=_fmt_metric(row["avg_candidates"]),
            )
        )

    ranked = [row for row in rows if row["verifier_accuracy"] is not None]
    ranked.sort(key=lambda row: row["verifier_accuracy"], reverse=True)
    lines.extend(["", "## Notes", ""])
    if ranked:
        best = ranked[0]
        lines.append(
            f"- Strongest verifier row: `{best['label']}` at `{best['verifier_accuracy']:.4f}` on `{best['num_examples']}` examples."
        )
    generated_rows = [row for row in rows if row["type"] == "generate_then_rerank"]
    if generated_rows:
        best_oracle = max(generated_rows, key=lambda row: row["oracle_coverage"] or -1)
        lines.append(
            f"- Strongest oracle coverage among generated branches: `{best_oracle['label']}` at `{(best_oracle['oracle_coverage'] or 0):.4f}`."
        )
    if len(ranked) >= 2:
        best, second = ranked[0], ranked[1]
        delta = (best["verifier_accuracy"] or 0.0) - (second["verifier_accuracy"] or 0.0)
        lines.append(
            f"- Margin over second-best verifier row: `{delta:.4f}` absolute accuracy (`{best['label']}` vs `{second['label']}`)."
        )

    write_text(args.output_md, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
