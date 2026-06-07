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
        description="Compare CACC summary JSON files with the paper Table 1 reproduction targets."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="Path to configs/reproduction_targets.json.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="ROW_ID=PATH",
        help="Summary JSON to compare, for example gsm8k/base=runs/foo/summary.json.",
    )
    parser.add_argument(
        "--close-final-delta",
        default=0.02,
        type=float,
        help="Absolute final-accuracy delta considered close.",
    )
    parser.add_argument(
        "--watch-final-delta",
        default=0.05,
        type=float,
        help="Absolute final-accuracy delta considered a watch item rather than a large gap.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for machine-readable comparison results.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any provided summary has a large final-accuracy gap.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def first_number(data: dict[str, Any], paths: tuple[tuple[str, ...], ...], label: str) -> float:
    for path in paths:
        value = get_nested(data, path)
        if isinstance(value, (int, float)):
            return float(value)
    joined = ", ".join(".".join(path) for path in paths)
    raise KeyError(f"Could not find numeric {label}; tried: {joined}")


def extract_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "oracle": first_number(
            summary,
            (
                ("oracle_coverage",),
                ("oracle",),
                ("metrics", "oracle"),
                ("paper", "oracle"),
            ),
            "oracle",
        ),
        "v_given_o": first_number(
            summary,
            (
                ("selection_efficiency_given_oracle", "verifier"),
                ("v_given_o",),
                ("verifier_efficiency_given_oracle",),
                ("metrics", "v_given_o"),
                ("paper", "v_given_o"),
            ),
            "v_given_o",
        ),
        "final": first_number(
            summary,
            (
                ("verifier_accuracy",),
                ("final",),
                ("metrics", "final"),
                ("paper", "final"),
            ),
            "final",
        ),
    }


def parse_summary_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected ROW_ID=PATH, got {value!r}")
    row_id, path = value.split("=", 1)
    row_id = row_id.strip()
    if not row_id:
        raise ValueError(f"Missing row id in {value!r}")
    return row_id, Path(path)


def classify_final_delta(delta: float, close: float, watch: float) -> str:
    magnitude = abs(delta)
    if magnitude <= close:
        return "close"
    if magnitude <= watch:
        return "watch"
    return "large_gap"


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def manifest_overview(rows: list[dict[str, Any]]) -> None:
    print("| Row | Paper O/V/F | Release status | Release reference | Fallback reference |")
    print("| --- | ---: | --- | ---: | ---: |")
    for row in rows:
        paper = row["paper"]
        release_reference = row.get("release_reference")
        fallback_reference = row.get("fallback_reference")
        release_text = "-"
        fallback_text = "-"
        if release_reference:
            release_text = (
                f"{fmt(release_reference['oracle'])} / "
                f"{fmt(release_reference['v_given_o'])} / "
                f"{fmt(release_reference['final'])}"
            )
        if fallback_reference:
            fallback_text = (
                f"{fmt(fallback_reference['oracle'])} / "
                f"{fmt(fallback_reference['v_given_o'])} / "
                f"{fmt(fallback_reference['final'])}"
            )
        print(
            "| {row_id} | {paper_ovf} | {status} | {release} | {fallback} |".format(
                row_id=row["row_id"],
                paper_ovf=f"{fmt(paper['oracle'])} / {fmt(paper['v_given_o'])} / {fmt(paper['final'])}",
                status=row["release_status"],
                release=release_text,
                fallback=fallback_text,
            )
        )


def compare_summary(
    row: dict[str, Any],
    summary_path: Path,
    close_final_delta: float,
    watch_final_delta: float,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    observed = extract_metrics(summary)
    paper = row["paper"]
    deltas = {key: observed[key] - float(paper[key]) for key in ("oracle", "v_given_o", "final")}
    bucket = classify_final_delta(deltas["final"], close_final_delta, watch_final_delta)
    result: dict[str, Any] = {
        "row_id": row["row_id"],
        "dataset": row["dataset"],
        "variant": row["variant"],
        "summary_path": str(summary_path),
        "release_status": row["release_status"],
        "paper": paper,
        "observed": observed,
        "delta_to_paper": deltas,
        "final_bucket": bucket,
    }
    reference = row.get("release_reference") or row.get("fallback_reference")
    if reference:
        result["delta_to_release_reference"] = {
            key: observed[key] - float(reference[key]) for key in ("oracle", "v_given_o", "final")
        }
    return result


def print_comparison(results: list[dict[str, Any]]) -> None:
    print("| Row | Observed O/V/F | Paper O/V/F | Delta O/V/F | Final bucket | Release status |")
    print("| --- | ---: | ---: | ---: | --- | --- |")
    for result in results:
        observed = result["observed"]
        paper = result["paper"]
        delta = result["delta_to_paper"]
        print(
            "| {row_id} | {observed} | {paper} | {delta} | {bucket} | {status} |".format(
                row_id=result["row_id"],
                observed=f"{fmt(observed['oracle'])} / {fmt(observed['v_given_o'])} / {fmt(observed['final'])}",
                paper=f"{fmt(paper['oracle'])} / {fmt(paper['v_given_o'])} / {fmt(paper['final'])}",
                delta=f"{fmt_delta(delta['oracle'])} / {fmt_delta(delta['v_given_o'])} / {fmt_delta(delta['final'])}",
                bucket=result["final_bucket"],
                status=result["release_status"],
            )
        )


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    rows = manifest.get("rows", [])
    row_by_id = {row["row_id"]: row for row in rows}

    if not args.summary:
        manifest_overview(rows)
        return 0

    results: list[dict[str, Any]] = []
    for item in args.summary:
        row_id, summary_path = parse_summary_arg(item)
        if row_id not in row_by_id:
            known = ", ".join(sorted(row_by_id))
            raise KeyError(f"Unknown row id {row_id!r}. Known row ids: {known}")
        results.append(
            compare_summary(row_by_id[row_id], summary_path, args.close_final_delta, args.watch_final_delta)
        )

    print_comparison(results)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    has_large_gap = any(result["final_bucket"] == "large_gap" for result in results)
    return 1 if args.strict and has_large_gap else 0


if __name__ == "__main__":
    sys.exit(main())
