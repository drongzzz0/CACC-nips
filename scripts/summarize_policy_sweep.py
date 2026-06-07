from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import write_json, write_text


DEFAULT_POLICY_ORDER = [
    "append_if_room",
    "replace_random_nonprefix",
    "replace_fragments_first",
    "replace_partials_first",
    "replace_random_fragment_only",
    "replace_random_partial_only",
    "replace_invalid_first",
    "replace_compatibility_risk_first",
    "replace_hybrid_salvageability",
    "replace_closure_score_first",
    "replace_verifier_uncertainty_first",
    "replace_margin_risk_hybrid",
    "replace_margin_risk_no_salvage",
    "replace_margin_salvage_no_risk",
    "replace_stratified_risk_preserve",
    "replace_margin_stratified_risk_preserve",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Route C+ policy sweep result JSON files into a compact leaderboard.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--summary-json", action="append", default=[])
    parser.add_argument("--baseline-policy", default="replace_random_nonprefix")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Summary spec must be POLICY=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def _policy_rank(policy: str) -> tuple[int, str]:
    try:
        return (DEFAULT_POLICY_ORDER.index(policy), policy)
    except ValueError:
        return (len(DEFAULT_POLICY_ORDER), policy)


def _row_from_summary(policy: str, path: Path) -> dict:
    summary = _load_json(path)
    hygiene = summary.get("prediction_hygiene", {})
    generation = summary.get("generation_hygiene", {})
    return {
        "policy": policy,
        "path": str(path),
        "run_label": summary.get("run_label"),
        "num_examples": summary.get("total_examples"),
        "oracle_coverage": summary.get("oracle_coverage"),
        "first_accuracy": summary.get("first_accuracy"),
        "base_accuracy": summary.get("base_accuracy"),
        "verifier_accuracy": summary.get("verifier_accuracy"),
        "verifier_given_oracle": summary.get("verifier_given_oracle"),
        "selected_parseable": hygiene.get("verifier_selected_parseable_rate"),
        "invalid_final": hygiene.get("verifier_invalid_final_rate"),
        "instruction_leak": hygiene.get("verifier_instruction_leak_rate"),
        "scaffold_residue": hygiene.get("verifier_scaffold_residue_rate"),
        "overrule_correct_first": hygiene.get("selector_overrule_correct_first_rate"),
        "verifier_margin_mean": hygiene.get("verifier_selected_margin_mean"),
        "examples_modified": generation.get("examples_modified"),
        "replaced_fragment": generation.get("replaced_fragment"),
        "replaced_partial_solution": generation.get("replaced_partial_solution"),
        "replaced_invalid_final": generation.get("replaced_invalid_final"),
        "replaced_instruction_leak": generation.get("replaced_instruction_leak"),
        "replaced_scaffold_residue": generation.get("replaced_scaffold_residue"),
    }


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    rows = [_row_from_summary(policy, path) for policy, path in (_parse_spec(spec) for spec in args.summary_json)]
    rows.sort(key=lambda row: _policy_rank(row["policy"]))

    baseline = next((row for row in rows if row["policy"] == args.baseline_policy), None)
    for row in rows:
        if baseline is None:
            row["delta_verifier_vs_baseline"] = None
            row["delta_oracle_vs_baseline"] = None
        else:
            row["delta_verifier_vs_baseline"] = (row["verifier_accuracy"] or 0.0) - (baseline["verifier_accuracy"] or 0.0)
            row["delta_oracle_vs_baseline"] = (row["oracle_coverage"] or 0.0) - (baseline["oracle_coverage"] or 0.0)

    ranked = sorted(rows, key=lambda row: (row["verifier_accuracy"] is not None, row["verifier_accuracy"]), reverse=True)
    best = ranked[0] if ranked else None

    payload = {
        "title": args.title,
        "benchmark": args.benchmark,
        "baseline_policy": args.baseline_policy,
        "best_policy_by_verifier": best["policy"] if best else None,
        "rows": rows,
    }
    write_json(args.output_json, payload)

    lines = [
        f"# {args.title}",
        "",
        f"- benchmark: `{args.benchmark}`",
        f"- baseline policy: `{args.baseline_policy}`",
        "",
        "| Policy | N | Oracle | Base | Verifier | V\\|Oracle | ΔVerifier | ΔOracle | Parseable | Invalid | Overrule-first |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {n} | {oracle} | {base} | {verifier} | {vgo} | {dv} | {do} | {parseable} | {invalid} | {overrule} |".format(
                policy=row["policy"],
                n=_fmt(row["num_examples"], digits=0),
                oracle=_fmt(row["oracle_coverage"]),
                base=_fmt(row["base_accuracy"]),
                verifier=_fmt(row["verifier_accuracy"]),
                vgo=_fmt(row["verifier_given_oracle"]),
                dv=_fmt(row["delta_verifier_vs_baseline"]),
                do=_fmt(row["delta_oracle_vs_baseline"]),
                parseable=_fmt(row["selected_parseable"]),
                invalid=_fmt(row["invalid_final"]),
                overrule=_fmt(row["overrule_correct_first"]),
            )
        )

    lines.extend(["", "## Notes", ""])
    if best is not None:
        lines.append(
            f"- Best verifier row: `{best['policy']}` at `{best['verifier_accuracy']:.4f}` with oracle `{(best['oracle_coverage'] or 0.0):.4f}`."
        )
    if baseline is not None and best is not None:
        lines.append(
            f"- Relative to baseline `{args.baseline_policy}`, best-row verifier delta is `{((best['verifier_accuracy'] or 0.0) - (baseline['verifier_accuracy'] or 0.0)):+.4f}`."
        )
    top_oracle = max(rows, key=lambda row: row["oracle_coverage"] if row["oracle_coverage"] is not None else float("-inf")) if rows else None
    if top_oracle is not None:
        lines.append(
            f"- Highest oracle coverage row: `{top_oracle['policy']}` at `{(top_oracle['oracle_coverage'] or 0.0):.4f}`."
        )
    write_text(args.output_md, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
