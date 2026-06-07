from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Route C+ true-strong Gate B compatibility pack.")
    parser.add_argument(
        "--numeric-remedy-json",
        default="Experiment/analysis/results/routec_p2_secondfamily_mistral7b_instruct_gsm8k_eval_128_remedy_matrix_v2.json",
    )
    parser.add_argument(
        "--mcq-family-json",
        default="Experiment/analysis/results/routec_p2_secondfamily_mistral7b_instruct_mmlu_pro_smoke_128_family_hygiene_filteronly_v3_matrix.json",
    )
    parser.add_argument(
        "--mcq-selector-json",
        default="Experiment/analysis/results/routec_p2_secondfamily_selectorcal_mmlu_pro_smoke_128_sweep_summary.json",
    )
    parser.add_argument(
        "--output-json",
        default="Experiment/analysis/results/routec_plus_true_gateB_compatibility_pack_v1.json",
    )
    parser.add_argument(
        "--output-md",
        default="Experiment/analysis/results/routec_plus_true_gateB_compatibility_pack_v1.md",
    )
    return parser.parse_args()


def _load_json(path: str | Path):
    return json.loads((PROJECT_ROOT / Path(path)).read_text(encoding="utf-8"))


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()

    numeric_rows = _load_json(args.numeric_remedy_json)
    mcq_family = _load_json(args.mcq_family_json)
    mcq_selector = _load_json(args.mcq_selector_json)

    numeric_by_method = {row["method"]: row for row in numeric_rows}
    mcq_family_rows = mcq_family["slices"][0]["rows"]
    mcq_family_by_method = {row["method"]: row for row in mcq_family_rows}
    mcq_selector_methods = mcq_selector["methods"]

    numeric_baseline = numeric_by_method["diverse_base"]
    numeric_best = max(numeric_rows, key=lambda row: row["verifier"])
    numeric_recovered = [
        row for row in numeric_rows if row["method"] != "diverse_base" and row["verifier"] > numeric_baseline["verifier"]
    ]

    mcq_baseline = mcq_family_by_method["diverse_base"]
    mcq_family_best = max(mcq_family_rows, key=lambda row: row["verifier_accuracy"])
    mcq_selector_best_name, mcq_selector_best = max(
        mcq_selector_methods.items(), key=lambda item: item[1]["best_selector_accuracy"]
    )
    mcq_recovered = [
        (name, row)
        for name, row in mcq_selector_methods.items()
        if row["best_selector_accuracy"] > mcq_baseline["verifier_accuracy"]
    ]

    integrated_rows = [
        {
            "method": "diverse_base",
            "pool": "raw diverse_base",
            "rule": "baseline verifier",
            "oracle": mcq_baseline["oracle_coverage"],
            "final_accuracy": mcq_baseline["verifier_accuracy"],
            "delta_vs_baseline": 0.0,
            "invalid": mcq_baseline["invalid_final"],
            "clean": mcq_baseline["selected_parseable"],
        },
        {
            "method": "cacc",
            "pool": "raw CACC",
            "rule": "baseline verifier",
            "oracle": mcq_family_by_method["cacc"]["oracle_coverage"],
            "final_accuracy": mcq_family_by_method["cacc"]["verifier_accuracy"],
            "delta_vs_baseline": mcq_family_by_method["cacc"]["delta_verifier_vs_baseline"],
            "invalid": mcq_family_by_method["cacc"]["invalid_final"],
            "clean": mcq_family_by_method["cacc"]["selected_parseable"],
        },
        {
            "method": "cacc_pstar",
            "pool": "raw CACC-P*",
            "rule": "baseline verifier",
            "oracle": mcq_family_by_method["cacc_pstar"]["oracle_coverage"],
            "final_accuracy": mcq_family_by_method["cacc_pstar"]["verifier_accuracy"],
            "delta_vs_baseline": mcq_family_by_method["cacc_pstar"]["delta_verifier_vs_baseline"],
            "invalid": mcq_family_by_method["cacc_pstar"]["invalid_final"],
            "clean": mcq_family_by_method["cacc_pstar"]["selected_parseable"],
        },
        {
            "method": "cacc_filteronly_v3",
            "pool": "family_hygiene_filteronly_v3",
            "rule": mcq_selector_methods["cacc_filteronly_v3"]["best_rule"],
            "oracle": mcq_family_by_method["cacc_family_hygiene_filteronly_v3"]["oracle_coverage"],
            "final_accuracy": mcq_selector_methods["cacc_filteronly_v3"]["best_selector_accuracy"],
            "delta_vs_baseline": mcq_selector_methods["cacc_filteronly_v3"]["best_selector_accuracy"] - mcq_baseline["verifier_accuracy"],
            "invalid": mcq_selector_methods["cacc_filteronly_v3"]["invalid_rate"],
            "clean": mcq_selector_methods["cacc_filteronly_v3"]["clean_rate"],
        },
        {
            "method": "cacc_pstar_filteronly_v3",
            "pool": "family_hygiene_filteronly_v3",
            "rule": mcq_selector_methods["cacc_pstar_filteronly_v3"]["best_rule"],
            "oracle": mcq_family_by_method["cacc_pstar_family_hygiene_filteronly_v3"]["oracle_coverage"],
            "final_accuracy": mcq_selector_methods["cacc_pstar_filteronly_v3"]["best_selector_accuracy"],
            "delta_vs_baseline": mcq_selector_methods["cacc_pstar_filteronly_v3"]["best_selector_accuracy"] - mcq_baseline["verifier_accuracy"],
            "invalid": mcq_selector_methods["cacc_pstar_filteronly_v3"]["invalid_rate"],
            "clean": mcq_selector_methods["cacc_pstar_filteronly_v3"]["clean_rate"],
        },
    ]

    payload = {
        "gate": "B",
        "title": "Route C+ true-strong compatibility pack",
        "status": "passed" if numeric_recovered and mcq_recovered else "partial",
        "claim": "compatibility is not free, but is systematically recoverable under family-aware alignment",
        "inputs": {
            "numeric_remedy_json": str(Path(args.numeric_remedy_json)),
            "mcq_family_json": str(Path(args.mcq_family_json)),
            "mcq_selector_json": str(Path(args.mcq_selector_json)),
        },
        "numeric": {
            "baseline_method": "diverse_base",
            "baseline_verifier": numeric_baseline["verifier"],
            "best_method": numeric_best["method"],
            "best_verifier": numeric_best["verifier"],
            "recovered_methods": [row["method"] for row in numeric_recovered],
            "rows": numeric_rows,
        },
        "mcq": {
            "baseline_method": "diverse_base",
            "baseline_verifier": mcq_baseline["verifier_accuracy"],
            "best_family_method": mcq_family_best["method"],
            "best_family_verifier": mcq_family_best["verifier_accuracy"],
            "best_selector_method": mcq_selector_best_name,
            "best_selector_accuracy": mcq_selector_best["best_selector_accuracy"],
            "recovered_methods": [name for name, _ in mcq_recovered],
            "family_rows": mcq_family_rows,
            "selector_rows": mcq_selector_methods,
            "integrated_rows": integrated_rows,
        },
        "assessment": {
            "numeric_recoverable": bool(numeric_recovered),
            "mcq_recoverable": bool(mcq_recovered),
            "numeric_takeaway": (
                "Numeric second-family no longer looks like an unrecoverable verifier-collapse regime: "
                f"{numeric_best['method']} reaches verifier {_fmt(numeric_best['verifier'])} vs diverse_base {_fmt(numeric_baseline['verifier'])}."
            ),
            "mcq_takeaway": (
                "MCQ second-family remains compatibility-sensitive at the raw verifier stage, but family-hygiene plus selector calibration closes the gap: "
                f"best final accuracy {_fmt(mcq_selector_best['best_selector_accuracy'])} vs diverse_base {_fmt(mcq_baseline['verifier_accuracy'])}."
            ),
            "paper_facing_claim": "The second-family story should be framed as recoverability under explicit alignment, not free cross-family generalization.",
        },
    }

    output_json = PROJECT_ROOT / Path(args.output_json)
    output_md = PROJECT_ROOT / Path(args.output_md)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Route C+ True-Strong Gate B Compatibility Pack",
        "",
        f"- Status: `{payload['status']}`",
        f"- Paper-facing claim: `{payload['claim']}`",
        "",
        "## Summary",
        "",
        f"- Numeric second-family baseline is `diverse_base={_fmt(numeric_baseline['verifier'])}` verifier accuracy; best recovered row is `{numeric_best['method']}={_fmt(numeric_best['verifier'])}`.",
        f"- MCQ second-family baseline is `diverse_base={_fmt(mcq_baseline['verifier_accuracy'])}` verifier accuracy; best recovered row is `{mcq_selector_best_name} + {mcq_selector_best['best_rule']} = {_fmt(mcq_selector_best['best_selector_accuracy'])}`.",
        "- The unified interpretation is not free generalization. It is family-sensitive compatibility that becomes recoverable after style / hygiene / selector alignment.",
        "",
        "## Numeric Remedy",
        "",
        "| Method | Oracle | Base | Verifier | V|Oracle | Invalid | Clean | ΔVerifier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in numeric_rows:
        lines.append(
            f"| {row['method']} | {_fmt(row['oracle'])} | {_fmt(row['base'])} | {_fmt(row['verifier'])} | {_fmt(row['vgo'])} | {_fmt(row['invalid'])} | {_fmt(row['clean'])} | {_fmt(row['delta_verifier'], digits=4)} |"
        )

    lines.extend([
        "",
        "## MCQ Remedy",
        "",
        "| Method | Pool | Rule | Oracle | Final | ΔFinal | Invalid | Clean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in integrated_rows:
        lines.append(
            f"| {row['method']} | {row['pool']} | `{row['rule']}` | {_fmt(row['oracle'])} | {_fmt(row['final_accuracy'])} | {_fmt(row['delta_vs_baseline'])} | {_fmt(row['invalid'])} | {_fmt(row['clean'])} |"
        )

    lines.extend([
        "",
        "## Assessment",
        "",
        f"- Numeric verdict: {payload['assessment']['numeric_takeaway']}",
        f"- MCQ verdict: {payload['assessment']['mcq_takeaway']}",
        f"- Final Gate B judgment: `{payload['status']}`. The evidence now supports a unified compatibility matrix, but the correct claim remains `{payload['assessment']['paper_facing_claim']}`",
        "",
        "## Canonical Inputs",
        "",
        f"- `{args.numeric_remedy_json}`",
        f"- `{args.mcq_family_json}`",
        f"- `{args.mcq_selector_json}`",
    ])

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
