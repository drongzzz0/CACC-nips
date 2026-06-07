from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Route C+ true-strong Gate C external frontier pack.")
    parser.add_argument(
        "--gsm8k-leaderboard-json",
        default="Experiment/analysis/results/routec_plus_policy_gsm8k_full_clean_leaderboard_v1.json",
    )
    parser.add_argument(
        "--gsm8k-self-refine-json",
        default="Experiment/analysis/results/experiment_12_gsm8k_full_self_refine_p1b_result_v1.json",
    )
    parser.add_argument(
        "--compmath-random-json",
        default="Experiment/analysis/results/routec_plus_policy_fullconfirm_v2_competition_math_numeric_random_nonprefix_v1.json",
    )
    parser.add_argument(
        "--compmath-main-json",
        default="Experiment/analysis/results/routec_plus_policy_fullconfirm_competition_math_numeric_hybrid_salvageability_v1.json",
    )
    parser.add_argument(
        "--self-refine-summary-json",
        default="Experiment/analysis/results/a800_self_refine_three_benchmark_transfer_b32_v2_summary.json",
    )
    parser.add_argument(
        "--mmlu-core-json",
        default="Experiment/analysis/results/routec_p1_mmlu_pro_cacc_v1.json",
    )
    parser.add_argument(
        "--mmlu-sp-json",
        default="Experiment/analysis/results/routec_p1_mmlu_pro_cacc_sp_v1.json",
    )
    parser.add_argument(
        "--mmlu-policy-subset-random-json",
        default="Experiment/analysis/results/routec_plus_policy_v2subset_mmlu_pro_test_random_nonprefix_v1.json",
    )
    parser.add_argument(
        "--mmlu-policy-subset-main-json",
        default="Experiment/analysis/results/routec_plus_policy_v2subset_mmlu_pro_test_verifier_uncertainty_first_v1.json",
    )
    parser.add_argument(
        "--gate-d-json",
        default="Experiment/analysis/results/routec_plus_true_gateD_stability_audit_pack_v1.json",
    )
    parser.add_argument(
        "--output-json",
        default="Experiment/analysis/results/routec_plus_true_gateC_external_frontier_pack_v1.json",
    )
    parser.add_argument(
        "--output-md",
        default="Experiment/analysis/results/routec_plus_true_gateC_external_frontier_pack_v1.md",
    )
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_json(path: str | Path):
    return json.loads(_project_path(path).read_text(encoding="utf-8"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _leaderboard_row(leaderboard: dict, policy: str) -> dict:
    for row in leaderboard.get("rows", []):
        if row.get("policy") == policy:
            return row
    raise KeyError(f"Policy not found in leaderboard: {policy}")


def _self_refine_benchmark(summary: dict, key: str) -> dict:
    benchmarks = summary.get("benchmarks", {})
    if key not in benchmarks:
        raise KeyError(f"Self-Refine summary missing benchmark: {key}")
    return benchmarks[key]


def _reported_pairrm(gate_d: dict, benchmark: str) -> dict | None:
    frontier = gate_d.get("external_frontier", {}).get("rows", [])
    for bench_row in frontier:
        if bench_row.get("benchmark") != benchmark:
            continue
        for row in bench_row.get("rows", []):
            if isinstance(row.get("label"), str) and row["label"].startswith("PairRM"):
                return {
                    "label": row.get("label"),
                    "accuracy": row.get("accuracy"),
                    "protocol": row.get("protocol"),
                    "num_examples": row.get("num_examples"),
                    "provenance": "reported_only_from_gateD_pack",
                    "report_path": row.get("report_path"),
                    "metrics_path": row.get("metrics_path"),
                }
    return None


def _competitive_tag(delta_vs_external: float) -> str:
    if delta_vs_external >= 0.03:
        return "clear_win"
    if delta_vs_external >= 0.0:
        return "edge_or_win"
    if delta_vs_external >= -0.015:
        return "near_parity"
    if delta_vs_external >= -0.03:
        return "competitive_but_trailing"
    return "clear_trail"


def _build_gsm8k(args: argparse.Namespace) -> dict:
    leaderboard = _load_json(args.gsm8k_leaderboard_json)
    self_refine = _load_json(args.gsm8k_self_refine_json)

    anchor = _leaderboard_row(leaderboard, "replace_random_nonprefix")
    promoted = _leaderboard_row(leaderboard, "replace_verifier_uncertainty_first")
    external = {
        "label": "Self-Refine",
        "accuracy": self_refine["accuracy"],
        "num_examples": self_refine["num_examples"],
        "parseable": self_refine.get("selected_prediction_parseable_rate"),
        "invalid": self_refine.get("invalid_final_answer_rate"),
        "protocol": "matched-budget self-refine external baseline",
        "artifact": str(Path(args.gsm8k_self_refine_json)),
    }
    delta_vs_external = promoted["verifier_accuracy"] - external["accuracy"]

    return {
        "benchmark": "gsm8k_full_clean",
        "regime": "numeric_open_ended",
        "evidence_scope": "full_primary",
        "anchor": {
            "label": "Internal random-nonprefix",
            "policy": anchor["policy"],
            "accuracy": anchor["verifier_accuracy"],
            "oracle": anchor["oracle_coverage"],
            "parseable": anchor.get("selected_parseable"),
            "invalid": anchor.get("invalid_final"),
            "artifact": anchor.get("path"),
        },
        "promoted": {
            "label": "Internal CACC-P*",
            "policy": promoted["policy"],
            "accuracy": promoted["verifier_accuracy"],
            "oracle": promoted["oracle_coverage"],
            "parseable": promoted.get("selected_parseable"),
            "invalid": promoted.get("invalid_final"),
            "artifact": promoted.get("path"),
            "role": "policy-promoted in-family main row",
        },
        "external": external,
        "delta_promoted_vs_anchor": promoted["verifier_accuracy"] - anchor["verifier_accuracy"],
        "delta_promoted_vs_external": delta_vs_external,
        "frontier_status": _competitive_tag(delta_vs_external),
        "paper_facing_note": "GSM8K full clean remains the clearest numeric regime where the current internal promoted row is statistically above the internal random control but still materially below strongest iterative external baseline.",
    }


def _build_compmath(args: argparse.Namespace, self_refine_summary: dict, gate_d: dict | None) -> dict:
    anchor = _load_json(args.compmath_random_json)
    promoted = _load_json(args.compmath_main_json)
    external_summary = _self_refine_benchmark(self_refine_summary, "competition_math_numeric")
    delta_vs_external = promoted["verifier_accuracy"] - external_summary["accuracy"]

    payload = {
        "benchmark": "competition_math_numeric",
        "regime": "numeric_open_ended",
        "evidence_scope": "full_primary",
        "anchor": {
            "label": "Internal random-nonprefix",
            "policy": anchor.get("generation_hygiene", {}).get("merge_policy"),
            "accuracy": anchor["verifier_accuracy"],
            "oracle": anchor["oracle_coverage"],
            "parseable": anchor.get("prediction_hygiene", {}).get("verifier_selected_parseable_rate"),
            "invalid": anchor.get("prediction_hygiene", {}).get("verifier_invalid_final_rate"),
            "artifact": str(Path(args.compmath_random_json)),
        },
        "promoted": {
            "label": "Internal best-landed full row",
            "policy": promoted.get("generation_hygiene", {}).get("merge_policy") or "hybrid_salvageability",
            "accuracy": promoted["verifier_accuracy"],
            "oracle": promoted["oracle_coverage"],
            "parseable": promoted.get("prediction_hygiene", {}).get("verifier_selected_parseable_rate"),
            "invalid": promoted.get("prediction_hygiene", {}).get("verifier_invalid_final_rate"),
            "artifact": str(Path(args.compmath_main_json)),
            "role": "best landed full numeric row; current workspace does not contain a full verifier_uncertainty_first artifact for this benchmark",
        },
        "external": {
            "label": "Self-Refine",
            "accuracy": external_summary["accuracy"],
            "num_examples": external_summary["total"],
            "protocol": "matched-budget self-refine external baseline",
            "artifact": str(Path(args.self_refine_summary_json)),
        },
        "delta_promoted_vs_anchor": promoted["verifier_accuracy"] - anchor["verifier_accuracy"],
        "delta_promoted_vs_external": delta_vs_external,
        "frontier_status": _competitive_tag(delta_vs_external),
        "paper_facing_note": "Competition-Math-Numeric is the cleanest near-frontier numeric row: the landed internal best row improves on internal random but still trails Self-Refine by about one point.",
    }
    if gate_d is not None:
        pairrm = _reported_pairrm(gate_d, "competition_math_numeric")
        if pairrm is not None:
            payload["reported_supporting_external_family"] = pairrm
    return payload


def _build_mmlu(args: argparse.Namespace, self_refine_summary: dict, gate_d: dict | None) -> dict:
    core = _load_json(args.mmlu_core_json)
    sp = _load_json(args.mmlu_sp_json)
    subset_random = _load_json(args.mmlu_policy_subset_random_json)
    subset_main = _load_json(args.mmlu_policy_subset_main_json)
    external_summary = _self_refine_benchmark(self_refine_summary, "mmlu_pro")
    delta_vs_external = sp["verifier_accuracy"] - external_summary["accuracy"]

    payload = {
        "benchmark": "mmlu_pro_test",
        "regime": "mcq_compatibility_limited",
        "evidence_scope": "full_primary_plus_subset_policy_note",
        "anchor": {
            "label": "Internal CACC",
            "policy": "cacc",
            "accuracy": core["verifier_accuracy"],
            "oracle": core["oracle_coverage"],
            "parseable": core.get("prediction_hygiene", {}).get("verifier_selected_parseable_rate"),
            "invalid": core.get("prediction_hygiene", {}).get("verifier_invalid_final_rate"),
            "artifact": str(Path(args.mmlu_core_json)),
        },
        "promoted": {
            "label": "Internal CACC+SP",
            "policy": "cacc_plus_sp",
            "accuracy": sp["verifier_accuracy"],
            "oracle": sp["oracle_coverage"],
            "parseable": sp.get("prediction_hygiene", {}).get("verifier_selected_parseable_rate"),
            "invalid": sp.get("prediction_hygiene", {}).get("verifier_invalid_final_rate"),
            "artifact": str(Path(args.mmlu_sp_json)),
            "role": "full landed MCQ strengthening row; this is stronger and better supported than the policy-only subset probe in the current workspace",
        },
        "external": {
            "label": "Self-Refine",
            "accuracy": external_summary["accuracy"],
            "num_examples": external_summary["total"],
            "protocol": "matched-budget self-refine external baseline",
            "artifact": str(Path(args.self_refine_summary_json)),
        },
        "subset_policy_note": {
            "scope": "subset128_only",
            "random_nonprefix_accuracy": subset_random["verifier_accuracy"],
            "verifier_uncertainty_first_accuracy": subset_main["verifier_accuracy"],
            "delta_policy_probe": subset_main["verifier_accuracy"] - subset_random["verifier_accuracy"],
            "note": "The current v2 subset probe ties random-nonprefix and verifier_uncertainty_first on MMLU-Pro, so it cannot replace the full-split CACC->CACC+SP frontier row.",
            "artifacts": [
                str(Path(args.mmlu_policy_subset_random_json)),
                str(Path(args.mmlu_policy_subset_main_json)),
            ],
        },
        "delta_promoted_vs_anchor": sp["verifier_accuracy"] - core["verifier_accuracy"],
        "delta_promoted_vs_external": delta_vs_external,
        "frontier_status": _competitive_tag(delta_vs_external),
        "paper_facing_note": "MMLU-Pro is the strongest MCQ frontier row currently landed in the workspace: CACC+SP clearly exceeds Self-Refine, while the subset policy probe shows no evidence that a policy-only CACC-P* upgrade is yet the right MCQ main row.",
    }
    if gate_d is not None:
        pairrm = _reported_pairrm(gate_d, "mmlu_pro_test")
        if pairrm is not None:
            payload["reported_supporting_external_family"] = pairrm
    return payload


def main() -> None:
    args = parse_args()
    self_refine_summary = _load_json(args.self_refine_summary_json)
    gate_d = _load_json(args.gate_d_json) if _project_path(args.gate_d_json).exists() else None

    benchmark_rows = [
        _build_gsm8k(args),
        _build_compmath(args, self_refine_summary, gate_d),
        _build_mmlu(args, self_refine_summary, gate_d),
    ]

    clear_wins = [row["benchmark"] for row in benchmark_rows if row["frontier_status"] in {"clear_win", "edge_or_win"}]
    near_parity = [row["benchmark"] for row in benchmark_rows if row["frontier_status"] == "near_parity"]
    clear_trails = [row["benchmark"] for row in benchmark_rows if row["frontier_status"] == "clear_trail"]

    status = "passed" if clear_wins and any(row["regime"].startswith("numeric") and row["frontier_status"] in {"clear_win", "edge_or_win", "near_parity"} for row in benchmark_rows) else "partial"
    if clear_trails:
        status = "partial"

    payload = {
        "gate": "C",
        "title": "Route C+ true-strong external frontier pack",
        "status": status,
        "claim": "Route C+ now has a regime-mixed external frontier profile: one clear MCQ competitive row, one near-frontier numeric row, and one still-open numeric failure regime.",
        "canonical_inputs": {
            "gsm8k_leaderboard_json": str(Path(args.gsm8k_leaderboard_json)),
            "gsm8k_self_refine_json": str(Path(args.gsm8k_self_refine_json)),
            "compmath_random_json": str(Path(args.compmath_random_json)),
            "compmath_main_json": str(Path(args.compmath_main_json)),
            "self_refine_summary_json": str(Path(args.self_refine_summary_json)),
            "mmlu_core_json": str(Path(args.mmlu_core_json)),
            "mmlu_sp_json": str(Path(args.mmlu_sp_json)),
            "mmlu_policy_subset_random_json": str(Path(args.mmlu_policy_subset_random_json)),
            "mmlu_policy_subset_main_json": str(Path(args.mmlu_policy_subset_main_json)),
            "gate_d_json": str(Path(args.gate_d_json)) if gate_d is not None else None,
        },
        "benchmarks": benchmark_rows,
        "assessment": {
            "clear_wins": clear_wins,
            "near_parity": near_parity,
            "clear_trails": clear_trails,
            "frontier_shape": {
                "mcq": "clear competitiveness exists on MMLU-Pro via the landed full CACC+SP row",
                "numeric": "competition-math is close but GSM8K full clean still trails strongest external baseline by a large margin",
            },
            "verdict": "Gate C remains partial: the workspace now supports a clean regime-mixed frontier summary, but not an overall frontier-win claim against strongest external iterative baselines.",
            "important_reconciliation": [
                "The new pack uses only landed artifacts and explicitly avoids the stale Gate D references to nonexistent routec_plus_true_gateA_* placeholder files.",
                "For Competition-Math-Numeric, the landed promoted full row is hybrid_salvageability, not a nonexistent full verifier_uncertainty_first artifact.",
                "For MMLU-Pro, the current full frontier should be reported as CACC -> CACC+SP; the subset policy probe does not justify replacing that row with a policy-only CACC-P* claim.",
            ],
        },
    }

    lines = [
        "# Route C+ True-Strong Gate C External Frontier Pack",
        "",
        f"- Status: `{payload['status']}`",
        f"- Paper-facing claim: `{payload['claim']}`",
        "",
        "## Summary",
        "",
        f"- Clear competitive / winning row(s): `{', '.join(clear_wins) if clear_wins else 'none'}`.",
        f"- Near-parity row(s): `{', '.join(near_parity) if near_parity else 'none'}`.",
        f"- Clear trailing row(s): `{', '.join(clear_trails) if clear_trails else 'none'}`.",
        "- This pack is built only from landed artifacts in the current workspace and explicitly avoids stale placeholder references serialized in older consolidation docs.",
        "",
        "## Unified Frontier Table",
        "",
        "| Benchmark | Scope | Anchor | Anchor Acc | Promoted | Promoted Acc | Strongest External | External Acc | Δ Promoted-Anchor | Δ Promoted-External | Frontier |",
        "| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]

    for row in benchmark_rows:
        lines.append(
            f"| {row['benchmark']} | {row['evidence_scope']} | {row['anchor']['label']} | {_fmt(row['anchor']['accuracy'])} | {row['promoted']['label']} | {_fmt(row['promoted']['accuracy'])} | {row['external']['label']} | {_fmt(row['external']['accuracy'])} | {_fmt(row['delta_promoted_vs_anchor'])} | {_fmt(row['delta_promoted_vs_external'])} | `{row['frontier_status']}` |"
        )

    lines.extend([
        "",
        "## Benchmark Notes",
        "",
    ])
    for row in benchmark_rows:
        lines.append(f"### {row['benchmark']}")
        lines.append("")
        lines.append(f"- Regime: `{row['regime']}`")
        lines.append(f"- Note: {row['paper_facing_note']}")
        lines.append(f"- Anchor artifact: `{row['anchor']['artifact']}`")
        lines.append(f"- Promoted artifact: `{row['promoted']['artifact']}`")
        lines.append(f"- External artifact: `{row['external']['artifact']}`")
        if "subset_policy_note" in row:
            subset = row["subset_policy_note"]
            lines.append(
                f"- Subset policy note: random_nonprefix={_fmt(subset['random_nonprefix_accuracy'])}, verifier_uncertainty_first={_fmt(subset['verifier_uncertainty_first_accuracy'])}, delta={_fmt(subset['delta_policy_probe'])}. {subset['note']}"
            )
        if "reported_supporting_external_family" in row:
            pairrm = row["reported_supporting_external_family"]
            lines.append(
                f"- Reported-only external family support: {pairrm['label']}={_fmt(pairrm['accuracy'])} (`{pairrm['provenance']}`; raw standalone PairRM artifact is not present in the current workspace)."
            )
        lines.append("")

    lines.extend([
        "## Assessment",
        "",
        f"- Overall Gate C verdict: {payload['assessment']['verdict']}",
        f"- MCQ frontier: {payload['assessment']['frontier_shape']['mcq']}",
        f"- Numeric frontier: {payload['assessment']['frontier_shape']['numeric']}",
        "- Reconciliation points:",
    ])
    for item in payload["assessment"]["important_reconciliation"]:
        lines.append(f"  - {item}")

    lines.extend([
        "",
        "## Canonical Inputs",
        "",
    ])
    for key, value in payload["canonical_inputs"].items():
        if value is None:
            continue
        lines.append(f"- `{key}`: `{value}`")

    output_json = _project_path(args.output_json)
    output_md = _project_path(args.output_md)
    _write_json_atomic(output_json, payload)
    _write_text_atomic(output_md, "\n".join(lines) + "\n")
    print(output_md.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
