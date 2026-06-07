from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from src.eval.evaluate_predictions import answers_match  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Route C+ oracle-to-final conversion audit pack.")
    parser.add_argument("--gsm8k-leaderboard-json", default="Experiment/analysis/results/routec_plus_policy_gsm8k_full_clean_leaderboard_v1.json")
    parser.add_argument("--gsm8k-anchor-preds", default="Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_random_nonprefix_v1_verifier_predictions.jsonl")
    parser.add_argument("--gsm8k-promoted-preds", default="Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_verifier_uncertainty_first_v1_verifier_predictions.jsonl")
    parser.add_argument("--gsm8k-external-preds", default="Experiment/core_code/logs/a800_self_refine_gsm8k_full_p1b_v1_predictions.jsonl")
    parser.add_argument("--gsm8k-context-json", default="Experiment/analysis/results/experiment_11_p1a_verifier_conversion_diagnosis_v1.json")
    parser.add_argument("--compmath-anchor-json", default="Experiment/analysis/results/routec_plus_policy_fullconfirm_v2_competition_math_numeric_random_nonprefix_v1.json")
    parser.add_argument("--compmath-promoted-json", default="Experiment/analysis/results/routec_plus_policy_fullconfirm_competition_math_numeric_hybrid_salvageability_v1.json")
    parser.add_argument("--compmath-promoted-preds", default="Experiment/core_code/logs/routec_plus_policy_fullconfirm_competition_math_numeric_hybrid_salvageability_v1_verifier_predictions.jsonl")
    parser.add_argument("--compmath-external-preds", default="Experiment/core_code/logs/a800_self_refine_competition_math_numeric_test_transfer_b32_v2_predictions.jsonl")
    parser.add_argument("--self-refine-summary-json", default="Experiment/analysis/results/a800_self_refine_three_benchmark_transfer_b32_v2_summary.json")
    parser.add_argument("--mmlu-anchor-json", default="Experiment/analysis/results/routec_p1_mmlu_pro_cacc_v1.json")
    parser.add_argument("--mmlu-promoted-json", default="Experiment/analysis/results/routec_p1_mmlu_pro_cacc_sp_v1.json")
    parser.add_argument("--mmlu-anchor-preds", default="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_v1_verifier_predictions.jsonl")
    parser.add_argument("--mmlu-promoted-preds", default="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_sp_v1_verifier_predictions.jsonl")
    parser.add_argument("--mmlu-external-preds", default="Experiment/core_code/logs/a800_self_refine_mmlu_pro_test_transfer_b32_v2_predictions.jsonl")
    parser.add_argument("--output-json", default="Experiment/analysis/results/routec_plus_oracle_final_conversion_audit_pack_v1.json")
    parser.add_argument("--output-md", default="Experiment/analysis/results/routec_plus_oracle_final_conversion_audit_pack_v1.md")
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
    raise KeyError(f"missing policy in leaderboard: {policy}")


def _load_predictions(path: str | Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    with _project_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prediction = str(row.get("prediction", ""))
            gold = str(row.get("gold_answer", ""))
            answer_mode = str(row.get("answer_mode", "numeric"))
            correct = bool(row.get("correct")) if "correct" in row else answers_match(prediction, gold, answer_mode=answer_mode)
            output[str(row["example_id"])] = {
                "example_id": str(row["example_id"]),
                "prediction": prediction,
                "gold_answer": gold,
                "answer_mode": answer_mode,
                "correct": correct,
            }
    return output


def _paired_counts(path_a: str | Path, path_b: str | Path) -> dict:
    map_a = _load_predictions(path_a)
    map_b = _load_predictions(path_b)
    shared_ids = sorted(set(map_a) & set(map_b))
    a_only = b_only = both = neither = 0
    a_only_examples = []
    b_only_examples = []
    for example_id in shared_ids:
        row_a = map_a[example_id]
        row_b = map_b[example_id]
        a_correct = bool(row_a["correct"])
        b_correct = bool(row_b["correct"])
        if a_correct and b_correct:
            both += 1
        elif a_correct and not b_correct:
            a_only += 1
            if len(a_only_examples) < 3:
                a_only_examples.append({
                    "example_id": example_id,
                    "winner_prediction": row_a["prediction"][:200],
                    "loser_prediction": row_b["prediction"][:200],
                    "gold": row_a["gold_answer"],
                })
        elif not a_correct and b_correct:
            b_only += 1
            if len(b_only_examples) < 3:
                b_only_examples.append({
                    "example_id": example_id,
                    "winner_prediction": row_b["prediction"][:200],
                    "loser_prediction": row_a["prediction"][:200],
                    "gold": row_a["gold_answer"],
                })
        else:
            neither += 1
    n = len(shared_ids)
    return {
        "n": n,
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "delta_b_minus_a": (b_only - a_only) / n if n else 0.0,
        "a_only_examples": a_only_examples,
        "b_only_examples": b_only_examples,
    }


def _exact_decomposition(o1: float, f1: float, v1: float, o2: float, f2: float, v2: float) -> dict:
    coverage = (o2 - o1) * v1
    conversion = o2 * (v2 - v1)
    delta = f2 - f1
    return {
        "delta_final": delta,
        "coverage_contribution": coverage,
        "conversion_contribution": conversion,
        "reconstructed_delta": coverage + conversion,
    }


def _benchmark_tag(delta_external: float, conv_contrib: float) -> str:
    if delta_external > 0 and conv_contrib > 0:
        return "both_positive_and_external_win"
    if delta_external > 0:
        return "coverage_positive_external_win"
    if conv_contrib < 0:
        return "conversion_limited"
    if delta_external >= -0.015:
        return "near_frontier_small_positive_conversion"
    return "external_gap_dominated"


def _gsm8k_section(args: argparse.Namespace) -> dict:
    leaderboard = _load_json(args.gsm8k_leaderboard_json)
    context = _load_json(args.gsm8k_context_json) if _project_path(args.gsm8k_context_json).exists() else None
    anchor = _leaderboard_row(leaderboard, "replace_random_nonprefix")
    promoted = _leaderboard_row(leaderboard, "replace_verifier_uncertainty_first")
    external_predictions = _load_predictions(args.gsm8k_external_preds)
    external_accuracy = sum(int(bool(r["correct"])) for r in external_predictions.values()) / len(external_predictions)

    anchor_oracle = anchor["oracle_coverage"]
    anchor_final = anchor["verifier_accuracy"]
    anchor_vgo = anchor["verifier_given_oracle"]
    promoted_oracle = promoted["oracle_coverage"]
    promoted_final = promoted["verifier_accuracy"]
    promoted_vgo = promoted["verifier_given_oracle"]

    decomposition = _exact_decomposition(anchor_oracle, anchor_final, anchor_vgo, promoted_oracle, promoted_final, promoted_vgo)
    paired_anchor_promoted = _paired_counts(args.gsm8k_anchor_preds, args.gsm8k_promoted_preds)
    paired_promoted_external = _paired_counts(args.gsm8k_promoted_preds, args.gsm8k_external_preds)

    return {
        "benchmark": "gsm8k_full_clean",
        "regime": "numeric_open_ended",
        "anchor": {
            "label": "random_nonprefix",
            "oracle": anchor_oracle,
            "final": anchor_final,
            "vgo": anchor_vgo,
            "parseable": anchor.get("selected_parseable"),
            "invalid": anchor.get("invalid_final"),
            "source": f"{args.gsm8k_leaderboard_json}#policy=replace_random_nonprefix",
        },
        "promoted": {
            "label": "verifier_uncertainty_first",
            "oracle": promoted_oracle,
            "final": promoted_final,
            "vgo": promoted_vgo,
            "parseable": promoted.get("selected_parseable"),
            "invalid": promoted.get("invalid_final"),
            "source": f"{args.gsm8k_leaderboard_json}#policy=replace_verifier_uncertainty_first",
        },
        "external": {
            "label": "Self-Refine",
            "final": external_accuracy,
            "source": str(Path(args.gsm8k_external_preds)),
        },
        "decomposition": decomposition,
        "paired": {
            "anchor_vs_promoted": paired_anchor_promoted,
            "promoted_vs_external": paired_promoted_external,
        },
        "historical_context": context,
        "verdict": {
            "tag": _benchmark_tag(promoted_final - external_accuracy, decomposition["conversion_contribution"]),
            "summary": "GSM8K promoted row is still the strongest numeric blocker: final gain over internal anchor is real, but almost all of it comes from additional coverage while conversion contribution is negative.",
            "next_step": "Do selector-side calibration / bucket-level external audit before any new candidate-side full rerun.",
        },
    }


def _summary_section(anchor: dict, promoted: dict, external_accuracy: float, benchmark: str, regime: str, source_anchor: str, source_promoted: str, external_source: str, *, paired_anchor_promoted: dict | None, paired_promoted_external: dict | None, note: str) -> dict:
    vgo_anchor = anchor.get("verifier_given_oracle", anchor.get("selection_efficiency_given_oracle", {}).get("verifier"))
    vgo_promoted = promoted.get("verifier_given_oracle", promoted.get("selection_efficiency_given_oracle", {}).get("verifier"))
    decomposition = _exact_decomposition(anchor["oracle_coverage"], anchor["verifier_accuracy"], vgo_anchor, promoted["oracle_coverage"], promoted["verifier_accuracy"], vgo_promoted)
    return {
        "benchmark": benchmark,
        "regime": regime,
        "anchor": {
            "label": anchor["run_label"],
            "oracle": anchor["oracle_coverage"],
            "final": anchor["verifier_accuracy"],
            "vgo": vgo_anchor,
            "source": source_anchor,
        },
        "promoted": {
            "label": promoted["run_label"],
            "oracle": promoted["oracle_coverage"],
            "final": promoted["verifier_accuracy"],
            "vgo": vgo_promoted,
            "source": source_promoted,
        },
        "external": {
            "label": "Self-Refine",
            "final": external_accuracy,
            "source": external_source,
        },
        "decomposition": decomposition,
        "paired": {
            "anchor_vs_promoted": paired_anchor_promoted,
            "promoted_vs_external": paired_promoted_external,
        },
        "note": note,
    }


def _compmath_section(args: argparse.Namespace, self_refine_summary: dict) -> dict:
    anchor = _load_json(args.compmath_anchor_json)
    promoted = _load_json(args.compmath_promoted_json)
    external_accuracy = self_refine_summary["benchmarks"]["competition_math_numeric"]["accuracy"]
    paired_promoted_external = _paired_counts(args.compmath_promoted_preds, args.compmath_external_preds)
    section = _summary_section(
        anchor,
        promoted,
        external_accuracy,
        "competition_math_numeric",
        "numeric_open_ended",
        str(Path(args.compmath_anchor_json)),
        str(Path(args.compmath_promoted_json)),
        str(Path(args.compmath_external_preds)),
        paired_anchor_promoted=None,
        paired_promoted_external=paired_promoted_external,
        note="Current workspace lacks random_nonprefix verifier predictions for a true paired anchor->promoted audit on Competition-Math, so the anchor->promoted diagnosis remains aggregate-only while promoted->external is paired.",
    )
    section["verdict"] = {
        "tag": _benchmark_tag(section["promoted"]["final"] - external_accuracy, section["decomposition"]["conversion_contribution"]),
        "summary": "Competition-Math is the cleanest near-frontier numeric regime: both coverage and conversion contributions are positive, and the remaining gap to Self-Refine is small.",
        "next_step": "Prefer bucket-level external audit over a new full rerun; current evidence does not justify making this the top execution priority.",
    }
    return section


def _mmlu_section(args: argparse.Namespace, self_refine_summary: dict) -> dict:
    anchor = _load_json(args.mmlu_anchor_json)
    promoted = _load_json(args.mmlu_promoted_json)
    external_accuracy = self_refine_summary["benchmarks"]["mmlu_pro"]["accuracy"]
    paired_anchor_promoted = _paired_counts(args.mmlu_anchor_preds, args.mmlu_promoted_preds)
    paired_promoted_external = _paired_counts(args.mmlu_promoted_preds, args.mmlu_external_preds)
    section = _summary_section(
        anchor,
        promoted,
        external_accuracy,
        "mmlu_pro_test",
        "mcq_compatibility_limited",
        str(Path(args.mmlu_anchor_json)),
        str(Path(args.mmlu_promoted_json)),
        str(Path(args.mmlu_external_preds)),
        paired_anchor_promoted=paired_anchor_promoted,
        paired_promoted_external=paired_promoted_external,
        note="For MMLU-Pro the current full frontier row should still be read as CACC -> CACC+SP. This is the landed full MCQ strengthening row, not the later subset-only policy probe.",
    )
    section["verdict"] = {
        "tag": _benchmark_tag(section["promoted"]["final"] - external_accuracy, section["decomposition"]["conversion_contribution"]),
        "summary": "MMLU-Pro is no longer a main blocker: both coverage and conversion contributions are strongly positive, and the landed promoted row beats Self-Refine on the full split.",
        "next_step": "Do not spend the next full experiment here unless the goal is appendix strengthening; current ROI is lower than GSM8K selector-side closure.",
    }
    return section


def main() -> None:
    args = parse_args()
    self_refine_summary = _load_json(args.self_refine_summary_json)

    sections = [
        _gsm8k_section(args),
        _compmath_section(args, self_refine_summary),
        _mmlu_section(args, self_refine_summary),
    ]

    strongest_blocker = max(
        sections,
        key=lambda row: (
            row["verdict"]["tag"] == "conversion_limited",
            abs(row["promoted"]["final"] - row["external"]["final"]),
        ),
    )

    payload = {
        "title": "Route C+ oracle-to-final conversion audit pack",
        "status": "completed",
        "sections": sections,
        "assessment": {
            "strongest_blocker": strongest_blocker["benchmark"],
            "global_verdict": "The remaining high-ROI blocker is not MCQ anymore. It is numeric conversion closure, especially GSM8K, where coverage gain is real but the promoted row still sacrifices conversion.",
            "recommended_next_step": "Prioritize GSM8K selector-side calibration or bucket-level external audit before launching any new full candidate-generation sweep.",
            "deprioritized_steps": [
                "Do not start a new MMLU-Pro full rerun as the next action.",
                "Do not prioritize Competition-Math full rerun before a GSM8K conversion-side audit.",
                "Do not continue blind candidate-side expansion until selector-side conversion issues are audited on GSM8K.",
            ],
        },
    }

    lines = [
        "# Route C+ Oracle-to-Final Conversion Audit Pack",
        "",
        "- Status: `completed`",
        f"- Global verdict: {payload['assessment']['global_verdict']}",
        f"- Recommended next step: {payload['assessment']['recommended_next_step']}",
        "",
        "## Cross-Regime Scorecard",
        "",
        "| Benchmark | Regime | Anchor Final | Promoted Final | External Final | ΔPromoted-Anchor | ΔPromoted-External | ΔOracle | ΔV|O | Coverage Contrib | Conversion Contrib | Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in sections:
        delta_oracle = row["promoted"]["oracle"] - row["anchor"]["oracle"]
        delta_vgo = row["promoted"]["vgo"] - row["anchor"]["vgo"]
        lines.append(
            f"| {row['benchmark']} | {row['regime']} | {_fmt(row['anchor']['final'])} | {_fmt(row['promoted']['final'])} | {_fmt(row['external']['final'])} | {_fmt(row['decomposition']['delta_final'])} | {_fmt(row['promoted']['final'] - row['external']['final'])} | {_fmt(delta_oracle)} | {_fmt(delta_vgo)} | {_fmt(row['decomposition']['coverage_contribution'])} | {_fmt(row['decomposition']['conversion_contribution'])} | `{row['verdict']['tag']}` |"
        )

    for row in sections:
        lines.extend([
            "",
            f"## {row['benchmark']}",
            "",
            f"- Regime: `{row['regime']}`",
            f"- Summary: {row['verdict']['summary']}",
            f"- Next step: {row['verdict']['next_step']}",
            f"- Note: {row.get('note', 'NA')}",
            f"- Anchor source: `{row['anchor']['source']}`",
            f"- Promoted source: `{row['promoted']['source']}`",
            f"- External source: `{row['external']['source']}`",
            "",
            "### Exact Decomposition",
            "",
            f"- Delta final: `{_fmt(row['decomposition']['delta_final'])}`",
            f"- Coverage contribution: `{_fmt(row['decomposition']['coverage_contribution'])}`",
            f"- Conversion contribution: `{_fmt(row['decomposition']['conversion_contribution'])}`",
            f"- Reconstructed delta check: `{_fmt(row['decomposition']['reconstructed_delta'])}`",
            "",
        ])
        if row['paired'].get('anchor_vs_promoted') is not None:
            paired = row['paired']['anchor_vs_promoted']
            lines.extend([
                "### Paired Anchor vs Promoted",
                "",
                f"- n=`{paired['n']}`, anchor-only=`{paired['a_only']}`, promoted-only=`{paired['b_only']}`, both=`{paired['both']}`, neither=`{paired['neither']}`",
                f"- Delta promoted-anchor from paired outcomes: `{_fmt(paired['delta_b_minus_a'])}`",
                "",
            ])
        paired_ext = row['paired'].get('promoted_vs_external')
        if paired_ext is not None:
            lines.extend([
                "### Paired Promoted vs Strongest External",
                "",
                f"- n=`{paired_ext['n']}`, promoted-only=`{paired_ext['a_only']}`, external-only=`{paired_ext['b_only']}`, both=`{paired_ext['both']}`, neither=`{paired_ext['neither']}`",
                f"- Delta external-promoted from paired outcomes: `{_fmt(-paired_ext['delta_b_minus_a'])}` for promoted; `{_fmt(paired_ext['delta_b_minus_a'])}` for external",
            ])
            if paired_ext['b_only_examples']:
                lines.append("- Example external-only wins:")
                for ex in paired_ext['b_only_examples']:
                    lines.append(f"  - `{ex['example_id']}` gold=`{ex['gold']}`")
            if paired_ext['a_only_examples']:
                lines.append("- Example promoted-only wins:")
                for ex in paired_ext['a_only_examples']:
                    lines.append(f"  - `{ex['example_id']}` gold=`{ex['gold']}`")
            lines.append("")

    lines.extend([
        "## Final Decision",
        "",
        f"- Strongest remaining blocker: `{payload['assessment']['strongest_blocker']}`",
        f"- Decision: {payload['assessment']['recommended_next_step']}",
        "- Deprioritized steps:",
    ])
    for item in payload['assessment']['deprioritized_steps']:
        lines.append(f"  - {item}")

    output_json = _project_path(args.output_json)
    output_md = _project_path(args.output_md)
    _write_json_atomic(output_json, payload)
    _write_text_atomic(output_md, "\n".join(lines) + "\n")
    print(output_md.relative_to(PROJECT_ROOT))


if __name__ == '__main__':
    main()
