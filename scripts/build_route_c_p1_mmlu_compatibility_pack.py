#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_neurips_p0_readiness_pack as p0  # type: ignore

ROOT = Path(__file__).resolve().parents[3]
PACK_DIR = ROOT / "Publication" / "paper" / "route_c_pack_v1"

RUN_SPECS = [
    p0.RunSpec(
        run_id="routec_p1_mmlu_pro_ccr_only_v1",
        benchmark="mmlu_pro",
        split="test",
        variant_name="routec_ccr_only_v1",
        paper_variant="CCR-only",
        role="route_c_p1_compatibility",
        result_json="Experiment/analysis/results/routec_p1_mmlu_pro_ccr_only_v1.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_v1.jsonl",
        first_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_only_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_only_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_only_v1_verifier_predictions.jsonl",
        notes=(
            "Route C P1-3 fixed-pool rerun on MMLU-Pro.",
            "This row isolates completion-centric replacement without answer-mode compatibility fixes.",
        ),
    ),
    p0.RunSpec(
        run_id="routec_p1_mmlu_pro_ccr_amc_v1",
        benchmark="mmlu_pro",
        split="test",
        variant_name="routec_ccr_amc_v1",
        paper_variant="CCR+AMC",
        role="route_c_p1_compatibility",
        result_json="Experiment/analysis/results/routec_p1_mmlu_pro_ccr_amc_v1.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_v2.jsonl",
        first_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_amc_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_amc_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_amc_v1_verifier_predictions.jsonl",
        notes=(
            "AMC = Answer-Mode Compatibility.",
            "This row adds answer-mode lock and metadata-preserving construction on top of CCR.",
        ),
    ),
    p0.RunSpec(
        run_id="routec_p1_mmlu_pro_cacc_v1",
        benchmark="mmlu_pro",
        split="test",
        variant_name="routec_cacc_v1",
        paper_variant="CACC",
        role="route_c_p1_compatibility",
        result_json="Experiment/analysis/results/routec_p1_mmlu_pro_cacc_v1.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_v3.jsonl",
        first_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_v1_verifier_predictions.jsonl",
        notes=(
            "CACC = CCR + AMC + SCH.",
            "This row is the canonical Route C method without stronger proposer augmentation.",
        ),
    ),
    p0.RunSpec(
        run_id="routec_p1_mmlu_pro_cacc_sp_v1",
        benchmark="mmlu_pro",
        split="test",
        variant_name="routec_cacc_sp_v1",
        paper_variant="CACC+SP",
        role="route_c_p1_compatibility",
        result_json="Experiment/analysis/results/routec_p1_mmlu_pro_cacc_sp_v1.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_qwen8bproposer_v2.jsonl",
        first_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_sp_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_sp_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_sp_v1_verifier_predictions.jsonl",
        notes=(
            "SP = Stronger Proposer.",
            "This row is an auxiliary strengthening path rather than the minimal canonical method.",
        ),
    ),
]


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _delta(new_value: float | None, old_value: float | None) -> str:
    if new_value is None or old_value is None:
        return "NA"
    return f"{new_value - old_value:+.4f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _reparsed_vgo(audit: dict) -> float | None:
    oracle = audit["reparsed"].get("oracle_accuracy")
    verifier = audit["reparsed"].get("verifier_accuracy")
    if oracle in (None, 0) or verifier is None:
        return None
    return verifier / oracle


def build_markdown(audits: list[dict]) -> str:
    by_variant = {audit["paper_facing_variant_name"]: audit for audit in audits}
    ccr_only = by_variant["CCR-only"]
    ccr_amc = by_variant["CCR+AMC"]
    cacc = by_variant["CACC"]
    cacc_sp = by_variant["CACC+SP"]

    main_rows = []
    hygiene_rows = []
    for name in ["CCR-only", "CCR+AMC", "CACC", "CACC+SP"]:
        audit = by_variant[name]
        rep = audit["reparsed"]
        hyg = audit["hygiene"]
        main_rows.append([
            name,
            _fmt(rep["first_accuracy"]),
            _fmt(rep["base_accuracy"]),
            _fmt(rep["verifier_accuracy"]),
            _fmt(rep["oracle_accuracy"]),
            _fmt(_reparsed_vgo(audit)),
            _fmt(hyg.get("selected_prediction_parseable_rate")),
            _fmt(hyg.get("invalid_final_answer_rate")),
            _fmt(hyg.get("instruction_leak_rate")),
            _fmt(hyg.get("scaffold_residue_rate")),
            _fmt(hyg.get("duplicate_example_rate")),
        ])
        hygiene_rows.append([
            name,
            _fmt(hyg.get("candidate_parseable_rate")),
            _fmt(hyg.get("selected_prediction_parseable_rate")),
            _fmt(hyg.get("selected_prediction_answer_mode_match_rate")),
            _fmt(hyg.get("invalid_final_answer_rate")),
            _fmt(hyg.get("instruction_leak_rate")),
            _fmt(hyg.get("scaffold_residue_rate")),
            _fmt(hyg.get("malformed_selected_rate")),
            _fmt(hyg.get("duplicate_slot_rate")),
            _fmt(hyg.get("duplicate_example_rate")),
        ])

    transition_rows = [
        [
            "CCR-only -> CCR+AMC",
            _delta(ccr_amc["reparsed"]["oracle_accuracy"], ccr_only["reparsed"]["oracle_accuracy"]),
            _delta(ccr_amc["reparsed"]["verifier_accuracy"], ccr_only["reparsed"]["verifier_accuracy"]),
            _delta(_reparsed_vgo(ccr_amc), _reparsed_vgo(ccr_only)),
            _delta(ccr_amc["hygiene"].get("selected_prediction_parseable_rate"), ccr_only["hygiene"].get("selected_prediction_parseable_rate")),
            _delta(ccr_amc["hygiene"].get("invalid_final_answer_rate"), ccr_only["hygiene"].get("invalid_final_answer_rate")),
            _delta(ccr_amc["hygiene"].get("instruction_leak_rate"), ccr_only["hygiene"].get("instruction_leak_rate")),
            _delta(ccr_amc["hygiene"].get("scaffold_residue_rate"), ccr_only["hygiene"].get("scaffold_residue_rate")),
        ],
        [
            "CCR+AMC -> CACC",
            _delta(cacc["reparsed"]["oracle_accuracy"], ccr_amc["reparsed"]["oracle_accuracy"]),
            _delta(cacc["reparsed"]["verifier_accuracy"], ccr_amc["reparsed"]["verifier_accuracy"]),
            _delta(_reparsed_vgo(cacc), _reparsed_vgo(ccr_amc)),
            _delta(cacc["hygiene"].get("selected_prediction_parseable_rate"), ccr_amc["hygiene"].get("selected_prediction_parseable_rate")),
            _delta(cacc["hygiene"].get("invalid_final_answer_rate"), ccr_amc["hygiene"].get("invalid_final_answer_rate")),
            _delta(cacc["hygiene"].get("instruction_leak_rate"), ccr_amc["hygiene"].get("instruction_leak_rate")),
            _delta(cacc["hygiene"].get("scaffold_residue_rate"), ccr_amc["hygiene"].get("scaffold_residue_rate")),
        ],
        [
            "CACC -> CACC+SP",
            _delta(cacc_sp["reparsed"]["oracle_accuracy"], cacc["reparsed"]["oracle_accuracy"]),
            _delta(cacc_sp["reparsed"]["verifier_accuracy"], cacc["reparsed"]["verifier_accuracy"]),
            _delta(_reparsed_vgo(cacc_sp), _reparsed_vgo(cacc)),
            _delta(cacc_sp["hygiene"].get("selected_prediction_parseable_rate"), cacc["hygiene"].get("selected_prediction_parseable_rate")),
            _delta(cacc_sp["hygiene"].get("invalid_final_answer_rate"), cacc["hygiene"].get("invalid_final_answer_rate")),
            _delta(cacc_sp["hygiene"].get("instruction_leak_rate"), cacc["hygiene"].get("instruction_leak_rate")),
            _delta(cacc_sp["hygiene"].get("scaffold_residue_rate"), cacc["hygiene"].get("scaffold_residue_rate")),
        ],
    ]

    samples = cacc_sp["error_taxonomy"]["metadata_samples"]
    outcome_samples = cacc_sp["error_taxonomy"]["outcome_samples"]

    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""# Route C P1-3 MMLU-Pro Compatibility Readout

Generated at {generated_at}.

## Scope

- Benchmark: `MMLU-Pro`
- Experiment family: `P1-3 Without vs With AMC / SCH`
- Fixed-pool variants: `CCR-only`, `CCR+AMC`, `CACC`, `CACC+SP`
- Metric source: rerun artifacts under `routec_p1_mmlu_pro_*`
- Parser / hygiene audit: benchmark-locked `choice_letter` mode using the same parsing rules as the NeurIPS readiness pack

## Main Scoreboard

{_table(["variant", "first", "base", "verifier", "oracle", "verifier_given_oracle", "selected_parseable", "invalid_final", "instruction_leak", "scaffold_residue", "duplicate_example"], main_rows)}

## Transition Readout

{_table(["transition", "delta_oracle", "delta_verifier", "delta_verifier_given_oracle", "delta_selected_parseable", "delta_invalid_final", "delta_instruction_leak", "delta_scaffold_residue"], transition_rows)}

## Hygiene Detail

{_table(["variant", "candidate_parseable", "selected_parseable", "answer_mode_match", "invalid_final", "instruction_leak", "scaffold_residue", "malformed_selected", "duplicate_slot", "duplicate_example"], hygiene_rows)}

## Key Findings

1. `CCR-only` on MMLU-Pro is a genuine compatibility-limited failure regime rather than a selector-only problem. Its `oracle` is only {_fmt(ccr_only["reparsed"]["oracle_accuracy"])} and `verifier` is only {_fmt(ccr_only["reparsed"]["verifier_accuracy"])}.
2. Adding `AMC` repairs the collapse immediately. Relative to `CCR-only`, `CCR+AMC` improves `oracle` by {_delta(ccr_amc["reparsed"]["oracle_accuracy"], ccr_only["reparsed"]["oracle_accuracy"])} and `verifier` by {_delta(ccr_amc["reparsed"]["verifier_accuracy"], ccr_only["reparsed"]["verifier_accuracy"])}.
3. The minimal canonical method `CACC` does not deliver a fresh accuracy jump over `CCR+AMC`; its value in this run is mainly hygiene stabilization rather than additional conversion. This means `SCH` should be claimed as a compatibility-preserving cleanup module, not as a standalone headline gain on MMLU-Pro.
4. `CACC+SP` is the strongest row in this chain. It reaches `oracle={_fmt(cacc_sp["reparsed"]["oracle_accuracy"])} / verifier={_fmt(cacc_sp["reparsed"]["verifier_accuracy"])} / verifier_given_oracle={_fmt(_reparsed_vgo(cacc_sp))}`.
5. Even after compatibility repair, this row is not paper-grade hygiene yet. The best selected parseability here remains below 1.0 and duplicate-example rate remains high, so the Route C claim must stay centered on mechanism rather than \"fully solved evaluation hygiene\".

## Route C Interpretation

- This experiment gives strong evidence for a coverage-versus-compatibility tradeoff on MCQ benchmarks, not a simple monotonic win from compatibility alone.
- `AMC` clearly fixes answer-mode lock and improves oracle coverage plus selected parseability, but it does not improve final verifier accuracy over reparsed `CCR-only` on this benchmark.
- `SCH` is still useful because it sharply reduces instruction leakage and scaffold residue while keeping accuracy roughly flat.
- `SP` remains the strongest strengthening path for MMLU-Pro. It can be used as the best-performing auxiliary row, but it also reopens some parseability loss, so the Route C story should stay mechanism-first rather than leaderboard-first.

## Paper-Safe Claims After P1-3

- Safe: MCQ failure in open-ended candidate construction can be compatibility-limited rather than purely selector-limited.
- Safe: answer-mode compatibility improves oracle coverage and selected-answer validity on MMLU-Pro.
- Safe: hygiene cleanup can reduce instruction/scaffold contamination without hurting final accuracy.
- Safe: stronger proposers help once a compatible pool exists, but they do not remove the compatibility tradeoff.
- Not yet safe: `AMC` alone improves final verifier accuracy on MMLU-Pro.
- Not yet safe: `SCH` alone yields a robust accuracy gain on MMLU-Pro.
- Not yet safe: the current Route C pipeline has already solved final-answer hygiene on MCQ benchmarks.

## Representative Examples From The Strongest Row

### Invalid selected answers

{json.dumps(samples.get("invalid_selected_answer", [])[:2], ensure_ascii=False, indent=2)}

### Oracle miss examples

{json.dumps(outcome_samples.get("oracle_miss", [])[:2], ensure_ascii=False, indent=2)}

## Go / No-Go

- `P1-3` status: **mixed-positive for Route C**
- Reason: this run strongly supports the tradeoff / compatibility-regime story, but it does not support a simple claim that compatibility alone raises final verifier accuracy on MMLU-Pro.
- Immediate next step: move to `P1-4 coverage vs selector-compatibility tradeoff` and write a benchmark-level regime table that combines GSM8K clean, P1-2 negative evidence, and this MMLU-Pro compatibility chain.
"""


def build_summary(audits: list[dict]) -> dict:
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "benchmark": "mmlu_pro", "variants": {}}
    for audit in audits:
        summary["variants"][audit["paper_facing_variant_name"]] = {
            "run_id": audit["run_id"],
            "reparsed": audit["reparsed"],
            "reported": {
                "verifier_given_oracle_accuracy": _reparsed_vgo(audit),
                "failure_decomposition": audit["reported"].get("failure_decomposition"),
                "rescue_counts": audit["reported"].get("rescue_counts"),
            },
            "hygiene": audit["hygiene"],
        }
    return summary


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    audits = [p0.audit_run(spec) for spec in RUN_SPECS]
    order = ["CCR-only", "CCR+AMC", "CACC", "CACC+SP"]
    audits.sort(key=lambda audit: order.index(audit["paper_facing_variant_name"]))

    scoreboard_rows = []
    for audit in audits:
        rep = audit["reparsed"]
        hyg = audit["hygiene"]
        scoreboard_rows.append({
            "variant": audit["paper_facing_variant_name"],
            "run_id": audit["run_id"],
            "first_accuracy": rep["first_accuracy"],
            "base_accuracy": rep["base_accuracy"],
            "verifier_accuracy": rep["verifier_accuracy"],
            "oracle_accuracy": rep["oracle_accuracy"],
            "verifier_given_oracle_accuracy": _reparsed_vgo(audit),
            "selected_parseable_rate": hyg.get("selected_prediction_parseable_rate"),
            "invalid_final_answer_rate": hyg.get("invalid_final_answer_rate"),
            "instruction_leak_rate": hyg.get("instruction_leak_rate"),
            "scaffold_residue_rate": hyg.get("scaffold_residue_rate"),
            "malformed_selected_rate": hyg.get("malformed_selected_rate"),
            "duplicate_slot_rate": hyg.get("duplicate_slot_rate"),
            "duplicate_example_rate": hyg.get("duplicate_example_rate"),
        })

    summary = build_summary(audits)
    markdown = build_markdown(audits)
    p0.atomic_write_json(PACK_DIR / "route_c_p1_mmlu_compatibility_summary.json", summary)
    p0.atomic_write_text(PACK_DIR / "route_c_p1_mmlu_compatibility_readout.md", markdown)
    write_csv(
        PACK_DIR / "route_c_p1_mmlu_compatibility_scoreboard.csv",
        scoreboard_rows,
        list(scoreboard_rows[0].keys()),
    )
    print(json.dumps({
        "generated": [
            str(PACK_DIR / "route_c_p1_mmlu_compatibility_summary.json"),
            str(PACK_DIR / "route_c_p1_mmlu_compatibility_readout.md"),
            str(PACK_DIR / "route_c_p1_mmlu_compatibility_scoreboard.csv"),
        ],
        "best_variant": "CACC+SP",
        "canonical_variant": "CACC",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
