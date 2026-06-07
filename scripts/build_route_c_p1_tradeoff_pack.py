#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_neurips_p0_readiness_pack as p0  # type: ignore

ROOT = Path(__file__).resolve().parents[3]
PACK_DIR = ROOT / "Publication" / "paper" / "route_c_pack_v1"

EXISTING = {spec.run_id: spec for spec in p0.RUN_SPECS}


def clone_spec(
    base_run_id: str,
    *,
    run_id: str,
    variant_name: str,
    paper_variant: str,
    role: str,
    result_json: str,
    first_predictions: str,
    base_predictions: str,
    verifier_predictions: str,
    notes: tuple[str, ...],
) -> p0.RunSpec:
    base = EXISTING[base_run_id]
    return replace(
        base,
        run_id=run_id,
        variant_name=variant_name,
        paper_variant=paper_variant,
        role=role,
        result_json=result_json,
        first_predictions=first_predictions,
        base_predictions=base_predictions,
        verifier_predictions=verifier_predictions,
        notes=notes,
    )


RUN_SPECS = [
    EXISTING["gsm8k_full_hybridp6_clean_p1a_v1"],
    EXISTING["gsm8k_full_completion_hybridp6_clean_p1a_v1"],
    p0.RunSpec(
        run_id="routec_p1_gsm8k_full_clean_ccr_randomslot_v1",
        benchmark="gsm8k",
        split="test_full_clean",
        variant_name="routec_p1_gsm8k_full_clean_ccr_randomslot_v1",
        paper_variant="CCR-random-slot",
        role="route_c_p1_targeted_vs_random",
        result_json="Experiment/analysis/results/routec_p1_gsm8k_full_clean_ccr_randomslot_v1.json",
        generation_json="Experiment/analysis/results/routec_p1_gsm8k_full_clean_ccr_randomslot_v1_generation.json",
        candidate_pool="Experiment/datasets/processed/routec_p1_gsm8k_full_clean_ccr_randomslot_v1_candidates.jsonl",
        first_predictions="Experiment/core_code/logs/routec_p1_gsm8k_full_clean_ccr_randomslot_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_gsm8k_full_clean_ccr_randomslot_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_gsm8k_full_clean_ccr_randomslot_v1_verifier_predictions.jsonl",
        notes=(
            "Route C P1-2 control row on GSM8K full clean.",
            "Uses random non-prefix replacement to test whether current targeted replacement is genuinely necessary.",
        ),
    ),
    p0.RunSpec(
        run_id="routec_p1_gsm8k_full_clean_ccr_targeted_v1",
        benchmark="gsm8k",
        split="test_full_clean",
        variant_name="routec_p1_gsm8k_full_clean_ccr_targeted_v1",
        paper_variant="CCR-targeted",
        role="route_c_p1_targeted_vs_random",
        result_json="Experiment/analysis/results/routec_p1_gsm8k_full_clean_ccr_targeted_v1.json",
        generation_json="Experiment/analysis/results/routec_p1_gsm8k_full_clean_ccr_targeted_v1_generation.json",
        candidate_pool="Experiment/datasets/processed/routec_p1_gsm8k_full_clean_ccr_targeted_v1_candidates.jsonl",
        first_predictions="Experiment/core_code/logs/routec_p1_gsm8k_full_clean_ccr_targeted_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_gsm8k_full_clean_ccr_targeted_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_gsm8k_full_clean_ccr_targeted_v1_verifier_predictions.jsonl",
        notes=(
            "Route C P1-2 intended mechanism row on GSM8K full clean.",
            "Current fragment-first targeted replacement underperforms the random-slot control and should not be promoted.",
        ),
    ),
    EXISTING["competition_math_numeric_test_hybridp6_v1"],
    EXISTING["competition_math_numeric_test_completion_hybridp6_v1"],
    EXISTING["mmlu_pro_test_hybridp6_v1"],
    clone_spec(
        "mmlu_pro_test_completion_hybridp6_v1",
        run_id="routec_p1_mmlu_pro_ccr_only_v1",
        variant_name="routec_p1_mmlu_pro_ccr_only_v1",
        paper_variant="CCR-only",
        role="route_c_p1_compatibility",
        result_json="Experiment/analysis/results/routec_p1_mmlu_pro_ccr_only_v1.json",
        first_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_only_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_only_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_only_v1_verifier_predictions.jsonl",
        notes=(
            "Route C P1-3 fixed-pool rerun on MMLU-Pro.",
            "This row isolates completion-centric replacement without explicit compatibility fixes.",
        ),
    ),
    clone_spec(
        "mmlu_pro_test_completion_hybridp6_benchmarkaware_v2",
        run_id="routec_p1_mmlu_pro_ccr_amc_v1",
        variant_name="routec_p1_mmlu_pro_ccr_amc_v1",
        paper_variant="CCR+AMC",
        role="route_c_p1_compatibility",
        result_json="Experiment/analysis/results/routec_p1_mmlu_pro_ccr_amc_v1.json",
        first_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_amc_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_amc_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_ccr_amc_v1_verifier_predictions.jsonl",
        notes=(
            "AMC = Answer-Mode Compatibility.",
            "This row adds answer-mode lock and metadata-preserving construction on top of CCR.",
        ),
    ),
    clone_spec(
        "mmlu_pro_test_completion_hybridp6_benchmarkaware_v3",
        run_id="routec_p1_mmlu_pro_cacc_v1",
        variant_name="routec_p1_mmlu_pro_cacc_v1",
        paper_variant="CACC",
        role="route_c_p1_compatibility",
        result_json="Experiment/analysis/results/routec_p1_mmlu_pro_cacc_v1.json",
        first_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_v1_verifier_predictions.jsonl",
        notes=(
            "CACC = CCR + AMC + SCH.",
            "This is the minimal canonical Route C method.",
        ),
    ),
    clone_spec(
        "mmlu_pro_test_completion_hybridp6_benchmarkaware_qwen8bproposer_v2",
        run_id="routec_p1_mmlu_pro_cacc_sp_v1",
        variant_name="routec_p1_mmlu_pro_cacc_sp_v1",
        paper_variant="CACC+SP",
        role="route_c_p1_compatibility",
        result_json="Experiment/analysis/results/routec_p1_mmlu_pro_cacc_sp_v1.json",
        first_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_sp_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_sp_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/routec_p1_mmlu_pro_cacc_sp_v1_verifier_predictions.jsonl",
        notes=(
            "SP = Stronger Proposer.",
            "Auxiliary strengthening path rather than the canonical minimal method.",
        ),
    ),
    EXISTING["gpqa_diamond_train_hybridp6_v1"],
    EXISTING["gpqa_diamond_train_completion_hybridp6_v1"],
]


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def delta(new_value: float | None, old_value: float | None, digits: int = 4) -> str:
    if new_value is None or old_value is None:
        return "NA"
    return f"{new_value - old_value:+.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    return p0.markdown_table(headers, rows)


def effective_vgo(audit: dict) -> float | None:
    oracle = audit["reparsed"].get("oracle_accuracy")
    verifier = audit["reparsed"].get("verifier_accuracy")
    if oracle in (None, 0) or verifier is None:
        return None
    return verifier / oracle


def total_examples(audit: dict) -> int:
    total = audit["reported"].get("total_examples")
    if isinstance(total, int):
        return total
    candidate_examples = audit["hygiene"].get("candidate_examples")
    if isinstance(candidate_examples, int) and candidate_examples:
        return candidate_examples
    verifier_examples = audit["hygiene"].get("verifier_examples")
    if isinstance(verifier_examples, int):
        return verifier_examples
    return 0


def count_of(audit: dict, section: str, key: str) -> int:
    return int(audit["error_taxonomy"].get(section, {}).get(key, 0))


def dominant_bucket(audit: dict) -> str:
    candidates = {
        "oracle_miss": count_of(audit, "outcome_counts", "oracle_miss"),
        "oracle_hit_but_verifier_wrong": count_of(audit, "outcome_counts", "oracle_hit_but_verifier_wrong"),
        "answer_mode_mismatch": max(
            count_of(audit, "metadata_counts", "candidate_answer_mode_mismatch"),
            count_of(audit, "metadata_counts", "prediction_answer_mode_mismatch"),
        ),
        "invalid_selected_answer": count_of(audit, "metadata_counts", "invalid_selected_answer"),
        "verifier_overruled_correct_first": count_of(audit, "outcome_counts", "verifier_overruled_correct_first"),
        "instruction_leak": count_of(audit, "metadata_counts", "instruction_leak"),
        "scaffold_residue": count_of(audit, "metadata_counts", "scaffold_residue"),
    }
    return max(candidates.items(), key=lambda item: item[1])[0]


def scoreboard_rows(audits: list[dict]) -> list[dict]:
    rows = []
    for audit in audits:
        rep = audit["reparsed"]
        hyg = audit["hygiene"]
        rows.append(
            {
                "run_id": audit["run_id"],
                "benchmark": audit["benchmark"],
                "split": audit["split"],
                "variant": audit["paper_facing_variant_name"],
                "role": audit["role"],
                "first": rep.get("first_accuracy"),
                "oracle": rep.get("oracle_accuracy"),
                "verifier": rep.get("verifier_accuracy"),
                "verifier_given_oracle": effective_vgo(audit),
                "candidate_parseable": hyg.get("candidate_parseable_rate"),
                "selected_parseable": hyg.get("selected_prediction_parseable_rate"),
                "answer_mode_match": hyg.get("selected_prediction_answer_mode_match_rate"),
                "invalid_final": hyg.get("invalid_final_answer_rate"),
                "instruction_leak": hyg.get("instruction_leak_rate"),
                "scaffold_residue": hyg.get("scaffold_residue_rate"),
                "duplicate_example": hyg.get("duplicate_example_rate"),
                "malformed_selected": hyg.get("malformed_selected_rate"),
                "dominant_failure_bucket": dominant_bucket(audit),
                "artifact_status": audit["artifact_status"],
                "summary_json": audit["summary_json_path"],
            }
        )
    return rows


def build_parser_audit(by_id: dict[str, dict]) -> str:
    rows = []
    selected = [
        ("gsm8k_full_hybridp6_clean_p1a_v1", "GSM8K clean Diverse-Base"),
        ("routec_p1_gsm8k_full_clean_ccr_targeted_v1", "GSM8K CCR-targeted"),
        ("competition_math_numeric_test_completion_hybridp6_v1", "competition_math:numeric CCR"),
        ("routec_p1_mmlu_pro_ccr_only_v1", "MMLU-Pro CCR-only"),
        ("routec_p1_mmlu_pro_ccr_amc_v1", "MMLU-Pro CCR+AMC"),
        ("routec_p1_mmlu_pro_cacc_v1", "MMLU-Pro CACC"),
        ("gpqa_diamond_train_completion_hybridp6_v1", "GPQA Diamond CCR"),
    ]
    for run_id, label in selected:
        audit = by_id[run_id]
        hyg = audit["hygiene"]
        rep = audit["reported"]
        reparsed = audit["reparsed"]
        parser_status = "stable"
        if (hyg.get("selected_prediction_answer_mode_match_rate") or 0.0) < 0.5:
            parser_status = "legacy answer-mode mismatch"
        elif (hyg.get("selected_prediction_parseable_rate") or 0.0) < 0.99:
            parser_status = "locked but parseability incomplete"
        rows.append(
            [
                label,
                audit["expected_answer_mode"],
                fmt(hyg.get("candidate_row_answer_mode_match_rate")),
                fmt(hyg.get("selected_prediction_answer_mode_match_rate")),
                fmt(rep.get("verifier_accuracy")),
                fmt(reparsed.get("verifier_accuracy")),
                fmt(rep.get("oracle_accuracy")),
                fmt(reparsed.get("oracle_accuracy")),
                fmt(hyg.get("selected_prediction_parseable_rate")),
                parser_status,
            ]
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""# Route C Parser Audit

Generated at {generated_at}.

## Scope

- Parser implementation follows the benchmark-locked audit path already used in the NeurIPS readiness pack.
- Numeric tasks are locked to `numeric`; multiple-choice tasks are locked to `choice_letter`.
- Route C P1-4 only treats reparsed metrics as authoritative whenever archived run metadata carries the wrong answer mode.

## Route C Critical Rows

{table(["row", "expected_mode", "candidate_mode_match", "selected_mode_match", "reported_verifier", "reparsed_verifier", "reported_oracle", "reparsed_oracle", "selected_parseable", "parser_status"], rows)}

## Findings

1. Numeric rows are parser-stable. `GSM8K` clean and `competition_math:numeric` show no answer-mode mismatch, so their Route C evidence is not an artifact of parser semantics.
2. The original MCQ collapse was partly a parser artifact. `MMLU-Pro CCR-only` and `GPQA Diamond CCR` still carry legacy `numeric` answer-mode metadata, which drives the reported near-zero rows; reparsing recovers them to analyzable levels.
3. `AMC` repairs the answer-mode lock itself. In the Route C chain, `CCR+AMC` and `CACC` both bring selected answer-mode match to `1.0000` on `MMLU-Pro`.
4. Parser lock is now sufficient for mechanism analysis but still not sufficient for paper-grade hygiene. Even the repaired MCQ rows remain below the `selected_parseable >= 0.99` target.

## Route C Parser Gate

- `answer-mode lock for canonical MCQ rows`: pass for `CCR+AMC`, `CACC`, and `CACC+SP`
- `legacy MCQ rows usable without reparsing`: fail
- `final selected parseability at paper-grade threshold`: fail

## Decision

Route C parser hygiene is now good enough to support the `coverage vs compatibility` mechanism argument, but not good enough to claim that MCQ final-answer hygiene is solved.
"""


def build_candidate_hygiene(by_id: dict[str, dict]) -> str:
    rows = []
    selected = [
        ("gsm8k_full_hybridp6_clean_p1a_v1", "GSM8K", "Diverse-Base"),
        ("routec_p1_gsm8k_full_clean_ccr_targeted_v1", "GSM8K", "CCR-targeted"),
        ("routec_p1_gsm8k_full_clean_ccr_randomslot_v1", "GSM8K", "CCR-random-slot"),
        ("competition_math_numeric_test_completion_hybridp6_v1", "competition_math:numeric", "CCR"),
        ("routec_p1_mmlu_pro_ccr_only_v1", "MMLU-Pro", "CCR-only"),
        ("routec_p1_mmlu_pro_ccr_amc_v1", "MMLU-Pro", "CCR+AMC"),
        ("routec_p1_mmlu_pro_cacc_v1", "MMLU-Pro", "CACC"),
        ("routec_p1_mmlu_pro_cacc_sp_v1", "MMLU-Pro", "CACC+SP"),
        ("gpqa_diamond_train_completion_hybridp6_v1", "GPQA Diamond", "CCR"),
    ]
    for run_id, benchmark, label in selected:
        audit = by_id[run_id]
        hyg = audit["hygiene"]
        rows.append(
            [
                benchmark,
                label,
                fmt(hyg.get("candidate_parseable_rate")),
                fmt(hyg.get("selected_prediction_parseable_rate")),
                fmt(hyg.get("invalid_final_answer_rate")),
                fmt(hyg.get("instruction_leak_rate")),
                fmt(hyg.get("scaffold_residue_rate")),
                fmt(hyg.get("duplicate_example_rate")),
                fmt(hyg.get("malformed_selected_rate")),
            ]
        )

    randomslot = by_id["routec_p1_gsm8k_full_clean_ccr_randomslot_v1"]["hygiene"]
    targeted = by_id["routec_p1_gsm8k_full_clean_ccr_targeted_v1"]["hygiene"]
    ccr_only = by_id["routec_p1_mmlu_pro_ccr_only_v1"]["hygiene"]
    ccr_amc = by_id["routec_p1_mmlu_pro_ccr_amc_v1"]["hygiene"]
    cacc = by_id["routec_p1_mmlu_pro_cacc_v1"]["hygiene"]
    cacc_sp = by_id["routec_p1_mmlu_pro_cacc_sp_v1"]["hygiene"]

    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""# Route C Candidate Hygiene Audit

Generated at {generated_at}.

## Scope

- This audit focuses only on the rows that matter for Route C mechanism claims.
- Metrics are computed under benchmark-locked answer modes.
- The goal is not to prove paper-grade cleanliness; the goal is to identify where hygiene interacts with coverage and selector conversion.

## Hygiene Summary

{table(["benchmark", "variant", "candidate_parseable", "selected_parseable", "invalid_final", "instruction_leak", "scaffold_residue", "duplicate_example", "malformed_selected"], rows)}

## Key Readouts

1. `GSM8K targeted vs random` is not a cleanliness win for targeted replacement. `CCR-targeted` only slightly improves candidate parseability relative to `CCR-random-slot` (`{fmt(targeted.get('candidate_parseable_rate'))}` vs `{fmt(randomslot.get('candidate_parseable_rate'))}`), but it is worse on duplicate-example rate (`{fmt(targeted.get('duplicate_example_rate'))}` vs `{fmt(randomslot.get('duplicate_example_rate'))}`), instruction leak (`{fmt(targeted.get('instruction_leak_rate'))}` vs `{fmt(randomslot.get('instruction_leak_rate'))}`), and malformed selected outputs (`{fmt(targeted.get('malformed_selected_rate'))}` vs `{fmt(randomslot.get('malformed_selected_rate'))}`).
2. `AMC` repairs mode compatibility but does not automatically clean everything else. On `MMLU-Pro`, `CCR+AMC` sharply improves selected parseability over `CCR-only` (`{fmt(ccr_only.get('selected_prediction_parseable_rate'))}` -> `{fmt(ccr_amc.get('selected_prediction_parseable_rate'))}`) and reduces invalid finals (`{fmt(ccr_only.get('invalid_final_answer_rate'))}` -> `{fmt(ccr_amc.get('invalid_final_answer_rate'))}`), but instruction leakage temporarily rises (`{fmt(ccr_only.get('instruction_leak_rate'))}` -> `{fmt(ccr_amc.get('instruction_leak_rate'))}`).
3. `SCH` is a real hygiene module. `CACC` leaves final accuracy roughly flat relative to `CCR+AMC`, but it cuts instruction leakage from `{fmt(ccr_amc.get('instruction_leak_rate'))}` to `{fmt(cacc.get('instruction_leak_rate'))}` and scaffold residue from `{fmt(ccr_amc.get('scaffold_residue_rate'))}` to `{fmt(cacc.get('scaffold_residue_rate'))}`.
4. `CACC+SP` reopens part of the compatibility problem. It gives the strongest `MMLU-Pro` final row, but selected parseability falls from `{fmt(cacc.get('selected_prediction_parseable_rate'))}` to `{fmt(cacc_sp.get('selected_prediction_parseable_rate'))}` and invalid final answers rise from `{fmt(cacc.get('invalid_final_answer_rate'))}` to `{fmt(cacc_sp.get('invalid_final_answer_rate'))}`.
5. `GPQA Diamond` remains the dirtiest MCQ slice. Legacy `CCR` still shows only `{fmt(by_id['gpqa_diamond_train_completion_hybridp6_v1']['hygiene'].get('selected_prediction_parseable_rate'))}` selected parseability and `{fmt(by_id['gpqa_diamond_train_completion_hybridp6_v1']['hygiene'].get('invalid_final_answer_rate'))}` invalid-final rate, so GPQA is still appendix-only until a clean confirmatory row exists.

## Route C Hygiene Verdict

The hygiene evidence now supports a paper-safe statement that `compatibility` and `selector-compatible cleanup` are method components rather than engineering afterthoughts. It still does **not** support a statement that the canonical pipeline has already reached paper-grade final-answer hygiene across all benchmarks.
"""


def build_error_taxonomy(by_id: dict[str, dict]) -> str:
    rows = []
    selected = [
        ("gsm8k_full_hybridp6_clean_p1a_v1", "GSM8K Diverse-Base"),
        ("routec_p1_gsm8k_full_clean_ccr_targeted_v1", "GSM8K CCR-targeted"),
        ("routec_p1_gsm8k_full_clean_ccr_randomslot_v1", "GSM8K CCR-random-slot"),
        ("competition_math_numeric_test_completion_hybridp6_v1", "competition_math:numeric CCR"),
        ("routec_p1_mmlu_pro_ccr_only_v1", "MMLU-Pro CCR-only"),
        ("routec_p1_mmlu_pro_cacc_v1", "MMLU-Pro CACC"),
        ("routec_p1_mmlu_pro_cacc_sp_v1", "MMLU-Pro CACC+SP"),
        ("gpqa_diamond_train_completion_hybridp6_v1", "GPQA Diamond CCR"),
    ]
    for run_id, label in selected:
        audit = by_id[run_id]
        metadata = audit["error_taxonomy"]["metadata_counts"]
        outcome = audit["error_taxonomy"]["outcome_counts"]
        rows.append(
            [
                label,
                str(total_examples(audit)),
                str(outcome.get("oracle_miss", 0)),
                str(outcome.get("oracle_hit_but_verifier_wrong", 0)),
                str(max(metadata.get("candidate_answer_mode_mismatch", 0), metadata.get("prediction_answer_mode_mismatch", 0))),
                str(metadata.get("invalid_selected_answer", 0)),
                str(outcome.get("verifier_overruled_correct_first", 0)),
                str(metadata.get("instruction_leak", 0)),
                str(metadata.get("scaffold_residue", 0)),
                dominant_bucket(audit),
            ]
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""# Route C Error Taxonomy Audit

Generated at {generated_at}.

## Scope

- The taxonomy below uses exact counts from the benchmark-locked audit path.
- `oracle_miss` and `oracle_hit_but_verifier_wrong` are outcome buckets.
- `answer_mode_mismatch`, `invalid_selected_answer`, `instruction_leak`, and `scaffold_residue` are metadata / compatibility buckets.

## Critical Bucket Summary

{table(["row", "n", "oracle_miss", "oracle_hit_verifier_wrong", "answer_mode_mismatch", "invalid_selected", "overrule_correct_first", "instruction_leak", "scaffold_residue", "dominant_bucket"], rows)}

## Cross-Run Interpretation

1. `GSM8K clean` remains predominantly coverage-limited. Even after `CCR-targeted`, the largest failure bucket is still `oracle_miss`, not parse failure.
2. `P1-2` negative signal is a coverage loss, not a selector collapse. The paired Route C diagnosis shows that `CCR-random-slot` gains most of its verifier advantage through preserving more oracle hits.
3. `MMLU-Pro CCR-only` is compatibility-limited. Its dominant bucket is answer-mode mismatch rather than selector weakness on a clean compatible pool.
4. `CACC` shifts the dominant MMLU bucket back to `oracle_miss`, which is exactly the transition Route C needs: once compatibility is repaired, the benchmark becomes analyzable as a real candidate-construction problem again.
5. `CACC+SP` increases coverage and final accuracy on `MMLU-Pro`, but `oracle_miss` still dominates the remaining failures. Stronger proposers help, but they do not remove the mechanism tension.
6. `GPQA Diamond` is still mixed-fragile. The legacy answer-mode mismatch is catastrophic, and even the reparsed row keeps large invalid-final and oracle-miss burdens.

## Route C Taxonomy Verdict

The Route C taxonomy is now sufficiently sharp to distinguish three different failure sources: genuine coverage shortage, compatibility collapse, and post-compatibility mixed regimes. That is enough for a mechanism paper, but it still leaves GPQA as a weak spot rather than a polished positive result.
"""


def build_regime_analysis(by_id: dict[str, dict]) -> str:
    gsm8k_base = by_id["gsm8k_full_hybridp6_clean_p1a_v1"]
    gsm8k_ccr = by_id["gsm8k_full_completion_hybridp6_clean_p1a_v1"]
    gsm8k_random = by_id["routec_p1_gsm8k_full_clean_ccr_randomslot_v1"]
    gsm8k_targeted = by_id["routec_p1_gsm8k_full_clean_ccr_targeted_v1"]
    comp_base = by_id["competition_math_numeric_test_hybridp6_v1"]
    comp_ccr = by_id["competition_math_numeric_test_completion_hybridp6_v1"]
    mmlu_base = by_id["mmlu_pro_test_hybridp6_v1"]
    mmlu_ccr = by_id["routec_p1_mmlu_pro_ccr_only_v1"]
    mmlu_amc = by_id["routec_p1_mmlu_pro_ccr_amc_v1"]
    mmlu_cacc = by_id["routec_p1_mmlu_pro_cacc_v1"]
    mmlu_sp = by_id["routec_p1_mmlu_pro_cacc_sp_v1"]
    gpqa_base = by_id["gpqa_diamond_train_hybridp6_v1"]
    gpqa_ccr = by_id["gpqa_diamond_train_completion_hybridp6_v1"]

    regime_rows = [
        [
            "GSM8K clean",
            "numeric open-ended",
            "Diverse-Base -> CCR-targeted",
            f"{fmt(gsm8k_base['reparsed']['oracle_accuracy'])} -> {fmt(gsm8k_ccr['reparsed']['oracle_accuracy'])}",
            f"{fmt(gsm8k_base['reparsed']['verifier_accuracy'])} -> {fmt(gsm8k_ccr['reparsed']['verifier_accuracy'])}",
            f"{fmt(effective_vgo(gsm8k_base))} -> {fmt(effective_vgo(gsm8k_ccr))}",
            "coverage rises, but conversion softens",
            "coverage-limited with compatibility drag",
        ],
        [
            "GSM8K P1-2",
            "numeric open-ended",
            "CCR-targeted -> CCR-random-slot",
            f"{fmt(gsm8k_targeted['reparsed']['oracle_accuracy'])} -> {fmt(gsm8k_random['reparsed']['oracle_accuracy'])}",
            f"{fmt(gsm8k_targeted['reparsed']['verifier_accuracy'])} -> {fmt(gsm8k_random['reparsed']['verifier_accuracy'])}",
            f"{fmt(effective_vgo(gsm8k_targeted))} -> {fmt(effective_vgo(gsm8k_random))}",
            "coverage gain beats tiny cleanliness gain",
            "tradeoff probe, negative for targeted heuristic",
        ],
        [
            "competition_math:numeric",
            "numeric open-ended",
            "Diverse-Base -> CCR",
            f"{fmt(comp_base['reparsed']['oracle_accuracy'])} -> {fmt(comp_ccr['reparsed']['oracle_accuracy'])}",
            f"{fmt(comp_base['reparsed']['verifier_accuracy'])} -> {fmt(comp_ccr['reparsed']['verifier_accuracy'])}",
            f"{fmt(effective_vgo(comp_base))} -> {fmt(effective_vgo(comp_ccr))}",
            "coverage and conversion both improve",
            "clean coverage-limited regime",
        ],
        [
            "MMLU-Pro legacy-to-reparsed",
            "multiple-choice",
            "CCR-only reported -> reparsed",
            f"{fmt(mmlu_ccr['reported']['oracle_accuracy'])} -> {fmt(mmlu_ccr['reparsed']['oracle_accuracy'])}",
            f"{fmt(mmlu_ccr['reported']['verifier_accuracy'])} -> {fmt(mmlu_ccr['reparsed']['verifier_accuracy'])}",
            f"{fmt(mmlu_ccr['reported'].get('verifier_given_oracle_accuracy'))} -> {fmt(effective_vgo(mmlu_ccr))}",
            "parser artifact masked true pool quality",
            "compatibility-limited",
        ],
        [
            "MMLU-Pro repaired chain",
            "multiple-choice",
            "CCR-only -> CCR+AMC -> CACC -> CACC+SP",
            f"{fmt(mmlu_ccr['reparsed']['oracle_accuracy'])} -> {fmt(mmlu_amc['reparsed']['oracle_accuracy'])} -> {fmt(mmlu_cacc['reparsed']['oracle_accuracy'])} -> {fmt(mmlu_sp['reparsed']['oracle_accuracy'])}",
            f"{fmt(mmlu_ccr['reparsed']['verifier_accuracy'])} -> {fmt(mmlu_amc['reparsed']['verifier_accuracy'])} -> {fmt(mmlu_cacc['reparsed']['verifier_accuracy'])} -> {fmt(mmlu_sp['reparsed']['verifier_accuracy'])}",
            f"{fmt(effective_vgo(mmlu_ccr))} -> {fmt(effective_vgo(mmlu_amc))} -> {fmt(effective_vgo(mmlu_cacc))} -> {fmt(effective_vgo(mmlu_sp))}",
            "compatibility repair raises coverage; SP restores part of final conversion",
            "mixed regime after compatibility repair",
        ],
        [
            "GPQA Diamond legacy-to-reparsed",
            "multiple-choice",
            "CCR reported -> reparsed",
            f"{fmt(gpqa_ccr['reported']['oracle_accuracy'])} -> {fmt(gpqa_ccr['reparsed']['oracle_accuracy'])}",
            f"{fmt(gpqa_ccr['reported']['verifier_accuracy'])} -> {fmt(gpqa_ccr['reparsed']['verifier_accuracy'])}",
            f"{fmt(gpqa_ccr['reported'].get('verifier_given_oracle_accuracy'))} -> {fmt(effective_vgo(gpqa_ccr))}",
            "collapse is partly parser-driven, but benchmark remains brittle",
            "compatibility-limited then mixed-fragile",
        ],
    ]

    probe_rows = [
        [
            "GSM8K clean: Diverse-Base -> CCR-targeted",
            delta(gsm8k_ccr['reparsed']['oracle_accuracy'], gsm8k_base['reparsed']['oracle_accuracy']),
            delta(gsm8k_ccr['reparsed']['verifier_accuracy'], gsm8k_base['reparsed']['verifier_accuracy']),
            delta(effective_vgo(gsm8k_ccr), effective_vgo(gsm8k_base)),
            delta(gsm8k_ccr['hygiene']['selected_prediction_parseable_rate'], gsm8k_base['hygiene']['selected_prediction_parseable_rate']),
            "coverage gain is real, but shared-hit conversion softens",
        ],
        [
            "GSM8K P1-2: CCR-targeted -> CCR-random-slot",
            delta(gsm8k_random['reparsed']['oracle_accuracy'], gsm8k_targeted['reparsed']['oracle_accuracy']),
            delta(gsm8k_random['reparsed']['verifier_accuracy'], gsm8k_targeted['reparsed']['verifier_accuracy']),
            delta(effective_vgo(gsm8k_random), effective_vgo(gsm8k_targeted)),
            delta(gsm8k_random['hygiene']['selected_prediction_parseable_rate'], gsm8k_targeted['hygiene']['selected_prediction_parseable_rate']),
            "current targeted heuristic sacrifices too much coverage",
        ],
        [
            "competition_math:numeric: Diverse-Base -> CCR",
            delta(comp_ccr['reparsed']['oracle_accuracy'], comp_base['reparsed']['oracle_accuracy']),
            delta(comp_ccr['reparsed']['verifier_accuracy'], comp_base['reparsed']['verifier_accuracy']),
            delta(effective_vgo(comp_ccr), effective_vgo(comp_base)),
            delta(comp_ccr['hygiene']['selected_prediction_parseable_rate'], comp_base['hygiene']['selected_prediction_parseable_rate']),
            "clean numeric evidence that completion-oriented construction is not a parser artifact",
        ],
        [
            "MMLU-Pro: CCR-only -> CCR+AMC",
            delta(mmlu_amc['reparsed']['oracle_accuracy'], mmlu_ccr['reparsed']['oracle_accuracy']),
            delta(mmlu_amc['reparsed']['verifier_accuracy'], mmlu_ccr['reparsed']['verifier_accuracy']),
            delta(effective_vgo(mmlu_amc), effective_vgo(mmlu_ccr)),
            delta(mmlu_amc['hygiene']['selected_prediction_parseable_rate'], mmlu_ccr['hygiene']['selected_prediction_parseable_rate']),
            "mode compatibility repairs pool usability, but not final conversion",
        ],
        [
            "MMLU-Pro: CCR+AMC -> CACC",
            delta(mmlu_cacc['reparsed']['oracle_accuracy'], mmlu_amc['reparsed']['oracle_accuracy']),
            delta(mmlu_cacc['reparsed']['verifier_accuracy'], mmlu_amc['reparsed']['verifier_accuracy']),
            delta(effective_vgo(mmlu_cacc), effective_vgo(mmlu_amc)),
            delta(mmlu_cacc['hygiene']['selected_prediction_parseable_rate'], mmlu_amc['hygiene']['selected_prediction_parseable_rate']),
            "SCH is mainly hygiene stabilization, not a fresh accuracy jump",
        ],
        [
            "MMLU-Pro: CACC -> CACC+SP",
            delta(mmlu_sp['reparsed']['oracle_accuracy'], mmlu_cacc['reparsed']['oracle_accuracy']),
            delta(mmlu_sp['reparsed']['verifier_accuracy'], mmlu_cacc['reparsed']['verifier_accuracy']),
            delta(effective_vgo(mmlu_sp), effective_vgo(mmlu_cacc)),
            delta(mmlu_sp['hygiene']['selected_prediction_parseable_rate'], mmlu_cacc['hygiene']['selected_prediction_parseable_rate']),
            "stronger proposers help, but they partially reopen the compatibility cost",
        ],
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""# Route C Regime Analysis

Generated at {generated_at}.

## Goal

This note closes `P1-4 coverage vs selector-compatibility tradeoff` using only auditable artifacts already present in the repository. The question is not whether Route C is already a finished best-method story. The question is whether the existing evidence now forms a clear mechanism map.

## Benchmark-Level Regime Table

{table(["slice", "answer_mode", "comparison", "oracle", "verifier", "verifier_given_oracle", "diagnostic", "regime"], regime_rows)}

## Tradeoff Probes

{table(["probe", "delta_oracle", "delta_verifier", "delta_verifier_given_oracle", "delta_selected_parseable", "paper-safe interpretation"], probe_rows)}

## Main Findings

1. `Route C now has a clear regime map.` Numeric open-ended benchmarks (`GSM8K`, `competition_math:numeric`) are primarily coverage-limited. MCQ benchmarks start as compatibility-limited and only become analyzable candidate-construction problems after answer-mode repair.
2. `Coverage gain is real but not free.` On `GSM8K clean`, `CCR-targeted` raises oracle by {delta(gsm8k_ccr['reparsed']['oracle_accuracy'], gsm8k_base['reparsed']['oracle_accuracy'])} and final verifier by {delta(gsm8k_ccr['reparsed']['verifier_accuracy'], gsm8k_base['reparsed']['verifier_accuracy'])}, but `verifier_given_oracle` falls by {delta(effective_vgo(gsm8k_ccr), effective_vgo(gsm8k_base))}. This is the core `coverage vs compatibility` signal.
3. `The current targeted heuristic is not the answer.` The `P1-2` control shows that `CCR-random-slot` actually beats `CCR-targeted` on both oracle and final verifier. Route C should therefore claim that replacement policy matters, not that the current fragment-first targeting rule is correct.
4. `AMC is necessary but not sufficient.` On `MMLU-Pro`, `CCR+AMC` restores answer-mode compatibility and raises oracle, yet final verifier does not improve over reparsed `CCR-only`. Compatibility repair unlocks analysis; it does not automatically solve selection.
5. `SCH is method content, not implementation noise.` `CACC` meaningfully suppresses instruction leakage and scaffold residue while keeping the repaired MCQ row stable.
6. `SP is a strengthening path, not the canonical story.` `CACC+SP` is the strongest `MMLU-Pro` row, but it reopens some parseability loss. That is exactly why `CACC` stays the canonical main variant and `CACC+SP` stays backup.
7. `GPQA is no longer a Route-A-style collapse headline.` Reparsing shows the old zero row was partly an evaluation artifact. But GPQA still remains mixed-fragile and is not strong enough to headline as a positive result.

## P1-4 Verdict

- `P1-4 status`: **pass as a mechanism pack**
- `What passed`: the repository now supports a coherent paper-facing decomposition into coverage-limited, compatibility-limited, and mixed regimes.
- `What did not pass`: Route C still lacks paper-grade GPQA evidence and still lacks a positive targeted-vs-random result.

## Route C Go / No-Go After P1-4

- `Route C`: **strengthened conditional-go**
- Reason: the mechanism story is now coherent enough to support a real analysis-driven methods paper, but not yet strong enough to remove all NeurIPS-main-track risk.

## Recommendation On GPQA Confirmatory Closure

`Do not treat a new GPQA row as the immediate blocker for writing.` The current regime map is already clear enough to continue paper-facing structuring, claim pruning, and main-table drafting.

`Do treat GPQA as the highest-value optional confirmatory experiment if time remains.` A clean GPQA compatibility row would strengthen the claim that MCQ brittleness is not unique to MMLU-Pro, but Route C no longer needs GPQA to rescue a broad transfer headline because that headline is already abandoned.

## Immediate Next Step

Move from experiment-heavy work to paper assembly:

1. freeze the Route C main table around `GSM8K clean`, `competition_math:numeric`, and `MMLU-Pro compatibility chain`;
2. update the claim ledger and readiness review around the new regime map;
3. schedule GPQA only as an appendix-strengthening confirmatory row if resources remain after writing assets are in place.
"""


def build_summary(rows: list[dict]) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pack": "route_c_p1_tradeoff_pack",
        "status": "p1_4_mechanism_pack_complete",
        "route_c_go_no_go": "strengthened_conditional_go",
        "gpqa_confirmatory_recommendation": "optional_appendix_confirmatory_row_only",
        "artifacts": {
            "scoreboard": str(PACK_DIR / "route_c_p1_tradeoff_scoreboard.csv"),
            "parser_audit": str(PACK_DIR / "parser_audit_route_c.md"),
            "candidate_hygiene": str(PACK_DIR / "candidate_hygiene_route_c.md"),
            "error_taxonomy": str(PACK_DIR / "error_taxonomy_route_c.md"),
            "regime_analysis": str(PACK_DIR / "regime_analysis_route_c.md"),
        },
        "rows": rows,
    }


def main() -> None:
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    audits = [p0.audit_run(spec) for spec in RUN_SPECS]
    by_id = {audit["run_id"]: audit for audit in audits}
    rows = scoreboard_rows(audits)

    p0.write_csv(
        PACK_DIR / "route_c_p1_tradeoff_scoreboard.csv",
        rows,
        [
            "run_id",
            "benchmark",
            "split",
            "variant",
            "role",
            "first",
            "oracle",
            "verifier",
            "verifier_given_oracle",
            "candidate_parseable",
            "selected_parseable",
            "answer_mode_match",
            "invalid_final",
            "instruction_leak",
            "scaffold_residue",
            "duplicate_example",
            "malformed_selected",
            "dominant_failure_bucket",
            "artifact_status",
            "summary_json",
        ],
    )
    p0.atomic_write_text(PACK_DIR / "parser_audit_route_c.md", build_parser_audit(by_id))
    p0.atomic_write_text(PACK_DIR / "candidate_hygiene_route_c.md", build_candidate_hygiene(by_id))
    p0.atomic_write_text(PACK_DIR / "error_taxonomy_route_c.md", build_error_taxonomy(by_id))
    p0.atomic_write_text(PACK_DIR / "regime_analysis_route_c.md", build_regime_analysis(by_id))
    p0.atomic_write_json(PACK_DIR / "route_c_p1_tradeoff_summary.json", build_summary(rows))


if __name__ == "__main__":
    main()
