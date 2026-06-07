#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
CORE_CODE_ROOT = ROOT / "Experiment" / "core_code"
if str(CORE_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_CODE_ROOT))

from src.eval.evaluate_predictions import (  # type: ignore
    answers_match,
    extract_choice_answer,
    extract_numeric_answer,
    normalize_answer,
)

PACK_DIR = ROOT / "Publication" / "paper" / "neurips_readiness_pack_v1"

EXPECTED_MODES = {
    "gsm8k": "numeric",
    "competition_math_numeric": "numeric",
    "mmlu_pro": "choice_letter",
    "gpqa_diamond": "choice_letter",
}

INSTRUCTION_PATTERNS = (
    "use at most",
    "just output the final answer",
    "just output the answer",
    "do not include any explanation",
    "do not include any reasoning",
    "do not produce any other text",
    "the last line must contain",
    "the last line should be",
    "this is a multiple-choice task",
    "you may reason about the options briefly",
    "the correct answer is one of the options listed",
    "the final answer must be",
    "your answer must be",
    "the answer must be",
    "single option letter",
    "single option-letter",
    "do not write anything else",
)

SCAFFOLD_REGEXES = (
    re.compile(r"^(?:here'?s the thought process:\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:let'?s think(?: about this)?(?: step by step)?[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:let me think(?: about this)?(?: step by step)?[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:okay,\s*)?let'?s tackle this problem(?: step by step)?[.:]?\s*", flags=re.IGNORECASE),
    re.compile(r"^(?:okay,\s*)?let'?s break down the question again[.:]?\s*", flags=re.IGNORECASE),
    re.compile(r"^(?:wait,\s*but that'?s not correct[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:wait,\s*that'?s not correct[.!]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:the solution (?:should|must)(?: not)? be [^.!?\n]*(?:[.!?]\s*|$))+", flags=re.IGNORECASE),
    re.compile(r"^(?:the last line should be [^.!?\n]*(?:[.!?]\s*|$))+", flags=re.IGNORECASE),
    re.compile(r"^(?:use the options given(?: above)?[.!?]?\s*)+", flags=re.IGNORECASE),
    re.compile(r"^(?:do not output the letter of the option until the last line[.!?]?\s*)+", flags=re.IGNORECASE),
)

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Dr Claw NeurIPS Run Manifest",
    "type": "object",
    "required": [
        "run_id",
        "benchmark",
        "variant_name",
        "paper_facing_variant_name",
        "candidate_budget",
        "generator",
        "proposer",
        "verifier",
        "prompt_family",
        "sanitizer_version",
        "scoring_script_version",
        "seed",
        "first_accuracy",
        "base_accuracy",
        "verifier_accuracy",
        "oracle_accuracy",
        "verifier_given_oracle_accuracy",
        "answer_format_validity",
        "scaffold_leakage_rate",
        "invalid_final_answer_rate",
        "duplicate_candidate_rate",
        "output_files",
    ],
    "properties": {
        "run_id": {"type": "string"},
        "benchmark": {"type": "string"},
        "variant_name": {"type": "string"},
        "paper_facing_variant_name": {"type": "string"},
        "candidate_budget": {"type": ["integer", "null"], "minimum": 1},
        "generator": {"type": ["string", "null"]},
        "proposer": {"type": ["string", "null"]},
        "verifier": {"type": ["string", "null"]},
        "prompt_family": {"type": ["string", "null"]},
        "sanitizer_version": {"type": ["string", "null"]},
        "scoring_script_version": {"type": ["string", "null"]},
        "seed": {"type": ["integer", "string", "null"]},
        "first_accuracy": {"type": ["number", "null"]},
        "base_accuracy": {"type": ["number", "null"]},
        "verifier_accuracy": {"type": ["number", "null"]},
        "oracle_accuracy": {"type": ["number", "null"]},
        "verifier_given_oracle_accuracy": {"type": ["number", "null"]},
        "answer_format_validity": {"type": ["number", "null"]},
        "scaffold_leakage_rate": {"type": ["number", "null"]},
        "invalid_final_answer_rate": {"type": ["number", "null"]},
        "duplicate_candidate_rate": {"type": ["number", "null"]},
        "output_files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "path", "sha256", "exists"],
                "properties": {
                    "label": {"type": "string"},
                    "path": {"type": "string"},
                    "sha256": {"type": ["string", "null"]},
                    "exists": {"type": "boolean"},
                },
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    benchmark: str
    split: str
    variant_name: str
    paper_variant: str
    role: str
    result_json: str | None = None
    generation_json: str | None = None
    candidate_pool: str | None = None
    first_predictions: str | None = None
    base_predictions: str | None = None
    verifier_predictions: str | None = None
    base_predictions_glob: str | None = None
    verifier_predictions_glob: str | None = None
    audit_scope_hint: str = "full_local"
    notes: tuple[str, ...] = ()


RUN_SPECS = [
    RunSpec(
        run_id="gsm8k_full_hybridp6_v1",
        benchmark="gsm8k",
        split="test_full",
        variant_name="full_hybridp6_v1",
        paper_variant="Diverse-Base",
        role="internal_baseline_legacy_summary",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_full_hybridp6_v1.json",
        candidate_pool="Experiment/datasets/processed/gsm8k_eval_full_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_hybridp6_v1.jsonl",
        audit_scope_hint="summary_only",
        notes=(
            "Archived pre-clean GSM8K matched-budget baseline row kept only for lineage.",
            "Superseded by the auditable P1-A clean rerun for paper-facing baseline use.",
        ),
    ),
    RunSpec(
        run_id="gsm8k_full_hybridp6_clean_p1a_v1",
        benchmark="gsm8k",
        split="test_full_clean",
        variant_name="hybridp6_clean_p1a_v1",
        paper_variant="Diverse-Base",
        role="internal_baseline_main_clean",
        result_json="Experiment/analysis/results/ser_p1a_gsm8k_full_hybridp6_clean_v1.json",
        generation_json="Experiment/analysis/results/ser_p1a_gsm8k_full_hybridp6_clean_v1_generation.json",
        candidate_pool="Experiment/datasets/processed/gsm8k_eval_full_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_hybridp6_clean_p1a_v1.jsonl",
        first_predictions="Experiment/core_code/logs/ser_p1a_gsm8k_full_hybridp6_clean_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_p1a_gsm8k_full_hybridp6_clean_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_p1a_gsm8k_full_hybridp6_clean_v1_verifier_predictions.jsonl",
        notes=(
            "P1-A GSM8K full clean rerun baseline row completed on 2026-04-07.",
            "Use this auditable run rather than archived summary-only baseline rows for paper-facing GSM8K comparisons.",
        ),
    ),
    RunSpec(
        run_id="gsm8k_full_eval_v1",
        benchmark="gsm8k",
        split="test_full",
        variant_name="filtered_full_v1",
        paper_variant="Diverse-Base",
        role="internal_baseline_audit_anchor",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_full_eval_v1.json",
        candidate_pool="Experiment/datasets/processed/gsm8k_eval_full_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_v1.jsonl",
        base_predictions_glob="Experiment/core_code/logs/shards/gsm8k_eval_full_base_rerank_qwen3_17b_filtered_t07_s16k8_v1/shard_*_predictions.jsonl",
        verifier_predictions_glob="Experiment/core_code/logs/shards/gsm8k_eval_full_verifier512_rerank_qwen3_17b_filtered_t07_s16k8_v1/shard_*_predictions.jsonl",
        audit_scope_hint="full_local",
        notes=(
            "Legacy auditable GSM8K anchor within the same Diverse-Base family.",
            "Use only for hygiene/error analysis; main paper baseline should remain full_hybridp6 after clean rerun.",
        ),
    ),
    RunSpec(
        run_id="gsm8k_full_completion_hybridp6_v1",
        benchmark="gsm8k",
        split="test_full",
        variant_name="full_completion_hybridp6_v1",
        paper_variant="CCR",
        role="internal_main_gsm8k_legacy_summary",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_full_completion_hybridp6_v1.json",
        generation_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_full_completion_hybridp6_generation_v1.json",
        candidate_pool="Experiment/datasets/processed/gsm8k_eval_full_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_v1.jsonl",
        audit_scope_hint="summary_only",
        notes=(
            "Archived pre-clean CCR GSM8K row kept only for lineage.",
            "Superseded by the auditable P1-A clean rerun for paper-facing GSM8K claims.",
        ),
    ),
    RunSpec(
        run_id="gsm8k_full_completion_hybridp6_clean_p1a_v1",
        benchmark="gsm8k",
        split="test_full_clean",
        variant_name="completion_hybridp6_clean_p1a_v1",
        paper_variant="CCR",
        role="internal_main_gsm8k_clean",
        result_json="Experiment/analysis/results/ser_p1a_gsm8k_full_completion_hybridp6_clean_v1.json",
        generation_json="Experiment/analysis/results/ser_p1a_gsm8k_full_completion_hybridp6_clean_v1_generation.json",
        candidate_pool="Experiment/datasets/processed/gsm8k_eval_full_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_clean_p1a_v1.jsonl",
        first_predictions="Experiment/core_code/logs/ser_p1a_gsm8k_full_completion_hybridp6_clean_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_p1a_gsm8k_full_completion_hybridp6_clean_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_p1a_gsm8k_full_completion_hybridp6_clean_v1_verifier_predictions.jsonl",
        notes=(
            "P1-A GSM8K full clean rerun main CCR row completed on 2026-04-07.",
            "Relative to the clean Diverse-Base baseline, this row improves oracle by +0.1509 and verifier by +0.0758, but verifier=0.3632 remains below the 0.40 promotion gate.",
        ),
    ),
    RunSpec(
        run_id="competition_math_numeric_test_hybridp6_v1",
        benchmark="competition_math_numeric",
        split="test",
        variant_name="hybridp6_v1",
        paper_variant="Diverse-Base",
        role="internal_baseline_transfer",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_hybridp6_v1.json",
        candidate_pool="Experiment/datasets/processed/competition_math_numeric_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_hybridp6_v1.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_hybridp6_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_hybridp6_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_hybridp6_v1_verifier_predictions.jsonl",
    ),
    RunSpec(
        run_id="competition_math_numeric_test_completion_hybridp6_v1",
        benchmark="competition_math_numeric",
        split="test",
        variant_name="completion_hybridp6_v1",
        paper_variant="CCR",
        role="internal_transfer",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_completion_hybridp6_v1.json",
        generation_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_completion_hybridp6_v1_generation.json",
        candidate_pool="Experiment/datasets/processed/competition_math_numeric_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_v1.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_completion_hybridp6_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_completion_hybridp6_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_completion_hybridp6_v1_verifier_predictions.jsonl",
    ),
    RunSpec(
        run_id="mmlu_pro_test_hybridp6_v1",
        benchmark="mmlu_pro",
        split="test",
        variant_name="hybridp6_v1",
        paper_variant="Diverse-Base",
        role="internal_baseline_transfer",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_hybridp6_v1.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_hybridp6_v1.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_hybridp6_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_hybridp6_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_hybridp6_v1_verifier_predictions.jsonl",
    ),
    RunSpec(
        run_id="mmlu_pro_test_completion_hybridp6_v1",
        benchmark="mmlu_pro",
        split="test",
        variant_name="completion_hybridp6_v1",
        paper_variant="CCR",
        role="internal_transfer_legacy",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_v1.json",
        generation_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_v1_generation.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_v1.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_v1_verifier_predictions.jsonl",
        notes=("Legacy archived summary is parser-corrupted; rely on reparsed expected-mode metrics instead.",),
    ),
    RunSpec(
        run_id="mmlu_pro_test_completion_hybridp6_benchmarkaware_v2",
        benchmark="mmlu_pro",
        split="test",
        variant_name="benchmarkaware_v2",
        paper_variant="CCR+Compat",
        role="ablation_transfer",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v2.json",
        generation_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v2_generation.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_v2.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v2_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v2_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v2_verifier_predictions.jsonl",
    ),
    RunSpec(
        run_id="mmlu_pro_test_completion_hybridp6_benchmarkaware_v3",
        benchmark="mmlu_pro",
        split="test",
        variant_name="benchmarkaware_v3",
        paper_variant="CCR+Compat+CH",
        role="ablation_transfer",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v3.json",
        generation_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v3_generation.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_v3.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v3_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v3_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_v3_verifier_predictions.jsonl",
    ),
    RunSpec(
        run_id="mmlu_pro_test_completion_hybridp6_benchmarkaware_qwen8bproposer_v2",
        benchmark="mmlu_pro",
        split="test",
        variant_name="benchmarkaware_qwen8bproposer_v2",
        paper_variant="CCR+Compat+SP",
        role="canonical_candidate",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_qwen8bproposer_v2.json",
        generation_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_qwen8bproposer_v2_generation.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_qwen8bproposer_v2.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_qwen8bproposer_v2_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_qwen8bproposer_v2_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_qwen8bproposer_v2_verifier_predictions.jsonl",
    ),
    RunSpec(
        run_id="mmlu_pro_test_completion_hybridp6_benchmarkaware_promptmixture_qwen8bproposer_v1_batched_v1",
        benchmark="mmlu_pro",
        split="test",
        variant_name="promptmixture_qwen8bproposer_v1_batched_v1",
        paper_variant="CCR+Compat+SP+PM",
        role="backup_candidate",
        result_json="Experiment/analysis/results/a800_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_promptmixture_qwen8bproposer_v1_batched_v1.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_promptmixture_qwen8bproposer_v1_batched_v1.jsonl",
        verifier_predictions="Experiment/core_code/logs/a800_generate_then_rerank_qwen3_17b_mmlu_pro_test_completion_hybridp6_benchmarkaware_promptmixture_qwen8bproposer_v1_batched_v1_verifier_predictions.jsonl",
        audit_scope_hint="partial_local",
        notes=(
            "Verifier prediction artifact exists locally.",
            "Raw candidate pool is missing; only partial hygiene audit is possible.",
        ),
    ),
    RunSpec(
        run_id="mmlu_pro_test_hetero_qwen8bproposerv2plusv3_benchmarkaware_v1",
        benchmark="mmlu_pro",
        split="test",
        variant_name="hetero_qwen8bproposerv2plusv3_benchmarkaware_v1",
        paper_variant="CCR+Compat+SP-Hetero",
        role="appendix_only",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_hetero_qwen8bproposerv2plusv3_benchmarkaware_v1.json",
        candidate_pool="Experiment/datasets/processed/mmlu_pro_test_generated_candidates_qwen3_17b_base_hetero_qwen8bproposerv2plusv3_benchmarkaware_v1.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_hetero_qwen8bproposerv2plusv3_benchmarkaware_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_mmlu_pro_test_hetero_qwen8bproposerv2plusv3_benchmarkaware_v1_verifier_predictions.jsonl",
        notes=("High oracle, but final verifier gain is not promotion-worthy.",),
    ),
    RunSpec(
        run_id="gpqa_diamond_train_hybridp6_v1",
        benchmark="gpqa_diamond",
        split="train",
        variant_name="hybridp6_v1",
        paper_variant="Diverse-Base",
        role="internal_baseline_transfer",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_hybridp6_v1.json",
        candidate_pool="Experiment/datasets/processed/gpqa_diamond_train_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_hybridp6_v1.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_hybridp6_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_hybridp6_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_hybridp6_v1_verifier_predictions.jsonl",
    ),
    RunSpec(
        run_id="gpqa_diamond_train_completion_hybridp6_v1",
        benchmark="gpqa_diamond",
        split="train",
        variant_name="completion_hybridp6_v1",
        paper_variant="CCR",
        role="internal_transfer_legacy",
        result_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_completion_hybridp6_v1.json",
        generation_json="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_completion_hybridp6_v1_generation.json",
        candidate_pool="Experiment/datasets/processed/gpqa_diamond_train_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_v1.jsonl",
        first_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_completion_hybridp6_v1_first_predictions.jsonl",
        base_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_completion_hybridp6_v1_base_rerank_predictions.jsonl",
        verifier_predictions="Experiment/core_code/logs/ser_generate_then_rerank_qwen3_17b_gpqa_diamond_train_completion_hybridp6_v1_verifier_predictions.jsonl",
        notes=("Legacy archived summary is parser-corrupted; rely on reparsed expected-mode metrics instead.",),
    ),
]


def resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_many(pattern: str | None) -> list[dict]:
    if pattern is None:
        return []
    path = resolve_path(pattern)
    if path is None:
        return []
    if "*" in pattern:
        records: list[dict] = []
        for item in sorted(ROOT.glob(pattern)):
            records.extend(read_jsonl(item))
        return records
    if path.exists():
        return read_jsonl(path)
    return []


def expected_mode(benchmark: str) -> str:
    return EXPECTED_MODES[benchmark]


def parse_answer(text: str, answer_mode: str) -> str | None:
    if answer_mode == "choice_letter":
        return extract_choice_answer(text)
    return extract_numeric_answer(text)


def canonical_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", normalize_answer(text))
    return collapsed.strip()


def dedupe_key(text: str, answer_mode: str) -> str:
    parsed = parse_answer(text, answer_mode)
    if parsed is not None:
        return f"parsed::{parsed}"
    return f"text::{canonical_text(text)}"


def contains_instruction_leak(text: str) -> bool:
    lowered = text.strip().lower()
    return any(pattern in lowered for pattern in INSTRUCTION_PATTERNS)


def contains_scaffold_residue(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(regex.search(stripped) for regex in SCAFFOLD_REGEXES):
        return True
    return lowered.startswith((
        "let me think",
        "let's think",
        "okay, let's",
        "here's the thought process",
        "i will now provide",
        "the sentence structure",
        "use at most",
    ))


def is_obviously_malformed(text: str, answer_mode: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return True
    if lowered in {"correct", "incorrect", "true", "false", "none", "n/a"}:
        return True
    if lowered.count("final answer:") > 1:
        return True
    if contains_instruction_leak(stripped):
        return True
    if stripped.endswith((":", "-", "=", "(", "[", "{", ",", " or", " and", " the")):
        return True
    if re.search(r"\[[^\]]*$|\([^\)]*$|\{[^\}]*$", stripped):
        return True
    if parse_answer(stripped, answer_mode) is None and len(stripped.split()) <= 2:
        return True
    return False


def clip(text: str, limit: int = 140) -> str:
    flat = re.sub(r"\s+", " ", text.strip())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def candidate_budget_from_rows(rows: list[dict]) -> int | None:
    if not rows:
        return None
    candidates = rows[0].get("candidates") or []
    return len(candidates)


def file_record(label: str, path_str: str | None) -> dict:
    if path_str is None:
        return {"label": label, "path": "", "sha256": None, "exists": False}
    if "*" in path_str:
        matches = sorted(ROOT.glob(path_str))
        return {
            "label": label,
            "path": path_str,
            "sha256": None,
            "exists": bool(matches),
        }
    path = resolve_path(path_str)
    exists = bool(path and path.exists())
    return {
        "label": label,
        "path": str(path) if path is not None else "",
        "sha256": sha256_file(path) if exists else None,
        "exists": exists,
    }


def infer_prompt_family(variant_name: str) -> str:
    if "promptmixture" in variant_name:
        return "completion+benchmark-aware+prompt-mixture"
    if "hetero" in variant_name:
        return "completion+benchmark-aware+heterogeneous"
    if "qwen8bproposer" in variant_name:
        return "completion+benchmark-aware+strong-proposer"
    if "benchmarkaware_v3" in variant_name:
        return "completion+benchmark-aware+candidate-hygiene"
    if "benchmarkaware_v2" in variant_name:
        return "completion+benchmark-aware"
    if "completion" in variant_name:
        return "completion-hybridp6"
    if "hybridp6" in variant_name or "filtered_full" in variant_name:
        return "diverse-base-hybridp6"
    return "unknown"


def infer_sanitizer_version(variant_name: str) -> str:
    if "promptmixture" in variant_name:
        return "AML+CH+PM"
    if "hetero" in variant_name:
        return "AML+CH+SP-Hetero"
    if "qwen8bproposer" in variant_name:
        return "AML+CH+SP"
    if "benchmarkaware_v3" in variant_name:
        return "AML+CH"
    if "benchmarkaware_v2" in variant_name:
        return "AML"
    if "completion" in variant_name:
        return "CCR-only"
    return "none"


def infer_proposer(variant_name: str) -> str | None:
    if "hetero" in variant_name:
        return "Qwen 8B proposer v2 + v3 heterogeneous"
    if "qwen8bproposer" in variant_name:
        return "Qwen 8B proposer"
    if "completion" in variant_name:
        return "Qwen3-1.7B completion generator"
    if "hybridp6" in variant_name or "filtered_full" in variant_name:
        return "Qwen3-1.7B diverse generator"
    return None


def evaluate_accuracy(rows: list[dict], answer_mode: str) -> float | None:
    if not rows:
        return None
    correct = 0
    for row in rows:
        correct += int(answers_match(str(row.get("prediction", "")), str(row.get("gold_answer", "")), answer_mode=answer_mode))
    return correct / len(rows)


def oracle_accuracy(candidate_rows: list[dict], answer_mode: str) -> float | None:
    if not candidate_rows:
        return None
    correct = 0
    for row in candidate_rows:
        candidates = [str(item) for item in row.get("candidates") or []]
        gold = str(row.get("gold_answer", ""))
        if any(answers_match(candidate, gold, answer_mode=answer_mode) for candidate in candidates):
            correct += 1
    return correct / len(candidate_rows)


def audit_run(spec: RunSpec) -> dict:
    result_path = resolve_path(spec.result_json)
    candidate_path = resolve_path(spec.candidate_pool)
    first_path = resolve_path(spec.first_predictions)
    base_path = resolve_path(spec.base_predictions)
    verifier_path = resolve_path(spec.verifier_predictions)

    result = read_json(result_path) if result_path and result_path.exists() else {}
    candidate_rows = read_jsonl(candidate_path) if candidate_path and candidate_path.exists() else []
    first_rows = read_jsonl(first_path) if first_path and first_path.exists() else []
    base_rows = read_jsonl(base_path) if base_path and base_path.exists() else []
    verifier_rows = read_jsonl(verifier_path) if verifier_path and verifier_path.exists() else []
    if spec.base_predictions_glob:
        base_rows = read_jsonl_many(spec.base_predictions_glob)
    if spec.verifier_predictions_glob:
        verifier_rows = read_jsonl_many(spec.verifier_predictions_glob)
    if not first_rows and candidate_rows:
        first_rows = [
            {
                "example_id": row.get("example_id"),
                "prediction": str((row.get("candidates") or [""])[0]),
                "gold_answer": row.get("gold_answer", ""),
                "answer_mode": row.get("answer_mode"),
            }
            for row in candidate_rows
        ]

    mode = expected_mode(spec.benchmark)
    candidate_budget = candidate_budget_from_rows(candidate_rows)

    candidate_slot_total = 0
    candidate_parseable = 0
    candidate_mode_matches = 0
    duplicate_slot_total = 0
    duplicate_examples = 0
    malformed_slot_total = 0
    empty_candidate_total = 0
    example_ids_with_duplicates: set[str] = set()

    candidate_map: dict[str, dict] = {}
    for row in candidate_rows:
        example_id = str(row.get("example_id"))
        candidate_map[example_id] = row
        row_mode = str(row.get("answer_mode", "numeric")).strip().lower() or "numeric"
        candidate_mode_matches += int(row_mode == mode)
        candidates = [str(item) for item in row.get("candidates") or []]
        candidate_slot_total += len(candidates)
        keys = [dedupe_key(candidate, mode) for candidate in candidates]
        unique_count = len(set(keys))
        duplicate_slot_total += len(candidates) - unique_count
        if unique_count < len(candidates):
            duplicate_examples += 1
            example_ids_with_duplicates.add(example_id)
        for candidate in candidates:
            parsed = parse_answer(candidate, mode)
            candidate_parseable += int(parsed is not None)
            malformed_slot_total += int(is_obviously_malformed(candidate, mode))
            empty_candidate_total += int(not candidate.strip())

    verifier_parseable = 0
    verifier_mode_matches = 0
    verifier_instruction = 0
    verifier_scaffold = 0
    verifier_malformed = 0
    verifier_map: dict[str, dict] = {}
    for row in verifier_rows:
        example_id = str(row.get("example_id"))
        verifier_map[example_id] = row
        prediction = str(row.get("prediction", ""))
        verifier_parseable += int(parse_answer(prediction, mode) is not None)
        row_mode = str(row.get("answer_mode", "numeric")).strip().lower() or "numeric"
        verifier_mode_matches += int(row_mode == mode)
        verifier_instruction += int(contains_instruction_leak(prediction))
        verifier_scaffold += int(contains_scaffold_residue(prediction))
        verifier_malformed += int(is_obviously_malformed(prediction, mode))

    base_map = {str(row.get("example_id")): row for row in base_rows}
    first_map = {str(row.get("example_id")): row for row in first_rows}

    metadata_counts = Counter()
    outcome_counts = Counter()
    metadata_samples: dict[str, list[dict]] = defaultdict(list)
    outcome_samples: dict[str, list[dict]] = defaultdict(list)

    common_ids: Iterable[str]
    if candidate_map and verifier_map:
        common_ids = sorted(set(candidate_map) & set(verifier_map))
    else:
        common_ids = []

    for example_id in common_ids:
        candidate_row = candidate_map[example_id]
        verifier_row = verifier_map[example_id]
        candidates = [str(item) for item in candidate_row.get("candidates") or []]
        gold_answer = str(candidate_row.get("gold_answer", ""))
        verifier_prediction = str(verifier_row.get("prediction", ""))
        first_prediction = str(first_map.get(example_id, {}).get("prediction", candidates[0] if candidates else ""))
        verifier_correct = answers_match(verifier_prediction, gold_answer, answer_mode=mode)
        first_correct = answers_match(first_prediction, gold_answer, answer_mode=mode)
        oracle_hit = any(answers_match(candidate, gold_answer, answer_mode=mode) for candidate in candidates)

        row_mode = str(candidate_row.get("answer_mode", "numeric")).strip().lower() or "numeric"
        verifier_row_mode = str(verifier_row.get("answer_mode", "numeric")).strip().lower() or "numeric"
        if row_mode != mode:
            metadata_counts["candidate_answer_mode_mismatch"] += 1
            if len(metadata_samples["candidate_answer_mode_mismatch"]) < 3:
                metadata_samples["candidate_answer_mode_mismatch"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                })
        if verifier_row_mode != mode:
            metadata_counts["prediction_answer_mode_mismatch"] += 1
            if len(metadata_samples["prediction_answer_mode_mismatch"]) < 3:
                metadata_samples["prediction_answer_mode_mismatch"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                })
        if parse_answer(verifier_prediction, mode) is None:
            metadata_counts["invalid_selected_answer"] += 1
            if len(metadata_samples["invalid_selected_answer"]) < 3:
                metadata_samples["invalid_selected_answer"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                })
        if contains_instruction_leak(verifier_prediction):
            metadata_counts["instruction_leak"] += 1
            if len(metadata_samples["instruction_leak"]) < 3:
                metadata_samples["instruction_leak"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                })
        if contains_scaffold_residue(verifier_prediction):
            metadata_counts["scaffold_residue"] += 1
            if len(metadata_samples["scaffold_residue"]) < 3:
                metadata_samples["scaffold_residue"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                })
        if example_id in example_ids_with_duplicates:
            metadata_counts["duplicate_candidate_example"] += 1
            if len(metadata_samples["duplicate_candidate_example"]) < 3:
                metadata_samples["duplicate_candidate_example"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                })

        if verifier_correct:
            outcome_counts["verifier_correct"] += 1
        if not oracle_hit:
            outcome_counts["oracle_miss"] += 1
            if len(outcome_samples["oracle_miss"]) < 3:
                outcome_samples["oracle_miss"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                    "first": clip(first_prediction),
                })
        if oracle_hit and not verifier_correct:
            outcome_counts["oracle_hit_but_verifier_wrong"] += 1
            if len(outcome_samples["oracle_hit_but_verifier_wrong"]) < 3:
                outcome_samples["oracle_hit_but_verifier_wrong"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                    "first": clip(first_prediction),
                })
        if first_correct and not verifier_correct:
            outcome_counts["verifier_overruled_correct_first"] += 1
            if len(outcome_samples["verifier_overruled_correct_first"]) < 3:
                outcome_samples["verifier_overruled_correct_first"].append({
                    "example_id": example_id,
                    "gold": gold_answer,
                    "prediction": clip(verifier_prediction),
                    "first": clip(first_prediction),
                })

    reported_first = result.get("first_accuracy")
    reported_base = result.get("base_accuracy")
    reported_verifier = result.get("verifier_accuracy")
    reported_oracle = result.get("oracle_coverage")
    verifier_given_oracle = None
    if isinstance(result.get("selection_efficiency_given_oracle"), dict):
        verifier_given_oracle = result["selection_efficiency_given_oracle"].get("verifier")
    p_base_vs_verifier = None
    if isinstance(result.get("paired_comparisons"), dict):
        base_vs_verifier = result["paired_comparisons"].get("base_vs_verifier") or {}
        p_base_vs_verifier = base_vs_verifier.get("p")

    reparsed_first = evaluate_accuracy(first_rows, mode)
    reparsed_base = evaluate_accuracy(base_rows, mode)
    reparsed_verifier = evaluate_accuracy(verifier_rows, mode)
    reparsed_oracle = oracle_accuracy(candidate_rows, mode)

    output_files = [
        file_record("summary_json", spec.result_json),
        file_record("generation_json", spec.generation_json),
        file_record("candidate_pool", spec.candidate_pool),
        file_record("first_predictions", spec.first_predictions),
        file_record("base_predictions", spec.base_predictions),
        file_record("verifier_predictions", spec.verifier_predictions),
        file_record("base_predictions_glob", spec.base_predictions_glob),
        file_record("verifier_predictions_glob", spec.verifier_predictions_glob),
    ]

    if spec.audit_scope_hint == "summary_only":
        artifact_status = "summary_only"
        audit_scope = "summary_only"
    elif spec.audit_scope_hint == "partial_local" or not candidate_rows or not verifier_rows:
        artifact_status = "partial_local"
        audit_scope = "partial_local"
    else:
        artifact_status = "full_local"
        audit_scope = "full_local"

    return {
        "run_id": spec.run_id,
        "benchmark": spec.benchmark,
        "split": spec.split,
        "variant_name": spec.variant_name,
        "paper_facing_variant_name": spec.paper_variant,
        "role": spec.role,
        "notes": list(spec.notes),
        "expected_answer_mode": mode,
        "candidate_budget": candidate_budget,
        "reported": {
            "total_examples": result.get("total_examples"),
            "first_accuracy": reported_first,
            "base_accuracy": reported_base,
            "verifier_accuracy": reported_verifier,
            "oracle_accuracy": reported_oracle,
            "verifier_given_oracle_accuracy": verifier_given_oracle,
            "p_base_vs_verifier": p_base_vs_verifier,
        },
        "reparsed": {
            "first_accuracy": reparsed_first,
            "base_accuracy": reparsed_base,
            "verifier_accuracy": reparsed_verifier,
            "oracle_accuracy": reparsed_oracle,
        },
        "hygiene": {
            "candidate_examples": len(candidate_rows),
            "candidate_slot_total": candidate_slot_total,
            "candidate_parseable_rate": (candidate_parseable / candidate_slot_total) if candidate_slot_total else None,
            "candidate_row_answer_mode_match_rate": (candidate_mode_matches / len(candidate_rows)) if candidate_rows else None,
            "verifier_examples": len(verifier_rows),
            "selected_prediction_parseable_rate": (verifier_parseable / len(verifier_rows)) if verifier_rows else None,
            "selected_prediction_answer_mode_match_rate": (verifier_mode_matches / len(verifier_rows)) if verifier_rows else None,
            "invalid_final_answer_rate": 1 - (verifier_parseable / len(verifier_rows)) if verifier_rows else None,
            "instruction_leak_rate": (verifier_instruction / len(verifier_rows)) if verifier_rows else None,
            "scaffold_residue_rate": (verifier_scaffold / len(verifier_rows)) if verifier_rows else None,
            "malformed_selected_rate": (verifier_malformed / len(verifier_rows)) if verifier_rows else None,
            "duplicate_slot_rate": (duplicate_slot_total / candidate_slot_total) if candidate_slot_total else None,
            "duplicate_example_rate": (duplicate_examples / len(candidate_rows)) if candidate_rows else None,
            "malformed_slot_rate": (malformed_slot_total / candidate_slot_total) if candidate_slot_total else None,
            "empty_candidate_rate": (empty_candidate_total / candidate_slot_total) if candidate_slot_total else None,
        },
        "error_taxonomy": {
            "metadata_counts": dict(metadata_counts),
            "outcome_counts": dict(outcome_counts),
            "metadata_samples": metadata_samples,
            "outcome_samples": outcome_samples,
        },
        "artifact_status": artifact_status,
        "audit_scope": audit_scope,
        "output_files": output_files,
        "summary_json_path": str(result_path) if result_path else None,
    }


def build_manifest_instance(run: dict) -> dict:
    hygiene = run["hygiene"]
    reparsed = run["reparsed"]
    return {
        "run_id": run["run_id"],
        "benchmark": run["benchmark"],
        "variant_name": run["variant_name"],
        "paper_facing_variant_name": run["paper_facing_variant_name"],
        "candidate_budget": run["candidate_budget"],
        "generator": "Qwen3-1.7B base filtered candidate generator",
        "proposer": infer_proposer(run["variant_name"]),
        "verifier": "Qwen3-1.7B yes/no verifier reranker",
        "prompt_family": infer_prompt_family(run["variant_name"]),
        "sanitizer_version": infer_sanitizer_version(run["variant_name"]),
        "scoring_script_version": str(ROOT / "Experiment" / "core_code" / "src" / "eval" / "evaluate_predictions.py"),
        "seed": None,
        "first_accuracy": reparsed.get("first_accuracy") if reparsed.get("first_accuracy") is not None else run["reported"].get("first_accuracy"),
        "base_accuracy": reparsed.get("base_accuracy") if reparsed.get("base_accuracy") is not None else run["reported"].get("base_accuracy"),
        "verifier_accuracy": reparsed.get("verifier_accuracy") if reparsed.get("verifier_accuracy") is not None else run["reported"].get("verifier_accuracy"),
        "oracle_accuracy": reparsed.get("oracle_accuracy") if reparsed.get("oracle_accuracy") is not None else run["reported"].get("oracle_accuracy"),
        "verifier_given_oracle_accuracy": run["reported"].get("verifier_given_oracle_accuracy"),
        "answer_format_validity": hygiene.get("selected_prediction_parseable_rate"),
        "scaffold_leakage_rate": hygiene.get("scaffold_residue_rate"),
        "invalid_final_answer_rate": hygiene.get("invalid_final_answer_rate"),
        "duplicate_candidate_rate": hygiene.get("duplicate_slot_rate"),
        "output_files": run["output_files"],
        "notes": run["notes"],
    }


def build_parser_report(runs: list[dict]) -> str:
    correction_rows: list[list[str]] = []
    for run in runs:
        if run["benchmark"] not in {"competition_math_numeric", "mmlu_pro", "gpqa_diamond"}:
            continue
        hygiene = run["hygiene"]
        correction_rows.append([
            run["benchmark"],
            run["paper_facing_variant_name"],
            run["variant_name"],
            fmt(run["reported"].get("verifier_accuracy")),
            fmt(run["reparsed"].get("verifier_accuracy")),
            fmt(run["reported"].get("oracle_accuracy")),
            fmt(run["reparsed"].get("oracle_accuracy")),
            fmt(hygiene.get("candidate_parseable_rate")),
            fmt(hygiene.get("selected_prediction_parseable_rate")),
            fmt(hygiene.get("candidate_row_answer_mode_match_rate")),
            fmt(hygiene.get("selected_prediction_answer_mode_match_rate")),
            run["audit_scope"],
        ])

    best_mode_match = max((run["hygiene"].get("selected_prediction_answer_mode_match_rate") or 0.0) for run in runs if run["benchmark"] in {"mmlu_pro", "gpqa_diamond"})
    best_selected_parse = max((run["hygiene"].get("selected_prediction_parseable_rate") or 0.0) for run in runs)
    best_invalid_rate = min((run["hygiene"].get("invalid_final_answer_rate") or 1.0) for run in runs if run["hygiene"].get("invalid_final_answer_rate") is not None)
    gate_rows = [
        [
            "Multiple-choice answer_mode locked to choice_letter",
            ">= 0.99 row-mode match",
            fmt(best_mode_match),
            "PASS" if best_mode_match >= 0.99 else "FAIL",
        ],
        [
            "Selected answer parseability",
            ">= 0.99",
            fmt(best_selected_parse),
            "PASS" if best_selected_parse >= 0.99 else "FAIL",
        ],
        [
            "Invalid selected answer rate",
            "<= 0.01",
            fmt(best_invalid_rate),
            "PASS" if best_invalid_rate <= 0.01 else "FAIL",
        ],
    ]

    return f'''# Parser Audit Report

Generated at {datetime.now(timezone.utc).isoformat()}.

## Scope

- Parser implementation audited from `{ROOT / "Experiment/core_code/src/eval/evaluate_predictions.py"}`.
- Benchmark answer-mode ground truth locked from `{ROOT / "Experiment/core_code/scripts/prepare_reasoning_benchmark.py"}`.
- Multiple-choice benchmarks are hard-locked to `choice_letter`; numeric benchmarks are hard-locked to `numeric`.

## Expected Answer Modes

{markdown_table(["benchmark", "expected answer_mode"], [[k, v] for k, v in EXPECTED_MODES.items() if k != "gsm8k"])}

## Reparsed Corrections

{markdown_table([
    "benchmark",
    "paper variant",
    "internal variant",
    "reported verifier",
    "reparsed verifier",
    "reported oracle",
    "reparsed oracle",
    "candidate parseable",
    "selected parseable",
    "candidate mode match",
    "prediction mode match",
    "audit scope",
], correction_rows)}

## Findings

- Legacy `mmlu_pro` and `gpqa_diamond` completion rows in archived result JSONs are not paper-safe as-is: the run artifacts carry `answer_mode=numeric`, while the benchmark ground truth is `choice_letter`.
- After reparsing with the correct benchmark mode, the earlier apparent near-zero collapse is largely an evaluation artifact. This is most visible for `mmlu_pro / CCR` and `gpqa_diamond / CCR`, whose reparsed verifier accuracies are materially higher than the archived summaries.
- Fixing `answer_mode` alone does not solve P0. Even the repaired `CCR+Compat+SP` row still has low selected-answer parseability and non-trivial scaffold residue.
- `competition_math_numeric` never suffered answer-mode mismatch, so its archived and reparsed scores align. The main remaining issue there is candidate/prediction cleanliness rather than parser semantics.

## P0 Gate Status

{markdown_table(["gate", "threshold", "best current", "status"], gate_rows)}

## Decision

P0 parser gate is **not passed**. No full rerun can be treated as paper-grade evidence until parser lock, answer-mode lock, and final-answer validity are jointly brought to the required range.
'''


def build_candidate_hygiene_report(runs: list[dict]) -> str:
    rows: list[list[str]] = []
    for run in runs:
        hygiene = run["hygiene"]
        if run["audit_scope"] == "summary_only":
            rows.append([
                run["benchmark"],
                run["paper_facing_variant_name"],
                run["variant_name"],
                run["audit_scope"],
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                run["artifact_status"],
            ])
            continue
        rows.append([
            run["benchmark"],
            run["paper_facing_variant_name"],
            run["variant_name"],
            run["audit_scope"],
            fmt(hygiene.get("candidate_parseable_rate")),
            fmt(hygiene.get("selected_prediction_parseable_rate")),
            fmt(hygiene.get("duplicate_slot_rate")),
            fmt(hygiene.get("duplicate_example_rate")),
            fmt(hygiene.get("instruction_leak_rate")),
            fmt(hygiene.get("scaffold_residue_rate")),
            fmt(hygiene.get("malformed_selected_rate")),
            run["artifact_status"],
        ])

    return f'''# Candidate Hygiene Report

Generated at {datetime.now(timezone.utc).isoformat()}.

## Scope

This report audits candidate and selected-prediction hygiene for the strongest currently available internal variants. Rates are computed against the benchmark-locked answer mode rather than the artifact-carried mode when those disagree.

## Hygiene Summary

{markdown_table([
    "benchmark",
    "paper variant",
    "internal variant",
    "audit scope",
    "candidate parseable",
    "selected parseable",
    "duplicate slot",
    "duplicate example",
    "instruction leak",
    "scaffold residue",
    "malformed selected",
    "artifact status",
], rows)}

## Findings

- Duplicate candidate rates remain high across the auditable pools, especially on `competition_math_numeric`, `mmlu_pro`, and the new GSM8K clean reruns. This weakens the effective candidate budget and makes oracle-coverage gains harder to convert into final verifier gains.
- Legacy multiple-choice completion pools still carry substantial instruction leakage and scaffold residue. The `CCR+Compat+SP` line is materially better, but still not within the P0 target band.
- `CCR+Compat+SP+PM` is **not fully auditable** because the raw candidate pool is missing locally. Its verifier score can be used as a pointer for backup-variant selection, but not as paper-grade main evidence.
- GSM8K `P1-A` clean rerun rows are now fully auditable locally. The remaining blocker is no longer missing artifacts; it is that final-answer hygiene and selector conversion are still below the paper-grade target band.

## P0 Gate Readout

- Required: multiple-choice legal parseability `>= 0.99`. Current best is well below threshold.
- Required: invalid final answer `<= 0.01`. Current best remains above threshold.
- Required: scaffold residue `<= 0.01`. Current best remains above threshold.
- Required: obvious malformed candidate/selection `<= 0.01`. Current best remains above threshold.

P0 hygiene gate is **not passed**.
'''



def build_error_taxonomy_report(selected_runs: list[dict]) -> str:
    sections: list[str] = [f"# Error Taxonomy Report\n\nGenerated at {datetime.now(timezone.utc).isoformat()}.\n"]
    for run in selected_runs:
        meta_counts = run["error_taxonomy"]["metadata_counts"]
        out_counts = run["error_taxonomy"]["outcome_counts"]
        total = max(run["reported"].get("total_examples") or 0, 1)
        failure_total = max(total - int((run["reparsed"].get("verifier_accuracy") or 0.0) * total), 1)
        meta_rows = []
        for key in [
            "candidate_answer_mode_mismatch",
            "prediction_answer_mode_mismatch",
            "invalid_selected_answer",
            "instruction_leak",
            "scaffold_residue",
            "duplicate_candidate_example",
        ]:
            count = int(meta_counts.get(key, 0))
            meta_rows.append([key, str(count), fmt(count / total)])
        outcome_rows = []
        for key in [
            "oracle_miss",
            "oracle_hit_but_verifier_wrong",
            "verifier_overruled_correct_first",
            "verifier_correct",
        ]:
            count = int(out_counts.get(key, 0))
            denom = total if key == "verifier_correct" else failure_total
            outcome_rows.append([key, str(count), fmt(count / denom)])

        sample_lines: list[str] = []
        for bucket, samples in run["error_taxonomy"]["metadata_samples"].items():
            for sample in samples[:1]:
                sample_lines.append(f"- metadata::{bucket}: `{sample['example_id']}` gold=`{sample['gold']}` pred=`{sample['prediction']}`")
        for bucket, samples in run["error_taxonomy"]["outcome_samples"].items():
            for sample in samples[:1]:
                sample_lines.append(f"- outcome::{bucket}: `{sample['example_id']}` gold=`{sample['gold']}` first=`{sample.get('first','')}` pred=`{sample['prediction']}`")

        sections.append(f'''## {run["benchmark"]} / {run["paper_facing_variant_name"]} / {run["variant_name"]}

- Reported verifier accuracy: {fmt(run["reported"].get("verifier_accuracy"))}
- Reparsed verifier accuracy: {fmt(run["reparsed"].get("verifier_accuracy"))}
- Reparsed oracle accuracy: {fmt(run["reparsed"].get("oracle_accuracy"))}
- Audit scope: `{run["audit_scope"]}`

### Metadata / Format Buckets

{markdown_table(["bucket", "count", "rate over total"], meta_rows)}

### Outcome Buckets

{markdown_table(["bucket", "count", "rate"], outcome_rows)}

### Representative Examples

{chr(10).join(sample_lines) if sample_lines else '- No local example samples available.'}
''')
    return "\n".join(sections)


def build_variant_registry() -> str:
    headers = ["internal name", "paper-facing module / variant", "status", "notes"]
    rows = [
        ["filtered_full_v1 / hybridp6_v1", "Diverse-Base", "keep as baseline family", "Use `full_hybridp6_v1` as main GSM8K baseline label; `filtered_full_v1` remains the local audit anchor."],
        ["completion_hybridp6_v1", "CCR", "keep", "Core completion-oriented candidate construction."],
        ["benchmarkaware_v2", "CCR+Compat", "keep as ablation", "Answer-mode lock / benchmark compatibility layer."],
        ["benchmarkaware_v3", "CCR+Compat+CH", "keep as ablation", "Adds candidate hygiene tightening on top of compatibility lock."],
        ["benchmarkaware_qwen8bproposer_v2", "CCR+Compat+SP", "freeze as canonical main variant", "Current strongest fully auditable strengthening path."],
        ["promptmixture_qwen8bproposer_v1_batched_v1", "CCR+Compat+SP+PM", "freeze as backup only", "Slightly stronger MMLU verifier, but raw candidate pool is missing locally."],
        ["hetero_qwen8bproposerv2plusv3_benchmarkaware_v1", "CCR+Compat+SP-Hetero", "appendix only", "Boosts oracle more than final verifier; do not promote to main variant."],
        ["repair / residual lines", "CH-only exploratory branches", "deprecate from headline", "Do not use as main contribution during NeurIPS main-track closing phase."],
    ]
    return f'''# Variant Registry

## Canonical Modules

- `CCR`: Core Completion Replacement.
- `Compat`: answer-mode lock and benchmark-aware compatibility handling.
- `CH`: candidate hygiene tightening.
- `SP`: stronger proposer.
- `PM`: prompt mixture.
- `VR`: verifier reranking. This is the constant selection stage, not a branching variant suffix.

## Mapping

{markdown_table(headers, rows)}

## Freeze Recommendation

- Canonical main variant: `CCR+Compat+SP`.
- Backup main variant: `CCR+Compat+SP+PM`.
- Do not actively maintain more than these two strengthening paths during the NeurIPS closing phase.
- `CCR+Compat+SP-Hetero` remains appendix-only until it improves final verifier accuracy rather than oracle alone.
'''


def build_claim_ledger() -> str:
    return '''# Paper Claim Ledger

## 可进入主文主体，但只能作为 Route B 风格的 analysis-driven 叙事

- 在可审计的 GSM8K full-split clean rerun 上，completion-oriented candidate construction 相对 matched-budget Diverse-Base clean baseline 将 oracle coverage 从 `0.4882` 提升到 `0.6391`，并将 final verifier accuracy 从 `0.2873` 提升到 `0.3632`。
- open-ended generate-then-rerank 的主要瓶颈仍然是 oracle miss / candidate coverage，而不仅仅是 selector weakness。
- 旧版 MMLU-Pro / GPQA 近零分结论并不可信；P0 审计表明其中很大一部分来自 answer-mode mismatch，而不是方法本体必然崩塌。

## 只能写进 discussion / appendix

- GSM8K clean rerun 仍未达到 headline 级门槛：`verifier=0.3632 < 0.40`，且 `verifier_given_oracle` 从 `0.5885` 降到 `0.5682`，说明 coverage 增益尚未被 selector 充分转化。
- `P1-B` full-split external baseline 扩展已经完成，且 `Self-Refine = 0.4875` 明显高于 clean `CCR = 0.3632`；因此任何 internal-vs-external superiority 叙事都不再成立。
- parser-corrected transfer evidence 目前对 completion family 是正向的，但 canonical same-version row 仍未闭合，不能升格为 headline transfer claim。
- `CCR+Compat+SP+PM` 在 MMLU 上略高于 `CCR+Compat+SP`，但因 raw candidate pool 缺失，只能作为 backup-variant 参考。
- heterogeneous proposer mixture 提升 oracle coverage，但 final verifier accuracy 不足以支持 main-variant promotion。

## 必须删除或降级

- `KV cache reuse` 作为主贡献。
- efficiency 作为 headline selling point。
- “当前 internal main variant 优于 strongest landed external baseline” 这种表述。
- “稳健通用的 broad reasoning transfer 方法” 这种大 claim。
- residual repair / repair hygiene branch 作为主结果。
- incomplete transfer row 或小样本 eval-128 结果作为 abstract headline。
- 用内部分支名直接进入正文方法命名。

## 当前结论

当前最安全的主线只剩下 Route B 风格的 analysis-driven 叙事：`completion-oriented candidate construction + verifier reranking` 在内部 matched-budget 比较中确实提升 coverage，但现有方法不能被描述为比 strongest landed external baseline 更强，也不能被写成 broad transfer headline。
'''



def build_ablation_summary(runs: list[dict]) -> str:
    wanted = {
        "Diverse-Base": None,
        "CCR": None,
        "CCR+Compat": None,
        "CCR+Compat+CH": None,
        "CCR+Compat+SP": None,
        "CCR+Compat+SP+PM": None,
        "CCR+Compat+SP-Hetero": None,
    }
    for run in runs:
        if run["benchmark"] == "mmlu_pro" and run["paper_facing_variant_name"] in wanted:
            wanted[run["paper_facing_variant_name"]] = run
    rows = []
    for key in wanted:
        run = wanted[key]
        if run is None:
            rows.append([key, "NA", "NA", "missing"])
            continue
        rows.append([
            key,
            fmt(run["reparsed"].get("verifier_accuracy")),
            fmt(run["reparsed"].get("oracle_accuracy")),
            run["audit_scope"],
        ])
    return f'''# Ablation Summary

## Current MMLU-Pro Ladder (Reparsed Expected-Mode Metrics)

{markdown_table(["paper variant", "verifier", "oracle", "audit scope"], rows)}

## Readout

- `Diverse-Base -> CCR` is the cleanest current evidence that gains come from completion-oriented construction rather than generic diversity alone.
- On the GSM8K `P1-A` clean rerun, `Diverse-Base -> CCR` yields `oracle +0.1509` and `verifier +0.0758`, but also `verifier_given_oracle -0.0203`; this supports the coverage story while showing the remaining selector-conversion gap.
- `CCR -> CCR+Compat / CCR+Compat+CH` improves benchmark compatibility and oracle coverage, but the strongest clean final-accuracy jump comes only after `SP`.
- `CCR+Compat+SP` is the current canonical promotion winner because it is both strongest among fully auditable variants and methodologically clean.
- `CCR+Compat+SP+PM` is a plausible backup, but raw-candidate incompleteness keeps it out of the main line for now.
- `CCR+Compat+SP-Hetero` improves oracle much more than final verifier; this makes it useful for analysis, not for the main variant slot.

## Missing P2 Ablations

- `CCR-targeted vs CCR-random-slot-replacement`.
- Clean AML/CH ablation outside MMLU-only evidence.
- Stronger proposer / prompt mixture promotion criterion tied to final verifier accuracy and macro robustness.
'''



def build_submission_readiness_review() -> str:
    return '''# Submission Readiness Review

## Verdict

- Recommended route: `Route B`.
- Current DoD status: `not met`.
- Current writing recommendation: only continue with a narrowed Route-B-style paper framing; do **not** write as if the current method is already a NeurIPS-main-track winning broad method paper.

## Why Route B Is Now Required

- P0 is now audited, but the P0 gate is not passed.
- `P1-A` GSM8K full clean rerun completed on `2026-04-07` and is now locally reproducible, but the main clean row still misses the promotion bar: `verifier=0.3632 < 0.40`, and `verifier_given_oracle` fell from `0.5885` to `0.5682`.
- `P1-B` GSM8K full Self-Refine expansion also completed on `2026-04-07` and materially outperforms the clean internal main row: `Self-Refine = 0.4875` vs `clean CCR = 0.3632`, with paired exact McNemar `p = 8.62e-13`.
- Canonical same-version transfer closure is still incomplete: the preferred canonical variant is `CCR+Compat+SP`, but the repository does not yet contain a matching completed row for all target transfer benchmarks.

## What Can Enter Abstract Right Now

- No broad transfer claim is abstract-safe.
- The only defensible abstract direction is a narrow analysis-driven GSM8K statement: under matched internal candidate budget, completion-oriented candidate construction raises oracle coverage from `0.4882` to `0.6391` and verifier accuracy from `0.2873` to `0.3632` relative to the clean Diverse-Base baseline.
- The abstract must **not** imply that the current method beats the strongest landed external baseline or that the project already has broad transfer closure.

## What Can Enter Discussion / Appendix

- `P1-B` provides a useful negative-control result: Self-Refine is stronger on GSM8K full even though its selected-answer parseability is only `0.8961` and invalid final-answer rate is `0.1039`. This constrains the paper to an evidence-limited analysis story rather than a best-method story.
- Parser-corrected MMLU-Pro and GPQA evidence indicates that the earlier apparent collapse was largely an evaluation hygiene failure.
- Transfer family evidence is positive enough to justify continued closure work, but not positive enough to headline broad generalization yet.
- Backup and heterogeneous branches may be discussed as exploratory strengthening paths.
- The `P1-A` clean rerun remains auditable evidence that coverage gains are real but incompletely converted by the current selector.

## What Must Be Deleted From Headline Narrative

- KV cache reuse.
- efficiency-first framing.
- any claim that the current internal main variant is stronger than the strongest landed external baseline on GSM8K.
- broad universal reasoning-transfer claims.
- residual repair as core contribution.
- incomplete transfer rows as robustness evidence.

## Immediate Blockers

- Final-answer validity and scaffold residue remain above P0 limits.
- `P1-A` clean rerun missed the GSM8K promotion gate on final verifier accuracy.
- `P1-B` shows the strongest landed external baseline still beats the current internal main row on GSM8K full.
- Canonical transfer closure for `CCR+Compat+SP` is incomplete.

## Freeze Recommendation

- Canonical main variant: `CCR+Compat+SP`.
- Backup variant: `CCR+Compat+SP+PM`.
- Do not promote hetero or repair-only lines.

## Go / No-Go

- Go for Route-B-only paper framing plus any narrowly targeted selector-conversion repair that can be cleanly explained.
- No-go for Route A and no-go for any headline that frames the current method as stronger than Self-Refine on GSM8K full.
'''



def load_external_gsm8k_full_self_refine_summary() -> dict | None:
    path = ROOT / "Experiment" / "analysis" / "results" / "experiment_12_gsm8k_full_self_refine_p1b_result_v1.json"
    if not path.exists():
        return None
    payload = read_json(path)
    payload["_summary_path"] = str(path)
    return payload


def build_external_gsm8k_full_manifest(summary: dict) -> dict:
    metrics_path = ROOT / "Experiment" / "core_code" / "logs" / "a800_self_refine_gsm8k_full_p1b_v1_metrics.json"
    predictions_path = ROOT / "Experiment" / "core_code" / "logs" / "a800_self_refine_gsm8k_full_p1b_v1_predictions.jsonl"
    trace_path = ROOT / "Experiment" / "core_code" / "logs" / "a800_self_refine_gsm8k_full_p1b_v1_trace.jsonl"
    report_path = ROOT / "Experiment" / "analysis" / "results" / "a800_self_refine_gsm8k_full_p1b_v1_eval.md"
    output_files = []
    for label, path in [
        ("summary_json", Path(summary["_summary_path"])),
        ("report", report_path),
        ("metrics", metrics_path),
        ("predictions", predictions_path),
        ("trace", trace_path),
    ]:
        output_files.append({
            "label": label,
            "path": str(path),
            "sha256": sha256_file(path),
            "exists": path.exists(),
        })
    return {
        "run_id": "gsm8k_full_self_refine_p1b_v1",
        "benchmark": "gsm8k",
        "variant_name": "self_refine_full_p1b_v1",
        "paper_facing_variant_name": "Self-Refine",
        "candidate_budget": None,
        "generator": "Qwen3-1.7B base causal LM",
        "proposer": "Self-Refine iterative refinement",
        "verifier": None,
        "prompt_family": "self_refine_round2",
        "sanitizer_version": "none",
        "scoring_script_version": str(ROOT / "Experiment" / "core_code" / "src" / "eval" / "evaluate_predictions.py"),
        "seed": 7,
        "first_accuracy": None,
        "base_accuracy": None,
        "verifier_accuracy": summary.get("accuracy"),
        "oracle_accuracy": None,
        "verifier_given_oracle_accuracy": None,
        "answer_format_validity": summary.get("selected_prediction_parseable_rate"),
        "scaffold_leakage_rate": summary.get("scaffold_residue_rate"),
        "invalid_final_answer_rate": summary.get("invalid_final_answer_rate"),
        "duplicate_candidate_rate": None,
        "output_files": output_files,
        "notes": [
            "P1-B GSM8K full Self-Refine expansion completed on 2026-04-07.",
            f"Self-Refine full accuracy={summary.get('accuracy'):.4f}; delta vs clean CCR={summary.get('comparisons', {}).get('delta_vs_clean_ccr'):+.4f}.",
            "Use this row to block any unsupported internal-vs-external superiority claim on GSM8K full.",
        ],
    }


def build_reproducibility_manifest(generated_files: list[Path], source_files: list[Path]) -> str:
    output_rows = [[str(path), sha256_file(path) or "NA"] for path in generated_files if path.exists()]
    source_rows = [[str(path), sha256_file(path) or "NA"] for path in source_files if path.exists()]
    return f'''# Reproducibility Pack Manifest

## Generated Files

{markdown_table(["path", "sha256"], output_rows)}

## Source Scripts / Artifacts Referenced

{markdown_table(["path", "sha256"], source_rows)}
'''


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def build_scoreboards(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    main_rows: list[dict] = []
    transfer_rows: list[dict] = []

    for run in runs:
        row = {
            "benchmark": run["benchmark"],
            "split": run["split"],
            "variant_name": run["variant_name"],
            "paper_facing_variant_name": run["paper_facing_variant_name"],
            "role": run["role"],
            "reported_first_accuracy": run["reported"].get("first_accuracy"),
            "reported_base_accuracy": run["reported"].get("base_accuracy"),
            "reported_verifier_accuracy": run["reported"].get("verifier_accuracy"),
            "reported_oracle_accuracy": run["reported"].get("oracle_accuracy"),
            "reparsed_first_accuracy": run["reparsed"].get("first_accuracy"),
            "reparsed_base_accuracy": run["reparsed"].get("base_accuracy"),
            "reparsed_verifier_accuracy": run["reparsed"].get("verifier_accuracy"),
            "reparsed_oracle_accuracy": run["reparsed"].get("oracle_accuracy"),
            "verifier_given_oracle_accuracy": run["reported"].get("verifier_given_oracle_accuracy"),
            "p_base_vs_verifier": run["reported"].get("p_base_vs_verifier"),
            "candidate_parseable_rate": run["hygiene"].get("candidate_parseable_rate"),
            "selected_prediction_parseable_rate": run["hygiene"].get("selected_prediction_parseable_rate"),
            "duplicate_slot_rate": run["hygiene"].get("duplicate_slot_rate"),
            "scaffold_residue_rate": run["hygiene"].get("scaffold_residue_rate"),
            "artifact_status": run["artifact_status"],
            "audit_scope": run["audit_scope"],
            "summary_json": run["summary_json_path"],
        }
        if run["benchmark"] == "gsm8k":
            main_rows.append(row)
        else:
            transfer_rows.append(row)

    external_full = load_external_gsm8k_full_self_refine_summary()
    if external_full is not None:
        main_rows.append({
            "benchmark": "gsm8k",
            "split": "full",
            "variant_name": "Self-Refine",
            "paper_facing_variant_name": "Self-Refine",
            "role": "external_baseline_full",
            "reported_first_accuracy": None,
            "reported_base_accuracy": None,
            "reported_verifier_accuracy": external_full.get("accuracy"),
            "reported_oracle_accuracy": None,
            "reparsed_first_accuracy": None,
            "reparsed_base_accuracy": None,
            "reparsed_verifier_accuracy": external_full.get("accuracy"),
            "reparsed_oracle_accuracy": None,
            "verifier_given_oracle_accuracy": None,
            "p_base_vs_verifier": None,
            "candidate_parseable_rate": None,
            "selected_prediction_parseable_rate": external_full.get("selected_prediction_parseable_rate"),
            "duplicate_slot_rate": None,
            "scaffold_residue_rate": external_full.get("scaffold_residue_rate"),
            "artifact_status": "full_local",
            "audit_scope": "full_local",
            "summary_json": external_full.get("_summary_path"),
        })

    external_gsm8k = read_json(ROOT / "Experiment/analysis/results/ser_external_baseline_comparison_gsm8k_eval128_v1.json")
    for item in external_gsm8k.get("rows", []):
        main_rows.append({
            "benchmark": "gsm8k",
            "split": "eval128",
            "variant_name": item.get("label", "unknown"),
            "paper_facing_variant_name": item.get("label", "unknown"),
            "role": item.get("kind", "external"),
            "reported_first_accuracy": external_gsm8k.get("internal_first_accuracy") if item.get("label") == "Internal completion verifier" else None,
            "reported_base_accuracy": None,
            "reported_verifier_accuracy": item.get("accuracy"),
            "reported_oracle_accuracy": None,
            "reparsed_first_accuracy": None,
            "reparsed_base_accuracy": None,
            "reparsed_verifier_accuracy": item.get("accuracy"),
            "reparsed_oracle_accuracy": None,
            "verifier_given_oracle_accuracy": None,
            "p_base_vs_verifier": None,
            "candidate_parseable_rate": None,
            "selected_prediction_parseable_rate": None,
            "duplicate_slot_rate": None,
            "scaffold_residue_rate": None,
            "artifact_status": "summary_only",
            "audit_scope": "summary_only",
            "summary_json": item.get("summary_json_path") or item.get("report_path"),
        })

    external_transfer = read_json(ROOT / "Experiment/analysis/results/a800_self_refine_three_benchmark_transfer_b32_v2_summary.json")
    for benchmark, item in external_transfer.get("benchmarks", {}).items():
        transfer_rows.append({
            "benchmark": benchmark,
            "split": "transfer",
            "variant_name": "Self-Refine",
            "paper_facing_variant_name": "Self-Refine",
            "role": "external_baseline",
            "reported_first_accuracy": None,
            "reported_base_accuracy": None,
            "reported_verifier_accuracy": item.get("accuracy"),
            "reported_oracle_accuracy": None,
            "reparsed_first_accuracy": None,
            "reparsed_base_accuracy": None,
            "reparsed_verifier_accuracy": item.get("accuracy"),
            "reparsed_oracle_accuracy": None,
            "verifier_given_oracle_accuracy": None,
            "p_base_vs_verifier": None,
            "candidate_parseable_rate": None,
            "selected_prediction_parseable_rate": None,
            "duplicate_slot_rate": None,
            "scaffold_residue_rate": None,
            "artifact_status": "summary_only",
            "audit_scope": "summary_only",
            "summary_json": str(ROOT / "Experiment/analysis/results/a800_self_refine_three_benchmark_transfer_b32_v2_summary.json"),
        })

    return main_rows, transfer_rows


def main() -> None:
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    audits = [audit_run(spec) for spec in RUN_SPECS]
    audits_by_id = {run["run_id"]: run for run in audits}

    manifest_instances = [build_manifest_instance(run) for run in audits]
    external_full_summary = load_external_gsm8k_full_self_refine_summary()
    if external_full_summary is not None:
        manifest_instances.append(build_external_gsm8k_full_manifest(external_full_summary))
    main_rows, transfer_rows = build_scoreboards(audits)

    generated_paths = [
        PACK_DIR / "paper_claim_ledger.md",
        PACK_DIR / "variant_registry.md",
        PACK_DIR / "run_manifest_schema.json",
        PACK_DIR / "run_manifest_instances.jsonl",
        PACK_DIR / "parser_audit_report.md",
        PACK_DIR / "candidate_hygiene_report.md",
        PACK_DIR / "error_taxonomy_report.md",
        PACK_DIR / "scoreboard_main.csv",
        PACK_DIR / "scoreboard_transfer.csv",
        PACK_DIR / "ablation_summary.md",
        PACK_DIR / "submission_readiness_review.md",
        PACK_DIR / "reproducibility_pack_manifest.md",
    ]

    atomic_write_text(PACK_DIR / "paper_claim_ledger.md", build_claim_ledger())
    atomic_write_text(PACK_DIR / "variant_registry.md", build_variant_registry())
    atomic_write_json(PACK_DIR / "run_manifest_schema.json", SCHEMA)
    atomic_write_text(PACK_DIR / "run_manifest_instances.jsonl", "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest_instances))
    atomic_write_text(PACK_DIR / "parser_audit_report.md", build_parser_report(audits))
    atomic_write_text(PACK_DIR / "candidate_hygiene_report.md", build_candidate_hygiene_report(audits))
    error_taxonomy_targets = [
        audits_by_id["gsm8k_full_hybridp6_clean_p1a_v1"],
        audits_by_id["gsm8k_full_completion_hybridp6_clean_p1a_v1"],
        audits_by_id["competition_math_numeric_test_completion_hybridp6_v1"],
        audits_by_id["mmlu_pro_test_completion_hybridp6_v1"],
        audits_by_id["mmlu_pro_test_completion_hybridp6_benchmarkaware_qwen8bproposer_v2"],
        audits_by_id["gpqa_diamond_train_completion_hybridp6_v1"],
    ]
    atomic_write_text(PACK_DIR / "error_taxonomy_report.md", build_error_taxonomy_report(error_taxonomy_targets))
    write_csv(PACK_DIR / "scoreboard_main.csv", main_rows, list(main_rows[0].keys()))
    write_csv(PACK_DIR / "scoreboard_transfer.csv", transfer_rows, list(transfer_rows[0].keys()))
    atomic_write_text(PACK_DIR / "ablation_summary.md", build_ablation_summary(audits))
    atomic_write_text(PACK_DIR / "submission_readiness_review.md", build_submission_readiness_review())

    source_files = [
        ROOT / "neurips_main_track_upgrade_plan_for_codex_v1.md",
        ROOT / "Experiment/core_code/src/eval/evaluate_predictions.py",
        ROOT / "Experiment/core_code/scripts/prepare_reasoning_benchmark.py",
        ROOT / "Experiment/core_code/scripts/generate_motif_completion_candidates.py",
        ROOT / "Experiment/core_code/scripts/apply_repair_hygiene_filter.py",
        ROOT / "Experiment/core_code/scripts/score_verifier_candidates.py",
    ]
    atomic_write_text(PACK_DIR / "reproducibility_pack_manifest.md", build_reproducibility_manifest(generated_paths[:-1], source_files))

    summary = {
        "generated_dir": str(PACK_DIR),
        "generated_files": [str(path) for path in generated_paths if path.exists()],
        "canonical_main_variant_recommendation": "CCR+Compat+SP",
        "backup_variant_recommendation": "CCR+Compat+SP+PM",
        "route_recommendation": "Route B",
        "p0_gate_passed": False,
    }
    atomic_write_json(PACK_DIR / "pack_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
