# Reproduction Status

This code release is intended to make the CACC pipeline runnable and auditable.
It does not include benchmark data, model weights, verifier adapters, generated
candidate pools, prediction logs, or paper metric JSON files. Full paper-table
reproduction therefore requires either the original artifacts or a fresh full
rerun with matched models and decoding settings.

## Current Internal Audit

Internal audit date: 2026-06-07.

The main paper table has 16 dataset/variant rows. At the current checkpoint:

- 9 rows are reproduced directly from recovered metric artifacts, or from the
  documented display-ratio convention.
- 1 row has provenance but lacks the current prediction log needed for full
  recomputation.
- 5 SPP-family rows have completed reconstructed reruns. They do not exactly
  reproduce the original oracle/verifier decomposition, but their final
  accuracies are close enough to characterize open-source rerun behavior.
- 1 SPP-family row is still running in the internal audit.

The practical release question is whether fresh reruns collapse relative to the
paper. GSM8K and GPQA reconstructed reruns are close on final accuracy, but the
CompMath SPP-family reruns are lower than the paper. A fresh CompMath
CACC+SPP completion reconstruction was lower by about 5.9 percentage points,
although a recovered full-size salvage/repair fallback is lower by about 1.8
percentage points.

## Completed Reconstructed Reruns

| Dataset | Variant | Paper final | Reconstructed final | Delta | Status |
| --- | --- | ---: | ---: | ---: | --- |
| GSM8K | SPP | 0.3698 | 0.3920 | +0.0222 | close final, different O/V split |
| GSM8K | CACC+SPP | 0.4189 | 0.4215 | +0.0026 | close final, different O/V split |
| CompMath | SPP | 0.2776 | 0.2454 | -0.0322 | main current release-risk row |
| CompMath | CACC+SPP | 0.3101 | 0.2513 | -0.0588 | largest current release-risk row |
| CompMath | salvage/repair fallback | 0.3101 | 0.2926 | -0.0175 | recovered full-size fallback, not exact table provenance |
| GPQA | SPP | 0.1983 | 0.1970 | -0.0013 | close final, different O/V split |

These reconstructed rows are fresh generation/reranking evidence, not original
paper provenance. Exact reproduction of oracle coverage and verifier efficiency
still depends on the original candidate pools or a bitwise-equivalent generation
path.

The CompMath salvage/repair fallback is based on a recovered full-size candidate
repair run over 3199 examples. It is closer to the paper CACC+SPP final accuracy
than the fresh completion reconstruction, but it should be documented as a
fallback protocol rather than the exact paper-row source.

Recovered provenance for that fallback:

- Base pool: `competition_math_numeric_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_v1.jsonl`
- Reranker predictions used for repair targeting: `ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_completion_hybridp6_v1_verifier_predictions.jsonl`
- Repair protocol: `samples_per_target=1`, `max_repair_targets=2`, `max_candidates=8`, `protect_prefix_candidates=1`, strict hygiene, no replacement of complete attempts, numeric repairs only.
- Recorded full generation cost: about 23,954 seconds for 3,199 examples.

## Pending Rows

| Dataset | Variant | Paper final | Internal status |
| --- | --- | ---: | --- |
| MMLU-Pro | SPP | 0.2663 | long Qwen3-8B generation still running |

Update this file after those runs finish.

## Command Templates

Direct SPP-style generation plus reranking:

```bash
python scripts/run_generate_then_rerank_eval.py \
  --run-label RUN_LABEL \
  --dataset /path/to/benchmark.jsonl \
  --candidate-output runs/RUN_LABEL/candidates.jsonl \
  --candidate-metrics-output runs/RUN_LABEL/generation.json \
  --generator-model-path /path/to/generator-model \
  --supervision-type filtered_cot \
  --samples-per-example 16 \
  --max-candidates 8 \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.95 \
  --dedupe-mode numeric_or_text \
  --selection-strategy dedupe_only \
  --seed 7 \
  --base-reranker-model-path /path/to/reranker-model \
  --verifier-adapter-path /path/to/verifier-adapter \
  --verifier-base-model /path/to/verifier-base-model \
  --analysis-report runs/RUN_LABEL/report.md \
  --analysis-summary-json runs/RUN_LABEL/summary.json
```

CACC+SPP completion reranking over an existing candidate pool:

```bash
python scripts/run_completion_rerank_eval.py \
  --run-label RUN_LABEL \
  --base-candidates /path/to/base_candidates.jsonl \
  --motif-tags /path/to/motif_tags.jsonl \
  --completion-candidates-output runs/RUN_LABEL/candidates.jsonl \
  --completion-metrics-output runs/RUN_LABEL/generation.json \
  --completion-prompt-preview-output runs/RUN_LABEL/prompts.jsonl \
  --generator-model-path /path/to/generator-model \
  --samples-per-example 4 \
  --max-candidates 8 \
  --max-context-candidates 3 \
  --protect-prefix-candidates 1 \
  --max-new-tokens 160 \
  --temperature 0.7 \
  --top-p 0.95 \
  --completion-dedupe-mode numeric_or_text \
  --merge-policy replace_fragments_first \
  --seed 7 \
  --base-reranker-model-path /path/to/reranker-model \
  --verifier-adapter-path /path/to/verifier-adapter \
  --verifier-base-model /path/to/verifier-base-model \
  --analysis-report runs/RUN_LABEL/report.md \
  --analysis-summary-json runs/RUN_LABEL/summary.json
```

CompMath salvage/repair fallback over a completion candidate pool:

```bash
python scripts/run_repair_rerank_eval.py \
  --run-label compmath_salvage_repair_fallback \
  --base-candidates /path/to/competition_math_completion_candidates.jsonl \
  --reranker-predictions /path/to/completion_verifier_predictions.jsonl \
  --repair-candidates-output runs/compmath_salvage_repair_fallback/candidates.jsonl \
  --repair-metrics-output runs/compmath_salvage_repair_fallback/generation.json \
  --repair-prompt-preview-output runs/compmath_salvage_repair_fallback/prompts.jsonl \
  --generator-model-path /path/to/generator-model \
  --samples-per-target 1 \
  --max-repair-targets 2 \
  --max-candidates 8 \
  --max-new-tokens 160 \
  --temperature 0.7 \
  --top-p 0.95 \
  --protect-prefix-candidates 1 \
  --repair-dedupe-mode numeric_or_text \
  --strict-hygiene \
  --seed 7 \
  --first-predictions-output runs/compmath_salvage_repair_fallback/first_predictions.jsonl \
  --base-predictions-output runs/compmath_salvage_repair_fallback/base_rerank_predictions.jsonl \
  --base-metrics-output runs/compmath_salvage_repair_fallback/base_rerank_metrics.json \
  --base-reranker-model-path /path/to/reranker-model \
  --verifier-predictions-output runs/compmath_salvage_repair_fallback/verifier_predictions.jsonl \
  --verifier-metrics-output runs/compmath_salvage_repair_fallback/verifier_metrics.json \
  --verifier-adapter-path /path/to/verifier-adapter \
  --verifier-base-model /path/to/verifier-base-model \
  --analysis-report runs/compmath_salvage_repair_fallback/report.md \
  --analysis-summary-json runs/compmath_salvage_repair_fallback/summary.json
```

## Interpreting Differences

Report oracle coverage, verifier efficiency given oracle, and final verifier
accuracy together. A regenerated pool can have a similar final accuracy but a
different oracle/verifier decomposition because generation diversity, prompt
formatting, parser behavior, deduplication, and model checkpoint revisions all
change the candidate pool before reranking.

For public reproducibility, do not claim exact table reproduction unless the
same candidate artifacts and metric JSONs are available. For code-level reruns,
compare final accuracy and document the candidate-generation protocol used.
