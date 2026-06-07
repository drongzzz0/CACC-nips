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
  reproduce the original oracle/verifier decomposition. GSM8K and GPQA are
  close on final accuracy; CompMath is lower and should be treated as the main
  open-source reproduction risk.
- 1 SPP-family row is still running in the internal audit.

The practical release question is whether fresh reruns collapse relative to the
paper. GSM8K and GPQA reconstructed reruns are close on final accuracy, but the
CompMath SPP-family reruns are lower than the paper. A fresh CompMath
CACC+SPP completion reconstruction was lower by about 5.9 percentage points,
although a recovered full-size salvage/repair fallback is lower by about 1.8
percentage points. A recovered full-size MMLU-Pro heterogeneous-pool fallback
is within about 0.5 percentage points of the paper SPP final accuracy.

## Completed Reconstructed Reruns

| Dataset | Variant | Paper final | Reconstructed final | Delta | Status |
| --- | --- | ---: | ---: | ---: | --- |
| GSM8K | SPP | 0.3698 | 0.3920 | +0.0222 | close final, different O/V split |
| GSM8K | CACC+SPP | 0.4189 | 0.4215 | +0.0026 | close final, different O/V split |
| CompMath | SPP | 0.2776 | 0.2454 | -0.0322 | main current release-risk row |
| CompMath | CACC+SPP | 0.3101 | 0.2513 | -0.0588 | largest current release-risk row |
| CompMath | salvage/repair fallback | 0.3101 | 0.2926 | -0.0175 | recovered full-size fallback, not exact table provenance |
| MMLU-Pro | heterogeneous-pool fallback | 0.2663 | 0.2708 | +0.0045 | recovered full-size fallback, different O/V split |
| GPQA | SPP | 0.1983 | 0.1970 | -0.0013 | close final, different O/V split |

These reconstructed rows are fresh generation/reranking evidence, not original
paper provenance. Exact reproduction of oracle coverage and verifier efficiency
still depends on the original candidate pools or a bitwise-equivalent generation
path.

The CompMath salvage/repair fallback is based on a recovered full-size candidate
repair run over 3199 examples. It is closer to the paper CACC+SPP final accuracy
than the fresh completion reconstruction, but it should be documented as a
fallback protocol rather than the exact paper-row source.

A refreshed internal scan of CompMath JSON metrics, candidate events, launch
logs, and table-value strings did not recover an independent full-size artifact
for the paper CACC+SPP O/V/F row (`0.5024 / 0.6173 / 0.3101`). The paper
numbers appear in audit/table surfaces, while nearby full-size artifacts remain
below the paper final. The main difference between the fresh CACC+SPP rerun and
the E01 fallback is candidate-pool oracle coverage: the fallback has 1,702
oracle hits over 3,199 examples, while the fresh rerun has 1,483; verifier
accuracy is 936/3,199 versus 804/3,199.

Recovered provenance for that fallback:

- Base pool: `competition_math_numeric_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_v1.jsonl`
- Reranker predictions used for repair targeting: `ser_generate_then_rerank_qwen3_17b_competition_math_numeric_test_completion_hybridp6_v1_verifier_predictions.jsonl`
- Repair protocol: `samples_per_target=1`, `max_repair_targets=2`, `max_candidates=8`, `protect_prefix_candidates=1`, strict hygiene, no replacement of complete attempts, numeric repairs only.
- Recorded full generation cost: about 23,954 seconds for 3,199 examples.

The MMLU-Pro heterogeneous-pool fallback is based on a recovered full-size pool
merge over 12,032 examples. It merges the base pool with two benchmark-aware
completion pools, then reranks the fixed merged pool. Its final accuracy is
close to the paper SPP row, but the oracle/verifier split is different. A
targeted exact-value search did not recover an independent full-size artifact
for the paper SPP O/V/F row, so this should remain fallback evidence rather
than exact SPP provenance.

A refreshed internal scan of MMLU-named JSON/JSONL artifacts found the same
boundary: the closest full-size final-accuracy artifact remains this
heterogeneous fallback. A half-dataset shard metric is almost exact on final
accuracy (`0.2662898936` over 6,016 examples), and some 128-example smoke runs
are closer on O/V/F, but neither is full-size direct SPP provenance.

Recovered provenance for that fallback:

- Base pool: `mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_hybridp6_v1.jsonl`
- Anchor completion pool: `mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_qwen8bproposer_v2.jsonl`
- Auxiliary completion pool: `mmlu_pro_test_generated_candidates_qwen3_17b_base_filtered_t07_s16k8_completion_hybridp6_benchmarkaware_v3.jsonl`
- Merge protocol: `max_candidates=8`, `protect_prefix_candidates=1`, `max_aux_insertions=3`, `dedupe_mode=numeric_or_text`.
- Rerank protocol: fixed-pool first/base/verifier evaluation with Qwen3-1.7B and the verifier adapter.

## Pending Rows

| Dataset | Variant | Paper final | Internal status |
| --- | --- | ---: | --- |
| MMLU-Pro | SPP | 0.2663 | long Qwen3-8B generation still running; last checked 2026-06-07 22:40 CST, parent PID `795448` and generator child PID `795452` alive on GPU0, no candidate pool or final summary written yet. Backup chunked-b4 run `repro20260607_mmlu_pro_spp_direct_qwen8b_full_reconstructed_chunked_b4_v1` is alive on GPU1 with PID `986646` and reached `processed 100 examples`. Backup chunked-b8 run `repro20260607_mmlu_pro_spp_direct_qwen8b_full_reconstructed_chunked_b8_v1` is alive on GPU2 with PID `995653` and reached `processed 150 examples`; the dataset has 12032 examples, so this is still a long-running generation job rather than a failed result. |

Update this file after those runs finish.

## Command Templates

Before comparing regenerated runs with the paper table, inspect the release
targets:

```bash
python scripts/compare_reproduction_metrics.py
```

After a run writes an analysis summary JSON, compare it with the corresponding
row id:

```bash
python scripts/compare_reproduction_metrics.py \
  --summary compmath/cacc_spp=runs/compmath_salvage_repair_fallback/summary.json
```

The target manifest is `configs/reproduction_targets.json`. It records paper
O/V/F values, current release status, and the closest release-facing rerun or
fallback reference where exact paper-row provenance is not currently recovered.
By default, the comparison tool labels an absolute final-accuracy delta of
`<=0.02` as `close`, `<=0.05` as `watch`, and anything larger as `large_gap`.
Use `--close-final-delta` and `--watch-final-delta` if your release criterion
needs different thresholds.

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

MMLU-Pro heterogeneous-pool fallback:

```bash
python scripts/merge_candidate_pools.py \
  --base-candidates /path/to/mmlu_base_candidates.jsonl \
  --anchor-candidates /path/to/mmlu_qwen8bproposer_v2_candidates.jsonl \
  --aux-candidates /path/to/mmlu_benchmarkaware_v3_candidates.jsonl \
  --output runs/mmlu_heterogeneous_fallback/candidates.jsonl \
  --metrics-output runs/mmlu_heterogeneous_fallback/pool_metrics.json \
  --max-candidates 8 \
  --protect-prefix-candidates 1 \
  --max-aux-insertions 3 \
  --dedupe-mode numeric_or_text

python scripts/run_fixed_pool_rerank_eval.py \
  --run-label mmlu_heterogeneous_fallback \
  --candidates runs/mmlu_heterogeneous_fallback/candidates.jsonl \
  --first-predictions-output runs/mmlu_heterogeneous_fallback/first_predictions.jsonl \
  --base-predictions-output runs/mmlu_heterogeneous_fallback/base_rerank_predictions.jsonl \
  --base-metrics-output runs/mmlu_heterogeneous_fallback/base_rerank_metrics.json \
  --base-reranker-model-path /path/to/reranker-model \
  --verifier-predictions-output runs/mmlu_heterogeneous_fallback/verifier_predictions.jsonl \
  --verifier-metrics-output runs/mmlu_heterogeneous_fallback/verifier_metrics.json \
  --verifier-adapter-path /path/to/verifier-adapter \
  --verifier-base-model /path/to/verifier-base-model \
  --analysis-report runs/mmlu_heterogeneous_fallback/report.md \
  --analysis-summary-json runs/mmlu_heterogeneous_fallback/summary.json
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
The helper script uses final accuracy only for the `close` / `watch` /
`large_gap` bucket because oracle and V|O can move in opposite directions under
new stochastic candidate pools. Large final gaps should be investigated before
using a rerun as release-facing evidence.
