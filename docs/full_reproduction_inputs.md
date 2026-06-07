# Full Reproduction Inputs

This repository contains code, small examples, and reproduction target metadata.
It does not contain the large or private artifacts needed for paper-level Table
1 reproduction.

Use this checklist before interpreting a full rerun. Missing any item can
change oracle coverage, verifier efficiency, or final accuracy.

## Required Artifact Classes

| Artifact class | Needed for | Notes |
| --- | --- | --- |
| Benchmark JSONL files | all full reruns | Must preserve ids, answer fields, split, and parser mode used by the paper run. |
| Generator model | candidate generation | Direct SPP-style rows used the configured Qwen generator path in the internal runs. |
| Reranker base model | base reranking | The released scripts expect an explicit model path. |
| Verifier base model | verifier reranking | Must match the adapter family. |
| Verifier adapter | verifier reranking | Main internal verifier id: `ser_a800_qwen3_17b_gsm8k512_verifier_yesno_v1`. |
| Candidate pools | exact paper-row reproduction | Required when avoiding stochastic regeneration drift. |
| Prediction JSONL logs | audit recomputation | Required to independently recompute row-level selected accuracy from predictions. |
| Metric summary JSON files | comparison | Used by `scripts/compare_reproduction_metrics.py`. |

## Exact vs Approximate Reproduction

Exact Table 1 reproduction requires the original candidate pools and metric
JSONs. A fresh rerun with the same script can still differ because decoding,
batching, model revisions, parser behavior, and candidate deduplication change
the retained pool before reranking.

For open-source reruns, compare:

1. Oracle coverage.
2. Verifier efficiency given oracle.
3. Final verifier-selected accuracy.

Use final accuracy for the release-facing `close` / `watch` / `large_gap`
bucket, but report all three metrics.

## Row Groups

| Row group | Current release expectation | Required inputs |
| --- | --- | --- |
| Base and CACC rows with recovered artifacts | Should match when the same candidate and metric artifacts are supplied. | Benchmark JSONL, candidate pool, reranker/verifier outputs or summary JSON. |
| Direct SPP rows | Fresh generation may have similar final accuracy but different O/V split. | Benchmark JSONL, generator model, reranker model, verifier adapter, generation settings, parser. |
| CACC+SPP completion rows | Depends on base pool plus CACC completion generation. | Base candidate pool, motif tags or compatible diagnostics, generator model, reranker model, verifier adapter. |
| CompMath CACC+SPP | Main current risk row. Fresh reconstruction was lower than paper; recovered repair fallback is closer but is not exact paper provenance. | Completion pool, verifier predictions for repair targeting, repair generator settings, reranker/verifier artifacts. |
| MMLU-Pro SPP | Direct rerun is pending in the internal audit; heterogeneous-pool fallback is close on final accuracy but differs in O/V split. | Full MMLU-Pro JSONL, base pool, optional benchmark-aware proposer pools, reranker/verifier artifacts. |

## CompMath Fallback Artifact Set

For the closest recovered CompMath CACC+SPP fallback, keep the full E01 bundle
together rather than only the summary JSON:

- `generation.json`: repair settings and generation accounting.
- `candidates.jsonl`: repaired candidate pool.
- `prompts.jsonl`: repair prompts.
- `first_predictions.jsonl`, `base_rerank_predictions.jsonl`, and
  `verifier_predictions.jsonl`: selected-answer audit logs.
- `base_rerank_metrics.json`, `verifier_metrics.json`, and `summary.json`:
  recomputation checkpoints and final O/V/F summary.

The internal E01 run used `samples_per_target=1`, `max_repair_targets=2`,
`max_candidates=8`, `protect_prefix_candidates=1`, strict hygiene,
numeric-only repairs, and no replacement of complete attempts. Its final
accuracy is closer to the paper than the fresh CACC+SPP rerun mainly because
its candidate pool has more oracle hits.

The bundle checker understands optional `expected_lines` for JSONL artifacts
and `expected_values` for JSON/metric-summary artifacts. The example manifest
uses those fields for the E01 bundle so a local artifact set can be checked for
the expected 3,199 examples, 6,396 repair prompts, and recorded O/V/F counts.

## Comparison Workflow

List paper targets and current release references:

```bash
python scripts/compare_reproduction_metrics.py
```

Check an artifact bundle before starting a long run:

```bash
python scripts/check_reproduction_bundle.py \
  --bundle configs/artifact_bundle.example.json \
  --root /path/to/local/artifact/root \
  --row compmath/cacc_spp
```

Copy `configs/artifact_bundle.example.json` and replace the paths with your
local dataset, model, adapter, candidate, prediction, and summary locations.
The checker validates path existence and basic JSON/JSONL/metric-summary
structure. When optional expected counts or values are present, it also checks
those. It does not validate that a model checkpoint is identical to the
internal paper run.

Compare one or more summary JSON files:

```bash
python scripts/compare_reproduction_metrics.py \
  --summary gsm8k/base=runs/gsm8k_base/summary.json \
  --summary compmath/cacc_spp=runs/compmath_cacc_spp/summary.json
```

The comparison tool defaults to:

- `close`: absolute final-accuracy delta `<=0.02`.
- `watch`: absolute final-accuracy delta `<=0.05`.
- `large_gap`: larger absolute final-accuracy delta.

Treat `large_gap` rows as release blockers unless you can explain them with a
different candidate-generation protocol or a known fallback row.
