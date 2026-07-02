# CACC Route-C Code

This repository contains the lightweight code release for the NeurIPS work on
Compatibility-Aware Candidate Construction (CACC) for multi-candidate
reasoning.

The code is intentionally separated from the full research workspace. It keeps
the runnable source, analysis utilities, and selected experiment entrypoints,
while excluding model weights, raw datasets, prediction logs, generated paper
artifacts, remote launch scripts, and large archives.

## Layout

- `src/`: reusable Python modules for prompts, candidate schemas, parsing,
  exact-match scoring, PEFT inference/training helpers, and JSONL utilities.
- `scripts/`: Python entrypoints for candidate generation, CACC-style
  completion construction, reranking/evaluation, analysis, and paper-table
  aggregation.
- `configs/`: small configuration templates and Table 1 reproduction targets.
- `examples/`: synthetic JSONL data for smoke testing the pipeline.
- `docs/release_inventory.md`: what was included, what was excluded, and why.
- `docs/reproduction_status.md`: current internal full-run reproduction status,
  expected rerun variance, and release-facing caveats.
- `docs/table1_reproduction_report.md`: compact generated Table 1
  reproduction-status report.
- `docs/full_reproduction_inputs.md`: checklist of benchmark/model/candidate
  artifacts needed for full Table 1 reproduction.
- `runs/`: local outputs from smoke tests or reproduction runs; ignored by git.

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a CPU-only smoke test, the full PEFT stack is not required if you use the
stub mode below.

## Smoke Test

Run the minimum reproducible pipeline on synthetic examples:

```bash
python scripts/run_minimum_pipeline.py --allow-stub
```

This writes outputs under `runs/minimum_pipeline/` and verifies that the JSONL
schemas, supervision conversion, stub training manifest, prediction file, and
exact-match report can be produced end to end.

To inspect the current Table 1 reproduction targets and compare a full-run
summary JSON against the paper values:

```bash
python scripts/compare_reproduction_metrics.py

python scripts/compare_reproduction_metrics.py \
  --summary gsm8k/base=examples/example_summary_metrics.json
```

Use row ids such as `gsm8k/cacc_spp`, `compmath/cacc_spp`,
`mmlu_pro/spp`, and `gpqa/spp`. The manifest is
`configs/reproduction_targets.json`. By default, final-accuracy deltas of
`<=0.02` are `close`; positive deltas above that threshold are
`higher_final`; below-paper deltas of `<=0.05` are `watch`; larger below-paper
gaps are `large_gap`.

To check whether a local artifact bundle has the datasets, models, adapters, or
summary files needed for selected rows:

```bash
python scripts/check_reproduction_bundle.py \
  --bundle configs/artifact_bundle.example.json \
  --row compmath/cacc_spp
```

The bundle checker also honors optional expected line counts and metric values
from the manifest, which are used for the documented CompMath E01 fallback
artifact set.

To export the current release-facing Table 1 reproduction report:

```bash
python scripts/export_reproduction_report.py \
  --markdown-output docs/table1_reproduction_report.md \
  --csv-output docs/table1_reproduction_report.csv
```

To check whether the current manifest has large release blockers:

```bash
python scripts/check_release_readiness.py
```

Use `--strict` when you want pending rows and caveats to fail the gate too.

## Main CACC Entry Points

Tag an existing candidate pool with heuristic motif and quality labels:

```bash
python scripts/tag_candidate_motifs.py \
  --candidates examples/synthetic_base_candidates.jsonl \
  --output runs/candidates/synthetic_motif_tags.jsonl \
  --summary-output runs/candidates/synthetic_motif_summary.json
```

Build completion-oriented candidates from an existing pool and motif tags.
The dry-run mode validates prompt construction, merge logic, and output schemas
without loading a model:

```bash
python scripts/generate_motif_completion_candidates.py \
  --base-candidates examples/synthetic_base_candidates.jsonl \
  --motif-tags runs/candidates/synthetic_motif_tags.jsonl \
  --output runs/candidates/cacc_candidates.jsonl \
  --metrics-output runs/candidates/cacc_metrics.json \
  --prompt-preview-output runs/candidates/cacc_prompts.jsonl \
  --dry-run
```

Full model-backed candidate generation requires an explicit model or adapter:

```bash
python scripts/generate_candidate_sets.py \
  --dataset examples/synthetic_reasoning_sample.jsonl \
  --model-path /path/to/base-or-instruction-model \
  --output runs/candidates/base_candidates.jsonl \
  --metrics-output runs/candidates/base_metrics.json
```

For long full-benchmark generation, prefer the checkpointed entrypoint so a
machine interruption does not discard all generated candidates:

```bash
python scripts/run_generate_then_rerank_eval_checkpointed.py \
  --run-label my_full_run \
  --dataset /path/to/benchmark.jsonl \
  --candidate-output runs/my_full_run/candidates.jsonl \
  --candidate-metrics-output runs/my_full_run/generation.json \
  --candidate-progress-output runs/my_full_run/generation.progress.json \
  --generator-model-path /path/to/generator-model \
  --samples-per-example 16 \
  --generation-batch-size 8 \
  --max-candidates 8 \
  --base-reranker-model-path /path/to/reranker-model \
  --verifier-adapter-path /path/to/verifier-adapter \
  --verifier-base-model /path/to/verifier-base-model \
  --first-predictions-output runs/my_full_run/first_predictions.jsonl \
  --base-predictions-output runs/my_full_run/base_predictions.jsonl \
  --base-metrics-output runs/my_full_run/base_metrics.json \
  --verifier-predictions-output runs/my_full_run/verifier_predictions.jsonl \
  --verifier-metrics-output runs/my_full_run/verifier_metrics.json \
  --analysis-report runs/my_full_run/report.md \
  --analysis-summary-json runs/my_full_run/summary.json
```

Pass `--resume` to continue from an existing partial candidate JSONL. The
resumed tail may not be bitwise-equivalent to one uninterrupted stochastic run,
but it preserves completed candidate rows and is safer for day-scale reruns.

Most experiment scripts accept explicit input/output paths. Keep generated
candidate pools, predictions, metrics, and reports under `runs/` or another
ignored output directory.

## Data and Weights

This release does not include benchmark datasets, raw candidate pools, model
checkpoints, PairRM weights, verifier adapters, or generated prediction JSONL
files. Recreate or supply those artifacts separately when running full
experiments.

See `docs/reproduction_status.md` before comparing regenerated full-run numbers
against the paper table. Candidate generation is stochastic and paper-level
reproduction depends on the exact candidate pools, model checkpoints, verifier
adapter, prompts, parser, and selection settings.

Known reproduction caveats (summarized; details and provenance in
`docs/reproduction_status.md`):

- CompMath SPP-family rows are the main open-source reproduction risk. Fresh
  reconstructed reruns landed below the paper final accuracy by about 3.2
  (SPP) and 5.9 (CACC+SPP) percentage points; the best recovered full-size
  fallback narrows the CACC+SPP gap to about 1.8 points but is not exact
  paper-row provenance.
- The MMLU-Pro SPP reference is a completed fresh direct rerun with a higher
  final accuracy than the paper row and a different oracle/verifier split.
  Optional backup corroboration runs were interrupted mid-generation and never
  finished.
- Exact oracle/verifier decompositions require the original candidate pools;
  code-level reruns should be compared on final accuracy with the generation
  protocol documented.

See `docs/full_reproduction_inputs.md` for the concrete artifact classes needed
for exact paper-row reproduction versus approximate code-level reruns.

## License

This repository is released under the MIT License. See `LICENSE` for the full
text.
