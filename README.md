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
`configs/reproduction_targets.json`.

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

## License

No open-source license has been selected yet. Choose a license before making the
GitHub repository public.
