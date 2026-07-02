# Release Inventory

## Source

The release candidate was assembled from the private experiment workspace's
`Experiment/core_code` directory.

Remote inspection was read-only. The source project is not a git repository.

## Included

- `src/`: core CACC and generate-then-rerank Python modules.
- `scripts/*.py`: portable Python entrypoints and analysis utilities.
- `configs/train_subgoals.yaml`: small training config template.
- `configs/reproduction_targets.json`: paper Table 1 targets and current
  release-facing rerun/fallback references.
- `configs/artifact_bundle.example.json`: editable template for checking local
  full-reproduction artifact bundles.
- `requirements.txt` and `requirements-peft.txt`: Python dependency hints.
- `examples/synthetic_reasoning_sample.jsonl`: tiny synthetic pipeline
  smoke-test data.
- `examples/synthetic_base_candidates.jsonl`: tiny synthetic candidate-pool
  data for motif tagging and CACC dry-run checks.
- `examples/example_summary_metrics.json`: minimal metric-summary schema for
  reproduction comparison.
- `README.md` and `.gitignore`: GitHub-facing repository scaffolding.
- `docs/reproduction_status.md`: audit snapshot and release-facing notes for
  interpreting full rerun differences.
- `docs/table1_reproduction_report.{md,csv}`: generated compact Table 1
  reproduction-status reports from `configs/reproduction_targets.json`.
- `docs/release_readiness.md`: generated release-readiness summary from the
  reproduction target manifest.
- `docs/full_reproduction_inputs.md`: checklist of non-released inputs needed
  for exact or approximate full reproduction.
- `scripts/compare_reproduction_metrics.py`: standard-library helper for
  comparing rerun summary JSON files with the target manifest.
- `scripts/check_reproduction_bundle.py`: standard-library helper for checking
  whether local datasets, models, adapters, and summaries exist before a full
  reproduction attempt.
- `scripts/export_reproduction_report.py`: standard-library helper for
  exporting a compact markdown/CSV Table 1 reproduction-status report from the
  target manifest.
- `scripts/check_release_readiness.py`: standard-library helper for checking
  whether the manifest still has large-gap blockers, pending rows, or caveats.
- `scripts/generate_candidate_sets_chunked_checkpointed.py` and
  `scripts/run_generate_then_rerank_eval_checkpointed.py`: resumable
  full-benchmark generation/reranking entrypoints for long stochastic reruns.

## Excluded

- Remote shell launchers and server-specific scripts with hard-coded private
  host labels or absolute machine paths.
- `Experiment/datasets`, raw benchmark data, candidate pools, prediction JSONL,
  and generated metrics/logs.
- `Experiment/core_code/checkpoints`, `models`, `PairRM-hf`, verifier adapters,
  and any model weights.
- Third-party vendored repositories such as full `LLM-Blender`, `FTTT`,
  `XBai-o4`, `Nabla-Reasoner`, and `verl`.
- Paper PDFs, TeX build outputs, local caches, and experiment archives.
- Local Codex, SSH, provider, or machine configuration files.

## Pre-Push Checklist

- License chosen: MIT, added as `LICENSE` on 2026-07-02.
- Decide whether the paper TeX package should live in the same repository or a
  separate manuscript repository.
- Run `python scripts/run_minimum_pipeline.py --allow-stub`.
- Run `python scripts/compare_reproduction_metrics.py`.
- Run `python scripts/check_reproduction_bundle.py --row compmath/cacc_spp`.
- Run `python scripts/export_reproduction_report.py --markdown-output docs/table1_reproduction_report.md --csv-output docs/table1_reproduction_report.csv`.
- Run `python scripts/check_release_readiness.py --markdown-output docs/release_readiness.md`.
- Run a large-file check before commit.
- Run a sensitive-string scan before commit.
