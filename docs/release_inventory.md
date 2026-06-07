# Release Inventory

## Source

The release candidate was assembled from the private experiment workspace's
`Experiment/core_code` directory.

Remote inspection was read-only. The source project is not a git repository.

## Included

- `src/`: core CACC and generate-then-rerank Python modules.
- `scripts/*.py`: portable Python entrypoints and analysis utilities.
- `configs/train_subgoals.yaml`: small training config template.
- `requirements.txt` and `requirements-peft.txt`: Python dependency hints.
- `examples/synthetic_reasoning_sample.jsonl`: tiny synthetic pipeline
  smoke-test data.
- `examples/synthetic_base_candidates.jsonl`: tiny synthetic candidate-pool
  data for motif tagging and CACC dry-run checks.
- `README.md` and `.gitignore`: GitHub-facing repository scaffolding.
- `docs/reproduction_status.md`: audit snapshot and release-facing notes for
  interpreting full rerun differences.

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

- Choose and add a real `LICENSE` if the repository will be public.
- Decide whether the paper TeX package should live in the same repository or a
  separate manuscript repository.
- Run `python scripts/run_minimum_pipeline.py --allow-stub`.
- Run a large-file check before commit.
- Run a sensitive-string scan before commit.
