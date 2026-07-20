from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TAG_SCRIPT = ROOT / "scripts" / "tag_candidate_motifs.py"
CHECKPOINT_SCRIPT = ROOT / "scripts" / "generate_candidate_sets_chunked_checkpointed.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ReleaseRegressionTests(unittest.TestCase):
    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_release_data_modules_are_importable(self) -> None:
        from src.data.schema import VerifierCandidateSet
        from src.data.transform_supervision import build_processed_record

        self.assertTrue(callable(VerifierCandidateSet))
        self.assertTrue(callable(build_processed_record))

    def test_motif_tagging_uses_each_records_answer_mode(self) -> None:
        rows = [
            {
                "example_id": "mcq-1",
                "dataset": "mmlu_pro",
                "problem": "Which option is correct?",
                "gold_answer": "B",
                "answer_mode": "choice_letter",
                "candidates": ["Reasoning. Final answer: B", "Final answer: A"],
            },
            {
                "example_id": "numeric-1",
                "dataset": "gsm8k",
                "problem": "What is six times seven?",
                "gold_answer": "42",
                "answer_mode": "numeric",
                "candidates": ["6 times 7 is 42. Final answer: 42.0", "Final answer: 41"],
            },
            {
                "example_id": "legacy-numeric",
                "dataset": "gsm8k",
                "problem": "What is three plus four?",
                "gold_answer": "7",
                "candidates": ["Final answer: 7"],
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            candidates = temporary / "candidates.jsonl"
            tags = temporary / "tags.jsonl"
            summary = temporary / "summary.json"
            _write_jsonl(candidates, rows)

            result = self._run(
                [
                    sys.executable,
                    str(TAG_SCRIPT),
                    "--candidates",
                    str(candidates),
                    "--output",
                    str(tags),
                    "--summary-output",
                    str(summary),
                ]
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            tag_rows = _read_jsonl(tags)
            self.assertEqual(
                [row["candidate_is_correct"] for row in tag_rows],
                [True, False, True, False, True],
            )
            self.assertEqual(
                [row["answer_mode"] for row in tag_rows],
                ["choice_letter", "choice_letter", "numeric", "numeric", "numeric"],
            )
            with summary.open("r", encoding="utf-8") as handle:
                summary_row = json.load(handle)
            self.assertEqual(summary_row["oracle_hit_examples"], 3)
            self.assertEqual(summary_row["oracle_miss_examples"], 0)

    def test_non_resume_refuses_to_overwrite_nonempty_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            dataset = temporary / "dataset.jsonl"
            output = temporary / "checkpoint.jsonl"
            _write_jsonl(
                dataset,
                [
                    {
                        "example_id": "example-1",
                        "problem": "1 + 1?",
                        "gold_answer": "2",
                    }
                ],
            )
            sentinel = b'{"sentinel": true}\n'
            output.write_bytes(sentinel)

            result = self._run(
                [
                    sys.executable,
                    str(CHECKPOINT_SCRIPT),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--model-path",
                    "unused-model",
                ]
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists; pass --resume", result.stderr + result.stdout)
            self.assertEqual(output.read_bytes(), sentinel)


if __name__ == "__main__":
    unittest.main()
