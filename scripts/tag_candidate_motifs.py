from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.motif_utils import infer_candidate_tag, infer_problem_motif  # noqa: E402
from src.eval.evaluate_predictions import answer_mode_for_record, answers_match  # noqa: E402
from src.utils.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tag verifier candidate pools with heuristic reasoning motifs and completion quality labels."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def _is_correct(prediction: str, gold_answer: str, answer_mode: str) -> bool:
    return answers_match(prediction, gold_answer, answer_mode=answer_mode)


def main() -> None:
    args = parse_args()

    tag_rows: list[dict] = []
    motif_counts = Counter()
    motif_counts_non_fragment = Counter()
    quality_counts = Counter()
    problem_motif_counts = Counter()
    oracle_hit_examples = 0
    example_count = 0
    candidate_count = 0
    unique_motifs_per_example: list[int] = []
    unique_non_fragment_motifs_per_example: list[int] = []
    quality_by_example = defaultdict(Counter)

    for row in read_jsonl(args.candidates):
        example_count += 1
        problem = str(row["problem"])
        gold_answer = str(row["gold_answer"])
        answer_mode = answer_mode_for_record(row)
        problem_motif = infer_problem_motif(problem)
        problem_motif_counts[problem_motif.label] += 1

        example_motifs = set()
        example_non_fragment_motifs = set()
        example_oracle_hit = False

        for candidate_index, candidate_text in enumerate(row["candidates"]):
            candidate_text = str(candidate_text)
            candidate_count += 1
            candidate_tag = infer_candidate_tag(problem, candidate_text)
            candidate_correct = _is_correct(candidate_text, gold_answer, answer_mode)
            example_oracle_hit = example_oracle_hit or candidate_correct

            motif_counts[candidate_tag.motif.label] += 1
            quality_counts[candidate_tag.quality.label] += 1
            quality_by_example[str(row["example_id"])][candidate_tag.quality.label] += 1
            example_motifs.add(candidate_tag.motif.label)
            if candidate_tag.quality.label != "fragment":
                motif_counts_non_fragment[candidate_tag.motif.label] += 1
                example_non_fragment_motifs.add(candidate_tag.motif.label)

            tag_rows.append(
                {
                    "example_id": str(row["example_id"]),
                    "dataset": str(row.get("dataset", "unknown")),
                    "answer_mode": answer_mode,
                    "candidate_index": candidate_index,
                    "candidate_text": candidate_text,
                    "candidate_is_correct": candidate_correct,
                    "problem_motif_label": problem_motif.label,
                    "problem_motif_confidence": problem_motif.confidence,
                    "problem_motif_cues": problem_motif.matched_cues,
                    "motif_label": candidate_tag.motif.label,
                    "motif_confidence": candidate_tag.motif.confidence,
                    "motif_cues": candidate_tag.motif.matched_cues,
                    "quality_label": candidate_tag.quality.label,
                    "quality_confidence": candidate_tag.quality.confidence,
                    "quality_cues": candidate_tag.quality.matched_cues,
                }
            )

        if example_oracle_hit:
            oracle_hit_examples += 1
        unique_motifs_per_example.append(len(example_motifs))
        unique_non_fragment_motifs_per_example.append(len(example_non_fragment_motifs))

    write_jsonl(args.output, tag_rows)
    write_json(
        args.summary_output,
        {
            "candidates_path": str(args.candidates),
            "tags_path": str(args.output),
            "total_examples": example_count,
            "total_candidates": candidate_count,
            "avg_candidates_per_example": round(candidate_count / example_count, 6) if example_count else 0.0,
            "oracle_hit_examples": oracle_hit_examples,
            "oracle_miss_examples": example_count - oracle_hit_examples,
            "problem_motif_counts": dict(sorted(problem_motif_counts.items())),
            "candidate_motif_counts": dict(sorted(motif_counts.items())),
            "candidate_motif_counts_non_fragment": dict(sorted(motif_counts_non_fragment.items())),
            "quality_counts": dict(sorted(quality_counts.items())),
            "avg_unique_motifs_per_example": round(sum(unique_motifs_per_example) / example_count, 6)
            if example_count
            else 0.0,
            "avg_unique_non_fragment_motifs_per_example": round(
                sum(unique_non_fragment_motifs_per_example) / example_count, 6
            )
            if example_count
            else 0.0,
            "examples_with_all_fragment_candidates": sum(
                1
                for counter in quality_by_example.values()
                if counter and counter["fragment"] == sum(counter.values())
            ),
        },
    )


if __name__ == "__main__":
    main()
