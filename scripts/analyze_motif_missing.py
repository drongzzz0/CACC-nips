from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import read_jsonl, write_json, write_jsonl, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze heuristic motif coverage and fragment rates on a verifier candidate pool."
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--tags", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--audit-sample-output", type=Path)
    parser.add_argument("--audit-sample-size", default=50, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--run-label", default="motif_audit")
    return parser.parse_args()


def _mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _share(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _shorten(text: str, limit: int = 180) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    candidate_rows = {str(row["example_id"]): row for row in read_jsonl(args.candidates)}
    tags_by_example: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.tags):
        tags_by_example[str(row["example_id"])].append(row)

    example_rows: list[dict] = []
    oracle_miss_problem_motifs = Counter()
    oracle_hit_problem_motifs = Counter()
    quality_counts = Counter()
    quality_counts_oracle_miss = Counter()
    quality_counts_oracle_hit = Counter()
    non_fragment_motif_counts = Counter()
    non_fragment_motif_counts_oracle_miss = Counter()

    for example_id, candidate_row in candidate_rows.items():
        tag_rows = sorted(tags_by_example[example_id], key=lambda row: row["candidate_index"])
        if not tag_rows:
            raise ValueError(f"Missing motif tags for {example_id}")

        problem_motif = str(tag_rows[0]["problem_motif_label"])
        oracle_hit = any(bool(tag_row["candidate_is_correct"]) for tag_row in tag_rows)
        motifs_all = {str(tag_row["motif_label"]) for tag_row in tag_rows}
        non_fragment_rows = [tag_row for tag_row in tag_rows if tag_row["quality_label"] != "fragment"]
        motifs_non_fragment = {str(tag_row["motif_label"]) for tag_row in non_fragment_rows}
        problem_motif_present_non_fragment = problem_motif in motifs_non_fragment
        problem_motif_present_any = problem_motif in motifs_all
        all_fragment = bool(tag_rows) and all(tag_row["quality_label"] == "fragment" for tag_row in tag_rows)

        for tag_row in tag_rows:
            quality_counts[str(tag_row["quality_label"])] += 1
            if oracle_hit:
                quality_counts_oracle_hit[str(tag_row["quality_label"])] += 1
            else:
                quality_counts_oracle_miss[str(tag_row["quality_label"])] += 1
            if tag_row["quality_label"] != "fragment":
                non_fragment_motif_counts[str(tag_row["motif_label"])] += 1
                if not oracle_hit:
                    non_fragment_motif_counts_oracle_miss[str(tag_row["motif_label"])] += 1

        if oracle_hit:
            oracle_hit_problem_motifs[problem_motif] += 1
        else:
            oracle_miss_problem_motifs[problem_motif] += 1

        example_rows.append(
            {
                "example_id": example_id,
                "problem": str(candidate_row["problem"]),
                "gold_answer": str(candidate_row["gold_answer"]),
                "oracle_hit": oracle_hit,
                "problem_motif_label": problem_motif,
                "problem_motif_present_any": problem_motif_present_any,
                "problem_motif_present_non_fragment": problem_motif_present_non_fragment,
                "all_fragment": all_fragment,
                "candidate_count": len(tag_rows),
                "fragment_count": sum(1 for tag_row in tag_rows if tag_row["quality_label"] == "fragment"),
                "partial_count": sum(1 for tag_row in tag_rows if tag_row["quality_label"] == "partial_solution"),
                "complete_count": sum(1 for tag_row in tag_rows if tag_row["quality_label"] == "complete_attempt"),
                "unique_motifs_all": sorted(motifs_all),
                "unique_motifs_non_fragment": sorted(motifs_non_fragment),
                "candidates": [
                    {
                        "candidate_index": tag_row["candidate_index"],
                        "candidate_text": str(tag_row["candidate_text"]),
                        "candidate_is_correct": bool(tag_row["candidate_is_correct"]),
                        "motif_label": str(tag_row["motif_label"]),
                        "quality_label": str(tag_row["quality_label"]),
                    }
                    for tag_row in tag_rows
                ],
            }
        )

    total_examples = len(example_rows)
    oracle_hit_examples = sum(1 for row in example_rows if row["oracle_hit"])
    oracle_miss_examples = total_examples - oracle_hit_examples
    heuristic_problem_motif_missing_examples = [
        row for row in example_rows if not row["problem_motif_present_non_fragment"]
    ]
    heuristic_problem_motif_missing_oracle_miss = [
        row for row in heuristic_problem_motif_missing_examples if not row["oracle_hit"]
    ]
    all_fragment_examples = [row for row in example_rows if row["all_fragment"]]
    all_fragment_oracle_miss = [row for row in all_fragment_examples if not row["oracle_hit"]]
    no_complete_attempt_examples = [row for row in example_rows if row["complete_count"] == 0]
    no_complete_attempt_oracle_miss = [row for row in no_complete_attempt_examples if not row["oracle_hit"]]

    sampled_audit_rows = heuristic_problem_motif_missing_oracle_miss[:]
    sampled_audit_rows.sort(
        key=lambda row: (
            -row["fragment_count"],
            row["problem_motif_label"],
            row["example_id"],
        )
    )
    if args.audit_sample_output is not None:
        random.shuffle(sampled_audit_rows)
        sampled = sorted(sampled_audit_rows[: args.audit_sample_size], key=lambda row: row["example_id"])
        write_jsonl(args.audit_sample_output, sampled)
    else:
        sampled = sampled_audit_rows[: min(5, len(sampled_audit_rows))]

    summary = {
        "run_label": args.run_label,
        "candidates_path": str(args.candidates),
        "tags_path": str(args.tags),
        "total_examples": total_examples,
        "oracle_hit_examples": oracle_hit_examples,
        "oracle_miss_examples": oracle_miss_examples,
        "heuristic_problem_motif_missing_examples": len(heuristic_problem_motif_missing_examples),
        "heuristic_problem_motif_missing_rate": _share(
            len(heuristic_problem_motif_missing_examples), total_examples
        ),
        "heuristic_problem_motif_missing_oracle_miss_examples": len(heuristic_problem_motif_missing_oracle_miss),
        "heuristic_problem_motif_missing_rate_given_oracle_miss": _share(
            len(heuristic_problem_motif_missing_oracle_miss), oracle_miss_examples
        ),
        "all_fragment_examples": len(all_fragment_examples),
        "all_fragment_rate": _share(len(all_fragment_examples), total_examples),
        "all_fragment_oracle_miss_examples": len(all_fragment_oracle_miss),
        "all_fragment_rate_given_oracle_miss": _share(len(all_fragment_oracle_miss), oracle_miss_examples),
        "examples_with_no_complete_attempt": len(no_complete_attempt_examples),
        "rate_with_no_complete_attempt": _share(len(no_complete_attempt_examples), total_examples),
        "oracle_miss_examples_with_no_complete_attempt": len(no_complete_attempt_oracle_miss),
        "rate_with_no_complete_attempt_given_oracle_miss": _share(
            len(no_complete_attempt_oracle_miss), oracle_miss_examples
        ),
        "avg_unique_motifs_all": _mean([len(row["unique_motifs_all"]) for row in example_rows]),
        "avg_unique_motifs_non_fragment": _mean([len(row["unique_motifs_non_fragment"]) for row in example_rows]),
        "quality_counts": dict(sorted(quality_counts.items())),
        "quality_counts_given_oracle_miss": dict(sorted(quality_counts_oracle_miss.items())),
        "quality_counts_given_oracle_hit": dict(sorted(quality_counts_oracle_hit.items())),
        "oracle_miss_problem_motif_counts": dict(sorted(oracle_miss_problem_motifs.items())),
        "oracle_hit_problem_motif_counts": dict(sorted(oracle_hit_problem_motifs.items())),
        "non_fragment_motif_counts": dict(sorted(non_fragment_motif_counts.items())),
        "non_fragment_motif_counts_given_oracle_miss": dict(sorted(non_fragment_motif_counts_oracle_miss.items())),
        "sampled_audit_example_ids": [row["example_id"] for row in sampled],
    }
    write_json(args.summary_json, summary)

    problem_motif_lines = []
    for motif_label, count in oracle_miss_problem_motifs.most_common(5):
        problem_motif_lines.append(f"- `{motif_label}`: {count} oracle-miss examples")
    if not problem_motif_lines:
        problem_motif_lines.append("- no oracle-miss examples")

    sampled_lines = []
    for row in sampled[:5]:
        candidate_preview = "; ".join(
            f"[{candidate['quality_label']}/{candidate['motif_label']}] {_shorten(candidate['candidate_text'], 80)}"
            for candidate in row["candidates"][:3]
        )
        sampled_lines.append(
            f"- `{row['example_id']}` | expected `{row['problem_motif_label']}` | "
            f"fragments {row['fragment_count']}/{row['candidate_count']} | {candidate_preview}"
        )
    if not sampled_lines:
        sampled_lines.append("- no sampled heuristic-missing oracle-miss examples")

    report = (
        f"# Motif Audit Report: {args.run_label}\n\n"
        "## Scope\n\n"
        "This report audits motif coverage on an existing candidate pool using a heuristic problem-level motif predictor "
        "and heuristic candidate motif / quality tags. The resulting `heuristic_problem_motif_missing_rate` is therefore "
        "a measurement-layer proxy rather than a validated ground-truth motif-missing metric.\n\n"
        "## Headline Metrics\n\n"
        f"- total examples: {total_examples}\n"
        f"- oracle-hit examples: {oracle_hit_examples}\n"
        f"- oracle-miss examples: {oracle_miss_examples}\n"
        f"- heuristic problem-motif-missing examples: {len(heuristic_problem_motif_missing_examples)} "
        f"({summary['heuristic_problem_motif_missing_rate']:.4f})\n"
        f"- heuristic problem-motif-missing rate given oracle miss: "
        f"{summary['heuristic_problem_motif_missing_rate_given_oracle_miss']:.4f}\n"
        f"- all-fragment examples: {len(all_fragment_examples)} ({summary['all_fragment_rate']:.4f})\n"
        f"- all-fragment rate given oracle miss: {summary['all_fragment_rate_given_oracle_miss']:.4f}\n"
        f"- examples with no complete attempt: {len(no_complete_attempt_examples)} "
        f"({summary['rate_with_no_complete_attempt']:.4f})\n"
        f"- no-complete-attempt rate given oracle miss: "
        f"{summary['rate_with_no_complete_attempt_given_oracle_miss']:.4f}\n"
        f"- avg unique motifs per example: {summary['avg_unique_motifs_all']:.4f}\n"
        f"- avg unique non-fragment motifs per example: {summary['avg_unique_motifs_non_fragment']:.4f}\n\n"
        "## Interpretation\n\n"
        "Phase 1 should treat two signals as especially important:\n\n"
        f"- motif-missing proxy: {len(heuristic_problem_motif_missing_oracle_miss)}/{oracle_miss_examples} oracle-miss "
        "examples do not contain the heuristic problem motif among non-fragment candidates.\n"
        f"- fragmentation proxy: {len(all_fragment_oracle_miss)}/{oracle_miss_examples} oracle-miss examples contain only "
        "fragment candidates.\n"
        f"- completion scarcity proxy: {len(no_complete_attempt_oracle_miss)}/{oracle_miss_examples} oracle-miss examples "
        "do not contain any candidate tagged as a complete attempt.\n\n"
        "If the motif-missing proxy is low but the completion-scarcity proxy is high, the next branch should emphasize "
        "finishing or repairing incomplete solution attempts rather than searching for entirely new motifs.\n\n"
        "## Oracle-Miss Problem Motifs\n\n"
        + "\n".join(problem_motif_lines)
        + "\n\n## Sampled Heuristic-Missing Oracle-Miss Examples\n\n"
        + "\n".join(sampled_lines)
        + "\n"
    )
    write_text(args.report, report)


if __name__ == "__main__":
    main()
