from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import VerifierCandidateSet
from src.eval.evaluate_predictions import extract_choice_answer, extract_numeric_answer, normalize_answer
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E02 matched fresh-resample pool by replacing the same slots repaired in E00.")
    parser.add_argument("--base-pool", required=True, type=Path)
    parser.add_argument("--fresh-source-pool", required=True, type=Path)
    parser.add_argument("--candidate-events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--max-candidates", default=8, type=int)
    return parser.parse_args()


def candidate_key(text: str, answer_mode: str) -> str:
    text = str(text)
    if answer_mode == "choice_letter":
        choice = extract_choice_answer(text)
        if choice is not None:
            return f"choice:{choice}"
    numeric = extract_numeric_answer(text)
    if numeric is not None:
        return f"num:{numeric}"
    return f"text:{normalize_answer(text)}"


def load_rows(path: Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in read_jsonl(path)}


def main() -> None:
    args = parse_args()
    base_by_id = load_rows(args.base_pool)
    fresh_by_id = load_rows(args.fresh_source_pool)
    repair_slots: dict[str, list[int]] = defaultdict(list)
    for event in read_jsonl(args.candidate_events):
        if event.get("repair_applied"):
            repair_slots[str(event["example_id"])].append(int(event["source_slot_id"]))
    for slots in repair_slots.values():
        slots.sort()

    output_rows = []
    stats = Counter()
    failed_examples = []
    for example_id, base_row in base_by_id.items():
        answer_mode = str(base_row.get("answer_mode", "numeric"))
        candidates = [str(candidate) for candidate in base_row.get("candidates", [])][: args.max_candidates]
        while len(candidates) < args.max_candidates:
            candidates.append("")
        fresh_candidates = [str(candidate) for candidate in fresh_by_id.get(example_id, {}).get("candidates", [])]
        fresh_cursor = 0
        slots = repair_slots.get(example_id, [])
        stats["examples_total"] += 1
        if slots:
            stats["examples_with_repair_slots"] += 1
        for slot in slots:
            if slot >= args.max_candidates:
                stats["skipped_slot_out_of_range"] += 1
                continue
            existing_keys = {
                candidate_key(candidate, answer_mode)
                for idx, candidate in enumerate(candidates)
                if idx != slot and candidate.strip()
            }
            replacement = None
            while fresh_cursor < len(fresh_candidates):
                candidate = fresh_candidates[fresh_cursor].strip()
                fresh_cursor += 1
                if not candidate:
                    continue
                key = candidate_key(candidate, answer_mode)
                if key in existing_keys:
                    stats["fresh_duplicate_skipped"] += 1
                    continue
                replacement = candidate
                break
            if replacement is None:
                stats["missing_fresh_replacement"] += 1
                failed_examples.append({"example_id": example_id, "slot": slot})
                continue
            candidates[slot] = replacement
            stats["fresh_replacements_applied"] += 1
        candidates = candidates[: args.max_candidates]
        output_rows.append(
            VerifierCandidateSet(
                example_id=example_id,
                dataset=str(base_row.get("dataset", "unknown")),
                problem=str(base_row.get("problem", "")),
                gold_answer=str(base_row.get("gold_answer", "")),
                candidates=candidates,
                answer_mode=answer_mode,
                choices=[str(choice) for choice in base_row.get("choices", [])],
                metadata={
                    **dict(base_row.get("metadata", {})),
                    "candidate_salvage_e02": {
                        "source": "matched_fresh_resample",
                        "fresh_source_pool": str(args.fresh_source_pool),
                        "repair_slots_from": str(args.candidate_events),
                        "repair_slots": slots,
                    },
                },
            ).to_dict()
        )
    write_jsonl(args.output, output_rows)
    write_json(
        args.metrics_output,
        {
            "base_pool": str(args.base_pool),
            "fresh_source_pool": str(args.fresh_source_pool),
            "candidate_events": str(args.candidate_events),
            "output": str(args.output),
            "max_candidates": args.max_candidates,
            "stats": dict(stats),
            "failed_examples_preview": failed_examples[:20],
            "replacement_completion_rate": (
                stats["fresh_replacements_applied"] / (stats["fresh_replacements_applied"] + stats["missing_fresh_replacement"])
                if (stats["fresh_replacements_applied"] + stats["missing_fresh_replacement"])
                else 1.0
            ),
        },
    )
    print(json.dumps(dict(stats), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
