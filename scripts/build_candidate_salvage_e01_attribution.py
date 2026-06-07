from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io_utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E01 salvage attribution artifacts from E00 candidate events.")
    parser.add_argument("--candidate-events", required=True, type=Path)
    parser.add_argument("--summary-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="gsm8k_clean_eval128")
    return parser.parse_args()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = list(read_jsonl(args.candidate_events))
    by_example: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_example[str(event["example_id"])].append(event)

    summary = read_summary(args.summary_metrics)
    base_final_raw = summary.get("base_pool", {}).get("final", "")
    salvage_final = float(summary.get("salvage_amc_sch", {}).get("final") or 0.0)
    base_final = float(base_final_raw) if str(base_final_raw).strip() else None
    delta_final = salvage_final - base_final if base_final is not None else "not_applicable_for_E01_base_pool_without_independent_verifier_row"

    final_source = Counter()
    selected_correct_examples = []
    repaired_selected_correct = []
    original_selected_correct = []
    repaired_any_correct_examples = []
    completion_by_bucket = defaultdict(lambda: Counter())
    case_rows = []

    for example_id, rows in by_example.items():
        selected = [row for row in rows if row.get("selected_by_verifier")]
        if not selected:
            final_source["unknown_source"] += 1
            continue
        selected_row = selected[0]
        if selected_row.get("selected_correct"):
            selected_correct_examples.append(example_id)
            if selected_row.get("repair_applied"):
                final_source["from_repaired_correct"] += 1
                repaired_selected_correct.append(example_id)
            else:
                final_source["from_original_correct"] += 1
                original_selected_correct.append(example_id)
        else:
            final_source["final_wrong"] += 1
        if any(row.get("repair_applied") and row.get("after_correct") for row in rows):
            repaired_any_correct_examples.append(example_id)

        for row in rows:
            if not row.get("repair_applied"):
                continue
            bucket = str(row.get("source_bucket", "unknown"))
            completion_by_bucket[bucket]["count"] += 1
            completion_by_bucket[bucket]["before_parseable"] += int(bool(row.get("before_parseable")))
            completion_by_bucket[bucket]["after_parseable"] += int(bool(row.get("after_parseable")))
            completion_by_bucket[bucket]["before_correct"] += int(bool(row.get("before_correct")))
            completion_by_bucket[bucket]["after_correct"] += int(bool(row.get("after_correct")))
            completion_by_bucket[bucket]["selected"] += int(bool(row.get("selected_by_verifier")))
            if len(case_rows) < 40 and (row.get("after_correct") or row.get("selected_by_verifier")):
                case_rows.append(
                    {
                        "example_id": example_id,
                        "source_slot_id": row.get("source_slot_id"),
                        "source_bucket": bucket,
                        "repair_type": row.get("repair_type"),
                        "before_parseable": row.get("before_parseable"),
                        "after_parseable": row.get("after_parseable"),
                        "before_correct": row.get("before_correct"),
                        "after_correct": row.get("after_correct"),
                        "selected_by_verifier": row.get("selected_by_verifier"),
                        "selected_correct": row.get("selected_correct"),
                        "verifier_score": row.get("verifier_score"),
                    }
                )

    total_examples = len(by_example)
    final_correct = len(selected_correct_examples)
    repaired_selected_correct_count = len(repaired_selected_correct)
    repaired_attribution_share_of_final = repaired_selected_correct_count / final_correct if final_correct else 0.0
    repaired_attribution_share_of_examples = repaired_selected_correct_count / total_examples if total_examples else 0.0
    repaired_any_correct_share = len(set(repaired_any_correct_examples)) / total_examples if total_examples else 0.0

    final_correct_rows = [
        {
            "method": "salvage_amc_sch",
            "final_correct": final_correct,
            "from_original_correct": final_source["from_original_correct"],
            "from_repaired_correct": final_source["from_repaired_correct"],
            "from_fresh_or_new": 0,
            "unknown_source": final_source["unknown_source"],
        }
    ]
    attribution_rows = [
        {
            "comparison": "base_pool_vs_salvage_amc_sch_eval128_existing_artifact",
            "delta_final": delta_final,
            "delta_from_repaired": repaired_attribution_share_of_examples,
            "delta_from_new_coverage": "not_estimated_without_fresh_resample_or_generation_provenance",
            "delta_from_verifier_switch": "not_applicable_same_verifier_for_salvage_row",
            "unexplained": "requires E02 matched fresh-resample and/or generation-level provenance",
            "repaired_attribution_share_of_final_correct": repaired_attribution_share_of_final,
            "repaired_selected_correct_count": repaired_selected_correct_count,
            "final_correct": final_correct,
            "total_examples": total_examples,
        }
    ]
    conversion_rows = []
    for bucket, counts in sorted(completion_by_bucket.items()):
        count = counts["count"]
        conversion_rows.append(
            {
                "source_bucket": bucket,
                "count": count,
                "before_parseable": counts["before_parseable"] / count if count else 0.0,
                "after_parseable": counts["after_parseable"] / count if count else 0.0,
                "before_correct": counts["before_correct"] / count if count else 0.0,
                "after_correct": counts["after_correct"] / count if count else 0.0,
                "selected_rate": counts["selected"] / count if count else 0.0,
            }
        )

    final_correct_path = args.output_dir / "final_correct_source_E01_gsm8k_clean_eval128.csv"
    attribution_path = args.output_dir / "attribution_E01_gsm8k_clean_eval128.csv"
    conversion_path = args.output_dir / "conversion_E01_gsm8k_clean_eval128.csv"
    cases_path = args.output_dir / "cases_E01_gsm8k_clean_eval128.md"
    summary_path = args.output_dir / "summary_E01_gsm8k_clean_eval128.json"

    write_csv(final_correct_path, list(final_correct_rows[0].keys()), final_correct_rows)
    write_csv(attribution_path, list(attribution_rows[0].keys()), attribution_rows)
    write_csv(conversion_path, list(conversion_rows[0].keys()) if conversion_rows else ["source_bucket"], conversion_rows)

    lines = [
        "# E01 GSM8K clean eval128 Salvage Attribution Cases",
        "",
        "Scope: offline attribution from E00 candidate events over existing completion_repair_hygieneonline_hybridp6_v2 artifacts.",
        "",
        "| example_id | slot | source_bucket | before_parseable | after_parseable | before_correct | after_correct | selected_by_verifier | selected_correct | verifier_score |",
        "|---|---:|---|---|---|---|---|---|---|---:|",
    ]
    for row in case_rows:
        lines.append(
            f"| {row['example_id']} | {row['source_slot_id']} | {row['source_bucket']} | {row['before_parseable']} | {row['after_parseable']} | {row['before_correct']} | {row['after_correct']} | {row['selected_by_verifier']} | {row['selected_correct']} | {row['verifier_score']} |"
        )
    cases_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "split": args.split,
        "status": "completed_offline_from_E00_events",
        "total_examples": total_examples,
        "final_correct": final_correct,
        "from_original_correct": final_source["from_original_correct"],
        "from_repaired_correct": final_source["from_repaired_correct"],
        "repaired_attribution_share_of_final_correct": repaired_attribution_share_of_final,
        "repaired_attribution_share_of_examples": repaired_attribution_share_of_examples,
        "repaired_any_correct_share_of_examples": repaired_any_correct_share,
        "base_final": base_final,
        "salvage_final": salvage_final,
        "delta_final": delta_final,
        "paths": {
            "final_correct_source": str(final_correct_path),
            "attribution": str(attribution_path),
            "conversion": str(conversion_path),
            "cases": str(cases_path),
        },
        "scope_note": "This E01 attribution is computed from slot-level before/after events reconstructed from existing eval128 artifacts. It does not replace the E02 matched fresh-resample comparison.",
    }
    write_json(summary_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
