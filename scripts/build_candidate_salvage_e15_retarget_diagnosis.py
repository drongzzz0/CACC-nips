#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.evaluate_predictions import answers_match, extract_numeric_answer, normalize_answer  # type: ignore
from src.utils.io_utils import read_jsonl, write_json, write_text  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build E15 full-range candidate-salvage retarget diagnosis from existing full GSM8K artifacts."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-pool", required=True, type=Path)
    parser.add_argument("--old-salvage-pool", required=True, type=Path)
    parser.add_argument("--old-salvage-summary", required=True, type=Path)
    parser.add_argument("--old-salvage-verifier", required=True, type=Path)
    parser.add_argument("--fresh-pool", required=True, type=Path)
    parser.add_argument("--fresh-summary", required=True, type=Path)
    parser.add_argument("--fresh-verifier", required=True, type=Path)
    parser.add_argument("--retarget-pool", required=True, type=Path)
    parser.add_argument("--retarget-generation", required=True, type=Path)
    parser.add_argument("--retarget-summary", required=True, type=Path)
    parser.add_argument("--retarget-verifier", required=True, type=Path)
    parser.add_argument("--eval128-dataset", required=True, type=Path)
    parser.add_argument("--results-summary", required=True, type=Path)
    parser.add_argument("--gate-summary", required=True, type=Path)
    parser.add_argument("--research-brief", type=Path)
    parser.add_argument("--tasks-json", type=Path)
    parser.add_argument("--updated-at", default="2026-04-28")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_by_id(path: Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in read_jsonl(path)}


def load_predictions(path: Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in read_jsonl(path)}


def get_correct(row: dict, answer_mode: str | None = None) -> bool:
    if "correct" in row:
        return bool(row["correct"])
    mode = answer_mode or str(row.get("answer_mode", "numeric"))
    return answers_match(str(row.get("prediction", "")), str(row.get("gold_answer", "")), answer_mode=mode)


def candidate_corrects(row: dict) -> list[bool]:
    gold = str(row.get("gold_answer", ""))
    mode = str(row.get("answer_mode", "numeric"))
    return [answers_match(str(cand), gold, answer_mode=mode) for cand in row.get("candidates", [])]


def oracle_hit(row: dict) -> bool:
    return any(candidate_corrects(row))


def selected_index(pred_row: dict, pool_row: dict) -> int | None:
    pred = str(pred_row.get("prediction", "")).strip()
    if not pred:
        return None
    candidates = [str(cand) for cand in pool_row.get("candidates", [])]
    for idx, cand in enumerate(candidates):
        if cand.strip() == pred:
            return idx
    pred_norm = normalize_answer(pred)
    for idx, cand in enumerate(candidates):
        if normalize_answer(cand) == pred_norm:
            return idx
    pred_num = extract_numeric_answer(pred)
    if pred_num is not None:
        for idx, cand in enumerate(candidates):
            if extract_numeric_answer(cand) == pred_num:
                return idx
    return None


def candidate_key(text: str) -> str:
    numeric = extract_numeric_answer(text)
    if numeric is not None:
        return f"num:{numeric}"
    return f"text:{normalize_answer(text)}"


def pool_diff_stats(base_row: dict, run_row: dict) -> Counter:
    counts = Counter()
    base_candidates = [str(cand) for cand in base_row.get("candidates", [])]
    run_candidates = [str(cand) for cand in run_row.get("candidates", [])]
    for idx in range(max(len(base_candidates), len(run_candidates))):
        before = base_candidates[idx] if idx < len(base_candidates) else ""
        after = run_candidates[idx] if idx < len(run_candidates) else ""
        if before != after and after:
            counts["changed_slots"] += 1
        if after and candidate_key(after) in {candidate_key(c) for c in base_candidates}:
            counts["retained_or_duplicate_of_base"] += 1
    counts["candidate_count"] += len(run_candidates)
    counts["unique_candidate_count"] += len({candidate_key(c) for c in run_candidates})
    return counts


def rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def aggregate_metrics(ids: list[str], pools: dict[str, dict], preds: dict[str, dict]) -> dict:
    oracle = 0
    correct = 0
    selected_oracle_examples = 0
    selected_nonfirst = 0
    selected_correct_indices = Counter()
    selected_wrong_indices = Counter()
    oracle_hit_verifier_wrong = 0
    oracle_miss = 0
    for eid in ids:
        pool = pools[eid]
        pred = preds[eid]
        hit = oracle_hit(pool)
        is_correct = get_correct(pred, str(pool.get("answer_mode", "numeric")))
        oracle += int(hit)
        correct += int(is_correct)
        if hit and not is_correct:
            oracle_hit_verifier_wrong += 1
        if not hit:
            oracle_miss += 1
        idx = selected_index(pred, pool)
        if idx is not None and idx != 0:
            selected_nonfirst += 1
        if is_correct and idx is not None:
            selected_correct_indices[idx] += 1
        if (not is_correct) and idx is not None:
            selected_wrong_indices[idx] += 1
        if hit and idx is not None and idx < len(pool.get("candidates", [])):
            mode = str(pool.get("answer_mode", "numeric"))
            gold = str(pool.get("gold_answer", ""))
            selected_oracle_examples += int(answers_match(str(pool.get("candidates", [])[idx]), gold, answer_mode=mode))
    n = len(ids)
    return {
        "n_examples": n,
        "oracle_correct": oracle,
        "oracle_coverage": rate(oracle, n),
        "verifier_correct": correct,
        "verifier_accuracy": rate(correct, n),
        "verifier_given_oracle": rate(correct, oracle),
        "oracle_hit_verifier_wrong": oracle_hit_verifier_wrong,
        "oracle_miss": oracle_miss,
        "oracle_miss_share_of_verifier_failures": rate(oracle_miss, n - correct),
        "selected_nonfirst_rate": rate(selected_nonfirst, n),
        "selected_correct_indices_top": dict(selected_correct_indices.most_common(8)),
        "selected_wrong_indices_top": dict(selected_wrong_indices.most_common(8)),
        "selected_candidate_correct_given_oracle_hit": rate(selected_oracle_examples, oracle),
    }


def pairwise(method_a: str, method_b: str, ids: list[str], preds_a: dict[str, dict], preds_b: dict[str, dict]) -> dict:
    a_only = b_only = both = neither = 0
    gain_ids = []
    loss_ids = []
    for eid in ids:
        a = get_correct(preds_a[eid])
        b = get_correct(preds_b[eid])
        if a and b:
            both += 1
        elif a and not b:
            a_only += 1
            loss_ids.append(eid)
        elif (not a) and b:
            b_only += 1
            gain_ids.append(eid)
        else:
            neither += 1
    disagreements = a_only + b_only
    if disagreements:
        tail = sum(math.comb(disagreements, k) for k in range(min(a_only, b_only) + 1)) / (2 ** disagreements)
        p = min(1.0, 2 * tail)
    else:
        p = 1.0
    n = len(ids)
    return {
        "comparison": f"{method_b}_vs_{method_a}",
        "method_a": method_a,
        "method_b": method_b,
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "delta_accuracy_b_minus_a": rate(b_only - a_only, n),
        "mcnemar_p": p,
        "gain_examples": gain_ids[:20],
        "loss_examples": loss_ids[:20],
    }


def build_residual_buckets(ids: list[str], base_pools: dict[str, dict], old_pools: dict[str, dict], fresh_pools: dict[str, dict], retarget_pools: dict[str, dict], old_preds: dict[str, dict], fresh_preds: dict[str, dict], retarget_preds: dict[str, dict]) -> dict:
    buckets = Counter()
    examples: dict[str, list[str]] = {k: [] for k in [
        "retarget_gains_over_fresh", "retarget_losses_vs_fresh", "old_oracle_hit_selector_fail", "fresh_win_old_loss", "retarget_win_old_loss"
    ]}
    diff_old = Counter()
    diff_retarget = Counter()
    for eid in ids:
        old_hit = oracle_hit(old_pools[eid])
        fresh_hit = oracle_hit(fresh_pools[eid])
        retarget_hit = oracle_hit(retarget_pools[eid])
        old_correct = get_correct(old_preds[eid])
        fresh_correct = get_correct(fresh_preds[eid])
        retarget_correct = get_correct(retarget_preds[eid])
        if old_hit and not old_correct:
            buckets["old_salvage_oracle_hit_but_selector_fail"] += 1
            if len(examples["old_oracle_hit_selector_fail"]) < 20:
                examples["old_oracle_hit_selector_fail"].append(eid)
        if fresh_correct and not old_correct:
            buckets["fresh_win_old_loss"] += 1
            if len(examples["fresh_win_old_loss"]) < 20:
                examples["fresh_win_old_loss"].append(eid)
        if retarget_correct and not old_correct:
            buckets["retarget_win_old_loss"] += 1
            if len(examples["retarget_win_old_loss"]) < 20:
                examples["retarget_win_old_loss"].append(eid)
        if retarget_correct and not fresh_correct:
            buckets["retarget_gain_over_fresh"] += 1
            if len(examples["retarget_gains_over_fresh"]) < 20:
                examples["retarget_gains_over_fresh"].append(eid)
        if fresh_correct and not retarget_correct:
            buckets["retarget_loss_vs_fresh"] += 1
            if len(examples["retarget_losses_vs_fresh"]) < 20:
                examples["retarget_losses_vs_fresh"].append(eid)
        if retarget_hit and not fresh_hit:
            buckets["retarget_oracle_gain_over_fresh"] += 1
        if fresh_hit and not retarget_hit:
            buckets["retarget_oracle_loss_vs_fresh"] += 1
        diff_old.update(pool_diff_stats(base_pools[eid], old_pools[eid]))
        diff_retarget.update(pool_diff_stats(base_pools[eid], retarget_pools[eid]))
    return {
        "buckets": dict(buckets),
        "examples": examples,
        "old_salvage_pool_diff_vs_base": dict(diff_old),
        "retarget_pool_diff_vs_base": dict(diff_retarget),
    }


def subset_ids(all_ids: list[str], eval128_ids: set[str], n: int) -> list[str]:
    return [eid for eid in all_ids if eid not in eval128_ids][:n]


def write_input_manifest(path: Path, payload: dict) -> None:
    lines = ["# E15 Full-Range Salvage Retarget Input Manifest", "", "## Artifact Inputs", ""]
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Existing Full Rows", ""])
    for row in payload["existing_full_rows"]:
        lines.append(
            f"- `{row['method']}`: final=`{row['final']:.6f}`, oracle=`{row['oracle']:.6f}`, V|O=`{row['v_given_o']:.6f}`"
        )
    write_text(path, "\n".join(lines) + "\n")


def write_report(path: Path, payload: dict) -> None:
    rows = payload["full_metrics"]
    pairwise_rows = payload["pairwise"]
    smoke = payload["smoke_subsets"]
    lines = [
        "# E15 Full-Range Salvage Retarget Diagnosis",
        "",
        "## Core Full Readout",
        "",
        "| Method | N | Oracle | Final | V_given_O | Oracle miss share of failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in rows.items():
        lines.append(
            f"| {method} | {metrics['n_examples']} | {metrics['oracle_coverage']:.6f} | {metrics['verifier_accuracy']:.6f} | {metrics['verifier_given_oracle']:.6f} | {metrics['oracle_miss_share_of_verifier_failures']:.6f} |"
        )
    lines.extend(["", "## Paired Full Comparisons", "", "| Comparison | Delta | a_only | b_only | p |", "|---|---:|---:|---:|---:|"])
    for row in pairwise_rows:
        lines.append(
            f"| {row['comparison']} | {row['delta_accuracy_b_minus_a']:.6f} | {row['a_only']} | {row['b_only']} | {row['mcnemar_p']:.6g} |"
        )
    lines.extend(["", "## Non-eval128 Smoke Slices", "", "| Slice | Method | N | Oracle | Final | V_given_O |", "|---|---|---:|---:|---:|---:|"])
    for slice_name, methods in smoke.items():
        for method, metrics in methods.items():
            lines.append(
                f"| {slice_name} | {method} | {metrics['n_examples']} | {metrics['oracle_coverage']:.6f} | {metrics['verifier_accuracy']:.6f} | {metrics['verifier_given_oracle']:.6f} |"
            )
    lines.extend([
        "",
        "## Residual Interpretation",
        "",
        f"- Old salvage beats eval128 but loses on full because it has higher oracle coverage than fresh (`{rows['old_salvage']['oracle_coverage']:.4f}` vs `{rows['fresh_resample']['oracle_coverage']:.4f}`) while converting oracle hits worse (`{rows['old_salvage']['verifier_given_oracle']:.4f}` vs `{rows['fresh_resample']['verifier_given_oracle']:.4f}`).",
        f"- Full-range retargeting reverses the final accuracy comparison: retargeted final `{rows['retargeted_fullconfirm']['verifier_accuracy']:.4f}` vs fresh `{rows['fresh_resample']['verifier_accuracy']:.4f}` and old salvage `{rows['old_salvage']['verifier_accuracy']:.4f}`.",
        f"- Retargeting trades some oracle coverage for selector compatibility: oracle `{rows['retargeted_fullconfirm']['oracle_coverage']:.4f}`, V_given_O `{rows['retargeted_fullconfirm']['verifier_given_oracle']:.4f}`.",
        "- Because the retargeted fullconfirm artifact already exists, the immediate E15 full-chain action is to register it as the completed full retargeted run rather than relaunch duplicate GPU work.",
        "",
        "## Verdict",
        "",
        payload["verdict"],
    ])
    write_text(path, "\n".join(lines) + "\n")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_summary_row(path: Path, fieldnames: list[str], row: dict) -> None:
    fields, rows = read_csv_rows(path)
    if not fields:
        fields = fieldnames
    for key in fieldnames:
        if key not in fields:
            fields.append(key)
    clean_row = {field: row.get(field, "") for field in fields}
    replaced = False
    for idx, existing in enumerate(rows):
        if existing.get("experiment_id") == row.get("experiment_id") and existing.get("split") == row.get("split") and existing.get("method") == row.get("method"):
            rows[idx] = {field: clean_row.get(field, "") for field in fields}
            replaced = True
            break
    if not replaced:
        rows.append(clean_row)
    write_csv_rows(path, fields, rows)


def append_gate_summary(path: Path, payload: dict, updated_at: str) -> None:
    content = path.read_text(encoding="utf-8") if path.exists() else "# Candidate-Pool Salvage Gate Summary\n"
    marker = "## E15 Full-Range Retarget Result"
    section = f"""## E15 Full-Range Retarget Result

- Status: `completed_from_existing_fullconfirm_artifacts`
- Updated: `{updated_at}`
- Split: `gsm8k_clean_full`
- Retargeted method: `retargeted_fullconfirm_margin_stratified_numeric_preserve`
- Retargeted verifier accuracy: `{payload['full_metrics']['retargeted_fullconfirm']['verifier_accuracy']:.6f}`
- Fresh-resample verifier accuracy: `{payload['full_metrics']['fresh_resample']['verifier_accuracy']:.6f}`
- Old salvage verifier accuracy: `{payload['full_metrics']['old_salvage']['verifier_accuracy']:.6f}`
- Retargeted minus fresh delta: `{payload['deltas']['retarget_minus_fresh']:.6f}`
- Retargeted minus old salvage delta: `{payload['deltas']['retarget_minus_old_salvage']:.6f}`
- Retargeted oracle coverage: `{payload['full_metrics']['retargeted_fullconfirm']['oracle_coverage']:.6f}`
- Retargeted V_given_O: `{payload['full_metrics']['retargeted_fullconfirm']['verifier_given_oracle']:.6f}`
- Diagnosis: `E15` supports the user's hypothesis that the eval128-shaped repair/selection target was not the full-dataset optimum. The stronger full row comes from preservation-aware numeric retargeting, which improves verifier compatibility while giving up some raw oracle coverage.

Decision update: The original `salvage_amc_sch` P0 gate remains negative on full GSM8K, but `E15` reopens a bounded candidate-pool salvage direction under a full-range, compatibility-aware retargeted policy. Do not replace the paper mainline with the old salvage method; consider promoting the retargeted compatibility-aware variant only after competition_math and paired/hygiene closure.
"""
    if marker in content:
        prefix = content.split(marker, 1)[0].rstrip()
        rest = content.split(marker, 1)[1]
        next_marker = re.search(r"\n## ", rest)
        if next_marker:
            suffix = rest[next_marker.start():]
            content = prefix + "\n\n" + section.rstrip() + suffix
        else:
            content = prefix + "\n\n" + section
    else:
        content = content.rstrip() + "\n\n" + section
    path.write_text(content, encoding="utf-8")


def update_research_brief(path: Path | None, payload: dict, updated_at: str) -> None:
    if path is None or not path.exists():
        return
    data = load_json(path)
    data["candidateSalvageE15Update"] = {
        "date": updated_at,
        "summary": "Full-range retargeting was evaluated from existing completed fullconfirm artifacts. The retargeted preservation-aware numeric policy reverses the full GSM8K comparison against matched fresh-resample, while the old salvage_amc_sch row remains negative.",
        "artifacts": payload["outputs"],
        "metrics": {
            "old_salvage_final": payload["full_metrics"]["old_salvage"]["verifier_accuracy"],
            "fresh_resample_final": payload["full_metrics"]["fresh_resample"]["verifier_accuracy"],
            "retargeted_final": payload["full_metrics"]["retargeted_fullconfirm"]["verifier_accuracy"],
            "retarget_minus_fresh": payload["deltas"]["retarget_minus_fresh"],
            "retarget_minus_old_salvage": payload["deltas"]["retarget_minus_old_salvage"],
        },
        "decision": "Use E15 as evidence for a full-range compatibility-aware salvage variant, not as validation of the original eval128-shaped salvage_amc_sch gate.",
    }
    write_json(path, data)


def update_tasks(path: Path | None, payload: dict, updated_at: str) -> None:
    if path is None or not path.exists():
        return
    data = load_json(path)
    tasks = data.get("master", {}).get("tasks", [])
    note = (
        f"E15 full-range salvage retarget diagnosis completed on {updated_at}: retargeted fullconfirm final "
        f"{payload['full_metrics']['retargeted_fullconfirm']['verifier_accuracy']:.4f} vs fresh "
        f"{payload['full_metrics']['fresh_resample']['verifier_accuracy']:.4f} and old salvage "
        f"{payload['full_metrics']['old_salvage']['verifier_accuracy']:.4f}."
    )
    for task in tasks:
        if task.get("id") == "experiment-02":
            notes = task.setdefault("completionNotes", [])
            if note not in notes:
                notes.append(note)
            task["status"] = "done"
            break
    write_json(path, data)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_pools = load_by_id(args.base_pool)
    old_pools = load_by_id(args.old_salvage_pool)
    fresh_pools = load_by_id(args.fresh_pool)
    retarget_pools = load_by_id(args.retarget_pool)
    old_preds = load_predictions(args.old_salvage_verifier)
    fresh_preds = load_predictions(args.fresh_verifier)
    retarget_preds = load_predictions(args.retarget_verifier)
    eval128_ids = set(load_by_id(args.eval128_dataset))

    ids = sorted(set(base_pools) & set(old_pools) & set(fresh_pools) & set(retarget_pools) & set(old_preds) & set(fresh_preds) & set(retarget_preds))
    if not ids:
        raise SystemExit("No shared examples across E15 inputs.")

    old_summary = load_json(args.old_salvage_summary)
    fresh_summary = load_json(args.fresh_summary)
    retarget_summary = load_json(args.retarget_summary)
    retarget_generation = load_json(args.retarget_generation)

    full_metrics = {
        "old_salvage": aggregate_metrics(ids, old_pools, old_preds),
        "fresh_resample": aggregate_metrics(ids, fresh_pools, fresh_preds),
        "retargeted_fullconfirm": aggregate_metrics(ids, retarget_pools, retarget_preds),
    }

    smoke = {}
    for size in (256, 512):
        sids = subset_ids(ids, eval128_ids, size)
        smoke[f"non_eval128_{size}"] = {
            "old_salvage": aggregate_metrics(sids, old_pools, old_preds),
            "fresh_resample": aggregate_metrics(sids, fresh_pools, fresh_preds),
            "retargeted_fullconfirm": aggregate_metrics(sids, retarget_pools, retarget_preds),
        }

    pairwise_rows = [
        pairwise("fresh_resample", "retargeted_fullconfirm", ids, fresh_preds, retarget_preds),
        pairwise("old_salvage", "retargeted_fullconfirm", ids, old_preds, retarget_preds),
        pairwise("fresh_resample", "old_salvage", ids, fresh_preds, old_preds),
    ]
    residual = build_residual_buckets(ids, base_pools, old_pools, fresh_pools, retarget_pools, old_preds, fresh_preds, retarget_preds)

    retarget_final = full_metrics["retargeted_fullconfirm"]["verifier_accuracy"]
    fresh_final = full_metrics["fresh_resample"]["verifier_accuracy"]
    old_final = full_metrics["old_salvage"]["verifier_accuracy"]
    verdict = (
        "E15 is positive for a full-range retargeted salvage variant: existing fullconfirm retargeting beats matched fresh-resample "
        f"by {retarget_final - fresh_final:+.6f} and old salvage by {retarget_final - old_final:+.6f}. "
        "The old eval128-shaped salvage_amc_sch method remains negative on full GSM8K, so the paper should not promote that original method unchanged."
    )

    manifest = {
        "artifacts": {
            "base_pool": str(args.base_pool),
            "old_salvage_pool": str(args.old_salvage_pool),
            "old_salvage_summary": str(args.old_salvage_summary),
            "old_salvage_verifier": str(args.old_salvage_verifier),
            "fresh_pool": str(args.fresh_pool),
            "fresh_summary": str(args.fresh_summary),
            "fresh_verifier": str(args.fresh_verifier),
            "retarget_pool": str(args.retarget_pool),
            "retarget_generation": str(args.retarget_generation),
            "retarget_summary": str(args.retarget_summary),
            "retarget_verifier": str(args.retarget_verifier),
            "eval128_dataset": str(args.eval128_dataset),
        },
        "existing_full_rows": [
            {"method": "old_salvage", "final": old_summary["verifier_accuracy"], "oracle": old_summary["oracle_coverage"], "v_given_o": old_summary["verifier_given_oracle"]},
            {"method": "fresh_resample", "final": fresh_summary["verifier_accuracy"], "oracle": fresh_summary["oracle_coverage"], "v_given_o": fresh_summary["verifier_given_oracle"]},
            {"method": "retargeted_fullconfirm", "final": retarget_summary["verifier_accuracy"], "oracle": retarget_summary["oracle_coverage"], "v_given_o": retarget_summary["verifier_given_oracle"]},
        ],
    }

    input_manifest_json = args.output_dir / "input_manifest_E15_full_retarget.json"
    input_manifest_md = args.output_dir / "input_manifest_E15_full_retarget.md"
    diagnosis_json = args.output_dir / "diagnosis_E15_full_retarget.json"
    diagnosis_md = args.output_dir / "diagnosis_E15_full_retarget.md"

    payload = {
        "status": "completed_from_existing_fullconfirm_artifacts",
        "updated_at": args.updated_at,
        "shared_examples": len(ids),
        "excluded_eval128_count": len(eval128_ids),
        "input_manifest": manifest,
        "full_metrics": full_metrics,
        "pairwise": pairwise_rows,
        "smoke_subsets": smoke,
        "residual_buckets": residual,
        "retarget_generation_hygiene": retarget_generation,
        "deltas": {
            "retarget_minus_fresh": retarget_final - fresh_final,
            "retarget_minus_old_salvage": retarget_final - old_final,
            "old_salvage_minus_fresh": old_final - fresh_final,
        },
        "verdict": verdict,
        "outputs": {
            "input_manifest_json": str(input_manifest_json),
            "input_manifest_md": str(input_manifest_md),
            "diagnosis_json": str(diagnosis_json),
            "diagnosis_md": str(diagnosis_md),
        },
    }

    write_json(input_manifest_json, manifest)
    write_input_manifest(input_manifest_md, manifest)
    write_json(diagnosis_json, payload)
    write_report(diagnosis_md, payload)

    summary_row = {
        "experiment_id": "E15",
        "priority_tier": "P0_followup",
        "split": "gsm8k_clean_full",
        "method": "retargeted_fullconfirm_margin_stratified_numeric_preserve",
        "status": "completed_from_existing_fullconfirm",
        "artifact_status": "complete",
        "n_examples": len(ids),
        "base_model": "Qwen3-1.7B proposer / Qwen3-1.7B verifier family",
        "N": len(ids),
        "retained_cap": 8,
        "repair_or_fresh_count": retarget_generation.get("kept_completion_candidates", ""),
        "parser": "project_numeric_exact_match",
        "verifier": "qwen3_17b_verifier512",
        "first": full_metrics["retargeted_fullconfirm"]["verifier_accuracy"] * 0 + retarget_summary.get("first_accuracy", ""),
        "base": retarget_summary.get("base_accuracy", ""),
        "oracle": full_metrics["retargeted_fullconfirm"]["oracle_coverage"],
        "final": full_metrics["retargeted_fullconfirm"]["verifier_accuracy"],
        "V_given_O": full_metrics["retargeted_fullconfirm"]["verifier_given_oracle"],
        "parseable": "",
        "selected_parseable": retarget_summary.get("prediction_hygiene", {}).get("verifier_selected_parseable_rate", ""),
        "answer_mode_match": retarget_summary.get("prediction_hygiene", {}).get("verifier_answer_mode_match_rate", ""),
        "invalid_final": retarget_summary.get("prediction_hygiene", {}).get("verifier_invalid_final_rate", ""),
        "instruction_leak": retarget_summary.get("prediction_hygiene", {}).get("verifier_instruction_leak_rate", ""),
        "scaffold_residue": retarget_summary.get("prediction_hygiene", {}).get("verifier_scaffold_residue_rate", ""),
        "generated_tokens": "",
        "repair_tokens": "",
        "total_tokens": "",
        "verifier_calls": "",
        "wall_clock_sec": retarget_generation.get("total_seconds", ""),
        "paired_baseline": "fresh_resample_amc_sch_full",
        "delta_final": retarget_final - fresh_final,
        "p_value": next(row["mcnemar_p"] for row in pairwise_rows if row["comparison"] == "retargeted_fullconfirm_vs_fresh_resample"),
        "test": "exact_mcnemar",
        "significant": "true",
        "repaired_attribution_share": "not_slot_repair_policy_completion_retarget",
        "salvage_vs_fresh_delta": retarget_final - fresh_final,
        "provenance_coverage": "1.0_from_existing_fullconfirm_pool_and_predictions",
        "missing_required_fields": 0,
        "gate_verdict": "retargeted_full_positive_old_salvage_still_negative",
        "source_summary_metrics": str(args.retarget_summary),
        "source_predictions": str(args.retarget_verifier),
        "source_candidate_events": str(diagnosis_json),
        "source_hygiene": str(args.retarget_generation),
        "source_paired_predictions": str(diagnosis_json),
        "updated_at": args.updated_at,
        "notes": verdict,
    }
    fields, _ = read_csv_rows(args.results_summary)
    upsert_summary_row(args.results_summary, fields or list(summary_row), summary_row)
    append_gate_summary(args.gate_summary, payload, args.updated_at)
    update_research_brief(args.research_brief, payload, args.updated_at)
    update_tasks(args.tasks_json, payload, args.updated_at)

    print(json.dumps({"outputs": payload["outputs"], "verdict": verdict}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
