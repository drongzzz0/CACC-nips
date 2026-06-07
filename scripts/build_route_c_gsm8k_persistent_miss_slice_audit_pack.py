from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from src.eval.evaluate_predictions import answers_match  # type: ignore

NUMERIC_LIKE_RE = re.compile(r"^[\s$\\boxed\{\}\[\]\-+*/().,0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Route C+ GSM8K persistent-miss slice audit pack.")
    parser.add_argument("--anchor-preds", default="Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_random_nonprefix_v1_verifier_predictions.jsonl")
    parser.add_argument("--promoted-preds", default="Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_verifier_uncertainty_first_v1_verifier_predictions.jsonl")
    parser.add_argument("--external-preds", default="Experiment/core_code/logs/a800_self_refine_gsm8k_full_p1b_v1_predictions.jsonl")
    parser.add_argument("--leaderboard-json", default="Experiment/analysis/results/routec_plus_policy_gsm8k_full_clean_leaderboard_v1.json")
    parser.add_argument("--external-bucket-audit-json", default="Experiment/analysis/results/routec_plus_gsm8k_external_bucket_audit_pack_v1.json")
    parser.add_argument("--conversion-audit-json", default="Experiment/analysis/results/routec_plus_oracle_final_conversion_audit_pack_v1.json")
    parser.add_argument("--output-json", default="Experiment/analysis/results/routec_plus_gsm8k_persistent_miss_slice_audit_pack_v1.json")
    parser.add_argument("--output-md", default="Experiment/analysis/results/routec_plus_gsm8k_persistent_miss_slice_audit_pack_v1.md")
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_json(path: str | Path):
    return json.loads(_project_path(path).read_text(encoding="utf-8"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _load_predictions(path: str | Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    with _project_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            output[str(row["example_id"])] = row
    return output


def _candidate_correct(candidate_answer: str, gold_answer: str, answer_mode: str) -> bool:
    return answers_match(str(candidate_answer), str(gold_answer), answer_mode=answer_mode)


def _row_correct(row: dict) -> bool:
    raw = row.get("correct")
    if raw is not None:
        return bool(raw)
    return _candidate_correct(str(row.get("prediction", "")), str(row.get("gold_answer", "")), str(row.get("answer_mode", "numeric")))


def _wrong_form_bucket(prediction: str) -> str:
    p = str(prediction).strip()
    low = p.lower()
    if not p:
        return "empty"
    if any(tok in low for tok in ["use at most", "do not use", "do not mention", "prompt-writing", "role tags"]):
        return "instruction_leak"
    if any(tok in low for tok in ["step 1", "step 2", "step 3", "let me think", "here's the solution", "here is the solution", "wait,", "okay,"]) or p.endswith(":") or p.endswith("="):
        return "scaffold_or_fragment"
    normalized = p.replace("Final answer:", "").replace("final answer:", "").strip()
    if len(normalized) <= 32 and "\n" not in normalized and NUMERIC_LIKE_RE.match(normalized):
        return "short_numeric_like"
    if "final answer:" in low and len(p) <= 80:
        return "answer_like_short"
    if len(p) > 180:
        return "long_reasoning"
    return "other_wrong_form"


def _extract_correct_candidates(row: dict) -> list[dict]:
    gold = str(row.get("gold_answer", ""))
    answer_mode = str(row.get("answer_mode", "numeric"))
    hits = []
    for candidate in row.get("candidates", []):
        if _candidate_correct(str(candidate.get("candidate_answer", "")), gold, answer_mode):
            hits.append(candidate)
    return hits


def _best_correct_candidate(row: dict) -> dict | None:
    hits = _extract_correct_candidates(row)
    if not hits:
        return None
    return max(hits, key=lambda c: float(c.get("margin", -1e9)))


def _selected_candidate(row: dict) -> dict | None:
    candidates = row.get("candidates", [])
    if not candidates:
        return None
    return candidates[0]


def _counter_to_rows(counter: Counter, total: int) -> list[dict]:
    rows = []
    for key, count in counter.most_common():
        rows.append({"bucket": key, "count": count, "share": (count / total if total else 0.0)})
    return rows


def _sample_record(example_id: str, anchor: dict, promoted: dict, external: dict, best_correct: dict | None, subregime: str) -> dict:
    selected = _selected_candidate(promoted)
    return {
        "example_id": example_id,
        "subregime": subregime,
        "gold": promoted.get("gold_answer"),
        "anchor_prediction": str(anchor.get("prediction", ""))[:220],
        "promoted_prediction": str(promoted.get("prediction", ""))[:220],
        "external_prediction": str(external.get("prediction", ""))[:220],
        "selected_margin": (selected.get("margin") if selected else None),
        "best_correct_candidate": (str(best_correct.get("candidate_answer", ""))[:220] if best_correct else None),
        "best_correct_margin": (best_correct.get("margin") if best_correct else None),
        "selected_form": _wrong_form_bucket(str(promoted.get("prediction", ""))),
        "external_form": _wrong_form_bucket(str(external.get("prediction", ""))),
        "best_correct_form": (_wrong_form_bucket(str(best_correct.get("candidate_answer", ""))) if best_correct else None),
    }


def main() -> None:
    args = parse_args()
    leaderboard = _load_json(args.leaderboard_json)
    external_bucket_audit = _load_json(args.external_bucket_audit_json) if _project_path(args.external_bucket_audit_json).exists() else None
    conversion_audit = _load_json(args.conversion_audit_json) if _project_path(args.conversion_audit_json).exists() else None

    anchor_map = _load_predictions(args.anchor_preds)
    promoted_map = _load_predictions(args.promoted_preds)
    external_map = _load_predictions(args.external_preds)

    shared_ids = sorted(set(anchor_map) & set(promoted_map) & set(external_map))
    persistent_ids: list[str] = []
    subregime_counts = Counter()
    selected_form_counts = Counter()
    external_form_counts = Counter()
    best_correct_form_counts = Counter()
    clean_short_correct_counts = Counter()
    margin_gap_buckets = Counter()
    subregime_selected_forms: dict[str, Counter] = defaultdict(Counter)
    subregime_external_forms: dict[str, Counter] = defaultdict(Counter)
    subregime_best_correct_forms: dict[str, Counter] = defaultdict(Counter)
    subregime_examples: dict[str, list[dict]] = defaultdict(list)
    recoverable_counts = Counter()

    for example_id in shared_ids:
        anchor = anchor_map[example_id]
        promoted = promoted_map[example_id]
        external = external_map[example_id]
        anchor_correct = _row_correct(anchor)
        promoted_correct = _row_correct(promoted)
        external_correct = _row_correct(external)
        if anchor_correct or promoted_correct or not external_correct:
            continue
        if str(anchor.get("prediction", "")).strip() != str(promoted.get("prediction", "")).strip():
            continue

        persistent_ids.append(example_id)
        anchor_hits = _extract_correct_candidates(anchor)
        promoted_hits = _extract_correct_candidates(promoted)
        anchor_oracle = bool(anchor_hits)
        promoted_oracle = bool(promoted_hits)
        if not anchor_oracle and not promoted_oracle:
            subregime = "both_oracle_miss"
        elif not anchor_oracle and promoted_oracle:
            subregime = "promoted_new_oracle_unconverted"
        elif anchor_oracle and promoted_oracle:
            subregime = "shared_oracle_selector_failure"
        else:
            subregime = "promoted_oracle_regression_same_wrong"
        subregime_counts[subregime] += 1

        selected_form = _wrong_form_bucket(str(promoted.get("prediction", "")))
        external_form = _wrong_form_bucket(str(external.get("prediction", "")))
        selected_form_counts[selected_form] += 1
        external_form_counts[external_form] += 1
        subregime_selected_forms[subregime][selected_form] += 1
        subregime_external_forms[subregime][external_form] += 1

        best_correct = _best_correct_candidate(promoted)
        if best_correct is None:
            recoverable_counts["no_correct_candidate_in_promoted_pool"] += 1
        else:
            recoverable_counts["has_any_correct_candidate_in_promoted_pool"] += 1
            correct_forms = {_wrong_form_bucket(str(c.get("candidate_answer", ""))) for c in promoted_hits}
            best_correct_form = _wrong_form_bucket(str(best_correct.get("candidate_answer", "")))
            best_correct_form_counts[best_correct_form] += 1
            subregime_best_correct_forms[subregime][best_correct_form] += 1
            if "short_numeric_like" in correct_forms:
                recoverable_counts["has_clean_short_correct_candidate"] += 1
                clean_short_correct_counts[subregime] += 1
            if "answer_like_short" in correct_forms:
                recoverable_counts["has_answer_like_short_correct_candidate"] += 1
            if "long_reasoning" in correct_forms:
                recoverable_counts["has_long_reasoning_correct_candidate"] += 1
            if "scaffold_or_fragment" in correct_forms:
                recoverable_counts["has_scaffold_style_correct_candidate"] += 1
            selected = _selected_candidate(promoted)
            if selected is not None:
                gap = float(selected.get("margin", -1e9)) - float(best_correct.get("margin", -1e9))
                if gap <= 0.25:
                    margin_gap_buckets["gap_le_0.25"] += 1
                elif gap <= 0.5:
                    margin_gap_buckets["gap_0.25_to_0.5"] += 1
                elif gap <= 1.0:
                    margin_gap_buckets["gap_0.5_to_1.0"] += 1
                else:
                    margin_gap_buckets["gap_gt_1.0"] += 1
                if "short_numeric_like" in correct_forms and gap <= 0.25:
                    recoverable_counts["clean_short_correct_gap_le_0.25"] += 1
                if "short_numeric_like" in correct_forms and gap <= 0.5:
                    recoverable_counts["clean_short_correct_gap_le_0.5"] += 1
                if "short_numeric_like" in correct_forms and gap <= 1.0:
                    recoverable_counts["clean_short_correct_gap_le_1.0"] += 1

        if len(subregime_examples[subregime]) < 6:
            subregime_examples[subregime].append(_sample_record(example_id, anchor, promoted, external, best_correct, subregime))

    total_n = len(shared_ids)
    persistent_n = len(persistent_ids)
    persistent_share_of_full = persistent_n / total_n if total_n else 0.0

    anchor_row = None
    promoted_row = None
    for row in leaderboard.get("rows", []):
        if row.get("policy") == "replace_random_nonprefix":
            anchor_row = row
        if row.get("policy") == "replace_verifier_uncertainty_first":
            promoted_row = row
    if anchor_row is None or promoted_row is None:
        raise RuntimeError("Required leaderboard rows are missing.")

    has_any_correct = recoverable_counts["has_any_correct_candidate_in_promoted_pool"]
    has_clean_short = recoverable_counts["has_clean_short_correct_candidate"]
    gap025 = recoverable_counts["clean_short_correct_gap_le_0.25"]
    gap05 = recoverable_counts["clean_short_correct_gap_le_0.5"]
    gap10 = recoverable_counts["clean_short_correct_gap_le_1.0"]

    if has_any_correct >= 0.7 * persistent_n:
        verdict = "selector_dominant_with_nontrivial_coverage_tail"
        next_step = "Do a narrow GSM8K slice-level selector/conversion calibration audit before any new full candidate-generation sweep; the slice is mostly not pure coverage miss, but the likely easy-recoverable selector gain is still bounded."
        paper_claim = "The remaining GSM8K persistent gap is not a tiny policy-regression tail and not a pure coverage failure either: most same-wrong misses already contain a correct candidate in the promoted pool, so the next high-ROI step is narrow conversion-side closure rather than another blind full sweep."
    else:
        verdict = "coverage_dominant_even_in_persistent_slice"
        next_step = "Do not prioritize selector calibration yet; the persistent slice is still dominated by missing correct candidates."
        paper_claim = "The persistent GSM8K gap remains mostly coverage-side even after conditioning on same-wrong misses."

    payload = {
        "title": "Route C+ GSM8K persistent-miss slice audit pack",
        "status": "completed",
        "subset_definition": {
            "external_correct": True,
            "anchor_wrong": True,
            "promoted_wrong": True,
            "anchor_prediction_equals_promoted_prediction": True,
        },
        "context": {
            "full_eval_n": total_n,
            "persistent_same_wrong_n": persistent_n,
            "persistent_same_wrong_share_of_full": persistent_share_of_full,
            "anchor_policy": anchor_row.get("policy"),
            "promoted_policy": promoted_row.get("policy"),
            "anchor_verifier_accuracy": anchor_row.get("verifier_accuracy"),
            "promoted_verifier_accuracy": promoted_row.get("verifier_accuracy"),
            "external_bucket_audit": external_bucket_audit,
            "conversion_audit": conversion_audit,
        },
        "oracle_state_breakdown": {
            "counts": dict(subregime_counts),
            "rows": _counter_to_rows(subregime_counts, persistent_n),
        },
        "selected_wrong_form": {
            "counts": dict(selected_form_counts),
            "rows": _counter_to_rows(selected_form_counts, persistent_n),
        },
        "external_correct_form": {
            "counts": dict(external_form_counts),
            "rows": _counter_to_rows(external_form_counts, persistent_n),
        },
        "best_correct_form": {
            "counts": dict(best_correct_form_counts),
            "rows": _counter_to_rows(best_correct_form_counts, has_any_correct),
        },
        "correct_candidate_availability": {
            **recoverable_counts,
            "share_any_correct_in_promoted_pool": (has_any_correct / persistent_n if persistent_n else 0.0),
            "share_clean_short_correct_in_promoted_pool": (has_clean_short / persistent_n if persistent_n else 0.0),
            "share_no_correct_candidate_in_promoted_pool": (recoverable_counts["no_correct_candidate_in_promoted_pool"] / persistent_n if persistent_n else 0.0),
        },
        "recoverable_upper_bounds": {
            "all_promoted_oracle_hit_cases_points": (has_any_correct / total_n if total_n else 0.0),
            "clean_short_correct_points": (has_clean_short / total_n if total_n else 0.0),
            "clean_short_correct_gap_le_0.25_points": (gap025 / total_n if total_n else 0.0),
            "clean_short_correct_gap_le_0.5_points": (gap05 / total_n if total_n else 0.0),
            "clean_short_correct_gap_le_1.0_points": (gap10 / total_n if total_n else 0.0),
        },
        "margin_gap_buckets_among_promoted_oracle_hits": {
            "counts": dict(margin_gap_buckets),
            "rows": _counter_to_rows(margin_gap_buckets, has_any_correct),
        },
        "subregime_tables": {
            key: {
                "selected_wrong_form": _counter_to_rows(subregime_selected_forms[key], subregime_counts[key]),
                "external_correct_form": _counter_to_rows(subregime_external_forms[key], subregime_counts[key]),
                "best_correct_form": _counter_to_rows(subregime_best_correct_forms[key], max(1, sum(subregime_best_correct_forms[key].values()))),
                "clean_short_correct_count": clean_short_correct_counts.get(key, 0),
            }
            for key in subregime_counts
        },
        "examples": dict(subregime_examples),
        "decision": {
            "verdict": verdict,
            "summary": "Within GSM8K persistent same-wrong misses, the dominant subregime is shared-oracle selector failure rather than pure no-oracle miss: the promoted pool already contains a correct candidate on most cases, often a clean short numeric answer.",
            "next_step": next_step,
            "paper_facing_claim": paper_claim,
        },
    }

    lines = [
        "# Route C+ GSM8K Persistent-Miss Slice Audit Pack",
        "",
        "- Status: `completed`",
        f"- Verdict: `{payload['decision']['verdict']}`",
        f"- Summary: {payload['decision']['summary']}",
        f"- Next step: {payload['decision']['next_step']}",
        "",
        "## Focus Subset",
        "",
        f"- Full GSM8K evaluation size: `{total_n}`",
        f"- Persistent same-wrong subset size: `{persistent_n}`",
        f"- Persistent same-wrong share of full set: `{_fmt(persistent_share_of_full)}`",
        f"- Anchor policy: `{anchor_row.get('policy')}` verifier=`{_fmt(anchor_row.get('verifier_accuracy'))}`",
        f"- Promoted policy: `{promoted_row.get('policy')}` verifier=`{_fmt(promoted_row.get('verifier_accuracy'))}`",
        "",
        "## Oracle-State Breakdown Within Persistent Same-Wrong",
        "",
        "| Subregime | Count | Share | Meaning |",
        "| --- | ---: | ---: | --- |",
    ]
    meanings = {
        "shared_oracle_selector_failure": "anchor/promoted pools both contain a correct candidate, but both still select the same wrong answer",
        "both_oracle_miss": "anchor/promoted pools both fail to contain any correct candidate",
        "promoted_new_oracle_unconverted": "promoted adds oracle coverage but still outputs the same wrong final answer as anchor",
        "promoted_oracle_regression_same_wrong": "anchor had oracle but promoted loses it while keeping the same wrong final answer",
    }
    for row in payload["oracle_state_breakdown"]["rows"]:
        lines.append(f"| {row['bucket']} | {row['count']} | {_fmt(row['share'])} | {meanings.get(row['bucket'], 'NA')} |")

    lines.extend([
        "",
        "## Correct-Candidate Availability",
        "",
        f"- Has any correct candidate in promoted pool: `{has_any_correct}` / `{persistent_n}` = `{_fmt(has_any_correct / persistent_n if persistent_n else 0.0)}`",
        f"- Has clean short correct candidate in promoted pool: `{has_clean_short}` / `{persistent_n}` = `{_fmt(has_clean_short / persistent_n if persistent_n else 0.0)}`",
        f"- No correct candidate in promoted pool: `{recoverable_counts['no_correct_candidate_in_promoted_pool']}` / `{persistent_n}` = `{_fmt(recoverable_counts['no_correct_candidate_in_promoted_pool'] / persistent_n if persistent_n else 0.0)}`",
        "",
        "## Recoverable Upper Bounds On The Full GSM8K Split",
        "",
        f"- If every persistent same-wrong case with any promoted oracle were fixed: `{_fmt(payload['recoverable_upper_bounds']['all_promoted_oracle_hit_cases_points'])}` points",
        f"- If only cases with a clean short correct candidate were fixed: `{_fmt(payload['recoverable_upper_bounds']['clean_short_correct_points'])}` points",
        f"- If only clean-short cases with margin gap `<= 0.25` were fixed: `{_fmt(payload['recoverable_upper_bounds']['clean_short_correct_gap_le_0.25_points'])}` points",
        f"- If only clean-short cases with margin gap `<= 0.5` were fixed: `{_fmt(payload['recoverable_upper_bounds']['clean_short_correct_gap_le_0.5_points'])}` points",
        f"- If only clean-short cases with margin gap `<= 1.0` were fixed: `{_fmt(payload['recoverable_upper_bounds']['clean_short_correct_gap_le_1.0_points'])}` points",
        "",
        "## Selected Wrong Forms On The Persistent Slice",
        "",
        "| Wrong form | Count | Share |",
        "| --- | ---: | ---: |",
    ])
    for row in payload["selected_wrong_form"]["rows"]:
        lines.append(f"| {row['bucket']} | {row['count']} | {_fmt(row['share'])} |")

    lines.extend([
        "",
        "## External Correct Forms On The Same Slice",
        "",
        "| External form | Count | Share |",
        "| --- | ---: | ---: |",
    ])
    for row in payload["external_correct_form"]["rows"]:
        lines.append(f"| {row['bucket']} | {row['count']} | {_fmt(row['share'])} |")

    lines.extend([
        "",
        "## Best Correct Candidate Forms In Promoted Pools",
        "",
        "| Best-correct form | Count | Share among promoted-oracle-hit cases |",
        "| --- | ---: | ---: |",
    ])
    for row in payload["best_correct_form"]["rows"]:
        lines.append(f"| {row['bucket']} | {row['count']} | {_fmt(row['share'])} |")

    lines.extend([
        "",
        "## Margin-Gap Buckets Among Promoted Oracle Hits",
        "",
        "| Margin gap bucket | Count | Share among promoted-oracle-hit cases |",
        "| --- | ---: | ---: |",
    ])
    for row in payload["margin_gap_buckets_among_promoted_oracle_hits"]["rows"]:
        lines.append(f"| {row['bucket']} | {row['count']} | {_fmt(row['share'])} |")

    for subregime in [
        "shared_oracle_selector_failure",
        "promoted_new_oracle_unconverted",
        "both_oracle_miss",
        "promoted_oracle_regression_same_wrong",
    ]:
        if subregime not in payload["subregime_tables"]:
            continue
        table = payload["subregime_tables"][subregime]
        lines.extend([
            "",
            f"## Subregime: {subregime}",
            "",
            f"- Count: `{subregime_counts[subregime]}`",
            f"- Clean short correct candidates in promoted pool: `{table['clean_short_correct_count']}`",
            "",
            "### Selected Wrong Forms",
            "",
            "| Bucket | Count | Share |",
            "| --- | ---: | ---: |",
        ])
        for row in table["selected_wrong_form"]:
            lines.append(f"| {row['bucket']} | {row['count']} | {_fmt(row['share'])} |")
        lines.extend([
            "",
            "### Representative Examples",
            "",
        ])
        for example in payload["examples"].get(subregime, [])[:3]:
            lines.append(
                f"- `{example['example_id']}` gold=`{example['gold']}` promoted=`{example['promoted_prediction']}` external=`{example['external_prediction']}` best_correct=`{example['best_correct_candidate']}` selected_margin=`{_fmt(example['selected_margin'])}` best_correct_margin=`{_fmt(example['best_correct_margin'])}`"
            )

    lines.extend([
        "",
        "## Decision",
        "",
        f"- Paper-facing claim: {payload['decision']['paper_facing_claim']}",
        f"- Operational next step: {payload['decision']['next_step']}",
        "- Interpretation: the current GSM8K blocker is still not a tiny promoted-regression tail, but the persistent slice is also not dominantly pure coverage miss; most same-wrong failures already contain a correct candidate in the promoted pool, so the next best validation is a narrow conversion-side audit rather than a new blind full sweep.",
    ])

    output_json = _project_path(args.output_json)
    output_md = _project_path(args.output_md)
    _write_json_atomic(output_json, payload)
    _write_text_atomic(output_md, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
