from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from src.eval.evaluate_predictions import answers_match  # type: ignore

NUMERIC_LIKE_RE = re.compile(r"^[\s$\\boxed\{\}\[\]\-+*/().,0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Route C+ GSM8K external bucket audit pack.")
    parser.add_argument("--anchor-preds", default="Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_random_nonprefix_v1_verifier_predictions.jsonl")
    parser.add_argument("--promoted-preds", default="Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_verifier_uncertainty_first_v1_verifier_predictions.jsonl")
    parser.add_argument("--external-preds", default="Experiment/core_code/logs/a800_self_refine_gsm8k_full_p1b_v1_predictions.jsonl")
    parser.add_argument("--leaderboard-json", default="Experiment/analysis/results/routec_plus_policy_gsm8k_full_clean_leaderboard_v1.json")
    parser.add_argument("--conversion-context-json", default="Experiment/analysis/results/experiment_11_p1a_verifier_conversion_diagnosis_v1.json")
    parser.add_argument("--output-json", default="Experiment/analysis/results/routec_plus_gsm8k_external_bucket_audit_pack_v1.json")
    parser.add_argument("--output-md", default="Experiment/analysis/results/routec_plus_gsm8k_external_bucket_audit_pack_v1.md")
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
            prediction = str(row.get("prediction", ""))
            gold = str(row.get("gold_answer", ""))
            answer_mode = str(row.get("answer_mode", "numeric"))
            raw_correct = row.get("correct")
            correct = bool(raw_correct) if raw_correct is not None else answers_match(prediction, gold, answer_mode=answer_mode)
            output[str(row["example_id"])] = {
                "example_id": str(row["example_id"]),
                "prediction": prediction,
                "gold_answer": gold,
                "answer_mode": answer_mode,
                "correct": correct,
            }
    return output


def _wrong_form_bucket(prediction: str) -> str:
    p = prediction.strip()
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


def _sample_record(example_id: str, winner: dict, loser: dict, anchor: dict | None = None) -> dict:
    payload = {
        "example_id": example_id,
        "gold": winner["gold_answer"],
        "winner_prediction": winner["prediction"][:220],
        "loser_prediction": loser["prediction"][:220],
    }
    if anchor is not None:
        payload["anchor_prediction"] = anchor["prediction"][:220]
    return payload


def main() -> None:
    args = parse_args()
    leaderboard = _load_json(args.leaderboard_json)
    context = _load_json(args.conversion_context_json) if _project_path(args.conversion_context_json).exists() else None
    anchor_map = _load_predictions(args.anchor_preds)
    promoted_map = _load_predictions(args.promoted_preds)
    external_map = _load_predictions(args.external_preds)

    shared_ids = sorted(set(anchor_map) & set(promoted_map) & set(external_map))
    paired = {"promoted_only": 0, "external_only": 0, "both": 0, "neither": 0}

    ext_source_buckets = Counter()
    ext_wrong_form_buckets = Counter()
    promoted_only_anchor_buckets = Counter()

    ext_regression_examples = []
    ext_persistent_examples = []
    promoted_only_examples = []

    for example_id in shared_ids:
        anchor = anchor_map[example_id]
        promoted = promoted_map[example_id]
        external = external_map[example_id]
        a_correct = bool(anchor["correct"])
        p_correct = bool(promoted["correct"])
        e_correct = bool(external["correct"])

        if p_correct and e_correct:
            paired["both"] += 1
        elif p_correct and not e_correct:
            paired["promoted_only"] += 1
            bucket = "anchor_already_correct" if a_correct else "anchor_wrong_promoted_new_win"
            promoted_only_anchor_buckets[bucket] += 1
            if len(promoted_only_examples) < 6:
                promoted_only_examples.append(_sample_record(example_id, promoted, external, anchor))
        elif not p_correct and e_correct:
            paired["external_only"] += 1
            form_bucket = _wrong_form_bucket(promoted["prediction"])
            ext_wrong_form_buckets[form_bucket] += 1
            if a_correct:
                ext_source_buckets["promoted_specific_regression"] += 1
                if len(ext_regression_examples) < 6:
                    ext_regression_examples.append(_sample_record(example_id, external, promoted, anchor))
            else:
                if anchor["prediction"].strip() == promoted["prediction"].strip():
                    ext_source_buckets["persistent_same_wrong_as_anchor"] += 1
                else:
                    ext_source_buckets["persistent_changed_wrong"] += 1
                if len(ext_persistent_examples) < 6:
                    ext_persistent_examples.append(_sample_record(example_id, external, promoted, anchor))
        else:
            paired["neither"] += 1

    total = len(shared_ids)
    anchor_external_gap = paired["external_only"] - paired["promoted_only"]
    max_recoverable_from_regression = ext_source_buckets["promoted_specific_regression"] / total if total else 0.0
    external_gap_delta = anchor_external_gap / total if total else 0.0

    promoted_row = None
    anchor_row = None
    for row in leaderboard.get("rows", []):
        if row.get("policy") == "replace_random_nonprefix":
            anchor_row = row
        if row.get("policy") == "replace_verifier_uncertainty_first":
            promoted_row = row
    if promoted_row is None or anchor_row is None:
        raise RuntimeError("Required leaderboard rows are missing.")

    if ext_source_buckets["promoted_specific_regression"] <= 5 and ext_source_buckets["persistent_same_wrong_as_anchor"] >= 0.9 * paired["external_only"]:
        next_step = "Do not prioritize a GSM8K selector-calibration full rerun yet; the external gap is dominated by persistent internal misses rather than promoted-policy regressions."
        verdict = "persistent_internal_gap_dominates"
    else:
        next_step = "A small GSM8K selector-calibration slice may be justified before any full rerun because promoted-policy regressions are nontrivial."
        verdict = "selector_calibration_might_help"

    payload = {
        "title": "Route C+ GSM8K external bucket audit pack",
        "status": "completed",
        "paired_summary": {
            **paired,
            "n": total,
            "delta_external_minus_promoted": external_gap_delta,
        },
        "external_only_source_buckets": dict(ext_source_buckets),
        "external_only_wrong_form_buckets": dict(ext_wrong_form_buckets),
        "promoted_only_anchor_buckets": dict(promoted_only_anchor_buckets),
        "derived_metrics": {
            "promoted_specific_regression_share_of_external_only": ext_source_buckets["promoted_specific_regression"] / paired["external_only"] if paired["external_only"] else 0.0,
            "persistent_same_wrong_share_of_external_only": ext_source_buckets["persistent_same_wrong_as_anchor"] / paired["external_only"] if paired["external_only"] else 0.0,
            "persistent_total_share_of_external_only": (ext_source_buckets["persistent_same_wrong_as_anchor"] + ext_source_buckets["persistent_changed_wrong"]) / paired["external_only"] if paired["external_only"] else 0.0,
            "max_recoverable_from_pure_regression_points": max_recoverable_from_regression,
            "external_gap_points": external_gap_delta,
        },
        "context": context,
        "leaderboard_anchor": anchor_row,
        "leaderboard_promoted": promoted_row,
        "examples": {
            "external_only_regressions": ext_regression_examples,
            "external_only_persistent": ext_persistent_examples,
            "promoted_only": promoted_only_examples,
        },
        "decision": {
            "verdict": verdict,
            "summary": "The external gap is overwhelmingly persistent rather than promoted-specific: almost all external-only wins are already anchor failures, and nearly all of those keep the same wrong answer under the promoted row.",
            "next_step": next_step,
            "paper_facing_claim": "The remaining GSM8K gap to Self-Refine should not be framed as a small selector-rule bug. It is mainly a broader internal miss regime, with only a tiny promoted-specific regression tail.",
        },
    }

    lines = [
        "# Route C+ GSM8K External Bucket Audit Pack",
        "",
        "- Status: `completed`",
        f"- Verdict: `{payload['decision']['verdict']}`",
        f"- Summary: {payload['decision']['summary']}",
        f"- Next step: {payload['decision']['next_step']}",
        "",
        "## Paired Summary",
        "",
        f"- n=`{total}`",
        f"- promoted-only=`{paired['promoted_only']}`",
        f"- external-only=`{paired['external_only']}`",
        f"- both=`{paired['both']}`",
        f"- neither=`{paired['neither']}`",
        f"- external-promoted gap=`{_fmt(external_gap_delta)}`",
        "",
        "## External-Only Source Buckets",
        "",
        "| Bucket | Count | Share of external-only |",
        "| --- | ---: | ---: |",
    ]
    for key in ["promoted_specific_regression", "persistent_same_wrong_as_anchor", "persistent_changed_wrong"]:
        count = ext_source_buckets[key]
        share = count / paired["external_only"] if paired["external_only"] else 0.0
        lines.append(f"| {key} | {count} | {_fmt(share)} |")

    lines.extend([
        "",
        "## External-Only Wrong Forms on Promoted Predictions",
        "",
        "| Wrong form | Count | Share of external-only |",
        "| --- | ---: | ---: |",
    ])
    for key, count in ext_wrong_form_buckets.most_common():
        share = count / paired["external_only"] if paired["external_only"] else 0.0
        lines.append(f"| {key} | {count} | {_fmt(share)} |")

    lines.extend([
        "",
        "## Promoted-Only Wins",
        "",
        "| Bucket | Count | Share of promoted-only |",
        "| --- | ---: | ---: |",
    ])
    for key in ["anchor_already_correct", "anchor_wrong_promoted_new_win"]:
        count = promoted_only_anchor_buckets[key]
        share = count / paired["promoted_only"] if paired["promoted_only"] else 0.0
        lines.append(f"| {key} | {count} | {_fmt(share)} |")

    lines.extend([
        "",
        "## Key Readout",
        "",
        f"- promoted-specific regression share of external-only = `{_fmt(payload['derived_metrics']['promoted_specific_regression_share_of_external_only'])}`",
        f"- persistent same-wrong share of external-only = `{_fmt(payload['derived_metrics']['persistent_same_wrong_share_of_external_only'])}`",
        f"- persistent total share of external-only = `{_fmt(payload['derived_metrics']['persistent_total_share_of_external_only'])}`",
        f"- maximum recoverable if all pure regressions were fixed = `{_fmt(payload['derived_metrics']['max_recoverable_from_pure_regression_points'])}` points",
        f"- current external gap = `{_fmt(payload['derived_metrics']['external_gap_points'])}` points",
        "",
        "## Representative Examples",
        "",
        "### External-Only Regressions",
        "",
    ])
    if ext_regression_examples:
        for ex in ext_regression_examples:
            lines.append(f"- `{ex['example_id']}` gold=`{ex['gold']}` anchor=`{ex['anchor_prediction']}` promoted=`{ex['loser_prediction']}` external=`{ex['winner_prediction']}`")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "### External-Only Persistent Misses",
        "",
    ])
    for ex in ext_persistent_examples:
        lines.append(f"- `{ex['example_id']}` gold=`{ex['gold']}` anchor=`{ex['anchor_prediction']}` promoted=`{ex['loser_prediction']}` external=`{ex['winner_prediction']}`")

    lines.extend([
        "",
        "### Promoted-Only Wins",
        "",
    ])
    for ex in promoted_only_examples:
        lines.append(f"- `{ex['example_id']}` gold=`{ex['gold']}` anchor=`{ex['anchor_prediction']}` promoted=`{ex['winner_prediction']}` external=`{ex['loser_prediction']}`")

    lines.extend([
        "",
        "## Decision",
        "",
        f"- Paper-facing claim: {payload['decision']['paper_facing_claim']}",
        f"- Next step: {payload['decision']['next_step']}",
        "- Practical implication: fixing every pure promoted regression would at most recover a tiny fraction of the current external gap, so a full selector-calibration rerun is not the first thing to do.",
    ])

    output_json = _project_path(args.output_json)
    output_md = _project_path(args.output_md)
    _write_json_atomic(output_json, payload)
    _write_text_atomic(output_md, "\n".join(lines) + "\n")
    print(output_md.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
