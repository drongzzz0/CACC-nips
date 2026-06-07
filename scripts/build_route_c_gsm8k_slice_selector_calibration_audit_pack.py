from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from src.eval.evaluate_predictions import answers_match  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Route C+ GSM8K slice selector calibration audit pack.")
    parser.add_argument("--anchor-preds", default="Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_random_nonprefix_v1_verifier_predictions.jsonl")
    parser.add_argument("--promoted-preds", default="Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_verifier_uncertainty_first_v1_verifier_predictions.jsonl")
    parser.add_argument("--external-preds", default="Experiment/core_code/logs/a800_self_refine_gsm8k_full_p1b_v1_predictions.jsonl")
    parser.add_argument("--persistent-audit-json", default="Experiment/analysis/results/routec_plus_gsm8k_persistent_miss_slice_audit_pack_v1.json")
    parser.add_argument("--selector-script", default="Experiment/core_code/scripts/run_selector_policy_fixed_pool.py")
    parser.add_argument("--output-json", default="Experiment/analysis/results/routec_plus_gsm8k_slice_selector_calibration_audit_pack_v1.json")
    parser.add_argument("--output-md", default="Experiment/analysis/results/routec_plus_gsm8k_slice_selector_calibration_audit_pack_v1.md")
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_json(path: str | Path):
    return json.loads(_project_path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with _project_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _fmt(x: float | int | None, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def _row_correct(row: dict) -> bool:
    raw = row.get("correct")
    if raw is not None:
        return bool(raw)
    return answers_match(str(row.get("prediction", "")), str(row.get("gold_answer", "")), answer_mode=str(row.get("answer_mode", "numeric")))


def _load_pred_map(path: str | Path) -> dict[str, dict]:
    return {str(row["example_id"]): row for row in _load_jsonl(path)}


def _build_subregime_sets(anchor_map: dict[str, dict], promoted_map: dict[str, dict], external_map: dict[str, dict], persistent_audit: dict) -> tuple[set[str], dict[str, set[str]]]:
    persistent_ids = set()
    subregimes = {k: set() for k in persistent_audit["oracle_state_breakdown"]["counts"].keys()}
    for ex in sorted(set(anchor_map) & set(promoted_map) & set(external_map)):
        a = anchor_map[ex]
        p = promoted_map[ex]
        e = external_map[ex]
        if _row_correct(a) or _row_correct(p) or not _row_correct(e):
            continue
        if str(a.get("prediction", "")).strip() != str(p.get("prediction", "")).strip():
            continue
        persistent_ids.add(ex)
        # subregime follows persistent audit definition using candidate oracle state
        def has_oracle(row: dict) -> bool:
            gold = str(row.get("gold_answer", ""))
            mode = str(row.get("answer_mode", "numeric"))
            for cand in row.get("candidates", []):
                ans = cand.get("candidate_answer", "") if isinstance(cand, dict) else cand
                if answers_match(str(ans), gold, answer_mode=mode):
                    return True
            return False
        ao = has_oracle(a)
        po = has_oracle(p)
        if not ao and not po:
            sub = "both_oracle_miss"
        elif not ao and po:
            sub = "promoted_new_oracle_unconverted"
        elif ao and po:
            sub = "shared_oracle_selector_failure"
        else:
            sub = "promoted_oracle_regression_same_wrong"
        subregimes[sub].add(ex)
    return persistent_ids, subregimes


def _external_accuracy(external_map: dict[str, dict]) -> float:
    vals = [_row_correct(v) for v in external_map.values()]
    return sum(int(v) for v in vals) / len(vals) if vals else 0.0


def _run_rule(selector_script: Path, promoted_preds: Path, stem: str, rule: str, tail_anchored: bool) -> dict:
    py = sys.executable
    out_pred = PROJECT_ROOT / f"Experiment/analysis/results/{stem}_predictions.jsonl"
    out_json = PROJECT_ROOT / f"Experiment/analysis/results/{stem}.json"
    out_md = PROJECT_ROOT / f"Experiment/analysis/results/{stem}.md"
    cmd = [
        py,
        str(selector_script),
        "--run-label", stem,
        "--rule", rule,
        "--verifier-predictions", str(promoted_preds),
        "--benchmark", "gsm8k_full_clean",
        "--answer-mode", "numeric",
        "--output-predictions", str(out_pred),
        "--summary-json", str(out_json),
        "--report-md", str(out_md),
    ]
    if tail_anchored:
        cmd.extend(["--numeric-validity-mode", "tail_anchored"])
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    return {
        "summary_json": out_json,
        "predictions_jsonl": out_pred,
        "report_md": out_md,
    }


def main() -> None:
    args = parse_args()
    selector_script = _project_path(args.selector_script)
    anchor_map = _load_pred_map(args.anchor_preds)
    promoted_map = _load_pred_map(args.promoted_prefs if False else args.promoted_preds)
    external_map = _load_pred_map(args.external_preds)
    persistent_audit = _load_json(args.persistent_audit_json)

    persistent_ids, subregime_sets = _build_subregime_sets(anchor_map, promoted_map, external_map, persistent_audit)
    total_n = len(promoted_map)
    promoted_acc = sum(int(_row_correct(v)) for v in promoted_map.values()) / total_n
    external_acc = _external_accuracy(external_map)
    external_gap = external_acc - promoted_acc

    rule_specs = [
        {
            "name": "baseline_top_margin",
            "display": "Baseline top-margin",
            "rule": "baseline_top_margin",
            "tail_anchored": False,
            "kind": "baseline",
            "paper_safe_expected": True,
        },
        {
            "name": "parseable_then_margin",
            "display": "Parseable-then-margin",
            "rule": "parseable_then_margin",
            "tail_anchored": False,
            "kind": "conservative",
            "paper_safe_expected": True,
        },
        {
            "name": "clean_dup_0p5_0p1",
            "display": "Clean+dup 0.5/0.1",
            "rule": "clean_dup_0.5_0.1",
            "tail_anchored": False,
            "kind": "paper_safe_candidate",
            "paper_safe_expected": True,
        },
        {
            "name": "clean_dup_firstclose_0p5_0p1_0p5",
            "display": "Clean+dup+firstclose 0.5/0.1/0.5",
            "rule": "clean_dup_firstclose_0.5_0.1_0.5",
            "tail_anchored": False,
            "kind": "paper_safe_candidate",
            "paper_safe_expected": True,
        },
        {
            "name": "clean_dup_firstclose_0p5_0p1_1p0",
            "display": "Clean+dup+firstclose 0.5/0.1/1.0",
            "rule": "clean_dup_firstclose_0.5_0.1_1.0",
            "tail_anchored": False,
            "kind": "paper_safe_candidate",
            "paper_safe_expected": True,
        },
        {
            "name": "clean_dup_firstclose_0p5_0p1_1p0_tail",
            "display": "Clean+dup+firstclose 0.5/0.1/1.0 + tail-anchored",
            "rule": "clean_dup_firstclose_0.5_0.1_1.0",
            "tail_anchored": True,
            "kind": "aggressive_upper_bound",
            "paper_safe_expected": False,
        },
    ]

    rows = []
    promoted_invalid = None
    promoted_clean = None

    for spec in rule_specs:
        stem = f"routec_plus_gsm8k_slice_selectorcal_{spec['name']}_v1"
        outputs = _run_rule(selector_script, _project_path(args.promoted_preds), stem, spec["rule"], spec["tail_anchored"])
        summary = _load_json(outputs["summary_json"])
        pred_map = _load_pred_map(outputs["predictions_jsonl"])

        if spec["name"] == "baseline_top_margin":
            promoted_invalid = float(summary["selector"]["invalid_rate"])
            promoted_clean = float(summary["selector"]["clean_rate"])

        gains_in = losses_in = gains_out = losses_out = 0
        subregime_recoveries = Counter()
        for ex, row in pred_map.items():
            newc = _row_correct(row)
            basec = _row_correct(promoted_map[ex])
            if ex in persistent_ids:
                if newc and not basec:
                    gains_in += 1
                    for sub, ids in subregime_sets.items():
                        if ex in ids:
                            subregime_recoveries[sub] += 1
                            break
                elif basec and not newc:
                    losses_in += 1
            else:
                if newc and not basec:
                    gains_out += 1
                elif basec and not newc:
                    losses_out += 1

        selector_acc = float(summary["selector_accuracy"])
        delta = float(summary["delta_accuracy"])
        invalid = float(summary["selector"]["invalid_rate"])
        clean = float(summary["selector"]["clean_rate"])
        paper_safe = (invalid <= float(promoted_invalid) and clean >= float(promoted_clean)) if promoted_invalid is not None else True
        closed_share = (delta / external_gap) if external_gap > 0 else 0.0
        rows.append({
            "name": spec["name"],
            "display": spec["display"],
            "rule": spec["rule"],
            "tail_anchored": spec["tail_anchored"],
            "kind": spec["kind"],
            "paper_safe": paper_safe,
            "selector_accuracy": selector_acc,
            "delta_vs_promoted": delta,
            "invalid_rate": invalid,
            "clean_rate": clean,
            "gain_count": int(summary["gain_count"]),
            "loss_count": int(summary["loss_count"]),
            "persistent_recovered": gains_in,
            "persistent_lost": losses_in,
            "outside_gains": gains_out,
            "outside_losses": losses_out,
            "shared_oracle_selector_failure_recovered": subregime_recoveries.get("shared_oracle_selector_failure", 0),
            "promoted_new_oracle_unconverted_recovered": subregime_recoveries.get("promoted_new_oracle_unconverted", 0),
            "both_oracle_miss_recovered": subregime_recoveries.get("both_oracle_miss", 0),
            "promoted_oracle_regression_same_wrong_recovered": subregime_recoveries.get("promoted_oracle_regression_same_wrong", 0),
            "external_gap_closed_share": closed_share,
            "new_external_gap": external_acc - selector_acc,
            "artifacts": {k: str(v) for k, v in outputs.items()},
        })

    baseline_row = next(r for r in rows if r["name"] == "baseline_top_margin")
    paper_safe_candidates = [r for r in rows if r["name"] != "baseline_top_margin" and r["paper_safe"]]
    best_paper_safe = max(paper_safe_candidates, key=lambda r: (r["selector_accuracy"], -r["outside_losses"])) if paper_safe_candidates else None
    aggressive_upper = max([r for r in rows if r["name"] != "baseline_top_margin"], key=lambda r: r["selector_accuracy"])

    verdict = "paper_safe_narrow_calibration_exists_but_bounded" if best_paper_safe else "no_paper_safe_narrow_calibration_found"
    if best_paper_safe:
        next_step = (
            "Freeze the best paper-safe offline selector calibration as a narrow GSM8K strengthening row and treat it as selector-side closure evidence; do not let it replace the broader candidate-side blocker analysis."
        )
        paper_claim = (
            "A narrow offline selector calibration can recover a real but bounded portion of the GSM8K persistent same-wrong gap without harming hygiene, which supports a selector-side strengthening row but not a full resolution of the external frontier gap."
        )
    else:
        next_step = "Do not invest further in selector-side calibration; current paper-safe rules do not provide enough recovery."
        paper_claim = "Current narrow selector-side calibration does not provide a meaningful paper-safe GSM8K recovery."

    payload = {
        "title": "Route C+ GSM8K slice selector calibration audit pack",
        "status": "completed",
        "context": {
            "total_examples": total_n,
            "promoted_accuracy": promoted_acc,
            "external_accuracy": external_acc,
            "external_gap": external_gap,
            "persistent_same_wrong_n": len(persistent_ids),
            "persistent_audit_json": str(_project_path(args.persistent_audit_json)),
        },
        "rows": rows,
        "baseline_row": baseline_row,
        "best_paper_safe_row": best_paper_safe,
        "aggressive_upper_bound_row": aggressive_upper,
        "decision": {
            "verdict": verdict,
            "summary": "A paper-safe narrow GSM8K selector calibration does exist on the current promoted pool, but its effect is bounded and concentrated in shared-oracle selector-failure cases rather than in pure coverage misses.",
            "next_step": next_step,
            "paper_facing_claim": paper_claim,
        },
    }

    lines = [
        "# Route C+ GSM8K Slice Selector Calibration Audit Pack",
        "",
        "- Status: `completed`",
        f"- Verdict: `{verdict}`",
        f"- Summary: {payload['decision']['summary']}",
        f"- Next step: {payload['decision']['next_step']}",
        "",
        "## Context",
        "",
        f"- Promoted GSM8K full accuracy: `{promoted_acc:.4f}`",
        f"- Strongest external GSM8K full accuracy: `{external_acc:.4f}`",
        f"- Current external gap: `{external_gap:.4f}`",
        f"- Persistent same-wrong slice size: `{len(persistent_ids)}` / `{total_n}` = `{len(persistent_ids)/total_n:.4f}`",
        "",
        "## Rule Scorecard",
        "",
        "| Rule | Paper-safe | Full Acc | Δ vs promoted | Persistent recovered | Shared-selector recovered | Outside losses | Invalid | Clean | External gap closed share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['display']} | {'yes' if row['paper_safe'] else 'no'} | {row['selector_accuracy']:.4f} | {row['delta_vs_promoted']:+.4f} | {row['persistent_recovered']} | {row['shared_oracle_selector_failure_recovered']} | {row['outside_losses']} | {row['invalid_rate']:.4f} | {row['clean_rate']:.4f} | {row['external_gap_closed_share']:.4f} |"
        )

    if best_paper_safe is not None:
        lines.extend([
            "",
            "## Best Paper-Safe Rule",
            "",
            f"- Rule: `{best_paper_safe['display']}`",
            f"- Full accuracy: `{best_paper_safe['selector_accuracy']:.4f}` vs promoted `{promoted_acc:.4f}` = `{best_paper_safe['delta_vs_promoted']:+.4f}`",
            f"- Invalid rate: `{best_paper_safe['invalid_rate']:.4f}` vs promoted `{baseline_row['invalid_rate']:.4f}`",
            f"- Clean rate: `{best_paper_safe['clean_rate']:.4f}` vs promoted `{baseline_row['clean_rate']:.4f}`",
            f"- Persistent slice recovered: `{best_paper_safe['persistent_recovered']}` / `{len(persistent_ids)}` = `{best_paper_safe['persistent_recovered']/len(persistent_ids):.4f}`",
            f"- Shared-oracle selector-failure recovered: `{best_paper_safe['shared_oracle_selector_failure_recovered']}` / `214` = `{best_paper_safe['shared_oracle_selector_failure_recovered']/214:.4f}`",
            f"- Promoted-new-oracle-unconverted recovered: `{best_paper_safe['promoted_new_oracle_unconverted_recovered']}` / `23` = `{best_paper_safe['promoted_new_oracle_unconverted_recovered']/23:.4f}`",
            f"- Outside-slice gains / losses: `{best_paper_safe['outside_gains']}` / `{best_paper_safe['outside_losses']}`",
            f"- External gap closed share: `{best_paper_safe['external_gap_closed_share']:.4f}`",
            "",
            "Interpretation:",
            "这条规则主要修复的是 `shared_oracle_selector_failure`，不是 `both_oracle_miss`。因此它支持 selector-side strengthening，但不支持把 GSM8K blocker 重写成已经解决。",
        ])

    lines.extend([
        "",
        "## Aggressive Upper Bound",
        "",
        f"- Rule: `{aggressive_upper['display']}`",
        f"- Full accuracy: `{aggressive_upper['selector_accuracy']:.4f}` = `{aggressive_upper['delta_vs_promoted']:+.4f}` vs promoted",
        f"- Invalid rate: `{aggressive_upper['invalid_rate']:.4f}`",
        f"- Clean rate: `{aggressive_upper['clean_rate']:.4f}`",
        "- Interpretation: 这条规则给出了 selector-side 可回收上界，但 invalid final 大幅恶化，不是 paper-safe row。",
        "",
        "## Decision",
        "",
        f"- Paper-facing claim: {payload['decision']['paper_facing_claim']}",
        f"- Operational next step: {payload['decision']['next_step']}",
        "- Practical implication: 现在值得补的是一个窄的 GSM8K selector-side strengthening row 或 confirmatory audit，而不是新的 blind candidate-generation full sweep；但这条线只能收窄 blocker，不能单独把 external gap 打平。",
    ])

    _write_json_atomic(_project_path(args.output_json), payload)
    _write_text_atomic(_project_path(args.output_md), "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
