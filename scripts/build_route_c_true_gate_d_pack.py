from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from src.eval.evaluate_predictions import answers_match  # type: ignore


REPORT_ACCURACY_RE = re.compile(r"exact-match accuracy:\s*([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Route C+ true-strong Gate D stability/audit pack.")
    parser.add_argument(
        "--gsm8k-leaderboard-json",
        default="Experiment/analysis/results/routec_plus_policy_gsm8k_full_clean_leaderboard_v1.json",
    )
    parser.add_argument(
        "--gsm8k-self-refine-json",
        default="Experiment/analysis/results/experiment_12_gsm8k_full_self_refine_p1b_result_v1.json",
    )
    parser.add_argument(
        "--gatea-compmath-random-json",
        default="Experiment/analysis/results/routec_plus_true_gateA_competition_math_numeric_random_nonprefix_v1.json",
    )
    parser.add_argument(
        "--gatea-compmath-main-json",
        default="Experiment/analysis/results/routec_plus_true_gateA_competition_math_numeric_verifier_uncertainty_first_v1.json",
    )
    parser.add_argument(
        "--gatea-mmlu-random-json",
        default="Experiment/analysis/results/routec_plus_true_gateA_mmlu_pro_test_random_nonprefix_v1.json",
    )
    parser.add_argument(
        "--gatea-mmlu-main-json",
        default="Experiment/analysis/results/routec_plus_true_gateA_mmlu_pro_test_verifier_uncertainty_first_v1.json",
    )
    parser.add_argument(
        "--gatec-compmath-external-json",
        default="Experiment/analysis/results/routec_plus_true_gateC_competition_math_numeric_external_comparison_v1.json",
    )
    parser.add_argument(
        "--gatec-mmlu-external-json",
        default="Experiment/analysis/results/routec_plus_true_gateC_mmlu_pro_test_external_comparison_v1.json",
    )
    parser.add_argument(
        "--budget-json",
        default="Experiment/analysis/results/ser_generate_then_rerank_qwen3_17b_full_candidate_budget_v1.json",
    )
    parser.add_argument(
        "--gateb-json",
        default="Experiment/analysis/results/routec_plus_true_gateB_compatibility_pack_v1.json",
    )
    parser.add_argument(
        "--paired-audit-json",
        default="Experiment/analysis/results/routec_plus_policy_v2_vs_v3margin_paired_delta_audit_2026_04_12.json",
    )
    parser.add_argument(
        "--output-json",
        default="Experiment/analysis/results/routec_plus_true_gateD_stability_audit_pack_v1.json",
    )
    parser.add_argument(
        "--output-md",
        default="Experiment/analysis/results/routec_plus_true_gateD_stability_audit_pack_v1.md",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _exists(path: str | Path) -> bool:
    return _project_path(path).exists()


def _read_json(path: str | Path) -> dict | list:
    return json.loads(_project_path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with _project_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
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


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return statistics.stdev(values)


def _accuracy_from_report(path: str | Path) -> float:
    text = _project_path(path).read_text(encoding="utf-8")
    match = REPORT_ACCURACY_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse exact-match accuracy from {path}")
    return float(match.group(1))


def _slug_to_policy(policy: str) -> str:
    if policy.startswith("replace_"):
        return policy
    return f"replace_{policy}"


def _policy_row(leaderboard: dict, policy_name: str) -> dict | None:
    target = _slug_to_policy(policy_name)
    for row in leaderboard.get("rows", []):
        if row.get("policy") == target:
            return row
    return None


def _prediction_map(path: str | Path) -> dict[str, dict]:
    records = _read_jsonl(path)
    output: dict[str, dict] = {}
    for record in records:
        prediction = str(record.get("prediction", ""))
        gold = str(record.get("gold_answer", ""))
        answer_mode = str(record.get("answer_mode", "numeric"))
        correct = bool(record.get("correct")) if "correct" in record else answers_match(prediction, gold, answer_mode=answer_mode)
        output[str(record["example_id"])] = {
            "example_id": str(record["example_id"]),
            "prediction": prediction,
            "gold_answer": gold,
            "answer_mode": answer_mode,
            "correct": correct,
        }
    return output


def _exact_mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict[str, float | int]:
    a_only = 0
    b_only = 0
    both = 0
    neither = 0
    for row_a, row_b in zip(rows_a, rows_b):
        a_correct = bool(row_a["correct"])
        b_correct = bool(row_b["correct"])
        if a_correct and b_correct:
            both += 1
        elif a_correct and not b_correct:
            a_only += 1
        elif not a_correct and b_correct:
            b_only += 1
        else:
            neither += 1
    disagreements = a_only + b_only
    if disagreements == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(disagreements, k) for k in range(min(a_only, b_only) + 1)) / (2**disagreements)
        p_value = min(1.0, 2 * tail)
    return {
        "a_only": a_only,
        "b_only": b_only,
        "both": both,
        "neither": neither,
        "p": p_value,
    }


def _bootstrap_ci(rows_a: list[dict], rows_b: list[dict], samples: int, seed: int) -> dict[str, float]:
    if not rows_a or not rows_b:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    n = len(rows_a)
    deltas: list[float] = []
    for _ in range(samples):
        total = 0
        for _ in range(n):
            idx = rng.randrange(n)
            total += int(bool(rows_b[idx]["correct"])) - int(bool(rows_a[idx]["correct"]))
        deltas.append(total / n)
    deltas.sort()
    low_idx = max(0, int(0.025 * samples) - 1)
    high_idx = min(samples - 1, int(0.975 * samples) - 1)
    return {"low": deltas[low_idx], "high": deltas[high_idx]}


def _paired_stats(label: str, benchmark: str, path_a: str | Path, path_b: str | Path, *, samples: int, seed: int) -> dict:
    abs_a = _project_path(path_a)
    abs_b = _project_path(path_b)
    if not abs_a.exists() or not abs_b.exists():
        missing = []
        if not abs_a.exists():
            missing.append(str(path_a))
        if not abs_b.exists():
            missing.append(str(path_b))
        return {
            "label": label,
            "benchmark": benchmark,
            "status": "missing_artifacts",
            "missing": missing,
        }

    map_a = _prediction_map(path_a)
    map_b = _prediction_map(path_b)
    shared_ids = sorted(set(map_a) & set(map_b))
    rows_a = [map_a[example_id] for example_id in shared_ids]
    rows_b = [map_b[example_id] for example_id in shared_ids]
    acc_a = _mean([float(bool(row["correct"])) for row in rows_a]) or 0.0
    acc_b = _mean([float(bool(row["correct"])) for row in rows_b]) or 0.0
    ci = _bootstrap_ci(rows_a, rows_b, samples=samples, seed=seed)
    return {
        "label": label,
        "benchmark": benchmark,
        "status": "ready",
        "n": len(shared_ids),
        "a_path": str(path_a),
        "b_path": str(path_b),
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "delta_b_minus_a": acc_b - acc_a,
        "ci95": ci,
        "mcnemar": _exact_mcnemar(rows_a, rows_b),
    }


def _main_row_from_summary(benchmark: str, baseline_name: str, main_name: str, baseline: dict | None, main: dict | None) -> dict:
    if baseline is None or main is None:
        missing = []
        if baseline is None:
            missing.append(baseline_name)
        if main is None:
            missing.append(main_name)
        return {
            "benchmark": benchmark,
            "status": "waiting",
            "missing": missing,
        }
    baseline_acc = baseline.get("verifier_accuracy")
    main_acc = main.get("verifier_accuracy")
    return {
        "benchmark": benchmark,
        "status": "ready",
        "baseline": baseline_name,
        "main": main_name,
        "n": baseline.get("total_examples") or baseline.get("num_examples") or main.get("total_examples") or main.get("num_examples"),
        "baseline_verifier": baseline_acc,
        "main_verifier": main_acc,
        "delta_verifier": None if baseline_acc is None or main_acc is None else main_acc - baseline_acc,
        "baseline_oracle": baseline.get("oracle_coverage"),
        "main_oracle": main.get("oracle_coverage"),
        "baseline_parseable": baseline.get("selected_parseable") or baseline.get("selected_prediction_parseable_rate"),
        "main_parseable": main.get("selected_parseable") or main.get("selected_prediction_parseable_rate"),
        "baseline_artifact": baseline_name,
        "main_artifact": main_name,
    }


def _seed_summary() -> dict:
    groups = {
        "gsm8k_clean128_cacc": [
            "Experiment/analysis/results/routec_b1_gsm8k_clean128_cacc_seed7_v1.json",
            "Experiment/analysis/results/routec_b1_gsm8k_clean128_cacc_seed11_v1.json",
            "Experiment/analysis/results/routec_b1_gsm8k_clean128_cacc_seed19_v1.json",
        ],
        "gsm8k_clean128_cacc_sp": [
            "Experiment/analysis/results/routec_b1_gsm8k_clean128_cacc_sp_seed7_v1.json",
            "Experiment/analysis/results/routec_b1_gsm8k_clean128_cacc_sp_seed11_v1.json",
            "Experiment/analysis/results/routec_b1_gsm8k_clean128_cacc_sp_seed19_v1.json",
        ],
        "mmlu_pro_smoke128_cacc": [
            "Experiment/analysis/results/routec_b1_mmlu_pro_smoke128_cacc_seed7_v1.json",
            "Experiment/analysis/results/routec_b1_mmlu_pro_smoke128_cacc_seed11_v1.json",
            "Experiment/analysis/results/routec_b1_mmlu_pro_smoke128_cacc_seed19_v1.json",
        ],
        "mmlu_pro_smoke128_cacc_sp": [
            "Experiment/analysis/results/routec_b1_mmlu_pro_smoke128_cacc_sp_seed7_v1.json",
            "Experiment/analysis/results/routec_b1_mmlu_pro_smoke128_cacc_sp_seed11_v1.json",
            "Experiment/analysis/results/routec_b1_mmlu_pro_smoke128_cacc_sp_seed19_v1.json",
        ],
        "second_family_weakest_slice": [
            "Experiment/analysis/results/routec_b1_secondfamily_weakest_slice_seed7_v1.json",
            "Experiment/analysis/results/routec_b1_secondfamily_weakest_slice_seed11_v1.json",
            "Experiment/analysis/results/routec_b1_secondfamily_weakest_slice_seed19_v1.json",
        ],
    }

    summaries = []
    for label, paths in groups.items():
        if not paths:
            summaries.append({
                "label": label,
                "status": "missing",
                "missing": ["no configured seed runs yet"],
            })
            continue
        missing_paths = [path for path in paths if not _exists(path)]
        if missing_paths:
            summaries.append({
                "label": label,
                "status": "missing",
                "missing": missing_paths,
            })
            continue
        payloads = [_read_json(path) for path in paths]
        verifier = [float(item["verifier_accuracy"]) for item in payloads]
        oracle = [float(item["oracle_coverage"]) for item in payloads]
        base = [float(item["base_accuracy"]) for item in payloads]
        summaries.append({
            "label": label,
            "status": "ready",
            "num_seeds": len(paths),
            "paths": paths,
            "mean_verifier": _mean(verifier),
            "std_verifier": _std(verifier),
            "min_verifier": min(verifier),
            "max_verifier": max(verifier),
            "spread_verifier": max(verifier) - min(verifier),
            "mean_oracle": _mean(oracle),
            "std_oracle": _std(oracle),
            "mean_base": _mean(base),
            "std_base": _std(base),
        })
    return {
        "rows": summaries,
        "coverage_status": {
            "gsm8k_subset_multiseed": any(row.get("label") == "gsm8k_clean128_cacc" and row.get("status") == "ready" for row in summaries),
            "mmlu_subset_multiseed": any(row.get("label") == "mmlu_pro_smoke128_cacc" and row.get("status") == "ready" for row in summaries),
            "second_family_risky_slice_multiseed": any(row.get("label") == "second_family_weakest_slice" and row.get("status") == "ready" for row in summaries),
        },
    }


def _budget_summary(path: str | Path) -> dict:
    if not _exists(path):
        return {"status": "missing", "missing": [str(path)]}
    payload = _read_json(path)
    curve = payload.get("top_k_curve", [])
    rows = []
    for row in curve:
        oracle = row.get("oracle_accuracy")
        verifier = row.get("verifier_accuracy")
        rows.append({
            "k": row.get("k"),
            "oracle_accuracy": oracle,
            "verifier_accuracy": verifier,
            "verifier_given_oracle": (verifier / oracle) if oracle not in (None, 0) and verifier is not None else None,
            "invalid_final": None,
            "selected_parseable": None,
        })
    observed_k = [int(row["k"]) for row in rows if row.get("k") is not None]
    return {
        "status": "ready",
        "path": str(path),
        "rows": rows,
        "observed_k": observed_k,
        "required_k": [4, 8, 16, 32],
        "missing_required_k": [k for k in [4, 8, 16, 32] if k not in observed_k],
        "budget_implications": payload.get("budget_implications"),
    }


def _external_summary(path: str | Path, benchmark: str) -> dict:
    if not _exists(path):
        return {"benchmark": benchmark, "status": "waiting", "missing": [str(path)]}
    payload = _read_json(path)
    rows = payload.get("rows", [])
    best_external = max((row for row in rows if row.get("kind") == "external"), key=lambda row: row.get("accuracy", -1), default=None)
    return {
        "benchmark": benchmark,
        "status": "ready",
        "internal_first_accuracy": payload.get("internal_first_accuracy"),
        "internal_main_accuracy": payload.get("internal_main_accuracy"),
        "best_external": best_external,
        "rows": rows,
        "path": str(path),
    }


def _gsm8k_self_refine_summary(path: str | Path) -> dict:
    if not _exists(path):
        return {"status": "missing", "missing": [str(path)]}
    payload = _read_json(path)
    return {
        "status": "ready",
        "accuracy": payload.get("accuracy"),
        "num_examples": payload.get("num_examples"),
        "parseable": payload.get("selected_prediction_parseable_rate"),
        "invalid": payload.get("invalid_final_answer_rate"),
        "comparisons": payload.get("comparisons"),
        "artifacts": payload.get("artifacts"),
        "path": str(path),
    }


def _main_system_section(args: argparse.Namespace) -> dict:
    main_rows: list[dict] = []

    gsm8k_row = {"benchmark": "gsm8k_full_clean", "status": "waiting"}
    if _exists(args.gsm8k_leaderboard_json):
        leaderboard = _read_json(args.gsm8k_leaderboard_json)
        baseline = _policy_row(leaderboard, "random_nonprefix")
        main = _policy_row(leaderboard, "verifier_uncertainty_first") or _policy_row(leaderboard, leaderboard.get("best_policy_by_verifier", ""))
        if baseline and main:
            gsm8k_row = {
                "benchmark": "gsm8k_full_clean",
                "status": "ready",
                "baseline": baseline.get("policy"),
                "main": main.get("policy"),
                "n": baseline.get("num_examples"),
                "baseline_verifier": baseline.get("verifier_accuracy"),
                "main_verifier": main.get("verifier_accuracy"),
                "delta_verifier": main.get("verifier_accuracy") - baseline.get("verifier_accuracy"),
                "baseline_oracle": baseline.get("oracle_coverage"),
                "main_oracle": main.get("oracle_coverage"),
                "baseline_parseable": baseline.get("selected_parseable"),
                "main_parseable": main.get("selected_parseable"),
                "leaderboard_path": str(args.gsm8k_leaderboard_json),
            }
        else:
            gsm8k_row = {
                "benchmark": "gsm8k_full_clean",
                "status": "partial",
                "missing": ["replace_random_nonprefix row", "replace_verifier_uncertainty_first row"],
            }
    else:
        gsm8k_row = {"benchmark": "gsm8k_full_clean", "status": "missing", "missing": [str(args.gsm8k_leaderboard_json)]}
    main_rows.append(gsm8k_row)

    compmath_random = _read_json(args.gatea_compmath_random_json) if _exists(args.gatea_compmath_random_json) else None
    compmath_main = _read_json(args.gatea_compmath_main_json) if _exists(args.gatea_compmath_main_json) else None
    main_rows.append(_main_row_from_summary("competition_math_numeric", str(args.gatea_compmath_random_json), str(args.gatea_compmath_main_json), compmath_random, compmath_main))

    mmlu_random = _read_json(args.gatea_mmlu_random_json) if _exists(args.gatea_mmlu_random_json) else None
    mmlu_main = _read_json(args.gatea_mmlu_main_json) if _exists(args.gatea_mmlu_main_json) else None
    main_rows.append(_main_row_from_summary("mmlu_pro_test", str(args.gatea_mmlu_random_json), str(args.gatea_mmlu_main_json), mmlu_random, mmlu_main))
    return {"rows": main_rows}


def _paired_stats_section(args: argparse.Namespace) -> dict:
    comparisons = [
        _paired_stats(
            "gsm8k_full_clean: random-slot vs CACC-P*",
            "gsm8k_full_clean",
            "Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_random_nonprefix_v1_verifier_predictions.jsonl",
            "Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_verifier_uncertainty_first_v1_verifier_predictions.jsonl",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        _paired_stats(
            "gsm8k_full_clean: CACC-P* vs Self-Refine",
            "gsm8k_full_clean",
            "Experiment/core_code/logs/routec_plus_policy_gsm8k_full_clean_verifier_uncertainty_first_v1_verifier_predictions.jsonl",
            "Experiment/core_code/logs/a800_self_refine_gsm8k_full_p1b_v1_predictions.jsonl",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 1,
        ),
        _paired_stats(
            "competition_math_numeric: random-slot vs CACC-P*",
            "competition_math_numeric",
            "Experiment/core_code/logs/routec_plus_true_gateA_competition_math_numeric_random_nonprefix_v1_verifier_predictions.jsonl",
            "Experiment/core_code/logs/routec_plus_true_gateA_competition_math_numeric_verifier_uncertainty_first_v1_verifier_predictions.jsonl",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 2,
        ),
        _paired_stats(
            "competition_math_numeric: CACC-P* vs Self-Refine",
            "competition_math_numeric",
            "Experiment/core_code/logs/routec_plus_true_gateA_competition_math_numeric_verifier_uncertainty_first_v1_verifier_predictions.jsonl",
            "Experiment/core_code/logs/a800_self_refine_competition_math_numeric_test_transfer_b32_v2_predictions.jsonl",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 3,
        ),
        _paired_stats(
            "competition_math_numeric: CACC-P* vs PairRM best-of-N",
            "competition_math_numeric",
            "Experiment/core_code/logs/routec_plus_true_gateA_competition_math_numeric_verifier_uncertainty_first_v1_verifier_predictions.jsonl",
            "Experiment/core_code/logs/routec_plus_true_gateC_pairrm_competition_math_numeric_v1_predictions.jsonl",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 4,
        ),
        _paired_stats(
            "mmlu_pro_test: random-slot vs CACC-P*",
            "mmlu_pro_test",
            "Experiment/core_code/logs/routec_plus_true_gateA_mmlu_pro_test_random_nonprefix_v1_verifier_predictions.jsonl",
            "Experiment/core_code/logs/routec_plus_true_gateA_mmlu_pro_test_verifier_uncertainty_first_v1_verifier_predictions.jsonl",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 5,
        ),
        _paired_stats(
            "mmlu_pro_test: CACC-P* vs Self-Refine",
            "mmlu_pro_test",
            "Experiment/core_code/logs/routec_plus_true_gateA_mmlu_pro_test_verifier_uncertainty_first_v1_verifier_predictions.jsonl",
            "Experiment/core_code/logs/a800_self_refine_mmlu_pro_test_transfer_b32_v2_predictions.jsonl",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 6,
        ),
        _paired_stats(
            "mmlu_pro_test: CACC-P* vs PairRM best-of-N",
            "mmlu_pro_test",
            "Experiment/core_code/logs/routec_plus_true_gateA_mmlu_pro_test_verifier_uncertainty_first_v1_verifier_predictions.jsonl",
            "Experiment/core_code/logs/routec_plus_true_gateC_pairrm_mmlu_pro_test_v1_predictions.jsonl",
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + 7,
        ),
    ]
    ready = [row for row in comparisons if row.get("status") == "ready"]
    return {
        "rows": comparisons,
        "ready_count": len(ready),
        "target_count": len(comparisons),
    }


def _failure_audit_section(args: argparse.Namespace, gateb_payload: dict | None) -> dict:
    paired = _read_json(args.paired_audit_json) if _exists(args.paired_audit_json) else None
    rows = []
    if isinstance(paired, dict):
        for benchmark, payload in paired.items():
            counts = payload.get("counts", {})
            oracle_added = counts.get("oracle_added")
            converted = counts.get("oracle_added_and_converted_to_verifier_win")
            conversion = None
            if oracle_added not in (None, 0) and converted is not None:
                conversion = converted / oracle_added
            rows.append({
                "benchmark": benchmark,
                "n": payload.get("n"),
                "oracle_added": oracle_added,
                "oracle_lost": counts.get("oracle_lost"),
                "verifier_gain_total": counts.get("verifier_gain_total"),
                "verifier_loss_total": counts.get("verifier_loss_total"),
                "selected_parseable_improvement": counts.get("selected_parseable_improvement"),
                "selected_parseable_regression": counts.get("selected_parseable_regression"),
                "oracle_added_conversion": conversion,
            })
    return {
        "status": "ready" if rows else "partial",
        "paired_delta_audit_path": str(args.paired_audit_json),
        "rows": rows,
        "gateb_claim": None if gateb_payload is None else gateb_payload.get("claim"),
        "gateb_assessment": None if gateb_payload is None else gateb_payload.get("assessment"),
    }


def _scorecard(main_system: dict, external: dict, paired: dict, seed: dict, budget: dict, failure: dict) -> list[dict]:
    main_ready = sum(1 for row in main_system["rows"] if row.get("status") == "ready")
    external_ready = sum(1 for row in external["rows"] if row.get("status") == "ready")
    second_family_seed_ready = seed.get("coverage_status", {}).get("second_family_risky_slice_multiseed", False)
    budget_missing = budget.get("missing_required_k", [4, 8, 16, 32]) if budget.get("status") == "ready" else [4, 8, 16, 32]
    return [
        {
            "section": "main_system_confirmatory",
            "status": "ready" if main_ready == 3 else "partial",
            "detail": f"{main_ready}/3 主 confirmatory 行已就绪。",
        },
        {
            "section": "external_frontier",
            "status": "ready" if external_ready == 3 else "partial",
            "detail": f"{external_ready}/3 external frontier 行已就绪（GSM8K full self-refine + Gate C 两条）。",
        },
        {
            "section": "paired_statistics",
            "status": "ready" if paired.get("ready_count") == paired.get("target_count") else "partial",
            "detail": f"paired 主比较已完成 {paired.get('ready_count')}/{paired.get('target_count')}。",
        },
        {
            "section": "subset_multiseed",
            "status": "ready" if second_family_seed_ready else "partial",
            "detail": "GSM8K / MMLU subset 3-seed 已有，但 second-family weakest slice multiseed 仍缺。",
        },
        {
            "section": "budget_scaling",
            "status": "ready" if budget.get("status") == "ready" and not budget_missing else "partial",
            "detail": "已观测 budget 点：{}；缺失强稿目标点：{}。".format(
                budget.get("observed_k", []),
                budget_missing,
            ) if budget.get("status") == "ready" else "budget 曲线缺失。",
        },
        {
            "section": "failure_migration_audit",
            "status": "ready" if failure.get("rows") else "partial",
            "detail": "已有 v2→v3 paired delta audit 与 Gate B compatibility audit，但还不是 200-300 样本跨 benchmark 完整人工桶审计。",
        },
    ]


def _overall_status(scorecard: list[dict]) -> str:
    statuses = [row["status"] for row in scorecard]
    if all(status == "ready" for status in statuses):
        return "passed"
    if any(status == "ready" for status in statuses):
        return "partial"
    return "waiting"


def main() -> None:
    args = parse_args()

    gateb_payload = _read_json(args.gateb_json) if _exists(args.gateb_json) else None
    main_system = _main_system_section(args)
    external = {
        "rows": [
            {
                "benchmark": "gsm8k_full_clean",
                **_gsm8k_self_refine_summary(args.gsm8k_self_refine_json),
            },
            _external_summary(args.gatec_compmath_external_json, "competition_math_numeric"),
            _external_summary(args.gatec_mmlu_external_json, "mmlu_pro_test"),
        ]
    }
    paired = _paired_stats_section(args)
    seed = _seed_summary()
    budget = _budget_summary(args.budget_json)
    failure = _failure_audit_section(args, gateb_payload if isinstance(gateb_payload, dict) else None)
    scorecard = _scorecard(main_system, external, paired, seed, budget, failure)
    overall_status = _overall_status(scorecard)

    payload = {
        "gate": "D",
        "title": "Route C+ true-strong stability and audit pack",
        "status": overall_status,
        "target": "Upgrade Route C+ from conditional strong-paper evidence to reviewer-proof true strong-paper evidence.",
        "inputs": {
            "gsm8k_leaderboard_json": str(args.gsm8k_leaderboard_json),
            "gsm8k_self_refine_json": str(args.gsm8k_self_refine_json),
            "gatea_compmath_random_json": str(args.gatea_compmath_random_json),
            "gatea_compmath_main_json": str(args.gatea_compmath_main_json),
            "gatea_mmlu_random_json": str(args.gatea_mmlu_random_json),
            "gatea_mmlu_main_json": str(args.gatea_mmlu_main_json),
            "gatec_compmath_external_json": str(args.gatec_compmath_external_json),
            "gatec_mmlu_external_json": str(args.gatec_mmlu_external_json),
            "budget_json": str(args.budget_json),
            "gateb_json": str(args.gateb_json),
            "paired_audit_json": str(args.paired_audit_json),
        },
        "scorecard": scorecard,
        "main_system": main_system,
        "external_frontier": external,
        "paired_statistics": paired,
        "seed_stability": seed,
        "budget_scaling": budget,
        "failure_migration": failure,
        "compatibility_pack": gateb_payload,
        "assessment": {
            "reviewer_facing_verdict": (
                "Main-system, second-family, and audit evidence are now being consolidated into one Gate D pack, but true strong-paper closure still requires the pending Gate A/Gate C rows and broader stability coverage beyond the current subset-seed/budget footprint."
                if overall_status != "passed"
                else "The current pack now satisfies the reviewer-facing stability and audit requirements for a true strong-paper Route C+ submission."
            ),
            "largest_remaining_gaps": [
                row["detail"] for row in scorecard if row["status"] != "ready"
            ],
            "paper_safe_claim": "CACC-P* is strongest when framed as bounded, regime-aware candidate construction whose second-family transfer is recoverable under explicit alignment rather than free.",
        },
    }

    output_json = _project_path(args.output_json)
    output_md = _project_path(args.output_md)
    _write_json_atomic(output_json, payload)

    lines = [
        "# Route C+ True-Strong Gate D Stability and Audit Pack",
        "",
        f"- Status: `{overall_status}`",
        f"- Target: `{payload['target']}`",
        f"- Reviewer-facing verdict: {payload['assessment']['reviewer_facing_verdict']}",
        "",
        "## Gate D Scorecard",
        "",
        "| section | status | detail |",
        "| --- | --- | --- |",
    ]
    for row in scorecard:
        lines.append(f"| {row['section']} | `{row['status']}` | {row['detail']} |")

    lines.extend([
        "",
        "## Main-System Confirmatory Evidence",
        "",
        "| benchmark | status | baseline | main | n | baseline verifier | main verifier | delta | baseline oracle | main oracle |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in main_system["rows"]:
        if row.get("status") != "ready":
            lines.append(f"| {row['benchmark']} | `{row.get('status')}` | NA | NA | 0 | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {row['benchmark']} | `{row['status']}` | {row['baseline']} | {row['main']} | {_fmt(row.get('n'))} | {_fmt(row.get('baseline_verifier'))} | {_fmt(row.get('main_verifier'))} | {_fmt(row.get('delta_verifier'))} | {_fmt(row.get('baseline_oracle'))} | {_fmt(row.get('main_oracle'))} |"
        )

    lines.extend([
        "",
        "## External Frontier",
        "",
        "| benchmark | status | internal main | best external | delta vs internal main | note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ])
    for row in external["rows"]:
        if row.get("benchmark") == "gsm8k_full_clean":
            if row.get("status") == "ready":
                delta_vs_internal = None
                main_gsm8k = next((item for item in main_system["rows"] if item["benchmark"] == "gsm8k_full_clean" and item.get("status") == "ready"), None)
                if main_gsm8k is not None and row.get("accuracy") is not None:
                    delta_vs_internal = row["accuracy"] - main_gsm8k["main_verifier"]
                lines.append(f"| gsm8k_full_clean | `{row['status']}` | {_fmt(main_gsm8k.get('main_verifier') if main_gsm8k else None)} | {_fmt(row.get('accuracy'))} | {_fmt(delta_vs_internal)} | full self-refine reference |")
            else:
                lines.append("| gsm8k_full_clean | `missing` | NA | NA | NA | missing self-refine full summary |")
            continue
        if row.get("status") != "ready":
            lines.append(f"| {row['benchmark']} | `{row.get('status')}` | NA | NA | NA | waiting for Gate C summary |")
            continue
        best_external = row.get("best_external") or {}
        delta = None
        if best_external and row.get("internal_main_accuracy") is not None and best_external.get("accuracy") is not None:
            delta = best_external["accuracy"] - row["internal_main_accuracy"]
        lines.append(
            f"| {row['benchmark']} | `{row['status']}` | {_fmt(row.get('internal_main_accuracy'))} | {_fmt(best_external.get('accuracy'))} | {_fmt(delta)} | {best_external.get('label', 'NA')} |"
        )

    lines.extend([
        "",
        "## Paired Statistics",
        "",
        "| comparison | status | n | acc A | acc B | delta(B-A) | 95% CI | McNemar p |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ])
    for row in paired["rows"]:
        if row.get("status") != "ready":
            lines.append(f"| {row['label']} | `{row['status']}` | 0 | NA | NA | NA | NA | NA |")
            continue
        ci = row["ci95"]
        lines.append(
            f"| {row['label']} | `{row['status']}` | {_fmt(row.get('n'))} | {_fmt(row.get('accuracy_a'))} | {_fmt(row.get('accuracy_b'))} | {_fmt(row.get('delta_b_minus_a'))} | [{_fmt(ci.get('low'))}, {_fmt(ci.get('high'))}] | {_fmt(row.get('mcnemar', {}).get('p'), digits=6)} |"
        )

    lines.extend([
        "",
        "## Seed Stability",
        "",
        "| slice | status | seeds | mean verifier | std verifier | spread verifier | mean oracle | std oracle |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in seed["rows"]:
        if row.get("status") != "ready":
            lines.append(f"| {row['label']} | `{row['status']}` | 0 | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {row['label']} | `{row['status']}` | {_fmt(row.get('num_seeds'))} | {_fmt(row.get('mean_verifier'))} | {_fmt(row.get('std_verifier'))} | {_fmt(row.get('spread_verifier'))} | {_fmt(row.get('mean_oracle'))} | {_fmt(row.get('std_oracle'))} |"
        )

    lines.extend([
        "",
        "## Budget Scaling",
        "",
    ])
    if budget.get("status") == "ready":
        lines.extend([
            "| k | oracle | verifier | verifier|oracle |",
            "| ---: | ---: | ---: | ---: |",
        ])
        for row in budget.get("rows", []):
            lines.append(
                f"| {_fmt(row.get('k'))} | {_fmt(row.get('oracle_accuracy'))} | {_fmt(row.get('verifier_accuracy'))} | {_fmt(row.get('verifier_given_oracle'))} |"
            )
        lines.append("")
        lines.append(f"- Observed budget points: `{budget.get('observed_k')}`")
        lines.append(f"- Missing strong-paper target points: `{budget.get('missing_required_k')}`")
        if budget.get("budget_implications"):
            implications = budget["budget_implications"]
            lines.append(
                f"- Existing GSM8K full curve still tops out at `k=8`; current summary shows verifier gain `1→4 = {implications.get('verifier_gain_1_to_4')}` and `4→8 = {implications.get('verifier_gain_4_to_8')}`."
            )
    else:
        lines.append(f"- Budget summary missing: `{args.budget_json}`")

    lines.extend([
        "",
        "## Failure Migration and Compatibility Audit",
        "",
    ])
    if failure.get("rows"):
        for row in failure["rows"]:
            lines.append(
                f"- `{row['benchmark']}`: oracle-added=`{_fmt(row.get('oracle_added'))}`, verifier-gain=`{_fmt(row.get('verifier_gain_total'))}`, verifier-loss=`{_fmt(row.get('verifier_loss_total'))}`, oracle-added conversion=`{_fmt(row.get('oracle_added_conversion'))}`."
            )
    else:
        lines.append(f"- Missing paired delta audit: `{args.paired_audit_json}`")
    if isinstance(gateb_payload, dict):
        lines.append(f"- Gate B compatibility claim: `{gateb_payload.get('claim')}`")
        assessment = gateb_payload.get("assessment", {})
        if isinstance(assessment, dict):
            lines.append(f"- Gate B paper-facing interpretation: `{assessment.get('paper_facing_claim')}`")

    lines.extend([
        "",
        "## Remaining Gaps",
        "",
    ])
    if payload["assessment"]["largest_remaining_gaps"]:
        for item in payload["assessment"]["largest_remaining_gaps"]:
            lines.append(f"- {item}")
    else:
        lines.append("- No remaining major Gate D gaps are detected by the current pack.")

    lines.extend([
        "",
        "## Canonical Inputs",
        "",
    ])
    for key, value in payload["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")

    _write_text_atomic(output_md, "\n".join(lines) + "\n")
    print(output_md.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
