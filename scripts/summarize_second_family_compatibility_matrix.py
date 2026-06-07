from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from analyze_generate_then_rerank import (
    _prediction_has_instruction_leak,
    _prediction_has_scaffold_residue,
    _prediction_has_valid_final_answer,
)
from src.utils.io_utils import read_jsonl, write_json, write_text


DEFAULT_METHOD_ORDER = [
    "diverse_base",
    "cacc",
    "cacc_pstar",
    "cacc_hybrid_salvageability",
    "cacc_norm",
    "cacc_pstar_norm",
]

DEFAULT_SLICE_ORDER = [
    "gsm8k_eval_128",
    "competition_math_numeric",
    "mmlu_pro_smoke_128",
    "gpqa_diamond_train",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Route C second-family result JSON files into a compatibility matrix report."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary-json", action="append", default=[])
    parser.add_argument("--baseline-method", default="diverse_base")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_spec(spec: str) -> tuple[str, str, Path]:
    if "=" not in spec:
        raise ValueError(f"Summary spec must be [SLICE::]METHOD=PATH, got: {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if "::" in label:
        slice_name, method = label.split("::", 1)
    else:
        slice_name, method = "default", label
    return slice_name.strip(), method.strip(), Path(path.strip())


def _slice_rank(slice_name: str) -> tuple[int, str]:
    try:
        return (DEFAULT_SLICE_ORDER.index(slice_name), slice_name)
    except ValueError:
        return (len(DEFAULT_SLICE_ORDER), slice_name)


def _method_rank(method: str) -> tuple[int, str]:
    try:
        return (DEFAULT_METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(DEFAULT_METHOD_ORDER), method)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _prediction_hygiene_from_file(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = list(read_jsonl(path))
    if not rows:
        return {}

    parseable = []
    mode_match = []
    instruction_leak = []
    scaffold_residue = []
    answer_modes = []
    for row in rows:
        prediction = str(row.get("prediction", ""))
        answer_mode = str(row.get("answer_mode", "numeric"))
        answer_modes.append(answer_mode)
        valid = _prediction_has_valid_final_answer(prediction, answer_mode)
        parseable.append(1.0 if valid else 0.0)
        mode_match.append(1.0 if valid else 0.0)
        instruction_leak.append(1.0 if _prediction_has_instruction_leak(prediction) else 0.0)
        scaffold_residue.append(1.0 if _prediction_has_scaffold_residue(prediction) else 0.0)

    answer_mode = max(set(answer_modes), key=answer_modes.count) if answer_modes else None
    selected_parseable = _mean(parseable)
    return {
        "answer_mode": answer_mode,
        "verifier_selected_parseable_rate": selected_parseable,
        "verifier_invalid_final_rate": (1.0 - selected_parseable) if selected_parseable is not None else None,
        "verifier_answer_mode_match_rate": _mean(mode_match),
        "verifier_instruction_leak_rate": _mean(instruction_leak),
        "verifier_scaffold_residue_rate": _mean(scaffold_residue),
    }


def _merge_hygiene(summary_hygiene: dict, fallback_hygiene: dict) -> tuple[dict, str]:
    merged = dict(summary_hygiene)
    source = "summary"
    used_fallback = False
    for key, value in fallback_hygiene.items():
        if merged.get(key) is None:
            merged[key] = value
            used_fallback = True
    if not merged:
        source = "missing"
    elif used_fallback and summary_hygiene:
        source = "summary+verifier_predictions"
    elif used_fallback:
        source = "verifier_predictions"
    return merged, source


def _row_from_summary(slice_name: str, method: str, path: Path) -> dict:
    summary = _load_json(path)
    run_label = summary.get("run_label") or path.stem
    summary_hygiene = dict(summary.get("prediction_hygiene", {}))
    verifier_predictions = PROJECT_ROOT / "Experiment" / "core_code" / "logs" / f"{run_label}_verifier_predictions.jsonl"
    fallback_hygiene = _prediction_hygiene_from_file(verifier_predictions)
    hygiene, hygiene_source = _merge_hygiene(summary_hygiene, fallback_hygiene)

    oracle_coverage = summary.get("oracle_coverage")
    base_accuracy = summary.get("base_accuracy")
    verifier_accuracy = summary.get("verifier_accuracy")
    verifier_given_oracle = summary.get("verifier_given_oracle")
    if verifier_given_oracle is None and oracle_coverage not in (None, 0):
        verifier_given_oracle = verifier_accuracy / oracle_coverage if verifier_accuracy is not None else None

    selected_parseable = hygiene.get("verifier_selected_parseable_rate")
    invalid_final = hygiene.get("verifier_invalid_final_rate")
    answer_mode_match = hygiene.get("verifier_answer_mode_match_rate")
    instruction_leak = hygiene.get("verifier_instruction_leak_rate")
    scaffold_residue = hygiene.get("verifier_scaffold_residue_rate")

    return {
        "slice": slice_name,
        "method": method,
        "path": str(path),
        "run_label": run_label,
        "num_examples": summary.get("total_examples"),
        "answer_mode": fallback_hygiene.get("answer_mode"),
        "oracle_coverage": oracle_coverage,
        "base_accuracy": base_accuracy,
        "verifier_accuracy": verifier_accuracy,
        "verifier_given_oracle": verifier_given_oracle,
        "oracle_minus_verifier": (oracle_coverage - verifier_accuracy) if oracle_coverage is not None and verifier_accuracy is not None else None,
        "selected_parseable": selected_parseable,
        "invalid_final": invalid_final,
        "answer_mode_match": answer_mode_match,
        "instruction_leak": instruction_leak,
        "scaffold_residue": scaffold_residue,
        "hygiene_source": hygiene_source,
    }


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _slice_payload(slice_name: str, rows: list[dict], baseline_method: str) -> dict:
    ordered_rows = sorted(rows, key=lambda row: _method_rank(row["method"]))
    baseline = next((row for row in ordered_rows if row["method"] == baseline_method), None)
    for row in ordered_rows:
        if baseline is None:
            row["delta_oracle_vs_baseline"] = None
            row["delta_verifier_vs_baseline"] = None
            row["delta_vgo_vs_baseline"] = None
            row["delta_parseable_vs_baseline"] = None
        else:
            row["delta_oracle_vs_baseline"] = (
                row["oracle_coverage"] - baseline["oracle_coverage"]
                if row["oracle_coverage"] is not None and baseline["oracle_coverage"] is not None
                else None
            )
            row["delta_verifier_vs_baseline"] = (
                row["verifier_accuracy"] - baseline["verifier_accuracy"]
                if row["verifier_accuracy"] is not None and baseline["verifier_accuracy"] is not None
                else None
            )
            row["delta_vgo_vs_baseline"] = (
                row["verifier_given_oracle"] - baseline["verifier_given_oracle"]
                if row["verifier_given_oracle"] is not None and baseline["verifier_given_oracle"] is not None
                else None
            )
            row["delta_parseable_vs_baseline"] = (
                row["selected_parseable"] - baseline["selected_parseable"]
                if row["selected_parseable"] is not None and baseline["selected_parseable"] is not None
                else None
            )

    best_verifier = max(
        ordered_rows,
        key=lambda row: (row["verifier_accuracy"] is not None, row["verifier_accuracy"] if row["verifier_accuracy"] is not None else float("-inf")),
    ) if ordered_rows else None
    best_oracle = max(
        ordered_rows,
        key=lambda row: (row["oracle_coverage"] is not None, row["oracle_coverage"] if row["oracle_coverage"] is not None else float("-inf")),
    ) if ordered_rows else None

    mismatch_rows = []
    if baseline is not None:
        for row in ordered_rows:
            if row["method"] == baseline_method:
                continue
            if row["delta_oracle_vs_baseline"] is None or row["delta_verifier_vs_baseline"] is None:
                continue
            if row["delta_oracle_vs_baseline"] > 0 and row["delta_verifier_vs_baseline"] < 0:
                mismatch_rows.append(
                    {
                        "method": row["method"],
                        "delta_oracle_vs_baseline": row["delta_oracle_vs_baseline"],
                        "delta_verifier_vs_baseline": row["delta_verifier_vs_baseline"],
                        "delta_vgo_vs_baseline": row["delta_vgo_vs_baseline"],
                    }
                )

    return {
        "slice": slice_name,
        "baseline_method": baseline_method,
        "best_method_by_verifier": best_verifier["method"] if best_verifier else None,
        "best_method_by_oracle": best_oracle["method"] if best_oracle else None,
        "rows": ordered_rows,
        "compatibility_tension_rows": mismatch_rows,
    }


def main() -> None:
    args = parse_args()
    parsed_specs = [_parse_spec(spec) for spec in args.summary_json]
    grouped: dict[str, list[dict]] = {}
    for slice_name, method, path in parsed_specs:
        grouped.setdefault(slice_name, []).append(_row_from_summary(slice_name, method, path))

    slice_payloads = [_slice_payload(slice_name, rows, args.baseline_method) for slice_name, rows in grouped.items()]
    slice_payloads.sort(key=lambda payload: _slice_rank(payload["slice"]))

    payload = {
        "title": args.title,
        "baseline_method": args.baseline_method,
        "slices": slice_payloads,
    }
    write_json(args.output_json, payload)

    lines = [
        f"# {args.title}",
        "",
        f"- baseline method: `{args.baseline_method}`",
        f"- slices: `{', '.join(payload['slice'] for payload in slice_payloads)}`",
        "",
    ]

    for slice_payload in slice_payloads:
        slice_name = slice_payload["slice"]
        rows = slice_payload["rows"]
        lines.extend(
            [
                f"## {slice_name}",
                "",
                "| Method | N | Oracle | Base | Verifier | V\\|Oracle | Oracle-Verifier Gap | Parseable | Invalid | Mode Match | Instr Leak | Scaffold | ΔOracle | ΔVerifier |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| {method} | {n} | {oracle} | {base} | {verifier} | {vgo} | {gap} | {parseable} | {invalid} | {mode_match} | {instr_leak} | {scaffold} | {d_oracle} | {d_verifier} |".format(
                    method=row["method"],
                    n=_fmt(row["num_examples"], digits=0),
                    oracle=_fmt(row["oracle_coverage"]),
                    base=_fmt(row["base_accuracy"]),
                    verifier=_fmt(row["verifier_accuracy"]),
                    vgo=_fmt(row["verifier_given_oracle"]),
                    gap=_fmt(row["oracle_minus_verifier"]),
                    parseable=_fmt(row["selected_parseable"]),
                    invalid=_fmt(row["invalid_final"]),
                    mode_match=_fmt(row["answer_mode_match"]),
                    instr_leak=_fmt(row["instruction_leak"]),
                    scaffold=_fmt(row["scaffold_residue"]),
                    d_oracle=_fmt(row["delta_oracle_vs_baseline"]),
                    d_verifier=_fmt(row["delta_verifier_vs_baseline"]),
                )
            )

        lines.extend(["", "### Notes", ""])
        if slice_payload["best_method_by_oracle"] is not None:
            best_oracle_row = next(row for row in rows if row["method"] == slice_payload["best_method_by_oracle"])
            lines.append(
                f"- Highest oracle row: `{best_oracle_row['method']}` at `{_fmt(best_oracle_row['oracle_coverage'])}`."
            )
        if slice_payload["best_method_by_verifier"] is not None:
            best_verifier_row = next(row for row in rows if row["method"] == slice_payload["best_method_by_verifier"])
            lines.append(
                f"- Highest verifier row: `{best_verifier_row['method']}` at `{_fmt(best_verifier_row['verifier_accuracy'])}` with `V|Oracle={_fmt(best_verifier_row['verifier_given_oracle'])}`."
            )
        if slice_payload["compatibility_tension_rows"]:
            for row in slice_payload["compatibility_tension_rows"]:
                lines.append(
                    "- Compatibility tension: `{method}` raises oracle by `{d_oracle:+.4f}` but drops verifier by `{d_verifier:+.4f}` relative to `{baseline}`.".format(
                        method=row["method"],
                        d_oracle=row["delta_oracle_vs_baseline"],
                        d_verifier=row["delta_verifier_vs_baseline"],
                        baseline=args.baseline_method,
                    )
                )
        else:
            lines.append("- No row shows the clean `oracle up / verifier down` tension against the chosen baseline.")

        missing_hygiene_rows = [row["method"] for row in rows if row["hygiene_source"] == "missing"]
        if missing_hygiene_rows:
            lines.append(
                f"- Missing hygiene sidecars for: `{', '.join(missing_hygiene_rows)}`; parseability-style cells stay `NA`."
            )
        lines.append("")

    write_text(args.output_md, "\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
