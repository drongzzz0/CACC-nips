from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the second-family weakest-slice multiseed summary.")
    parser.add_argument(
        "--seed-json",
        action="append",
        default=[],
        help="Per-seed summary JSON path. Defaults to the canonical seed7/11/19 weakest-slice runs.",
    )
    parser.add_argument(
        "--output-json",
        default="Experiment/analysis/results/routec_b1_secondfamily_weakest_slice_multiseed_v1.json",
    )
    parser.add_argument(
        "--output-md",
        default="Experiment/analysis/results/routec_b1_secondfamily_weakest_slice_multiseed_v1.md",
    )
    return parser.parse_args()


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    tmp.replace(path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    seed_paths = args.seed_json or [
        "Experiment/analysis/results/routec_b1_secondfamily_weakest_slice_seed7_v1.json",
        "Experiment/analysis/results/routec_b1_secondfamily_weakest_slice_seed11_v1.json",
        "Experiment/analysis/results/routec_b1_secondfamily_weakest_slice_seed19_v1.json",
    ]
    abs_paths = [_project_path(path) for path in seed_paths]
    missing = [str(path) for path in seed_paths if not _project_path(path).exists()]

    payload: dict
    if missing:
        payload = {
            "label": "second_family_weakest_slice",
            "status": "missing",
            "missing": missing,
            "paths": seed_paths,
            "definition": "routec_p2_secondfamily_mistral7b_instruct_gsm8k_eval_128_cacc_v1 weakest negative slice",
        }
    else:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in abs_paths]
        verifier = [float(row["verifier_accuracy"]) for row in rows]
        oracle = [float(row["oracle_coverage"]) for row in rows]
        base = [float(row["base_accuracy"]) for row in rows]
        first = [float(row["first_accuracy"]) for row in rows]
        payload = {
            "label": "second_family_weakest_slice",
            "status": "ready",
            "definition": "routec_p2_secondfamily_mistral7b_instruct_gsm8k_eval_128_cacc_v1 weakest negative slice",
            "benchmark": "gsm8k_eval_128",
            "generator_family": "mistral7b_instruct",
            "method": "cacc",
            "num_seeds": len(rows),
            "paths": seed_paths,
            "mean_verifier": _mean(verifier),
            "std_verifier": _std(verifier),
            "min_verifier": min(verifier),
            "max_verifier": max(verifier),
            "spread_verifier": max(verifier) - min(verifier),
            "mean_oracle": _mean(oracle),
            "std_oracle": _std(oracle),
            "mean_base": _mean(base),
            "std_base": _std(base),
            "mean_first": _mean(first),
            "std_first": _std(first),
            "per_seed": rows,
        }

    output_json = _project_path(args.output_json)
    output_md = _project_path(args.output_md)
    _write_json_atomic(output_json, payload)

    lines = [
        "# Route C B1 Second-Family Weakest-Slice Multiseed Summary",
        "",
        f"- Label: `{payload['label']}`",
        f"- Status: `{payload['status']}`",
        f"- Definition: {payload['definition']}",
        "",
    ]
    if payload["status"] != "ready":
        lines.extend([
            "## Missing Inputs",
            "",
            *[f"- `{path}`" for path in payload.get("missing", [])],
        ])
    else:
        lines.extend([
            "## Aggregate Readout",
            "",
            "| metric | mean | std | min | max |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| verifier_accuracy | {_fmt(payload['mean_verifier'])} | {_fmt(payload['std_verifier'])} | {_fmt(payload['min_verifier'])} | {_fmt(payload['max_verifier'])} |",
            f"| oracle_coverage | {_fmt(payload['mean_oracle'])} | {_fmt(payload['std_oracle'])} | NA | NA |",
            f"| base_accuracy | {_fmt(payload['mean_base'])} | {_fmt(payload['std_base'])} | NA | NA |",
            f"| first_accuracy | {_fmt(payload['mean_first'])} | {_fmt(payload['std_first'])} | NA | NA |",
            "",
            "## Per-Seed Readout",
            "",
            "| seed | verifier | oracle | base | first | artifact |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in payload["per_seed"]:
            seed = row["run_label"].split("seed")[-1].split("_")[0]
            lines.append(
                f"| {seed} | {_fmt(row.get('verifier_accuracy'))} | {_fmt(row.get('oracle_coverage'))} | {_fmt(row.get('base_accuracy'))} | {_fmt(row.get('first_accuracy'))} | `{row.get('run_label')}` |"
            )

    _write_text_atomic(output_md, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
