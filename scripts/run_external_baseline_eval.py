from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


BASELINE_TO_SCRIPT = {
    "ccqa": "run_ccqa_baseline.py",
    "self_refine": "run_self_refine_baseline.py",
    "pairrm_best_of_n": "run_pairrm_best_of_n_baseline.py",
    "tot_lite": "run_tot_lite_baseline.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one external baseline and evaluate its predictions.")
    parser.add_argument("--baseline", required=True, choices=sorted(BASELINE_TO_SCRIPT))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--candidates-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--base-model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("baseline_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _run(command: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(printable)
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python_bin = sys.executable
    baseline_script = SCRIPTS_DIR / BASELINE_TO_SCRIPT[args.baseline]
    baseline_cmd = [
        python_bin,
        str(baseline_script),
        "--dataset",
        str(args.dataset),
        "--predictions",
        str(args.predictions),
    ]
    if args.metrics_output is not None:
        baseline_cmd.extend(["--metrics-output", str(args.metrics_output)])
    if args.adapter_path is not None:
        baseline_cmd.extend(["--adapter-path", str(args.adapter_path)])
    if args.model_path:
        baseline_cmd.extend(["--model-path", args.model_path])
    if args.base_model:
        baseline_cmd.extend(["--base-model", args.base_model])
    if args.candidates_output is not None and args.baseline in {"ccqa", "pairrm_best_of_n", "tot_lite"}:
        baseline_cmd.extend(["--candidates-output", str(args.candidates_output)])
    if args.trace_output is not None and args.baseline in {"ccqa", "self_refine"}:
        baseline_cmd.extend(["--trace-output", str(args.trace_output)])
    if args.baseline_args and args.baseline_args[0] == "--":
        baseline_cmd.extend(args.baseline_args[1:])
    else:
        baseline_cmd.extend(args.baseline_args)

    eval_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "evaluate_student.py"),
        "--predictions",
        str(args.predictions),
        "--report",
        str(args.report),
        "--run-name",
        args.run_name,
    ]

    _run(baseline_cmd, dry_run=args.dry_run)
    _run(eval_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
