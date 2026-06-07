from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.peft_generation import GenerationConfig, generate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions from a PEFT adapter checkpoint.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--do-sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.adapter_path is None and not args.model_path:
        raise SystemExit("Pass either --adapter-path for PEFT generation or --model-path for base-model generation.")
    generate_predictions(
        GenerationConfig(
            dataset_path=args.dataset,
            adapter_path=args.adapter_path,
            predictions_path=args.predictions,
            metrics_path=args.metrics_output,
            model_path=args.model_path,
            base_model=args.base_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
    )


if __name__ == "__main__":
    main()
