from __future__ import annotations

import json
import re
import time
from typing import Any
from dataclasses import dataclass
from pathlib import Path

from src.generation.prompts import build_inference_prompt
from src.utils.io_utils import read_jsonl, write_jsonl


BASE_REQUIRED_MODULES = ("torch", "transformers")
PEFT_REQUIRED_MODULES = ("peft",)


def missing_dependencies(require_peft: bool = True) -> list[str]:
    missing = []
    required = BASE_REQUIRED_MODULES + (PEFT_REQUIRED_MODULES if require_peft else ())
    for module_name in required:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return missing


@dataclass
class GenerationConfig:
    dataset_path: Path
    adapter_path: Path | None
    predictions_path: Path
    metrics_path: Path | None = None
    model_path: str | None = None
    base_model: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False


def _resolve_base_model(adapter_path: Path | None, base_model_override: str | None, model_path: str | None) -> str:
    if model_path:
        return model_path

    if base_model_override:
        return base_model_override

    if adapter_path is None:
        raise ValueError("adapter_path is required when model_path and base_model are not provided.")

    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Missing adapter config at {adapter_config_path}; pass --base-model explicitly."
        )

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"adapter_config.json at {adapter_config_path} does not define base_model_name_or_path."
        )
    return str(base_model)


def load_tokenizer_for_inference(*, adapter_path: Path | None, base_model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    sources: list[tuple[str, str]] = []
    if adapter_path is not None:
        sources.append(("adapter", str(adapter_path)))
    sources.append(("base", base_model_name_or_path))

    last_error: Exception | None = None
    for source_kind, source in sources:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except AttributeError as exc:
            last_error = exc
            # Newer transformers versions reject some adapter-side tokenizer configs
            # where extra_special_tokens is serialized as a list instead of a mapping.
            if source_kind == "adapter" and "keys" in str(exc):
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("failed to load tokenizer")


def _extract_prediction(text: str) -> str:
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        lowered = line.lower()
        if not (lowered.startswith("final answer:") or lowered.startswith("answer:") or lowered.startswith("option:")):
            continue
        candidate = line.split(":", maxsplit=1)[1].strip()
        if candidate and "<answer>" not in candidate.lower():
            return candidate
        for next_line in lines[index + 1 :]:
            normalized = next_line.strip()
            if not normalized:
                continue
            if normalized.lower().startswith("final answer:"):
                break
            if "<answer>" in normalized.lower():
                continue
            if len(normalized.split()) <= 6 or re.search(r"-?\d", normalized):
                return normalized
            break
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", stripped)
    if boxed_matches:
        return boxed_matches[-1].strip()
    answer_is_matches = re.findall(r"the answer is\s*[: ]\s*(.+)", stripped, flags=re.IGNORECASE)
    if answer_is_matches:
        return answer_is_matches[-1].strip().rstrip(".")
    labelled_matches = re.findall(r"(?:answer|option)\s*[:\-]\s*([A-J])\b", stripped, flags=re.IGNORECASE)
    if labelled_matches:
        return labelled_matches[-1].upper()
    return lines[-1] if lines else ""


def generate_predictions(config: GenerationConfig) -> None:
    require_peft = config.adapter_path is not None
    missing = missing_dependencies(require_peft=require_peft)
    if missing:
        raise RuntimeError(
            "Missing inference dependencies: "
            + ", ".join(missing)
            + ". Install the required HuggingFace inference stack to run generation."
        )

    import torch
    from transformers import AutoModelForCausalLM
    if require_peft:
        from peft import PeftModel

    base_model_name_or_path = _resolve_base_model(config.adapter_path, config.base_model, config.model_path)

    tokenizer = load_tokenizer_for_inference(
        adapter_path=config.adapter_path,
        base_model_name_or_path=base_model_name_or_path,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        dtype=dtype,
    )
    if config.adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, config.adapter_path)
    else:
        model = base_model
    model.to(device)
    model.eval()

    prediction_records = []
    aggregate_prompt_tokens = 0
    aggregate_generated_tokens = 0
    started_at = time.perf_counter()
    for example in read_jsonl(config.dataset_path):
        prompt = build_inference_prompt(
            str(example["prompt"]),
            str(example.get("supervision_type", "answer_only")),
            answer_mode=str(example.get("answer_mode", "numeric")),
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]
        example_started_at = time.perf_counter()

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = generated[0][prompt_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        generated_token_count = int(generated_tokens.shape[0])
        aggregate_prompt_tokens += int(prompt_length)
        aggregate_generated_tokens += generated_token_count
        prediction_records.append(
            {
                "example_id": example["example_id"],
                "prompt": prompt,
                "generated_text": generated_text,
                "prediction": _extract_prediction(generated_text),
                "gold_answer": str(example["gold_answer"]),
                "answer_mode": str(example.get("answer_mode", "numeric")),
                "prompt_token_count": int(prompt_length),
                "generated_token_count": generated_token_count,
                "generation_seconds": round(time.perf_counter() - example_started_at, 6),
            }
        )

    write_jsonl(config.predictions_path, prediction_records)
    if config.metrics_path is not None:
        total_seconds = time.perf_counter() - started_at
        example_count = len(prediction_records)
        metrics = {
            "dataset_path": str(config.dataset_path),
            "adapter_path": str(config.adapter_path) if config.adapter_path is not None else None,
            "base_model": base_model_name_or_path,
            "model_path": config.model_path,
            "num_examples": example_count,
            "total_prompt_tokens": aggregate_prompt_tokens,
            "total_generated_tokens": aggregate_generated_tokens,
            "avg_prompt_tokens": (aggregate_prompt_tokens / example_count) if example_count else 0.0,
            "avg_generated_tokens": (aggregate_generated_tokens / example_count) if example_count else 0.0,
            "total_generation_seconds": round(total_seconds, 6),
            "avg_generation_seconds": round(total_seconds / example_count, 6) if example_count else 0.0,
            "examples_per_second": round(example_count / total_seconds, 6) if total_seconds else 0.0,
            "generated_tokens_per_second": round(aggregate_generated_tokens / total_seconds, 6) if total_seconds else 0.0,
        }
        config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        config.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
