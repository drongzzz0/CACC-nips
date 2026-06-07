from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.generation.prompts import build_inference_prompt
from src.inference.peft_generation import _extract_prediction, _resolve_base_model, load_tokenizer_for_inference, missing_dependencies
from src.utils.io_utils import read_jsonl, write_json, write_jsonl


@dataclass
class LoadedCausalLM:
    model: object
    tokenizer: object
    device: object
    base_model_name_or_path: str


def seed_everything(seed: int) -> None:
    random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_causal_lm(
    adapter_path: Path | None,
    model_path: str | None,
    base_model: str | None,
) -> LoadedCausalLM:
    require_peft = adapter_path is not None
    missing = missing_dependencies(require_peft=require_peft)
    if missing:
        raise RuntimeError("Missing inference dependencies: " + ", ".join(missing))

    import torch
    from transformers import AutoModelForCausalLM
    if require_peft:
        from peft import PeftModel

    base_model_name_or_path = _resolve_base_model(adapter_path, base_model, model_path)
    tokenizer = load_tokenizer_for_inference(
        adapter_path=adapter_path,
        base_model_name_or_path=base_model_name_or_path,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model_obj, adapter_path) if adapter_path is not None else base_model_obj
    model.to(device)
    model.eval()
    return LoadedCausalLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        base_model_name_or_path=base_model_name_or_path,
    )


def build_problem_prompt(example: dict) -> str:
    if "prompt" in example:
        return str(example["prompt"])
    return f"Problem: {example['problem']}\nAnswer with the requested format."


def generate_batch_completions(
    lm: LoadedCausalLM,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    num_return_sequences: int = 1,
) -> list[list[dict]]:
    import torch

    if not prompts:
        return []

    original_padding_side = getattr(lm.tokenizer, "padding_side", "right")
    if not getattr(lm.model.config, "is_encoder_decoder", False):
        lm.tokenizer.padding_side = "left"

    try:
        encoded = lm.tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(lm.device) for key, value in encoded.items()}
        prompt_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        prompt_width = int(encoded["input_ids"].shape[1])

        generation_kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": lm.tokenizer.pad_token_id,
            "eos_token_id": lm.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            sequences = lm.model.generate(**generation_kwargs)
    finally:
        lm.tokenizer.padding_side = original_padding_side

    if getattr(lm.model.config, "is_encoder_decoder", False):
        generated_tokens = sequences
    else:
        generated_tokens = sequences[:, prompt_width:]

    generated_texts = [text.strip() for text in lm.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)]
    repeated_prompt_counts = [
        prompt_token_counts[index // num_return_sequences]
        for index in range(len(generated_texts))
    ]
    generated_token_counts = generated_tokens.ne(lm.tokenizer.pad_token_id).sum(dim=1).tolist()

    grouped_outputs = [[] for _ in prompts]
    for output_index, generated_text in enumerate(generated_texts):
        prompt_index = output_index // num_return_sequences
        grouped_outputs[prompt_index].append(
            {
                "prompt": prompts[prompt_index],
                "generated_text": generated_text,
                "prediction": _extract_prediction(generated_text).strip(),
                "prompt_token_count": int(repeated_prompt_counts[output_index]),
                "generated_token_count": int(generated_token_counts[output_index]),
            }
        )
    return grouped_outputs


def generate_completions(
    lm: LoadedCausalLM,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    num_return_sequences: int = 1,
) -> list[dict]:
    return generate_batch_completions(
        lm,
        [prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
    )[0]


def run_batch_generation(
    lm: LoadedCausalLM,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[dict]:
    return [
        rows[0]
        for rows in generate_batch_completions(
            lm,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=1,
        )
    ]


def run_generation(
    lm: LoadedCausalLM,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> dict:
    return run_batch_generation(
        lm,
        [prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )[0]


def dedupe_preserve_order(candidates: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for candidate in candidates:
        normalized = " ".join(candidate.strip().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def dataset_records(path: Path) -> list[dict]:
    return list(read_jsonl(path))


def write_predictions(path: Path, rows: list[dict]) -> None:
    write_jsonl(path, rows)


def write_metrics(path: Path | None, payload: dict) -> None:
    if path is not None:
        write_json(path, payload)


def finalize_metrics(
    *,
    dataset_path: Path,
    adapter_path: Path | None,
    base_model_name_or_path: str,
    started_at: float,
    example_count: int,
    extra: dict | None = None,
) -> dict:
    total_seconds = time.perf_counter() - started_at
    payload = {
        "dataset_path": str(dataset_path),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "base_model": base_model_name_or_path,
        "num_examples": example_count,
        "total_runtime_seconds": round(total_seconds, 6),
        "avg_runtime_seconds": round(total_seconds / example_count, 6) if example_count else 0.0,
    }
    if extra:
        payload.update(extra)
    return payload


def write_trace_jsonl(path: Path | None, rows: list[dict]) -> None:
    if path is not None:
        write_jsonl(path, rows)


def json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False)


def baseline_inference_prompt(example: dict, supervision_type: str) -> str:
    return build_inference_prompt(
        build_problem_prompt(example),
        supervision_type,
        answer_mode=str(example.get("answer_mode", "numeric")),
    )
