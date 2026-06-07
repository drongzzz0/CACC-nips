from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Iterable

from src.data.schema import TeacherTraceRecord
from src.generation.prompts import build_teacher_prompt


@dataclass
class TeacherGeneratorConfig:
    teacher_model: str = "qwen3.5-placeholder-teacher"
    backend: str = "template"
    reasoning_style: str = "template"
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False


BASE_REQUIRED_MODULES = ("torch", "transformers")


def missing_dependencies() -> list[str]:
    missing = []
    for module_name in BASE_REQUIRED_MODULES:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    return missing


def _extract_final_answer(reasoning: str) -> str:
    lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lower().startswith("final answer:"):
            candidate = line.split(":", maxsplit=1)[1].strip()
            if candidate and "<" not in candidate and "your answer" not in candidate.lower():
                return candidate
    matches = re.findall(r"final answer:\s*(.+)", reasoning, flags=re.IGNORECASE)
    for match in reversed(matches):
        candidate = match.strip()
        if candidate and "<" not in candidate and "your answer" not in candidate.lower():
            return candidate
    return lines[-1].strip() if lines else ""


def _normalize_answer(text: str) -> str:
    return text.strip().lower()


def _extract_numeric_answer(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    return matches[-1]


def _answers_match(prediction: str, gold_answer: str) -> bool:
    if _normalize_answer(prediction) == _normalize_answer(gold_answer):
        return True
    prediction_numeric = _extract_numeric_answer(prediction)
    gold_numeric = _extract_numeric_answer(gold_answer)
    return prediction_numeric is not None and gold_numeric is not None and prediction_numeric == gold_numeric


def synthesize_teacher_trace(problem: str, gold_answer: str) -> str:
    return (
        "1. Identify the quantities and the target value.\n"
        "2. Perform the intermediate computation needed to solve the problem.\n"
        f"3. Verify the result is consistent with the question.\nFinal answer: {gold_answer}"
    )


class HFTeacherGenerator:
    def __init__(self, config: TeacherGeneratorConfig) -> None:
        missing = missing_dependencies()
        if missing:
            raise RuntimeError(
                "Missing teacher-generation dependencies: "
                + ", ".join(missing)
                + ". Install torch and transformers to use the hf backend."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.teacher_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(config.teacher_model, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.config = config

    def generate_reasoning(self, problem: str) -> tuple[str, str]:
        prompt = build_teacher_prompt(problem)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful math reasoning assistant. "
                    "Solve the user's problem in at most 4 short steps. "
                    "Do not restate the request. "
                    "Do not use markdown bullets or headings. "
                    "End with exactly one line that starts with "
                    "'Final answer:' followed by the answer only."
                ),
            },
            {"role": "user", "content": f"Problem: {problem}"},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        with self._torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = generated[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return prompt, generated_text


def generate_teacher_record(example: dict, config: TeacherGeneratorConfig) -> TeacherTraceRecord:
    prompt = build_teacher_prompt(example["problem"])
    if config.backend == "template":
        reasoning = synthesize_teacher_trace(example["problem"], str(example["gold_answer"]))
    else:
        raise ValueError(
            "generate_teacher_record only supports the template backend. "
            "Use generate_teacher_records for hf-backed generation."
        )
    teacher_final_answer = _extract_final_answer(reasoning)
    return TeacherTraceRecord(
        example_id=example["example_id"],
        dataset=example["dataset"],
        problem=example["problem"],
        gold_answer=str(example["gold_answer"]),
        teacher_model=config.teacher_model,
        teacher_trace=reasoning,
        teacher_final_answer=teacher_final_answer,
        generation_prompt=prompt,
        trace_valid=_answers_match(teacher_final_answer, str(example["gold_answer"])),
    )


def generate_teacher_records(examples: Iterable[dict], config: TeacherGeneratorConfig) -> list[TeacherTraceRecord]:
    if config.backend == "template":
        return [generate_teacher_record(example, config) for example in examples]

    if config.backend != "hf":
        raise ValueError(f"Unsupported teacher backend: {config.backend}")

    generator = HFTeacherGenerator(config)
    records = []
    example_list = list(examples)
    total_examples = len(example_list)
    valid_count = 0
    for index, example in enumerate(example_list, start=1):
        prompt, reasoning = generator.generate_reasoning(example["problem"])
        teacher_final_answer = _extract_final_answer(reasoning)
        trace_valid = _answers_match(teacher_final_answer, str(example["gold_answer"]))
        if trace_valid:
            valid_count += 1
        records.append(
            TeacherTraceRecord(
                example_id=example["example_id"],
                dataset=example["dataset"],
                problem=example["problem"],
                gold_answer=str(example["gold_answer"]),
                teacher_model=config.teacher_model,
                teacher_trace=reasoning,
                teacher_final_answer=teacher_final_answer,
                generation_prompt=prompt,
                trace_valid=trace_valid,
            )
        )
        print(
            f"[teacher] {index}/{total_examples} example_id={example['example_id']} "
            f"trace_valid={trace_valid} valid_so_far={valid_count}",
            file=sys.stderr,
            flush=True,
        )
    return records
