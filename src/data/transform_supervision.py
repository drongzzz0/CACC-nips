from __future__ import annotations

import re

from src.data.schema import ProcessedTraceRecord, SFTExample


THINK_TAG_LINES = {"<think>", "</think>"}
FILLER_SENTENCE_PREFIXES = (
    "okay, let's see",
    "ok, let's see",
    "let's see",
    "let me think",
)


def normalize_reasoning(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _reasoning_lines(text: str) -> list[str]:
    lines = []
    for raw_line in normalize_reasoning(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("final answer:"):
            continue
        if line.lower() in THINK_TAG_LINES:
            continue
        lines.append(line)
    return lines


def extract_filtered_cot(text: str, max_lines: int = 3) -> str:
    filtered = []
    for line in _reasoning_lines(text):
        filtered.append(line)
        if len(filtered) >= max_lines:
            break
    return "\n".join(filtered)


def extract_brief_reasoning(text: str, max_lines: int = 2) -> str:
    cleaned_lines = []
    for line in _reasoning_lines(text):
        cleaned = re.sub(r"^\d+[\).\s-]*", "", line).strip()
        if cleaned:
            cleaned = re.sub(r"Final answer:\s*.*$", "", cleaned, flags=re.IGNORECASE).strip()
            if cleaned:
                cleaned_lines.append(cleaned)
    flattened = " ".join(cleaned_lines)
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", flattened):
        stripped = sentence.strip()
        if not stripped:
            continue
        lowered = stripped.lower().rstrip(".!?")
        if any(lowered.startswith(prefix) for prefix in FILLER_SENTENCE_PREFIXES):
            continue
        sentences.append(stripped)
    return "\n".join(sentences[:max_lines])


def extract_subgoals(text: str, max_items: int = 3) -> list[str]:
    subgoals = []
    for line in _reasoning_lines(text):
        cleaned = re.sub(r"^\d+[\).\s-]*", "", line).strip()
        if cleaned:
            subgoals.append(cleaned)
        if len(subgoals) >= max_items:
            break
    return subgoals


def build_processed_record(record: dict) -> ProcessedTraceRecord:
    filtered_cot = extract_filtered_cot(record["teacher_trace"])
    brief_reasoning = extract_brief_reasoning(record["teacher_trace"])
    subgoals = extract_subgoals(record["teacher_trace"])
    return ProcessedTraceRecord(
        example_id=record["example_id"],
        dataset=record["dataset"],
        problem=record["problem"],
        gold_answer=str(record["gold_answer"]),
        teacher_model=record["teacher_model"],
        teacher_trace=normalize_reasoning(record["teacher_trace"]),
        filtered_cot=filtered_cot,
        brief_reasoning=brief_reasoning,
        subgoals=subgoals,
        teacher_final_answer=str(record["teacher_final_answer"]),
        trace_valid=bool(record.get("trace_valid", True)),
        answer_mode=str(record.get("answer_mode", "numeric")),
        choices=[str(choice) for choice in record.get("choices", [])],
        metadata=dict(record.get("metadata", {})),
    )


def _base_prompt(problem: str) -> str:
    return f"Problem: {problem}\nAnswer with the requested format."


def build_answer_only_example(record: ProcessedTraceRecord) -> SFTExample:
    return SFTExample(
        example_id=record.example_id,
        dataset=record.dataset,
        prompt=_base_prompt(record.problem),
        response=f"Final answer: {record.gold_answer}",
        supervision_type="answer_only",
        gold_answer=record.gold_answer,
        answer_mode=record.answer_mode,
        choices=list(record.choices),
        metadata=dict(record.metadata),
    )


def build_filtered_cot_example(record: ProcessedTraceRecord) -> SFTExample:
    response = f"{record.filtered_cot}\nFinal answer: {record.gold_answer}".strip()
    return SFTExample(
        example_id=record.example_id,
        dataset=record.dataset,
        prompt=_base_prompt(record.problem),
        response=response,
        supervision_type="filtered_cot",
        gold_answer=record.gold_answer,
        answer_mode=record.answer_mode,
        choices=list(record.choices),
        metadata=dict(record.metadata),
    )


def build_subgoals_then_answer_example(record: ProcessedTraceRecord) -> SFTExample:
    steps = "\n".join(f"Sub-goal {idx + 1}: {goal}" for idx, goal in enumerate(record.subgoals))
    response = f"{steps}\nFinal answer: {record.gold_answer}".strip()
    return SFTExample(
        example_id=record.example_id,
        dataset=record.dataset,
        prompt=_base_prompt(record.problem),
        response=response,
        supervision_type="subgoals_then_answer",
        gold_answer=record.gold_answer,
        answer_mode=record.answer_mode,
        choices=list(record.choices),
        metadata=dict(record.metadata),
    )


def build_brief_reasoning_then_answer_example(record: ProcessedTraceRecord) -> SFTExample:
    response = f"{record.brief_reasoning}\nFinal answer: {record.gold_answer}".strip()
    return SFTExample(
        example_id=record.example_id,
        dataset=record.dataset,
        prompt=_base_prompt(record.problem),
        response=response,
        supervision_type="brief_reasoning_then_answer",
        gold_answer=record.gold_answer,
        answer_mode=record.answer_mode,
        choices=list(record.choices),
        metadata=dict(record.metadata),
    )
