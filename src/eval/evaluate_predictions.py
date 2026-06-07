from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


FINAL_ANSWER_PREFIX = re.compile(r"^final answer:\s*", flags=re.IGNORECASE)
CHOICE_PATTERN = re.compile(
    r"(?:final answer|answer|option|choice)\s*[:\-]?\s*\(?([A-J])\)?",
    flags=re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    lowered = text.strip().lower()
    lowered = FINAL_ANSWER_PREFIX.sub("", lowered)
    return lowered.strip()


def canonicalize_numeric_token(token: str) -> str | None:
    cleaned = token.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        decimal = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not decimal.is_finite():
        return None

    integral = decimal.to_integral_value()
    if decimal == integral:
        if abs(integral.adjusted()) > 30:
            return str(integral.normalize()).lower()
        try:
            normalized_integral = format(integral, "f")
        except (ValueError, InvalidOperation):
            normalized_integral = str(integral.normalize()).lower()
        if "." in normalized_integral:
            normalized_integral = normalized_integral.rstrip("0").rstrip(".")
        return normalized_integral or "0"

    if abs(decimal.adjusted()) > 30:
        return str(decimal.normalize()).lower()

    try:
        normalized = format(decimal.normalize(), "f")
    except (ValueError, InvalidOperation):
        normalized = str(decimal.normalize()).lower()
    normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def extract_numeric_answer(text: str) -> str | None:
    normalized = normalize_answer(text)
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", normalized)
    if boxed_matches:
        for boxed_match in reversed(boxed_matches):
            boxed = canonicalize_numeric_token(boxed_match)
            if boxed is not None:
                return boxed
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?", normalized, flags=re.IGNORECASE)
    if not matches:
        return None
    for match in reversed(matches):
        canonical = canonicalize_numeric_token(match)
        if canonical is not None:
            return canonical
    return None


def extract_choice_answer(text: str) -> str | None:
    stripped = text.strip()
    matches = CHOICE_PATTERN.findall(stripped)
    if matches:
        return matches[-1].upper()

    normalized = normalize_answer(stripped)
    single = re.fullmatch(r"\(?([a-j])\)?[.)]?", normalized)
    if single:
        return single.group(1).upper()

    tokens = re.findall(r"\b([A-J])\b", stripped.upper())
    if len(tokens) == 1:
        return tokens[0]
    return None


def answer_mode_for_record(record: dict) -> str:
    answer_mode = str(record.get("answer_mode", "numeric")).strip().lower()
    return answer_mode or "numeric"


def answers_match(prediction: str, gold_answer: str, answer_mode: str = "numeric") -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold_answer)
    if normalized_prediction == normalized_gold:
        return True

    if answer_mode == "choice_letter":
        prediction_choice = extract_choice_answer(prediction)
        gold_choice = extract_choice_answer(gold_answer) or normalize_answer(gold_answer).upper()
        return prediction_choice is not None and gold_choice is not None and prediction_choice == gold_choice

    prediction_numeric = extract_numeric_answer(prediction)
    gold_numeric = extract_numeric_answer(gold_answer)
    return prediction_numeric is not None and gold_numeric is not None and prediction_numeric == gold_numeric


@dataclass
class EvalMetrics:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def compute_exact_match(records: list[dict]) -> EvalMetrics:
    total = len(records)
    correct = 0
    for record in records:
        if answers_match(
            str(record["prediction"]),
            str(record["gold_answer"]),
            answer_mode=answer_mode_for_record(record),
        ):
            correct += 1
    return EvalMetrics(total=total, correct=correct)


def build_markdown_report(run_name: str, metrics: EvalMetrics) -> str:
    return (
        f"# Evaluation Report: {run_name}\n\n"
        f"- total examples: {metrics.total}\n"
        f"- correct: {metrics.correct}\n"
        f"- exact-match accuracy: {metrics.accuracy:.4f}\n"
    )
