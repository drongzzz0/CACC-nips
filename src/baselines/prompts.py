from __future__ import annotations

import re


def build_self_refine_init_prompt(problem_prompt: str, answer_mode: str) -> str:
    final_instruction = _final_answer_instruction(answer_mode)
    return (
        f"{problem_prompt.strip()}\n"
        "Solve the problem carefully using concise reasoning. "
        f"{final_instruction}"
    )


def build_self_refine_feedback_prompt(problem: str, attempt: str, answer_mode: str) -> str:
    answer_hint = (
        "Check whether the chosen option letter is justified by the reasoning."
        if answer_mode == "choice_letter"
        else "Check whether the numerical computation and last-step extraction are valid."
    )
    return (
        "You are reviewing a draft solution. "
        "Identify the main weaknesses briefly and concretely. "
        "Do not rewrite the solution yet. "
        "Use at most 3 short lines.\n\n"
        f"Problem: {problem}\n"
        f"Draft solution:\n{attempt}\n\n"
        f"Review focus: {answer_hint}"
    )


def build_self_refine_refine_prompt(problem: str, attempt: str, feedback: str, answer_mode: str) -> str:
    final_instruction = _final_answer_instruction(answer_mode)
    return (
        "Revise the draft solution using the feedback. "
        "Write a corrected, self-contained solution. "
        "Do not mention the feedback explicitly. "
        "Use at most 4 short reasoning lines. "
        f"{final_instruction}\n\n"
        f"Problem: {problem}\n"
        f"Draft solution:\n{attempt}\n\n"
        f"Feedback:\n{feedback}"
    )


def build_tot_expand_prompt(problem_prompt: str, partial_solution: str, answer_mode: str, depth: int, max_depth: int) -> str:
    final_instruction = _final_answer_instruction(answer_mode)
    if partial_solution.strip():
        return (
            f"{problem_prompt.strip()}\n"
            f"Current draft at search depth {depth - 1} of {max_depth}:\n"
            f"{partial_solution}\n\n"
            "Improve or complete this draft into a stronger solution candidate. "
            "You may rewrite it if needed. "
            "Use at most 4 short reasoning lines. "
            f"{final_instruction}"
        )
    return (
        f"{problem_prompt.strip()}\n"
        "Produce one promising solution candidate. "
        "Use at most 4 short reasoning lines. "
        f"{final_instruction}"
    )


def build_tot_value_prompt(problem: str, candidate: str, answer_mode: str) -> str:
    answer_hint = (
        "whether the selected option letter is well supported"
        if answer_mode == "choice_letter"
        else "whether the computation appears internally consistent"
    )
    return (
        "Score the following candidate solution for correctness and promise on a 1-5 scale. "
        "Reply with exactly one line in the format 'Score: N' where N is an integer from 1 to 5.\n\n"
        f"Problem: {problem}\n"
        f"Candidate solution:\n{candidate}\n\n"
        f"Focus on {answer_hint}."
    )


def parse_tot_score(text: str) -> int:
    matches = re.findall(r"score\s*[:\-]\s*([1-5])", text, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])
    digits = re.findall(r"\b([1-5])\b", text)
    if digits:
        return int(digits[-1])
    return 1


def build_ccqa_question_generation_prompt(candidate_solution: str, answer_mode: str) -> str:
    if answer_mode == "choice_letter":
        return (
            "Reconstruct the original multiple-choice question that this candidate answer is trying to solve. "
            "Preserve the semantic content and include the answer choices if they are recoverable. "
            "Write one concise question only.\n\n"
            f"Candidate solution:\n{candidate_solution}"
        )
    return (
        "CRITICAL: Do not change any numeric values that appear in the solution. "
        "Generate the original math word problem that would have this as its solution. "
        "Write one concise question only.\n\n"
        f"Candidate solution:\n{candidate_solution}"
    )


def build_ccqa_similarity_prompt(original_question: str, generated_questions: list[str]) -> str:
    lines = [
        "Which regenerated question is most similar to the original question?",
        "",
        f"Original question: {original_question}",
        "",
    ]
    for index, question in enumerate(generated_questions, start=1):
        normalized = question.strip() or "[No question generated]"
        lines.append(f"{index}. {normalized}")
    lines.extend(["", f"Answer with just one number from 1 to {len(generated_questions)}."])
    return "\n".join(lines)


def parse_ccqa_selected_index(text: str, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    digits = re.findall(r"\b(\d+)\b", text)
    for token in reversed(digits):
        value = int(token)
        if 1 <= value <= candidate_count:
            return value - 1
    return 0


def _final_answer_instruction(answer_mode: str) -> str:
    if answer_mode == "choice_letter":
        return "On the last line, write 'Final answer:' followed by the single option letter only."
    return "On the last line, write 'Final answer:' followed by the answer only."
