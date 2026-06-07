def build_teacher_prompt(problem: str) -> str:
    return (
        "Solve the following reasoning problem. "
        "Use at most 4 short steps. "
        "Do not restate the instructions or the problem. "
        "Do not use markdown bullets or headings. "
        "End with exactly one final line that starts with "
        "'Final answer:' followed by the answer only.\n\n"
        f"Problem: {problem}"
    )


def build_subgoal_prompt(problem: str, reasoning: str) -> str:
    return (
        "Compress the reasoning trace into short numbered sub-goals. "
        "Each sub-goal should be a concise action or planning state.\n\n"
        f"Problem: {problem}\n\n"
        f"Reasoning:\n{reasoning}"
    )


def build_verifier_prompt(problem: str, candidate_answer: str, answer_mode: str = "numeric") -> str:
    answer_label = "candidate option" if answer_mode == "choice_letter" else "candidate answer"
    return (
        f"Problem: {problem}\n"
        f"{answer_label.capitalize()}: {candidate_answer}\n"
        f"Is the {answer_label} correct? Reply with yes or no."
    )


def _normalize_completion_prompt_variant(variant: str) -> str:
    normalized = variant.strip().lower().replace("-", "_")
    alias_map = {
        "default": "default",
        "benchmarkaware": "default",
        "benchmark_aware": "default",
        "finish_best": "finish_best",
        "attempt_repair": "finish_best",
        "attempt_completion": "finish_best",
        "motif_guided": "motif_guided",
        "motif_first": "motif_guided",
    }
    if normalized not in alias_map:
        raise ValueError(f"Unsupported completion prompt variant: {variant}")
    return alias_map[normalized]



def build_completion_prompt(
    problem: str,
    problem_motif_label: str,
    observed_non_fragment_motifs: list[str],
    attempts: list[dict],
    answer_mode: str = "numeric",
    variant: str = "default",
) -> str:
    variant = _normalize_completion_prompt_variant(variant)
    motif_description = _describe_motif(problem_motif_label)
    final_answer_instruction = _final_answer_instruction(answer_mode)
    if answer_mode == "choice_letter":
        task_description = "multiple-choice reasoning problem"
        answer_mode_guidance = (
            "This is a multiple-choice task. You may reason about the options briefly, "
            "but the last line must contain the single option letter only, not the option text. "
        )
    else:
        task_description = "reasoning problem"
        answer_mode_guidance = ""
    observed_description = (
        ", ".join(_describe_motif(label) for label in observed_non_fragment_motifs)
        if observed_non_fragment_motifs
        else "none"
    )
    if attempts:
        attempt_lines = []
        for index, attempt in enumerate(attempts, start=1):
            attempt_lines.append(
                f"Attempt {index} "
                f"[quality={_describe_quality(attempt['quality_label'])}; "
                f"motif={_describe_motif(attempt['motif_label'])}]: "
                f"{attempt['candidate_text']}"
            )
        attempts_block = "\n".join(attempt_lines)
    else:
        attempts_block = "No useful partial attempts are available."

    if variant == "default":
        return (
            f"You are improving an incomplete candidate pool for a {task_description}. "
            "Most current candidates stop early or do not finish the reasoning. "
            "Use any useful partial steps if they help, but produce one complete alternative solution. "
            "Do not mention the attempts. Do not critique them. Do not stop mid-sentence. "
            "Do not output general advice, prompt-writing tips, role tags, or dialogue markers such as Human or Assistant. "
            f"{answer_mode_guidance}"
            "Use at most 4 short reasoning lines. "
            f"{final_answer_instruction}\n\n"
            f"Problem: {problem}\n"
            f"Likely useful reasoning motif: {motif_description}.\n"
            f"Observed non-fragment motifs already present: {observed_description}.\n\n"
            f"Incomplete attempts:\n{attempts_block}\n\n"
            "Write one complete solution now."
        )

    if variant == "finish_best":
        return (
            f"You are finishing the strongest incomplete attempt for a {task_description}. "
            "Pick one useful path from the incomplete attempts, but rewrite the final solution so it is fully self-contained. "
            "Do not mention the attempts. Do not critique them. Do not stop mid-sentence. "
            "Do not output general advice, prompt-writing tips, role tags, or dialogue markers such as Human or Assistant. "
            f"{answer_mode_guidance}"
            "Use at most 4 short reasoning lines. "
            f"{final_answer_instruction}\n\n"
            f"Problem: {problem}\n"
            f"Likely useful reasoning motif: {motif_description}.\n"
            f"Observed non-fragment motifs already present: {observed_description}.\n\n"
            f"Incomplete attempts:\n{attempts_block}\n\n"
            "Choose the most promising path, silently repair any local mistake, and write one complete solution now."
        )

    if variant == "motif_guided":
        return (
            f"You are writing one concise complete solution for a {task_description}. "
            "Use the likely reasoning motif as a planning cue and borrow only useful steps from the incomplete attempts. "
            "The final solution must be self-contained and decisive. "
            "Do not mention the attempts or motifs. Do not critique them. Do not stop mid-sentence. "
            "Do not output general advice, prompt-writing tips, role tags, or dialogue markers such as Human or Assistant. "
            f"{answer_mode_guidance}"
            "Use at most 4 short reasoning lines. "
            f"{final_answer_instruction}\n\n"
            f"Problem: {problem}\n"
            f"Likely useful reasoning motif: {motif_description}.\n"
            f"Observed non-fragment motifs already present: {observed_description}.\n\n"
            f"Incomplete attempts:\n{attempts_block}\n\n"
            "Write one complete solution that follows one coherent path now."
        )

    raise ValueError(f"Unsupported completion prompt variant after normalization: {variant}")

def build_repair_prompt(
    problem: str,
    candidate_text: str,
    motif_label: str,
    quality_label: str,
    repair_error_label: str,
    answer_mode: str = "numeric",
) -> str:
    final_answer_instruction = _final_answer_instruction(answer_mode)
    if answer_mode == "choice_letter":
        task_description = "multiple-choice reasoning problem"
        answer_mode_guidance = (
            "This is a multiple-choice task. You may briefly reason over the options, "
            "but the last line must contain the single option letter only, not the option text. "
        )
    else:
        task_description = "reasoning problem"
        answer_mode_guidance = ""
    return (
        f"You are revising one promising but likely incorrect candidate solution for a {task_description}. "
        "Keep any useful early reasoning only if it helps, but rewrite the solution so it is self-contained and complete. "
        "Do not mention the earlier attempt. Do not explain the error. "
        "Do not output critique, prompt-writing tips, role tags, or dialogue markers such as Human or Assistant. "
        f"{answer_mode_guidance}"
        "Use at most 4 short reasoning lines. "
        f"{final_answer_instruction}\n\n"
        f"Problem: {problem}\n"
        f"Candidate motif: {_describe_motif(motif_label)}.\n"
        f"Candidate quality: {_describe_quality(quality_label)}.\n"
        f"Likely issue to repair: {_describe_repair_error(repair_error_label)}.\n\n"
        f"Candidate to revise:\n{candidate_text}\n\n"
        "Write one repaired solution now."
    )


def build_inference_prompt(problem_prompt: str, supervision_type: str, answer_mode: str = "numeric") -> str:
    stripped = problem_prompt.strip()
    final_answer_instruction = _final_answer_instruction(answer_mode)
    if supervision_type == "answer_only":
        if answer_mode == "choice_letter":
            instruction = (
                "Solve the problem. "
                "Do not show intermediate reasoning. "
                f"{final_answer_instruction}"
            )
        else:
            instruction = (
                "Solve the problem. "
                "Do not show intermediate reasoning. "
                f"{final_answer_instruction}"
            )
    elif supervision_type == "filtered_cot":
        instruction = (
            "Solve the problem in at most 3 short numbered steps. "
            "Keep the reasoning concise. "
            f"{final_answer_instruction}"
        )
    elif supervision_type == "subgoals_then_answer":
        instruction = (
            "List up to 3 short planning sub-goals, one per line using the format "
            "'Sub-goal N: ...'. "
            f"Then {final_answer_instruction[0].lower() + final_answer_instruction[1:]}"
        )
    elif supervision_type == "brief_reasoning_then_answer":
        instruction = (
            "Solve the problem using at most 2 short reasoning lines. "
            "Do not use tags, bullets, or headings. "
            f"{final_answer_instruction}"
        )
    elif supervision_type == "verifier_yes_no":
        instruction = "Reply with yes or no only."
    else:
        instruction = (
            "Solve the problem briefly. "
            f"{final_answer_instruction}"
        )
    return f"{stripped}\n{instruction}"


def _final_answer_instruction(answer_mode: str) -> str:
    if answer_mode == "choice_letter":
        return (
            "On the last line, write 'Final answer:' followed by the single option letter only "
            "(for example A or I)."
        )
    return "On the last line, write 'Final answer:' followed by the answer only."


def _describe_motif(label: str) -> str:
    descriptions = {
        "equation_setup": "equation setup",
        "ratio_or_proportion": "ratio or proportion reasoning",
        "count_aggregation": "count aggregation",
        "unit_conversion": "unit conversion",
        "temporal_or_age_shift": "temporal or age shift reasoning",
        "reverse_reasoning": "reverse reasoning",
        "arithmetic_finish": "arithmetic finishing step",
        "other_or_unclear": "other or unclear reasoning",
    }
    return descriptions.get(label, label.replace("_", " "))


def _describe_quality(label: str) -> str:
    descriptions = {
        "fragment": "fragment",
        "partial_solution": "partial solution",
        "complete_attempt": "complete attempt",
    }
    return descriptions.get(label, label.replace("_", " "))


def _describe_repair_error(label: str) -> str:
    descriptions = {
        "equation_setup_error": "the setup or variable equation is probably wrong",
        "wrong_base_or_reference_value": "the reasoning likely uses the wrong base value or reference quantity",
        "temporal_shift_error": "the time or age shift is likely handled incorrectly",
        "arithmetic_finish_error": "the final arithmetic or last-step conversion is likely wrong",
        "other": "the attempt is promising but needs a conservative correction",
    }
    return descriptions.get(label, label.replace("_", " "))
