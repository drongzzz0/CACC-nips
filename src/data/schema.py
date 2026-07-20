from dataclasses import asdict, dataclass, field


@dataclass
class TeacherTraceRecord:
    example_id: str
    dataset: str
    problem: str
    gold_answer: str
    teacher_model: str
    teacher_trace: str
    teacher_final_answer: str
    generation_prompt: str
    trace_valid: bool
    answer_mode: str = "numeric"
    choices: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessedTraceRecord:
    example_id: str
    dataset: str
    problem: str
    gold_answer: str
    teacher_model: str
    teacher_trace: str
    filtered_cot: str
    brief_reasoning: str
    subgoals: list[str]
    teacher_final_answer: str
    trace_valid: bool
    answer_mode: str = "numeric"
    choices: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SFTExample:
    example_id: str
    dataset: str
    prompt: str
    response: str
    supervision_type: str
    gold_answer: str
    answer_mode: str = "numeric"
    choices: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerifierCandidateSet:
    example_id: str
    dataset: str
    problem: str
    gold_answer: str
    candidates: list[str]
    answer_mode: str = "numeric"
    choices: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
