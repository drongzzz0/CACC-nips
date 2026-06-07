from __future__ import annotations

import re
from dataclasses import asdict, dataclass


MOTIF_LABELS = (
    "equation_setup",
    "ratio_or_proportion",
    "count_aggregation",
    "unit_conversion",
    "temporal_or_age_shift",
    "reverse_reasoning",
    "arithmetic_finish",
    "other_or_unclear",
)

QUALITY_LABELS = (
    "fragment",
    "partial_solution",
    "complete_attempt",
)

_FRAGMENT_SUFFIXES = (
    "then",
    "so",
    "because",
    "which",
    "that",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "by",
    "with",
)

_MONEY_UNITS = {"dollar", "dollars", "cent", "cents"}
_LENGTH_UNITS = {"inch", "inches", "foot", "feet", "yard", "yards", "mile", "miles", "meter", "meters"}
_WEIGHT_UNITS = {"ounce", "ounces", "pound", "pounds", "gram", "grams", "kilogram", "kilograms"}
_VOLUME_UNITS = {"cup", "cups", "liter", "liters", "gallon", "gallons"}
_TIME_UNITS = {"minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years"}
_CONTAINER_UNITS = {"bag", "bags", "box", "boxes", "pack", "packs", "bottle", "bottles"}
_ALL_UNIT_GROUPS = {
    "money": _MONEY_UNITS,
    "length": _LENGTH_UNITS,
    "weight": _WEIGHT_UNITS,
    "volume": _VOLUME_UNITS,
    "time": _TIME_UNITS,
    "container": _CONTAINER_UNITS,
}


@dataclass
class MotifMatch:
    label: str
    confidence: float
    matched_cues: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityMatch:
    label: str
    confidence: float
    matched_cues: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidateTag:
    motif: MotifMatch
    quality: QualityMatch

    def to_dict(self) -> dict:
        return {
            "motif": self.motif.to_dict(),
            "quality": self.quality.to_dict(),
        }


def normalize_freeform(text: str) -> str:
    return " ".join(text.strip().lower().split())


def infer_problem_motif(problem_text: str) -> MotifMatch:
    return _infer_motif(problem_text, "")


def infer_candidate_tag(problem_text: str, candidate_text: str) -> CandidateTag:
    return CandidateTag(
        motif=_infer_motif(problem_text, candidate_text),
        quality=_infer_quality(candidate_text),
    )


def _infer_motif(problem_text: str, candidate_text: str) -> MotifMatch:
    normalized_problem = normalize_freeform(problem_text)
    normalized_candidate = normalize_freeform(candidate_text)
    combined = f"{normalized_problem}\n{normalized_candidate}".strip()
    motif_scores: dict[str, list[str]] = {label: [] for label in MOTIF_LABELS[:-1]}

    _extend_if_match(
        motif_scores["equation_setup"],
        normalized_candidate,
        [
            (r"\blet\b", "candidate:let"),
            (r"\bequation\b", "candidate:equation"),
            (r"\bsolve for\b", "candidate:solve_for"),
            (r"\b[xy]\s*=", "candidate:variable_equals"),
            (r"=", "candidate:equals"),
        ],
    )

    _extend_if_match(
        motif_scores["ratio_or_proportion"],
        combined,
        [
            (r"\bhalf\b", "half"),
            (r"\btwice\b", "twice"),
            (r"\bdouble\b", "double"),
            (r"\btriple\b", "triple"),
            (r"\bratio\b", "ratio"),
            (r"\bproportion\b", "proportion"),
            (r"\bpercent(?:age)?\b", "percent"),
            (r"\btimes as much\b", "times_as_much"),
            (r"\bper\b", "per"),
            (r"\beach\b", "each"),
        ],
    )

    _extend_if_match(
        motif_scores["count_aggregation"],
        combined,
        [
            (r"\btotal\b", "total"),
            (r"\baltogether\b", "altogether"),
            (r"\bin all\b", "in_all"),
            (r"\bcombined\b", "combined"),
            (r"\bsum\b", "sum"),
            (r"\bmore\b", "more"),
            (r"\bplus\b", "plus"),
            (r"\btogether\b", "together"),
        ],
    )

    _extend_if_match(
        motif_scores["temporal_or_age_shift"],
        combined,
        [
            (r"\bage\b", "age"),
            (r"\bolder\b", "older"),
            (r"\byounger\b", "younger"),
            (r"\bbefore\b", "before"),
            (r"\bafter\b", "after"),
            (r"\bremaining\b", "remaining"),
            (r"\bfirst\b", "first"),
            (r"\bthen\b", "then"),
        ],
    )

    _extend_if_match(
        motif_scores["reverse_reasoning"],
        combined,
        [
            (r"\boriginally\b", "originally"),
            (r"\bstarted with\b", "started_with"),
            (r"\bat first\b", "at_first"),
            (r"\bleft\b", "left"),
            (r"\bremain(?:ing)?\b", "remain"),
            (r"\bdifference\b", "difference"),
            (r"\bhow many were\b", "how_many_were"),
        ],
    )

    if _has_unit_conversion_signal(normalized_problem, normalized_candidate):
        motif_scores["unit_conversion"].append("unit_groups>=2")
    if _has_arithmetic_finish_signal(normalized_candidate):
        motif_scores["arithmetic_finish"].append("candidate:arithmetic_ops")

    # Bias the label toward what the candidate is actively doing, not only what the problem asks.
    if normalized_candidate:
        if "final answer:" in normalized_candidate:
            motif_scores["arithmetic_finish"].append("candidate:final_answer")
        if normalized_candidate.count("step") >= 2:
            motif_scores["count_aggregation"].append("candidate:multi_step")

    best_label = "other_or_unclear"
    best_cues: list[str] = []
    best_score = 0
    label_priority = {
        "equation_setup": 7,
        "ratio_or_proportion": 6,
        "temporal_or_age_shift": 5,
        "reverse_reasoning": 4,
        "unit_conversion": 3,
        "count_aggregation": 2,
        "arithmetic_finish": 1,
    }
    for label, cues in motif_scores.items():
        score = len(set(cues))
        if score > best_score:
            best_label = label
            best_cues = sorted(set(cues))
            best_score = score
            continue
        if score == best_score and score > 0 and label_priority[label] > label_priority.get(best_label, 0):
            best_label = label
            best_cues = sorted(set(cues))

    if best_score == 0:
        return MotifMatch(label="other_or_unclear", confidence=0.2, matched_cues=[])

    confidence = min(0.95, 0.35 + 0.12 * best_score)
    return MotifMatch(label=best_label, confidence=round(confidence, 4), matched_cues=best_cues)


def _infer_quality(candidate_text: str) -> QualityMatch:
    stripped = candidate_text.strip()
    normalized = normalize_freeform(candidate_text)
    words = stripped.split()
    fragment_cues: list[str] = []
    partial_cues: list[str] = []
    complete_cues: list[str] = []

    if not stripped:
        return QualityMatch(label="fragment", confidence=0.99, matched_cues=["empty"])

    if len(stripped) < 18:
        fragment_cues.append("short_chars")
    if len(words) <= 4:
        fragment_cues.append("short_words")
    if normalized.endswith(_FRAGMENT_SUFFIXES):
        fragment_cues.append("dangling_suffix")
    if re.search(r"\b(step\s*\d+|then|so)\b", normalized):
        partial_cues.append("reasoning_marker")
    if re.search(r"\bfinal answer:\b", normalized):
        complete_cues.append("final_answer")
    if re.search(r"[.!?]$", stripped):
        complete_cues.append("terminal_punctuation")
    if re.search(r"\btherefore\b|\bso the answer\b", normalized):
        complete_cues.append("final_conclusion")
    if re.search(r"-?\d+(?:\.\d+)?", normalized):
        partial_cues.append("contains_number")
    if len(words) >= 12:
        partial_cues.append("long_enough")
    if len(words) >= 20:
        complete_cues.append("substantial_length")

    fragment_score = len(set(fragment_cues))
    partial_score = len(set(partial_cues))
    complete_score = len(set(complete_cues))

    if complete_score >= 2 and complete_score >= partial_score:
        confidence = min(0.95, 0.45 + 0.12 * complete_score)
        return QualityMatch(
            label="complete_attempt",
            confidence=round(confidence, 4),
            matched_cues=sorted(set(complete_cues)),
        )

    if fragment_score >= 2 and fragment_score >= partial_score:
        confidence = min(0.95, 0.45 + 0.12 * fragment_score)
        return QualityMatch(
            label="fragment",
            confidence=round(confidence, 4),
            matched_cues=sorted(set(fragment_cues)),
        )

    confidence = min(0.9, 0.4 + 0.1 * max(partial_score, 1))
    return QualityMatch(
        label="partial_solution",
        confidence=round(confidence, 4),
        matched_cues=sorted(set(partial_cues or ["default_partial"])),
    )


def _extend_if_match(target: list[str], text: str, patterns: list[tuple[str, str]]) -> None:
    for pattern, cue in patterns:
        if re.search(pattern, text):
            target.append(cue)


def _has_unit_conversion_signal(problem_text: str, candidate_text: str) -> bool:
    combined = f"{problem_text} {candidate_text}"
    matched_groups = set()
    distinct_units = set()
    for group_name, units in _ALL_UNIT_GROUPS.items():
        for unit in units:
            if re.search(rf"\b{re.escape(unit)}\b", combined):
                matched_groups.add(group_name)
                distinct_units.add(unit)
    if len(matched_groups) >= 2:
        return True
    if len(distinct_units) >= 2 and re.search(r"\bconvert\b|\bcontains\b|\bper\b|\beach\b", combined):
        return True
    return False


def _has_arithmetic_finish_signal(candidate_text: str) -> bool:
    if not candidate_text:
        return False
    number_matches = re.findall(r"-?\d+(?:\.\d+)?", candidate_text)
    if len(number_matches) >= 2 and re.search(r"[+\-*/=]", candidate_text):
        return True
    if re.search(r"\bsubtract\b|\badd\b|\bmultiply\b|\bdivide\b", candidate_text):
        return True
    return False
