

from __future__ import annotations

from enum import Enum

from qprobe.errors import ConfigurationError, ModelCallError


class ParaphraseRole(str, Enum):
    LAYPERSON = "Layperson"
    STUDENT = "Student"
    EXPERT = "Expert"
    RESEARCHER = "Researcher"
    WILDCARD = "Wildcard"


ROLE_INSTRUCTIONS: dict[ParaphraseRole, str] = {
    ParaphraseRole.LAYPERSON: (
        "A casual user asking out of mild curiosity; use a short and informal wording."
    ),
    ParaphraseRole.STUDENT: (
        "A learner who does not know the topic well but wants to understand it; "
        "use a clear and standard wording."
    ),
    ParaphraseRole.EXPERT: (
        "A user with relevant domain knowledge; use appropriate field-specific "
        "terminology."
    ),
    ParaphraseRole.RESEARCHER: (
        "A user investigating the topic further; add necessary historical, temporal, "
        "or contextual details only when they preserve the same fact and do not carry "
        "the answer."
    ),
    ParaphraseRole.WILDCARD: (
        "Use a suitable formulation outside the above four roles to avoid homogeneous "
        "paraphrases."
    ),
}


def _role_block() -> str:
    return "\n".join(
        f"{role.value}: {ROLE_INSTRUCTIONS[role]}" for role in ParaphraseRole
    )


def paraphrase_prompt(original_prompt: str, role: ParaphraseRole) -> str:


    if not original_prompt.strip():
        raise ConfigurationError("Original prompt must be non-empty")
    return (
        "Instruction. Rewrite the original question into a semantically equivalent "
        "question for the assigned role. Preserve the queried subject, factual "
        "attribute, expected answer type, and constraint scope. Do not add constraints, "
        "remove constraints, change the queried fact, or introduce answer-bearing "
        "information.\n\n"
        f"Roles.\n{_role_block()}\n\n"
        f"Assigned role: {role.value}\n\n"
        f"Original prompt: {original_prompt}\n\n"
        "Output: One rewritten prompt for the assigned role."
    )


def equivalence_prompt(original_question: str, paraphrased_question: str) -> str:


    if not original_question.strip() or not paraphrased_question.strip():
        raise ConfigurationError("Judge questions must be non-empty")
    return (
        "Instruction. You are an equivalence judge for question paraphrases.\n\n"
        "Given two questions, decide whether they ask about the same underlying fact "
        "and would expect the same answer.\n\n"
        "Output exactly one of the following labels: EQUIVALENT or NOT_EQUIVALENT.\n\n"
        "Reject as NOT_EQUIVALENT if any of the following holds:\n\n"
        "Subject shift: The two questions concern different subjects.\n"
        "Attribute shift: The two questions ask for different attributes of the same "
        "subject.\n"
        "Scope shift: One question is significantly more specific than the other, "
        "e.g., asks for a strict subset of the answer.\n"
        "Answer-type shift: The expected answer types differ, e.g., one expects a "
        "person while the other expects a date.\n\n"
        f"Original question: {original_question}\n\n"
        f"Paraphrased question: {paraphrased_question}\n\n"
        "Decision:"
    )


def parse_equivalence_label(text: str) -> bool:


    label = text.strip()
    if label == "EQUIVALENT":
        return True
    if label == "NOT_EQUIVALENT":
        return False
    raise ModelCallError(f"Invalid equivalence label: {label!r}")
