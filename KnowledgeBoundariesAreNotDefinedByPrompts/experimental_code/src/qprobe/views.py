

from __future__ import annotations

from dataclasses import dataclass

from qprobe.errors import ScientificDefinitionError
from qprobe.semantic import PromptSemantics


@dataclass(frozen=True, slots=True)
class PromptView:


    view_id: str
    prompt: str
    answers: tuple[str, ...]
    semantics: PromptSemantics

    def __post_init__(self) -> None:
        if not self.view_id.strip() or not self.prompt.strip():
            raise ScientificDefinitionError("view_id and prompt must be non-empty")
        if len(self.answers) != self.semantics.weight_matrix.shape[0]:
            raise ScientificDefinitionError(
                "Answer count does not match the semantic analysis"
            )

    @property
    def representative_answer(self) -> str:
        return self.answers[self.semantics.representative_index]
