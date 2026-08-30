

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence

from qprobe.errors import ConfigurationError, ModelCallError


@dataclass(frozen=True, slots=True)
class TokenUsage:


    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        values = (self.input_tokens, self.output_tokens)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ModelCallError("Token counts cannot be negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class GenerationRequest:


    prompt: str
    model_id: str
    seed: int
    seed_ref: str
    temperature: float = 1.0
    max_new_tokens: int = 100
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ConfigurationError("Generation prompt must be non-empty")
        if not self.model_id.strip():
            raise ConfigurationError("model_id must be non-empty")
        if not self.seed_ref.strip():
            raise ConfigurationError("seed_ref must be non-empty")
        if self.temperature < 0:
            raise ConfigurationError("temperature cannot be negative")
        if self.max_new_tokens <= 0:
            raise ConfigurationError("max_new_tokens must be positive")


@dataclass(frozen=True, slots=True)
class GenerationResponse:


    text: str
    model_id: str
    finish_reason: str
    usage: TokenUsage
    latency_seconds: float
    provider_request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ModelCallError("Model returned an empty generation")
        if not self.model_id.strip():
            raise ModelCallError("Generation response is missing model_id")
        if not self.finish_reason.strip():
            raise ModelCallError("Generation response is missing finish_reason")
        if self.latency_seconds < 0:
            raise ModelCallError("latency_seconds cannot be negative")


class TextGenerationBackend(Protocol):


    def generate(self, request: GenerationRequest) -> GenerationResponse:
        pass

    def generate_many(
        self, requests: Sequence[GenerationRequest]
    ) -> tuple[GenerationResponse, ...]:
        pass


class NLILabel(str, Enum):


    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True, slots=True)
class NLIRequest:
    premise: str
    hypothesis: str
    model_id: str

    def __post_init__(self) -> None:
        if not self.premise.strip() or not self.hypothesis.strip():
            raise ConfigurationError("NLI texts must be non-empty")
        if not self.model_id.strip():
            raise ConfigurationError("NLI model_id must be non-empty")


@dataclass(frozen=True, slots=True)
class NLIResponse:
    label: NLILabel
    scores: Mapping[NLILabel, float]
    model_id: str
    latency_seconds: float

    def __post_init__(self) -> None:
        if set(self.scores) != set(NLILabel):
            raise ModelCallError("NLI response must contain all three label scores")
        if any(score < 0 for score in self.scores.values()):
            raise ModelCallError("NLI scores cannot be negative")
        total = sum(self.scores.values())
        if abs(total - 1.0) > 1e-6:
            raise ModelCallError("NLI scores must sum to one")
        if self.latency_seconds < 0:
            raise ModelCallError("latency_seconds cannot be negative")


class NLIBackend(Protocol):


    def classify(self, request: NLIRequest) -> NLIResponse:
        pass

    def classify_many(
        self, requests: Sequence[NLIRequest]
    ) -> tuple[NLIResponse, ...]:
        pass
