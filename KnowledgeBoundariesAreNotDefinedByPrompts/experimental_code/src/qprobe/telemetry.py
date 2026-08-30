

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping

from qprobe.backends import TokenUsage


class EventKind(str, Enum):
    TARGET_GENERATION = "target_generation"
    PARAPHRASE_GENERATION = "paraphrase_generation"
    EQUIVALENCE_JUDGE = "equivalence_judge"
    CORRECTNESS_JUDGE = "correctness_judge"
    NLI_FORWARD = "nli_forward"


@dataclass(frozen=True, slots=True)
class CallEvent:


    event_kind: EventKind
    model_id: str
    latency_seconds: float
    usage: TokenUsage | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_kind, EventKind):
            raise ValueError("event_kind must be an EventKind")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if (
            not isinstance(self.latency_seconds, (int, float))
            or isinstance(self.latency_seconds, bool)
            or not math.isfinite(self.latency_seconds)
            or self.latency_seconds < 0
        ):
            raise ValueError("latency_seconds cannot be negative")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise ValueError("usage must be TokenUsage or None")
        if not isinstance(self.metadata, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.metadata.items()
        ):
            raise ValueError("metadata must map strings to strings")


@dataclass(slots=True)
class TraceRecorder:


    events: list[CallEvent] = field(default_factory=list)

    def record(self, event: CallEvent) -> None:
        self.events.append(event)

    def count(self, event_kind: EventKind) -> int:
        return sum(event.event_kind == event_kind for event in self.events)

    @property
    def total_tokens(self) -> int:
        return sum(
            event.usage.total_tokens
            for event in self.events
            if event.usage is not None
        )

    @property
    def total_serial_latency_seconds(self) -> float:


        return sum(event.latency_seconds for event in self.events)
