

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

from qprobe.backends import GenerationRequest, TextGenerationBackend
from qprobe.errors import ConfigurationError, ModelCallError
from qprobe.telemetry import CallEvent, EventKind, TraceRecorder


@dataclass(frozen=True, slots=True)
class SamplingPlan:


    model_id: str
    base_seed: int
    seed_ref: str
    deployment_ref: str = "controlled-target-ref"
    sample_count: int = 10
    temperature: float = 1.0
    max_new_tokens: int = 100

    def __post_init__(self) -> None:
        identity_values = (self.model_id, self.deployment_ref, self.seed_ref)
        if any(
            not isinstance(value, str) or not value.strip()
            for value in identity_values
        ):
            raise ConfigurationError(
                "model_id, deployment_ref, and seed_ref must be non-empty"
            )
        if self.sample_count <= 0:
            raise ConfigurationError("sample_count must be positive")
        if self.temperature <= 0 or self.max_new_tokens <= 0:
            raise ConfigurationError("Generation settings must be positive")


def derive_call_seed(base_seed: int, namespace: str, call_index: int) -> int:


    if not namespace.strip():
        raise ConfigurationError("Seed namespace must be non-empty")
    if call_index < 0:
        raise ConfigurationError("call_index cannot be negative")
    payload = f"{base_seed}\x00{namespace}\x00{call_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_prompt(
    prompt: str,
    *,
    prompt_id: str,
    plan: SamplingPlan,
    backend: TextGenerationBackend,
    recorder: TraceRecorder | None = None,
    metadata: Mapping[str, str] | None = None,
) -> tuple[str, ...]:


    requests = tuple(
        GenerationRequest(
            prompt=prompt,
            model_id=plan.model_id,
            seed=derive_call_seed(plan.base_seed, prompt_id, sample_index),
            seed_ref=plan.seed_ref,
            temperature=plan.temperature,
            max_new_tokens=plan.max_new_tokens,
            metadata=metadata,
        )
        for sample_index in range(plan.sample_count)
    )
    responses = backend.generate_many(requests)
    if len(responses) != len(requests):
        raise ModelCallError("Generation backend returned the wrong response count")

    outputs: list[str] = []
    for sample_index, response in enumerate(responses):
        if response.model_id != plan.model_id:
            raise ModelCallError("Generation backend returned an unexpected model_id")
        outputs.append(response.text.strip())
        if recorder is not None:
            recorder.record(
                CallEvent(
                    event_kind=EventKind.TARGET_GENERATION,
                    model_id=response.model_id,
                    latency_seconds=response.latency_seconds,
                    usage=response.usage,
                    metadata={
                        "deployment_ref": plan.deployment_ref,
                        "prompt_id": prompt_id,
                        "sample_index": str(sample_index),
                        "seed_ref": plan.seed_ref,
                        "finish_reason": response.finish_reason,
                    },
                )
            )
    return tuple(outputs)
