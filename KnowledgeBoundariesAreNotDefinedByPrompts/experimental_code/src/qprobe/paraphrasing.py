

from __future__ import annotations

from dataclasses import dataclass

from qprobe.backends import GenerationRequest, TextGenerationBackend
from qprobe.errors import ConfigurationError, ModelCallError, ScientificDefinitionError
from qprobe.generation import derive_call_seed
from qprobe.prompts import (
    ParaphraseRole,
    equivalence_prompt,
    paraphrase_prompt,
    parse_equivalence_label,
)
from qprobe.telemetry import CallEvent, EventKind, TraceRecorder


@dataclass(frozen=True, slots=True)
class AuxiliaryCallPlan:


    model_id: str
    base_seed: int
    seed_ref: str
    temperature: float
    max_new_tokens: int
    deployment_ref: str = "controlled-claude-deployment-ref"

    def __post_init__(self) -> None:
        identity_values = (self.model_id, self.deployment_ref, self.seed_ref)
        if any(
            not isinstance(value, str) or not value.strip()
            for value in identity_values
        ):
            raise ConfigurationError(
                "model_id, deployment_ref, and seed_ref must be non-empty"
            )
        if self.temperature < 0:
            raise ConfigurationError("temperature cannot be negative")
        if self.max_new_tokens <= 0:
            raise ConfigurationError("max_new_tokens must be positive")


@dataclass(frozen=True, slots=True)
class RoleParaphrase:
    role: ParaphraseRole
    prompt: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, ParaphraseRole):
            raise ScientificDefinitionError("role must be a ParaphraseRole")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ModelCallError(f"Empty paraphrase for role {self.role.value}")


@dataclass(frozen=True, slots=True)
class VerifiedParaphrase:
    role: ParaphraseRole
    prompt: str
    equivalent: bool

    def __post_init__(self) -> None:
        if not isinstance(self.role, ParaphraseRole):
            raise ScientificDefinitionError("role must be a ParaphraseRole")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ScientificDefinitionError("Verified paraphrase must be non-empty")
        if not isinstance(self.equivalent, bool):
            raise ScientificDefinitionError("equivalent must be boolean")


def generate_role_paraphrases(
    original_prompt: str,
    *,
    example_id: str,
    plan: AuxiliaryCallPlan,
    backend: TextGenerationBackend,
    recorder: TraceRecorder | None = None,
) -> tuple[RoleParaphrase, ...]:


    roles = tuple(ParaphraseRole)
    requests = tuple(
        GenerationRequest(
            prompt=paraphrase_prompt(original_prompt, role),
            model_id=plan.model_id,
            seed=derive_call_seed(
                plan.base_seed, f"{example_id}:paraphrase:{role.value}", 0
            ),
            seed_ref=plan.seed_ref,
            temperature=plan.temperature,
            max_new_tokens=plan.max_new_tokens,
            metadata={"example_id": example_id, "role": role.value},
        )
        for role in roles
    )
    responses = backend.generate_many(requests)
    if len(responses) != len(requests):
        raise ModelCallError("Paraphraser returned the wrong response count")

    paraphrases: list[RoleParaphrase] = []
    for role, response in zip(roles, responses):
        if response.model_id != plan.model_id:
            raise ModelCallError("Paraphraser returned an unexpected model_id")
        paraphrases.append(RoleParaphrase(role=role, prompt=response.text.strip()))
        if recorder is not None:
            recorder.record(
                CallEvent(
                    event_kind=EventKind.PARAPHRASE_GENERATION,
                    model_id=response.model_id,
                    latency_seconds=response.latency_seconds,
                    usage=response.usage,
                    metadata={
                        "deployment_ref": plan.deployment_ref,
                        "example_id": example_id,
                        "role": role.value,
                        "seed_ref": plan.seed_ref,
                        "finish_reason": response.finish_reason,
                    },
                )
            )
    return tuple(paraphrases)


def verify_role_paraphrases(
    original_prompt: str,
    paraphrases: tuple[RoleParaphrase, ...],
    *,
    example_id: str,
    plan: AuxiliaryCallPlan,
    backend: TextGenerationBackend,
    recorder: TraceRecorder | None = None,
) -> tuple[VerifiedParaphrase, ...]:


    requests = tuple(
        GenerationRequest(
            prompt=equivalence_prompt(original_prompt, item.prompt),
            model_id=plan.model_id,
            seed=derive_call_seed(
                plan.base_seed, f"{example_id}:equivalence:{item.role.value}", 0
            ),
            seed_ref=plan.seed_ref,
            temperature=plan.temperature,
            max_new_tokens=plan.max_new_tokens,
            metadata={"example_id": example_id, "role": item.role.value},
        )
        for item in paraphrases
    )
    responses = backend.generate_many(requests)
    if len(responses) != len(requests):
        raise ModelCallError("Equivalence judge returned the wrong response count")

    verified: list[VerifiedParaphrase] = []
    for item, response in zip(paraphrases, responses):
        if response.model_id != plan.model_id:
            raise ModelCallError("Equivalence judge returned an unexpected model_id")
        equivalent = parse_equivalence_label(response.text)
        verified.append(
            VerifiedParaphrase(
                role=item.role,
                prompt=item.prompt,
                equivalent=equivalent,
            )
        )
        if recorder is not None:
            recorder.record(
                CallEvent(
                    event_kind=EventKind.EQUIVALENCE_JUDGE,
                    model_id=response.model_id,
                    latency_seconds=response.latency_seconds,
                    usage=response.usage,
                    metadata={
                        "deployment_ref": plan.deployment_ref,
                        "example_id": example_id,
                        "role": item.role.value,
                        "seed_ref": plan.seed_ref,
                        "decision": (
                            "EQUIVALENT" if equivalent else "NOT_EQUIVALENT"
                        ),
                    },
                )
            )
    return tuple(verified)
