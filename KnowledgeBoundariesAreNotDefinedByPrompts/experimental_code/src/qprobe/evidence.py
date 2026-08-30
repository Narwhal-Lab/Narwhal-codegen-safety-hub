

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from qprobe.config import NLI_MODEL_ID, TARGET_MODEL_IDS
from qprobe.datasets import DatasetExample
from qprobe.decision import MethodDecision, MethodVariant
from qprobe.errors import ConfigurationError, ScientificDefinitionError
from qprobe.paraphrasing import RoleParaphrase, VerifiedParaphrase
from qprobe.prompts import ParaphraseRole
from qprobe.semantic import (
    DirectionalLabels,
    analyze_prompt_semantics,
    validate_directional_labels,
)
from qprobe.views import PromptView


ROLE_ORDER = {role: index for index, role in enumerate(ParaphraseRole)}


def _require_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ScientificDefinitionError(f"{field_name} must be a non-empty string")


def validate_variant(variant: MethodVariant) -> None:


    if not isinstance(variant, MethodVariant):
        raise ConfigurationError("variant must be a MethodVariant")


@dataclass(frozen=True, slots=True)
class TargetEvidenceProvenance:


    target_model_id: str
    target_deployment_ref: str
    generation_seed_ref: str
    nli_model_id: str
    nli_checkpoint_ref: str

    def __post_init__(self) -> None:
        if self.target_model_id not in TARGET_MODEL_IDS:
            raise ScientificDefinitionError(
                f"Unsupported target model: {self.target_model_id!r}"
            )
        _require_non_empty_string(
            self.target_deployment_ref, "target_deployment_ref"
        )
        _require_non_empty_string(self.generation_seed_ref, "generation_seed_ref")
        if self.nli_model_id != NLI_MODEL_ID:
            raise ScientificDefinitionError(f"NLI model must be {NLI_MODEL_ID!r}")
        _require_non_empty_string(self.nli_checkpoint_ref, "nli_checkpoint_ref")


@dataclass(frozen=True, slots=True)
class ParaphraseEvidenceProvenance:


    paraphrase_model_id: str
    paraphrase_deployment_ref: str
    paraphrase_seed_ref: str
    equivalence_model_id: str | None
    equivalence_deployment_ref: str | None
    equivalence_seed_ref: str | None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.paraphrase_model_id, "paraphrase_model_id")
        _require_non_empty_string(
            self.paraphrase_deployment_ref, "paraphrase_deployment_ref"
        )
        _require_non_empty_string(self.paraphrase_seed_ref, "paraphrase_seed_ref")
        equivalence_values = (
            self.equivalence_model_id,
            self.equivalence_deployment_ref,
            self.equivalence_seed_ref,
        )
        if all(value is None for value in equivalence_values):
            return
        if any(value is None for value in equivalence_values):
            raise ScientificDefinitionError(
                "Equivalence provenance must be entirely present or absent"
            )
        _require_non_empty_string(self.equivalence_model_id, "equivalence_model_id")
        _require_non_empty_string(
            self.equivalence_deployment_ref, "equivalence_deployment_ref"
        )
        _require_non_empty_string(self.equivalence_seed_ref, "equivalence_seed_ref")

    @property
    def has_equivalence(self) -> bool:
        return self.equivalence_model_id is not None


@dataclass(frozen=True, slots=True)
class PromptEvidence:


    role: ParaphraseRole | None
    view: PromptView
    directional_labels: DirectionalLabels

    def __post_init__(self) -> None:
        if self.role is not None and not isinstance(self.role, ParaphraseRole):
            raise ScientificDefinitionError(
                "Prompt evidence role must be a ParaphraseRole or None"
            )
        if not isinstance(self.view, PromptView):
            raise ScientificDefinitionError("view must be a PromptView")
        size = validate_directional_labels(self.directional_labels)
        if size != len(self.view.answers):
            raise ScientificDefinitionError(
                "Prompt evidence labels do not match sampled answers"
            )
        expected = analyze_prompt_semantics(self.directional_labels)
        actual = self.view.semantics
        matrices_match = all(
            np.allclose(actual_matrix, expected_matrix, rtol=0.0, atol=1e-12)
            for actual_matrix, expected_matrix in (
                (actual.weight_matrix, expected.weight_matrix),
                (actual.laplacian, expected.laplacian),
                (actual.normalized_kernel, expected.normalized_kernel),
            )
        )
        scalar_values_match = all(
            math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12)
            for actual_value, expected_value in (
                (actual.uncertainty, expected.uncertainty),
                (actual.dominant_strength, expected.dominant_strength),
                (actual.competitor_strength, expected.competitor_strength),
            )
        )
        structure_matches = (
            actual.modes == expected.modes
            and actual.dominant_mode_index == expected.dominant_mode_index
        )
        if not matrices_match or not scalar_values_match or not structure_matches:
            raise ScientificDefinitionError(
                "Prompt semantics do not match the directional NLI labels"
            )


@dataclass(frozen=True, slots=True)
class SharedParaphraseEvidence:


    example_id: str
    original_question: str
    provenance: ParaphraseEvidenceProvenance
    candidates: tuple[RoleParaphrase, ...]
    verified: tuple[VerifiedParaphrase, ...] | None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.example_id, "example_id")
        _require_non_empty_string(self.original_question, "original_question")
        if not isinstance(self.provenance, ParaphraseEvidenceProvenance):
            raise ScientificDefinitionError(
                "provenance must be ParaphraseEvidenceProvenance"
            )
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(candidate, RoleParaphrase) for candidate in self.candidates
        ):
            raise ScientificDefinitionError(
                "Paraphrase candidates must be RoleParaphrase records"
            )
        roles = tuple(candidate.role for candidate in self.candidates)
        if roles != tuple(ParaphraseRole):
            raise ScientificDefinitionError(
                "Paraphrase candidates must contain all roles in fixed order"
            )
        if self.verified is None:
            if self.provenance.has_equivalence:
                raise ScientificDefinitionError(
                    "Equivalence provenance requires verified paraphrases"
                )
            return
        if not self.provenance.has_equivalence:
            raise ScientificDefinitionError(
                "Verified paraphrases require equivalence provenance"
            )
        if not isinstance(self.verified, tuple) or not all(
            isinstance(item, VerifiedParaphrase) for item in self.verified
        ):
            raise ScientificDefinitionError(
                "Verified paraphrases must be VerifiedParaphrase records"
            )
        verified_roles = tuple(item.role for item in self.verified)
        if verified_roles != roles:
            raise ScientificDefinitionError(
                "Verified paraphrases must match candidate role order"
            )
        for candidate, verified in zip(self.candidates, self.verified):
            if candidate.prompt != verified.prompt:
                raise ScientificDefinitionError(
                    "Verified paraphrase text does not match its candidate"
                )

    def roles_for_variant(self, variant: MethodVariant) -> tuple[ParaphraseRole, ...]:


        validate_variant(variant)
        if variant == MethodVariant.STAGE1_ONLY:
            return ()
        if variant == MethodVariant.VARIANT_B_NO_EQUIV:
            return tuple(candidate.role for candidate in self.candidates)
        if self.verified is None:
            raise ScientificDefinitionError(
                "Variant A and Variant B require equivalence evidence"
            )
        return tuple(item.role for item in self.verified if item.equivalent)


@dataclass(frozen=True, slots=True)
class ProbeEvidence:


    prompt_views: tuple[PromptEvidence, ...]
    representative_labels: DirectionalLabels

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_views, tuple) or not all(
            isinstance(item, PromptEvidence) for item in self.prompt_views
        ):
            raise ScientificDefinitionError(
                "prompt_views must be PromptEvidence records"
            )
        roles = tuple(item.role for item in self.prompt_views)
        if any(not isinstance(role, ParaphraseRole) for role in roles):
            raise ScientificDefinitionError(
                "Probe views must have valid paraphrase roles"
            )
        if len(set(roles)) != len(roles):
            raise ScientificDefinitionError("Probe view roles must be unique")
        if tuple(sorted(roles, key=lambda role: ROLE_ORDER[role])) != roles:
            raise ScientificDefinitionError("Probe views must use fixed role order")
        if not self.prompt_views:
            if self.representative_labels:
                raise ScientificDefinitionError(
                    "Empty probe evidence cannot contain representative labels"
                )
            return
        if validate_directional_labels(self.representative_labels) != len(
            self.prompt_views
        ):
            raise ScientificDefinitionError(
                "Representative labels do not match probe views"
            )


@dataclass(frozen=True, slots=True)
class ExampleEvidence:


    example: DatasetExample
    provenance: TargetEvidenceProvenance
    original: PromptEvidence
    shared_paraphrases: SharedParaphraseEvidence | None = None
    probe: ProbeEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example, DatasetExample):
            raise ScientificDefinitionError("example must be a DatasetExample")
        if not isinstance(self.provenance, TargetEvidenceProvenance):
            raise ScientificDefinitionError(
                "provenance must be TargetEvidenceProvenance"
            )
        if not isinstance(self.original, PromptEvidence):
            raise ScientificDefinitionError("original must be PromptEvidence")
        if self.original.role is not None:
            raise ScientificDefinitionError("Original prompt evidence cannot have a role")
        expected_original_id = f"{self.example.sample_id}:original"
        if self.original.view.view_id != expected_original_id:
            raise ScientificDefinitionError("Original view ID does not match the example")
        if self.original.view.prompt != self.example.question:
            raise ScientificDefinitionError(
                "Original prompt evidence does not match the dataset question"
            )
        if self.shared_paraphrases is not None:
            if not isinstance(self.shared_paraphrases, SharedParaphraseEvidence):
                raise ScientificDefinitionError(
                    "shared_paraphrases must be SharedParaphraseEvidence"
                )
            if self.shared_paraphrases.example_id != self.example.sample_id:
                raise ScientificDefinitionError(
                    "Paraphrase evidence sample ID does not match the example"
                )
            if self.shared_paraphrases.original_question != self.example.question:
                raise ScientificDefinitionError(
                    "Paraphrase evidence question does not match the example"
                )
        if self.probe is not None and not isinstance(self.probe, ProbeEvidence):
            raise ScientificDefinitionError("probe must be ProbeEvidence or None")
        if self.probe is not None and self.shared_paraphrases is None:
            raise ScientificDefinitionError(
                "Probe evidence requires shared paraphrase evidence"
            )
        if self.probe is not None and self.shared_paraphrases is not None:
            prompts_by_role = {
                candidate.role: candidate.prompt
                for candidate in self.shared_paraphrases.candidates
            }
            sample_count = len(self.original.view.answers)
            view_ids: set[str] = set()
            for prompt_evidence in self.probe.prompt_views:
                role = prompt_evidence.role
                expected_view_id = f"{self.example.sample_id}:role:{role.value}"
                if prompt_evidence.view.view_id != expected_view_id:
                    raise ScientificDefinitionError(
                        "Probe view ID does not match its example and role"
                    )
                if prompt_evidence.view.view_id in view_ids:
                    raise ScientificDefinitionError("Probe view IDs must be unique")
                view_ids.add(prompt_evidence.view.view_id)
                if prompt_evidence.view.prompt != prompts_by_role[role]:
                    raise ScientificDefinitionError(
                        "Probe prompt does not match shared paraphrase evidence"
                    )
                if len(prompt_evidence.view.answers) != sample_count:
                    raise ScientificDefinitionError(
                        "All target prompt views must use the same sample count"
                    )


class OutputKind(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class MethodOutput:


    kind: OutputKind
    answer: str | None

    def __post_init__(self) -> None:
        if self.kind == OutputKind.ANSWER:
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ScientificDefinitionError("Answer output must be non-empty")
        elif self.kind == OutputKind.ABSTAIN:
            if self.answer is not None:
                raise ScientificDefinitionError("Abstention cannot expose an answer")
        else:
            raise ScientificDefinitionError(f"Unsupported output kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class PolicyCallCounts:


    target_generation: int
    paraphrase_generation: int
    equivalence_judge: int

    def __post_init__(self) -> None:
        values = (
            self.target_generation,
            self.paraphrase_generation,
            self.equivalence_judge,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ScientificDefinitionError("Policy call counts must be non-negative")

    @property
    def total_method_calls(self) -> int:
        return (
            self.target_generation
            + self.paraphrase_generation
            + self.equivalence_judge
        )


@dataclass(frozen=True, slots=True)
class ReplayResult:


    decision: MethodDecision
    output: MethodOutput
    evaluation_candidate: str
    u_final: float
    confidence: float
    policy_calls: PolicyCallCounts
    entered_paraphrase_probe: bool

    def __post_init__(self) -> None:
        if not isinstance(self.decision, MethodDecision):
            raise ScientificDefinitionError("decision must be a MethodDecision")
        if not isinstance(self.output, MethodOutput):
            raise ScientificDefinitionError("output must be a MethodOutput")
        _require_non_empty_string(self.evaluation_candidate, "evaluation_candidate")
        if not isinstance(self.u_final, (int, float)) or not math.isfinite(self.u_final):
            raise ScientificDefinitionError("u_final must be finite")
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(
            self.confidence
        ):
            raise ScientificDefinitionError("confidence must be finite")
        if self.confidence != -self.u_final:
            raise ScientificDefinitionError("confidence must equal negative u_final")
        if not isinstance(self.policy_calls, PolicyCallCounts):
            raise ScientificDefinitionError("policy_calls must be PolicyCallCounts")
        if not isinstance(self.entered_paraphrase_probe, bool):
            raise ScientificDefinitionError(
                "entered_paraphrase_probe must be boolean"
            )
