

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qprobe.backends import NLIBackend, NLILabel, TextGenerationBackend
from qprobe.config import ModelRunConfig
from qprobe.datasets import DatasetExample
from qprobe.decision import (
    DecisionKind,
    MethodDecision,
    MethodVariant,
    decide_original_view,
    decide_recovery,
)
from qprobe.errors import ConfigurationError, ScientificDefinitionError
from qprobe.evidence import (
    ExampleEvidence,
    MethodOutput,
    OutputKind,
    ParaphraseEvidenceProvenance,
    PolicyCallCounts,
    ProbeEvidence,
    PromptEvidence,
    ROLE_ORDER,
    ReplayResult,
    SharedParaphraseEvidence,
    TargetEvidenceProvenance,
    validate_variant,
)
from qprobe.generation import SamplingPlan, sample_prompt
from qprobe.nli import classify_all_directions
from qprobe.paraphrasing import (
    AuxiliaryCallPlan,
    generate_role_paraphrases,
    verify_role_paraphrases,
)
from qprobe.prompts import ParaphraseRole
from qprobe.semantic import analyze_prompt_semantics
from qprobe.telemetry import TraceRecorder
from qprobe.views import PromptView


@dataclass(frozen=True, slots=True)
class PipelinePlans:


    models: ModelRunConfig
    target_sampling: SamplingPlan
    paraphrase: AuxiliaryCallPlan
    equivalence_judge: AuxiliaryCallPlan

    def __post_init__(self) -> None:
        if not isinstance(self.models, ModelRunConfig):
            raise ConfigurationError("models must be a ModelRunConfig")
        if not isinstance(self.target_sampling, SamplingPlan):
            raise ConfigurationError("target_sampling must be a SamplingPlan")
        if not isinstance(self.paraphrase, AuxiliaryCallPlan):
            raise ConfigurationError("paraphrase must be an AuxiliaryCallPlan")
        if not isinstance(self.equivalence_judge, AuxiliaryCallPlan):
            raise ConfigurationError(
                "equivalence_judge must be an AuxiliaryCallPlan"
            )
        if self.target_sampling.model_id != self.models.target_model_id:
            raise ConfigurationError(
                "Target sampling model does not match the run configuration"
            )
        if (
            self.target_sampling.deployment_ref
            != self.models.target_deployment_ref
        ):
            raise ConfigurationError(
                "Target deployment does not match the run configuration"
            )
        if self.paraphrase.model_id != self.models.auxiliary_model_id:
            raise ConfigurationError(
                "Paraphrase model does not match the run configuration"
            )
        if self.equivalence_judge.model_id != self.models.auxiliary_model_id:
            raise ConfigurationError(
                "Equivalence Judge model does not match the run configuration"
            )
        auxiliary_deployments = (
            self.paraphrase.deployment_ref,
            self.equivalence_judge.deployment_ref,
        )
        if any(
            deployment != self.models.auxiliary_deployment_ref
            for deployment in auxiliary_deployments
        ):
            raise ConfigurationError(
                "Auxiliary deployments do not match the run configuration"
            )
        target_settings = (
            self.target_sampling.sample_count,
            self.target_sampling.temperature,
            self.target_sampling.max_new_tokens,
        )
        if target_settings != (10, 1.0, 100):
            raise ConfigurationError(
                "Target sampling settings must be 10, 1.0, 100"
            )
        if (self.paraphrase.temperature, self.paraphrase.max_new_tokens) != (
            1.0,
            100,
        ):
            raise ConfigurationError(
                "Paraphrase settings must be temperature 1.0 and 100 output tokens"
            )
        if (
            self.equivalence_judge.temperature,
            self.equivalence_judge.max_new_tokens,
        ) != (0.0, 16):
            raise ConfigurationError(
                "Equivalence Judge settings must be 0.0 and 16 output tokens"
            )


@dataclass(frozen=True, slots=True)
class PipelineBackends:


    target_generation: TextGenerationBackend
    nli: NLIBackend
    paraphrase_generation: TextGenerationBackend
    equivalence_judge: TextGenerationBackend


def _target_provenance(plans: PipelinePlans) -> TargetEvidenceProvenance:
    return TargetEvidenceProvenance(
        target_model_id=plans.models.target_model_id,
        target_deployment_ref=plans.models.target_deployment_ref,
        generation_seed_ref=plans.target_sampling.seed_ref,
        nli_model_id=plans.models.nli_model_id,
        nli_checkpoint_ref=plans.models.nli_checkpoint_ref,
    )


def _paraphrase_provenance(
    plans: PipelinePlans,
    *,
    include_equivalence: bool,
) -> ParaphraseEvidenceProvenance:
    return ParaphraseEvidenceProvenance(
        paraphrase_model_id=plans.paraphrase.model_id,
        paraphrase_deployment_ref=plans.paraphrase.deployment_ref,
        paraphrase_seed_ref=plans.paraphrase.seed_ref,
        equivalence_model_id=(
            plans.equivalence_judge.model_id if include_equivalence else None
        ),
        equivalence_deployment_ref=(
            plans.equivalence_judge.deployment_ref if include_equivalence else None
        ),
        equivalence_seed_ref=(
            plans.equivalence_judge.seed_ref if include_equivalence else None
        ),
    )


def _validate_collection_inputs(
    example: DatasetExample,
    plans: PipelinePlans,
    backends: PipelineBackends,
) -> None:
    if not isinstance(example, DatasetExample):
        raise ConfigurationError("example must be a DatasetExample")
    if not isinstance(plans, PipelinePlans):
        raise ConfigurationError("plans must be PipelinePlans")
    if not isinstance(backends, PipelineBackends):
        raise ConfigurationError("backends must be PipelineBackends")


def _validate_shared_paraphrase_identity(
    shared: SharedParaphraseEvidence,
    plans: PipelinePlans,
) -> None:
    expected = _paraphrase_provenance(plans, include_equivalence=False)
    actual_identity = (
        shared.provenance.paraphrase_model_id,
        shared.provenance.paraphrase_deployment_ref,
        shared.provenance.paraphrase_seed_ref,
    )
    expected_identity = (
        expected.paraphrase_model_id,
        expected.paraphrase_deployment_ref,
        expected.paraphrase_seed_ref,
    )
    if actual_identity != expected_identity:
        raise ScientificDefinitionError(
            "Shared paraphrase provenance does not match the run"
        )


def _validate_shared_for_run(
    shared: SharedParaphraseEvidence,
    example: DatasetExample,
    plans: PipelinePlans,
    variant: MethodVariant,
) -> None:
    if not isinstance(shared, SharedParaphraseEvidence):
        raise ConfigurationError(
            "shared_paraphrases must be SharedParaphraseEvidence"
        )
    if shared.example_id != example.sample_id:
        raise ScientificDefinitionError(
            "Shared paraphrase sample ID does not match the example"
        )
    if shared.original_question != example.question:
        raise ScientificDefinitionError(
            "Shared paraphrase question does not match the example"
        )
    _validate_shared_paraphrase_identity(shared, plans)
    expected = _paraphrase_provenance(
        plans,
        include_equivalence=variant != MethodVariant.VARIANT_B_NO_EQUIV,
    )
    actual = shared.provenance
    if variant in {MethodVariant.VARIANT_A, MethodVariant.VARIANT_B}:
        equivalence_identity = (
            actual.equivalence_model_id,
            actual.equivalence_deployment_ref,
            actual.equivalence_seed_ref,
        )
        expected_equivalence_identity = (
            expected.equivalence_model_id,
            expected.equivalence_deployment_ref,
            expected.equivalence_seed_ref,
        )
        if equivalence_identity != expected_equivalence_identity:
            raise ScientificDefinitionError(
                "Shared equivalence provenance does not match the run"
            )


def collect_prompt_evidence(
    *,
    prompt: str,
    view_id: str,
    role: ParaphraseRole | None,
    plans: PipelinePlans,
    backends: PipelineBackends,
    recorder: TraceRecorder | None = None,
) -> PromptEvidence:


    if not isinstance(plans, PipelinePlans) or not isinstance(
        backends, PipelineBackends
    ):
        raise ConfigurationError("Invalid pipeline plans or backends")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigurationError("prompt must be a non-empty string")
    if not isinstance(view_id, str) or not view_id.strip():
        raise ConfigurationError("view_id must be a non-empty string")
    if role is not None and not isinstance(role, ParaphraseRole):
        raise ConfigurationError("role must be a ParaphraseRole or None")
    answers = sample_prompt(
        prompt,
        prompt_id=view_id,
        plan=plans.target_sampling,
        backend=backends.target_generation,
        recorder=recorder,
        metadata={"view_id": view_id},
    )
    labels = classify_all_directions(
        answers,
        backend=backends.nli,
        model_id=plans.models.nli_model_id,
        recorder=recorder,
        metadata={"view_id": view_id, "nli_stage": "within_view"},
        checkpoint_ref=plans.models.nli_checkpoint_ref,
    )
    return PromptEvidence(
        role=role,
        view=PromptView(
            view_id=view_id,
            prompt=prompt,
            answers=answers,
            semantics=analyze_prompt_semantics(labels),
        ),
        directional_labels=labels,
    )


def collect_shared_paraphrase_evidence(
    example: DatasetExample,
    *,
    require_equivalence: bool,
    plans: PipelinePlans,
    backends: PipelineBackends,
    recorder: TraceRecorder | None = None,
) -> SharedParaphraseEvidence:


    _validate_collection_inputs(example, plans, backends)
    if not isinstance(require_equivalence, bool):
        raise ConfigurationError("require_equivalence must be boolean")
    candidates = generate_role_paraphrases(
        example.question,
        example_id=example.sample_id,
        plan=plans.paraphrase,
        backend=backends.paraphrase_generation,
        recorder=recorder,
    )
    verified = None
    if require_equivalence:
        verified = verify_role_paraphrases(
            example.question,
            candidates,
            example_id=example.sample_id,
            plan=plans.equivalence_judge,
            backend=backends.equivalence_judge,
            recorder=recorder,
        )
    return SharedParaphraseEvidence(
        example_id=example.sample_id,
        original_question=example.question,
        provenance=_paraphrase_provenance(
            plans, include_equivalence=require_equivalence
        ),
        candidates=candidates,
        verified=verified,
    )


def collect_probe_evidence(
    shared: SharedParaphraseEvidence,
    roles: Sequence[ParaphraseRole],
    *,
    plans: PipelinePlans,
    backends: PipelineBackends,
    recorder: TraceRecorder | None = None,
) -> ProbeEvidence:


    if not isinstance(shared, SharedParaphraseEvidence):
        raise ConfigurationError("shared must be SharedParaphraseEvidence")
    if not isinstance(plans, PipelinePlans) or not isinstance(
        backends, PipelineBackends
    ):
        raise ConfigurationError("Invalid pipeline plans or backends")
    _validate_shared_paraphrase_identity(shared, plans)
    if not all(isinstance(role, ParaphraseRole) for role in roles):
        raise ConfigurationError("Probe roles must be ParaphraseRole values")
    if len(set(roles)) != len(roles):
        raise ConfigurationError("Probe roles must be unique")
    if tuple(sorted(roles, key=lambda role: ROLE_ORDER[role])) != tuple(roles):
        raise ConfigurationError("Probe roles must use fixed role order")
    candidates_by_role = {candidate.role: candidate for candidate in shared.candidates}
    if any(role not in candidates_by_role for role in roles):
        raise ConfigurationError("Probe role is missing from paraphrase evidence")

    prompt_views = tuple(
        collect_prompt_evidence(
            prompt=candidates_by_role[role].prompt,
            view_id=f"{shared.example_id}:role:{role.value}",
            role=role,
            plans=plans,
            backends=backends,
            recorder=recorder,
        )
        for role in roles
    )
    if not prompt_views:
        return ProbeEvidence(prompt_views=(), representative_labels=())
    representative_answers = tuple(
        item.view.representative_answer for item in prompt_views
    )
    representative_labels = classify_all_directions(
        representative_answers,
        backend=backends.nli,
        model_id=plans.models.nli_model_id,
        recorder=recorder,
        metadata={
            "example_id": shared.example_id,
            "nli_stage": "representative_answers",
        },
        checkpoint_ref=plans.models.nli_checkpoint_ref,
    )
    return ProbeEvidence(
        prompt_views=prompt_views,
        representative_labels=representative_labels,
    )


def collect_full_evidence(
    example: DatasetExample,
    *,
    variant: MethodVariant,
    plans: PipelinePlans,
    backends: PipelineBackends,
    recorder: TraceRecorder | None = None,
    shared_paraphrases: SharedParaphraseEvidence | None = None,
) -> ExampleEvidence:


    validate_variant(variant)
    _validate_collection_inputs(example, plans, backends)
    if shared_paraphrases is not None:
        _validate_shared_for_run(shared_paraphrases, example, plans, variant)
    original = collect_prompt_evidence(
        prompt=example.question,
        view_id=f"{example.sample_id}:original",
        role=None,
        plans=plans,
        backends=backends,
        recorder=recorder,
    )
    provenance = _target_provenance(plans)
    if variant == MethodVariant.STAGE1_ONLY:
        return ExampleEvidence(
            example=example,
            provenance=provenance,
            original=original,
        )
    shared = shared_paraphrases or collect_shared_paraphrase_evidence(
        example,
        require_equivalence=variant != MethodVariant.VARIANT_B_NO_EQUIV,
        plans=plans,
        backends=backends,
        recorder=recorder,
    )
    roles = shared.roles_for_variant(variant)
    probe = collect_probe_evidence(
        shared,
        roles,
        plans=plans,
        backends=backends,
        recorder=recorder,
    )
    return ExampleEvidence(
        example=example,
        provenance=provenance,
        original=original,
        shared_paraphrases=shared,
        probe=probe,
    )


def _slice_representative_labels(
    probe: ProbeEvidence,
    selected_indices: Sequence[int],
) -> tuple[tuple[NLILabel | None, ...], ...]:
    return tuple(
        tuple(
            None
            if left == right
            else probe.representative_labels[source_left][source_right]
            for right, source_right in enumerate(selected_indices)
        )
        for left, source_left in enumerate(selected_indices)
    )


def _policy_call_counts(
    evidence: ExampleEvidence,
    variant: MethodVariant,
    entered_probe: bool,
    retained_role_count: int,
) -> PolicyCallCounts:
    sample_count = len(evidence.original.view.answers)
    if not entered_probe:
        return PolicyCallCounts(sample_count, 0, 0)
    return PolicyCallCounts(
        target_generation=sample_count * (1 + retained_role_count),
        paraphrase_generation=len(tuple(ParaphraseRole)),
        equivalence_judge=(
            0
            if variant == MethodVariant.VARIANT_B_NO_EQUIV
            else len(tuple(ParaphraseRole))
        ),
    )


def _build_replay_result(
    evidence: ExampleEvidence,
    *,
    variant: MethodVariant,
    decision: MethodDecision,
    entered_probe: bool,
    retained_role_count: int,
) -> ReplayResult:
    if decision.kind == DecisionKind.RECOVERED:
        if evidence.probe is None or decision.selected_view_id is None:
            raise ScientificDefinitionError(
                "Recovered decision is missing selected probe evidence"
            )
        selected = next(
            (
                item.view
                for item in evidence.probe.prompt_views
                if item.view.view_id == decision.selected_view_id
            ),
            None,
        )
        if selected is None:
            raise ScientificDefinitionError(
                "Recovered decision selected an unknown probe view"
            )
        evaluation_candidate = selected.representative_answer
        u_final = selected.semantics.uncertainty
    else:
        evaluation_candidate = evidence.original.view.representative_answer
        u_final = evidence.original.view.semantics.uncertainty

    output = (
        MethodOutput(kind=OutputKind.ABSTAIN, answer=None)
        if decision.kind == DecisionKind.ABSTAIN
        else MethodOutput(kind=OutputKind.ANSWER, answer=decision.answer)
    )
    return ReplayResult(
        decision=decision,
        output=output,
        evaluation_candidate=evaluation_candidate,
        u_final=u_final,
        confidence=-u_final,
        policy_calls=_policy_call_counts(
            evidence,
            variant,
            entered_probe,
            retained_role_count,
        ),
        entered_paraphrase_probe=entered_probe,
    )


def replay_decision(
    evidence: ExampleEvidence,
    *,
    variant: MethodVariant,
    uncertainty_threshold: float,
) -> ReplayResult:


    if not isinstance(evidence, ExampleEvidence):
        raise ConfigurationError("evidence must be ExampleEvidence")
    validate_variant(variant)
    terminal = decide_original_view(
        evidence.original.view,
        variant=variant,
        uncertainty_threshold=uncertainty_threshold,
    )
    if terminal is not None:
        return _build_replay_result(
            evidence,
            variant=variant,
            decision=terminal,
            entered_probe=False,
            retained_role_count=0,
        )
    if evidence.shared_paraphrases is None or evidence.probe is None:
        raise ScientificDefinitionError(
            "Paraphrase and probe evidence are required for policy replay"
        )

    required_roles = evidence.shared_paraphrases.roles_for_variant(variant)
    available_indices = {
        item.role: index for index, item in enumerate(evidence.probe.prompt_views)
    }
    missing_roles = tuple(role for role in required_roles if role not in available_indices)
    if missing_roles:
        raise ScientificDefinitionError(
            f"Probe evidence is missing roles: {[role.value for role in missing_roles]}"
        )
    selected_indices = tuple(available_indices[role] for role in required_roles)
    selected_views = tuple(
        evidence.probe.prompt_views[index].view for index in selected_indices
    )
    if not selected_views:
        decision = MethodDecision(
            kind=DecisionKind.ABSTAIN,
            answer=None,
            selected_view_id=None,
            reason="no_retained_paraphrase_views",
        )
    else:
        decision = decide_recovery(
            selected_views,
            _slice_representative_labels(evidence.probe, selected_indices),
            uncertainty_threshold=uncertainty_threshold,
        )
    return _build_replay_result(
        evidence,
        variant=variant,
        decision=decision,
        entered_probe=True,
        retained_role_count=len(required_roles),
    )


def execute_example(
    example: DatasetExample,
    *,
    variant: MethodVariant,
    uncertainty_threshold: float,
    plans: PipelinePlans,
    backends: PipelineBackends,
    recorder: TraceRecorder | None = None,
    shared_paraphrases: SharedParaphraseEvidence | None = None,
) -> tuple[ExampleEvidence, ReplayResult]:


    validate_variant(variant)
    _validate_collection_inputs(example, plans, backends)
    if shared_paraphrases is not None:
        _validate_shared_for_run(shared_paraphrases, example, plans, variant)
    provenance = _target_provenance(plans)
    original = collect_prompt_evidence(
        prompt=example.question,
        view_id=f"{example.sample_id}:original",
        role=None,
        plans=plans,
        backends=backends,
        recorder=recorder,
    )
    partial = ExampleEvidence(
        example=example,
        provenance=provenance,
        original=original,
    )
    terminal = decide_original_view(
        original.view,
        variant=variant,
        uncertainty_threshold=uncertainty_threshold,
    )
    if terminal is not None:
        return partial, _build_replay_result(
            partial,
            variant=variant,
            decision=terminal,
            entered_probe=False,
            retained_role_count=0,
        )

    shared = shared_paraphrases or collect_shared_paraphrase_evidence(
        example,
        require_equivalence=variant != MethodVariant.VARIANT_B_NO_EQUIV,
        plans=plans,
        backends=backends,
        recorder=recorder,
    )
    roles = shared.roles_for_variant(variant)
    probe = collect_probe_evidence(
        shared,
        roles,
        plans=plans,
        backends=backends,
        recorder=recorder,
    )
    complete = ExampleEvidence(
        example=example,
        provenance=provenance,
        original=original,
        shared_paraphrases=shared,
        probe=probe,
    )
    return complete, replay_decision(
        complete,
        variant=variant,
        uncertainty_threshold=uncertainty_threshold,
    )
