

from dataclasses import replace

import pytest

from qprobe.backends import NLILabel
from qprobe.config import NLI_MODEL_ID, TARGET_MODEL_IDS, ModelRunConfig
from qprobe.datasets import DatasetExample
from qprobe.decision import DecisionKind, MethodVariant
from qprobe.evidence import (
    ExampleEvidence,
    OutputKind,
    ParaphraseEvidenceProvenance,
    ProbeEvidence,
    PromptEvidence,
    SharedParaphraseEvidence,
    TargetEvidenceProvenance,
)
from qprobe.errors import ConfigurationError, ScientificDefinitionError
from qprobe.paraphrasing import RoleParaphrase, VerifiedParaphrase
from qprobe.generation import SamplingPlan
from qprobe.paraphrasing import AuxiliaryCallPlan
from qprobe.pipeline import (
    PipelineBackends,
    PipelinePlans,
    collect_probe_evidence,
    replay_decision,
)
from qprobe.prompts import ParaphraseRole
from qprobe.semantic import analyze_prompt_semantics
from qprobe.views import PromptView


def make_target_provenance() -> TargetEvidenceProvenance:
    return TargetEvidenceProvenance(
        target_model_id=TARGET_MODEL_IDS[0],
        target_deployment_ref="controlled-target-ref",
        generation_seed_ref="generation-seed-ref",
        nli_model_id=NLI_MODEL_ID,
        nli_checkpoint_ref="controlled-nli-checkpoint-ref",
    )


def make_pipeline_plans() -> PipelinePlans:
    model_id = TARGET_MODEL_IDS[0]
    auxiliary_model_id = "controlled-claude-model-id"
    return PipelinePlans(
        models=ModelRunConfig(
            target_model_id=model_id,
            target_deployment_ref="controlled-target-ref",
            nli_checkpoint_ref="controlled-nli-checkpoint-ref",
            auxiliary_model_id=auxiliary_model_id,
            auxiliary_deployment_ref="controlled-claude-deployment-ref",
        ),
        target_sampling=SamplingPlan(
            model_id=model_id,
            base_seed=11,
            seed_ref="generation-seed-ref",
        ),
        paraphrase=AuxiliaryCallPlan(
            model_id=auxiliary_model_id,
            base_seed=22,
            seed_ref="paraphrase-seed-ref",
            temperature=1.0,
            max_new_tokens=100,
        ),
        equivalence_judge=AuxiliaryCallPlan(
            model_id=auxiliary_model_id,
            base_seed=33,
            seed_ref="equivalence-seed-ref",
            temperature=0.0,
            max_new_tokens=16,
        ),
    )


def test_pipeline_plans_reject_mismatched_auxiliary_deployment() -> None:
    plans = make_pipeline_plans()

    with pytest.raises(ConfigurationError):
        replace(
            plans,
            paraphrase=replace(
                plans.paraphrase,
                deployment_ref="different-auxiliary-deployment-ref",
            ),
        )


def test_pipeline_plans_reject_mismatched_target_deployment() -> None:
    plans = make_pipeline_plans()

    with pytest.raises(ConfigurationError):
        replace(
            plans,
            target_sampling=replace(
                plans.target_sampling,
                deployment_ref="different-target-deployment-ref",
            ),
        )


def make_prompt_evidence(
    view_id: str,
    prompt: str,
    answers: tuple[str, ...],
    labels,
    role: ParaphraseRole | None,
) -> PromptEvidence:
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


def make_singleton_view(
    view_id: str,
    prompt: str,
    answer: str,
    role: ParaphraseRole | None,
) -> PromptEvidence:
    return make_prompt_evidence(
        view_id,
        prompt,
        (answer,),
        ((None,),),
        role,
    )


def make_example() -> DatasetExample:
    return DatasetExample(
        dataset="triviaqa",
        sample_id="q-1",
        question="Original question?",
        reference_answer_groups=(("A",),),
        source_split="validation",
    )


def make_uncertain_original(example: DatasetExample) -> PromptEvidence:
    labels = (
        (None, NLILabel.CONTRADICTION),
        (NLILabel.CONTRADICTION, None),
    )
    return make_prompt_evidence(
        "q-1:original",
        example.question,
        ("original A", "original B"),
        labels,
        None,
    )


def make_early_exit_original(example: DatasetExample) -> PromptEvidence:
    labels = (
        (None, NLILabel.ENTAILMENT),
        (NLILabel.ENTAILMENT, None),
    )
    return make_prompt_evidence(
        "q-1:original",
        example.question,
        ("A", "A"),
        labels,
        None,
    )


def make_shared_evidence(
    equivalent_roles: frozenset[ParaphraseRole] | None = None,
) -> SharedParaphraseEvidence:
    candidates = tuple(
        RoleParaphrase(role=role, prompt=f"{role.value} question?")
        for role in ParaphraseRole
    )
    retained = (
        frozenset(ParaphraseRole)
        if equivalent_roles is None
        else equivalent_roles
    )
    verified = tuple(
        VerifiedParaphrase(
            role=candidate.role,
            prompt=candidate.prompt,
            equivalent=candidate.role in retained,
        )
        for candidate in candidates
    )
    return SharedParaphraseEvidence(
        example_id="q-1",
        original_question="Original question?",
        provenance=ParaphraseEvidenceProvenance(
            paraphrase_model_id="controlled-claude-model-id",
            paraphrase_deployment_ref="controlled-claude-deployment-ref",
            paraphrase_seed_ref="paraphrase-seed-ref",
            equivalence_model_id="controlled-claude-model-id",
            equivalence_deployment_ref="controlled-claude-deployment-ref",
            equivalence_seed_ref="equivalence-seed-ref",
        ),
        candidates=candidates,
        verified=verified,
    )


def labels_for_groups(group_ids: tuple[str, ...]):
    return tuple(
        tuple(
            None
            if left == right
            else (
                NLILabel.ENTAILMENT
                if group_ids[left] == group_ids[right]
                else NLILabel.CONTRADICTION
            )
            for right in range(len(group_ids))
        )
        for left in range(len(group_ids))
    )


def make_probe(
    shared: SharedParaphraseEvidence,
    answers: tuple[str, ...],
    group_ids: tuple[str, ...],
) -> ProbeEvidence:
    within_view_labels = (
        (None, NLILabel.ENTAILMENT),
        (NLILabel.ENTAILMENT, None),
    )
    prompt_views = tuple(
        make_prompt_evidence(
            f"q-1:role:{candidate.role.value}",
            candidate.prompt,
            (answer, answer),
            within_view_labels,
            candidate.role,
        )
        for candidate, answer in zip(shared.candidates, answers)
    )
    return ProbeEvidence(
        prompt_views=prompt_views,
        representative_labels=labels_for_groups(group_ids),
    )


def make_evidence(
    original: PromptEvidence,
    shared: SharedParaphraseEvidence | None = None,
    probe: ProbeEvidence | None = None,
) -> ExampleEvidence:
    return ExampleEvidence(
        example=make_example(),
        provenance=make_target_provenance(),
        original=original,
        shared_paraphrases=shared,
        probe=probe,
    )


def test_direct_accept_returns_the_original_representative() -> None:
    example = make_example()
    original = make_singleton_view(
        "q-1:original", example.question, "A", role=None
    )

    result = replay_decision(
        make_evidence(original),
        variant=MethodVariant.VARIANT_A,
        uncertainty_threshold=0.1,
    )

    assert result.decision.kind == DecisionKind.DIRECT_ACCEPT
    assert result.output.answer == "A"
    assert not result.entered_paraphrase_probe


def test_stage1_abstention_hides_its_evaluation_candidate() -> None:
    example = make_example()
    original = make_singleton_view(
        "q-1:original", example.question, "A", role=None
    )

    result = replay_decision(
        make_evidence(original),
        variant=MethodVariant.STAGE1_ONLY,
        uncertainty_threshold=0.0,
    )

    assert result.decision.kind == DecisionKind.ABSTAIN
    assert result.output.kind == OutputKind.ABSTAIN
    assert result.output.answer is None
    assert result.evaluation_candidate == "A"


def test_variant_b_early_exit_precedes_paraphrase_probing() -> None:
    example = make_example()
    original = make_early_exit_original(example)

    result = replay_decision(
        make_evidence(original),
        variant=MethodVariant.VARIANT_B,
        uncertainty_threshold=0.0,
    )

    assert result.decision.kind == DecisionKind.EARLY_EXIT
    assert result.output.answer == "A"
    assert result.policy_calls.paraphrase_generation == 0


def test_variant_a_filters_non_equivalent_roles() -> None:
    original = make_uncertain_original(make_example())
    retained = frozenset(
        (ParaphraseRole.LAYPERSON, ParaphraseRole.STUDENT)
    )
    shared = make_shared_evidence(retained)
    probe = make_probe(
        shared,
        answers=("A", "A", "B", "C", "D"),
        group_ids=("a", "a", "b", "c", "d"),
    )

    result = replay_decision(
        make_evidence(original, shared, probe),
        variant=MethodVariant.VARIANT_A,
        uncertainty_threshold=0.67,
    )

    assert result.decision.kind == DecisionKind.RECOVERED
    assert result.output.answer == "A"
    assert result.policy_calls.target_generation == 6
    assert result.policy_calls.equivalence_judge == 5


def test_noequiv_recovery_uses_unique_qualifying_group() -> None:
    original = make_uncertain_original(make_example())
    shared = make_shared_evidence()
    probe = make_probe(
        shared,
        answers=("A", "A", "B", "C", "D"),
        group_ids=("a", "a", "b", "c", "d"),
    )

    result = replay_decision(
        make_evidence(original, shared, probe),
        variant=MethodVariant.VARIANT_B_NO_EQUIV,
        uncertainty_threshold=0.67,
    )

    assert result.decision.kind == DecisionKind.RECOVERED
    assert result.output.answer == "A"
    assert result.policy_calls.equivalence_judge == 0


def test_competing_stable_groups_abstain_with_original_evaluation_candidate() -> None:
    original = make_uncertain_original(make_example())
    shared = make_shared_evidence()
    probe = make_probe(
        shared,
        answers=("A", "A", "B", "B", "C"),
        group_ids=("a", "a", "b", "b", "c"),
    )

    result = replay_decision(
        make_evidence(original, shared, probe),
        variant=MethodVariant.VARIANT_A,
        uncertainty_threshold=0.67,
    )

    assert result.decision.kind == DecisionKind.ABSTAIN
    assert result.output.answer is None
    assert result.evaluation_candidate == "original A"


def test_example_evidence_rejects_mixed_target_sample_counts() -> None:
    example = make_example()
    original = make_singleton_view(
        "q-1:original", example.question, "A", role=None
    )
    shared = make_shared_evidence()
    probe = make_probe(
        shared,
        answers=("A", "A", "B", "C", "D"),
        group_ids=("a", "a", "b", "c", "d"),
    )

    with pytest.raises(ScientificDefinitionError):
        make_evidence(original, shared, probe)


def test_prompt_evidence_rejects_semantics_from_different_labels() -> None:
    entailment_labels = (
        (None, NLILabel.ENTAILMENT),
        (NLILabel.ENTAILMENT, None),
    )
    contradiction_labels = (
        (None, NLILabel.CONTRADICTION),
        (NLILabel.CONTRADICTION, None),
    )
    mismatched_view = PromptView(
        view_id="q-1:original",
        prompt="Original question?",
        answers=("A", "B"),
        semantics=analyze_prompt_semantics(entailment_labels),
    )

    with pytest.raises(ScientificDefinitionError):
        PromptEvidence(
            role=None,
            view=mismatched_view,
            directional_labels=contradiction_labels,
        )


def test_probe_collection_rejects_mismatched_paraphrase_provenance() -> None:
    shared = make_shared_evidence()
    mismatched = SharedParaphraseEvidence(
        example_id=shared.example_id,
        original_question=shared.original_question,
        provenance=ParaphraseEvidenceProvenance(
            paraphrase_model_id="different-model-id",
            paraphrase_deployment_ref="different-deployment-ref",
            paraphrase_seed_ref="different-seed-ref",
            equivalence_model_id=shared.provenance.equivalence_model_id,
            equivalence_deployment_ref=(
                shared.provenance.equivalence_deployment_ref
            ),
            equivalence_seed_ref=shared.provenance.equivalence_seed_ref,
        ),
        candidates=shared.candidates,
        verified=shared.verified,
    )
    unused_backends = PipelineBackends(None, None, None, None)

    with pytest.raises(ScientificDefinitionError):
        collect_probe_evidence(
            mismatched,
            (),
            plans=make_pipeline_plans(),
            backends=unused_backends,
        )
