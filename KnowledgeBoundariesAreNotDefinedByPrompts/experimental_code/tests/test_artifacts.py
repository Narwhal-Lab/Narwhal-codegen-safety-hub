

from dataclasses import replace

import pytest

import qprobe.artifacts as artifacts_module

from qprobe.aggregation import (
    EvidenceOrigin,
    EvidenceSources,
    ExampleRunRecord,
    RunIdentity,
)
from qprobe.artifacts import (
    BoundCalibrationResult,
    EXAMPLES_FILE,
    CURRENT_FILE,
    GENERATIONS_DIRECTORY,
    ResultOrigin,
    RunMetadata,
    RunSplit,
    RunStatus,
    load_run_snapshot,
    public_output_record,
    write_run_snapshot,
)
from qprobe.backends import NLILabel, TokenUsage
from qprobe.calibration import CalibrationResult, TauCalibrationScore
from qprobe.config import (
    NLI_MODEL_ID,
    TARGET_MODEL_IDS,
    ModelRunConfig,
    PublicMethodConfig,
)
from qprobe.correctness import (
    CorrectnessProvenance,
    CorrectnessResult,
    grouped_exact_match,
)
from qprobe.datasets import DatasetExample
from qprobe.decision import DecisionKind, MethodVariant
from qprobe.errors import ArtifactValidationError, ScientificDefinitionError
from qprobe.evidence import (
    ExampleEvidence,
    ParaphraseEvidenceProvenance,
    ProbeEvidence,
    PromptEvidence,
    SharedParaphraseEvidence,
    TargetEvidenceProvenance,
)
from qprobe.paraphrasing import RoleParaphrase, VerifiedParaphrase
from qprobe.pipeline import replay_decision
from qprobe.prompts import ParaphraseRole
from qprobe.semantic import analyze_prompt_semantics
from qprobe.telemetry import CallEvent, EventKind
from qprobe.views import PromptView


def make_calibration() -> BoundCalibrationResult:
    scores = tuple(
        TauCalibrationScore(
            tau=tau,
            seed_aurocs=(0.5, 0.5, 0.5),
            seed_augrcs=(0.5, 0.5, 0.5),
            seed_mean_method_calls=(1.0, 1.0, 1.0),
        )
        for tau in tuple(index / 10 for index in range(1, 21))
    )
    return BoundCalibrationResult(
        manifest_fingerprint="a" * 64,
        controlled_config_fingerprint="b" * 64,
        result=CalibrationResult(
            target_model_id=TARGET_MODEL_IDS[0],
            dataset="triviaqa",
            generation_seed_refs=(
                "generation-seed-ref",
                "generation-seed-ref-2",
                "generation-seed-ref-3",
            ),
            selected_tau=0.1,
            candidate_scores=scores,
        ),
    )


def make_metadata(
    variant: MethodVariant = MethodVariant.STAGE1_ONLY,
) -> RunMetadata:
    return RunMetadata(
        run_id="new-run-1",
        dataset="triviaqa",
        split=RunSplit.DEV,
        variant=variant,
        models=ModelRunConfig(
            target_model_id=TARGET_MODEL_IDS[0],
            target_deployment_ref="controlled-target-ref",
            nli_checkpoint_ref="controlled-nli-checkpoint-ref",
            auxiliary_model_id="controlled-claude-model-id",
            auxiliary_deployment_ref="controlled-claude-deployment-ref",
        ),
        method=PublicMethodConfig(),
        generation_seed_ref="generation-seed-ref",
        paraphrase_seed_ref="paraphrase-seed-ref",
        equivalence_judge_seed_ref="equivalence-seed-ref",
        correctness_judge_seed_ref="correctness-seed-ref",
        calibration=make_calibration(),
        selected_tau=0.1,
        manifest_fingerprint="a" * 64,
        controlled_config_fingerprint="b" * 64,
        expected_sample_ids=tuple(f"q-{index}" for index in range(200)),
        result_origin=ResultOrigin.RECONSTRUCTED_RUN,
    )


def make_abstained_record(
    sample_id: str = "q-0",
    *,
    correct: bool = False,
) -> ExampleRunRecord:
    question = f"Question {sample_id}?"
    labels = tuple(
        tuple(
            None if left == right else NLILabel.CONTRADICTION
            for right in range(10)
        )
        for left in range(10)
    )
    evidence = ExampleEvidence(
        example=DatasetExample(
            dataset="triviaqa",
            sample_id=sample_id,
            question=question,
            reference_answer_groups=(("A",),),
            source_split="validation",
        ),
        provenance=TargetEvidenceProvenance(
            target_model_id=TARGET_MODEL_IDS[0],
            target_deployment_ref="controlled-target-ref",
            generation_seed_ref="generation-seed-ref",
            nli_model_id=NLI_MODEL_ID,
            nli_checkpoint_ref="controlled-nli-checkpoint-ref",
        ),
        original=PromptEvidence(
            role=None,
            view=PromptView(
                view_id=f"{sample_id}:original",
                prompt=question,
                answers=tuple(f"Candidate {index}" for index in range(10)),
                semantics=analyze_prompt_semantics(labels),
            ),
            directional_labels=labels,
        ),
    )
    replay = replay_decision(
        evidence,
        variant=MethodVariant.STAGE1_ONLY,
        uncertainty_threshold=0.1,
    )
    metadata = make_metadata()
    return ExampleRunRecord(
        run_identity=RunIdentity(
            run_id=metadata.run_id,
            dataset=metadata.dataset,
            split=metadata.split.value,
            variant=metadata.variant,
            models=metadata.models,
            generation_seed_ref=metadata.generation_seed_ref,
            paraphrase_seed_ref=metadata.paraphrase_seed_ref,
            equivalence_judge_seed_ref=metadata.equivalence_judge_seed_ref,
            correctness_judge_seed_ref=metadata.correctness_judge_seed_ref,
            uncertainty_threshold=metadata.selected_tau,
            calibration_fingerprint=metadata.calibration_fingerprint,
            manifest_fingerprint=metadata.manifest_fingerprint,
            controlled_config_fingerprint=metadata.controlled_config_fingerprint,
        ),
        evidence_sources=EvidenceSources(
            target=EvidenceOrigin.CACHED,
            nli=EvidenceOrigin.CACHED,
            paraphrase=None,
            target_cache_fingerprint="c" * 64,
            nli_cache_fingerprint="d" * 64,
        ),
        evidence=evidence,
        replay=replay,
        correctness_provenance=CorrectnessProvenance(
            candidate_answer=replay.evaluation_candidate,
            question=evidence.example.question,
            reference_answer_groups=evidence.example.reference_answer_groups,
            judge_model_id=metadata.models.auxiliary_model_id,
            judge_deployment_ref=metadata.models.auxiliary_deployment_ref,
            judge_seed_ref=metadata.correctness_judge_seed_ref,
            cache_fingerprint="e" * 64,
        ),
        correctness=CorrectnessResult(
            correct=correct,
            exact_match=False,
            matched_group_index=None,
            judge_called=False,
            judge_cache_hit=True,
        ),
        real_events=(),
    )


def make_collected_record(sample_id: str = "q-0") -> ExampleRunRecord:
    record = make_abstained_record(sample_id)
    view_id = record.evidence.original.view.view_id
    answer_count = len(record.evidence.original.view.answers)
    target_events = tuple(
        CallEvent(
            event_kind=EventKind.TARGET_GENERATION,
            model_id=record.run_identity.models.target_model_id,
            latency_seconds=0.1,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            metadata={
                "deployment_ref": record.evidence.provenance.target_deployment_ref,
                "prompt_id": view_id,
                "sample_index": str(sample_index),
                "seed_ref": record.evidence.provenance.generation_seed_ref,
                "finish_reason": "stop",
            },
        )
        for sample_index in range(answer_count)
    )
    nli_events = tuple(
        CallEvent(
            event_kind=EventKind.NLI_FORWARD,
            model_id=record.run_identity.models.nli_model_id,
            latency_seconds=0.1,
            metadata={
                "view_id": view_id,
                "nli_stage": "within_view",
                "premise_index": str(left),
                "hypothesis_index": str(right),
                "checkpoint_ref": record.evidence.provenance.nli_checkpoint_ref,
            },
        )
        for left in range(answer_count)
        for right in range(answer_count)
        if left != right
    )
    return replace(
        record,
        evidence_sources=EvidenceSources(
            target=EvidenceOrigin.COLLECTED,
            nli=EvidenceOrigin.COLLECTED,
            paraphrase=None,
        ),
        real_events=target_events + nli_events,
    )


def make_variant_b_early_exit_collected_record(
    sample_id: str = "q-0",
) -> ExampleRunRecord:
    record = make_collected_record(sample_id)
    answer_count = len(record.evidence.original.view.answers)
    labels = tuple(
        tuple(
            None
            if left == right
            else (
                NLILabel.ENTAILMENT
                if left < answer_count - 1 and right < answer_count - 1
                else NLILabel.CONTRADICTION
            )
            for right in range(answer_count)
        )
        for left in range(answer_count)
    )
    original = PromptEvidence(
        role=None,
        view=PromptView(
            view_id=record.evidence.original.view.view_id,
            prompt=record.evidence.example.question,
            answers=("A",) * (answer_count - 1) + ("B",),
            semantics=analyze_prompt_semantics(labels),
        ),
        directional_labels=labels,
    )
    evidence = replace(record.evidence, original=original)
    replay = replay_decision(
        evidence,
        variant=MethodVariant.VARIANT_B,
        uncertainty_threshold=record.run_identity.uncertainty_threshold,
    )
    return replace(
        record,
        run_identity=replace(
            record.run_identity,
            variant=MethodVariant.VARIANT_B,
        ),
        evidence=evidence,
        replay=replay,
        correctness_provenance=replace(
            record.correctness_provenance,
            candidate_answer=replay.evaluation_candidate,
            cache_fingerprint=None,
        ),
        correctness=CorrectnessResult(
            correct=True,
            exact_match=grouped_exact_match(
                replay.evaluation_candidate,
                evidence.example.reference_answer_groups,
            )
            is not None,
            matched_group_index=grouped_exact_match(
                replay.evaluation_candidate,
                evidence.example.reference_answer_groups,
            ),
            judge_called=False,
            judge_cache_hit=False,
        ),
    )


def make_probing_collected_record(
    sample_id: str = "q-0",
    *,
    variant: MethodVariant = MethodVariant.VARIANT_A,
    equivalent_roles: frozenset[ParaphraseRole] | None = None,
) -> ExampleRunRecord:
    base = make_abstained_record(sample_id)
    metadata = make_metadata(variant)
    candidates = tuple(
        RoleParaphrase(role=role, prompt=f"{role.value} question?")
        for role in ParaphraseRole
    )
    verified = (
        None
        if variant == MethodVariant.VARIANT_B_NO_EQUIV
        else tuple(
            VerifiedParaphrase(
                role=candidate.role,
                prompt=candidate.prompt,
                equivalent=(
                    equivalent_roles is None
                    or candidate.role in equivalent_roles
                ),
            )
            for candidate in candidates
        )
    )
    shared = SharedParaphraseEvidence(
        example_id=sample_id,
        original_question=base.evidence.example.question,
        provenance=ParaphraseEvidenceProvenance(
            paraphrase_model_id=metadata.models.auxiliary_model_id,
            paraphrase_deployment_ref=(
                metadata.models.auxiliary_deployment_ref
            ),
            paraphrase_seed_ref=metadata.paraphrase_seed_ref,
            equivalence_model_id=(
                None
                if verified is None
                else metadata.models.auxiliary_model_id
            ),
            equivalence_deployment_ref=(
                None
                if verified is None
                else metadata.models.auxiliary_deployment_ref
            ),
            equivalence_seed_ref=(
                None if verified is None else metadata.equivalence_judge_seed_ref
            ),
        ),
        candidates=candidates,
        verified=verified,
    )
    answer_count = len(base.evidence.original.view.answers)
    retained_candidates = (
        candidates
        if verified is None
        else tuple(
            candidate
            for candidate, result in zip(candidates, verified)
            if result.equivalent
        )
    )
    entailment_labels = tuple(
        tuple(
            None if left == right else NLILabel.ENTAILMENT
            for right in range(answer_count)
        )
        for left in range(answer_count)
    )
    probe_views = tuple(
        PromptEvidence(
            role=candidate.role,
            view=PromptView(
                view_id=f"{sample_id}:role:{candidate.role.value}",
                prompt=candidate.prompt,
                answers=("A",) * answer_count,
                semantics=analyze_prompt_semantics(entailment_labels),
            ),
            directional_labels=entailment_labels,
        )
        for candidate in retained_candidates
    )
    representative_count = len(probe_views)
    representative_labels = tuple(
        tuple(
            None if left == right else NLILabel.ENTAILMENT
            for right in range(representative_count)
        )
        for left in range(representative_count)
    )
    evidence = replace(
        base.evidence,
        shared_paraphrases=shared,
        probe=ProbeEvidence(
            prompt_views=probe_views,
            representative_labels=representative_labels,
        ),
    )
    replay = replay_decision(
        evidence,
        variant=variant,
        uncertainty_threshold=metadata.selected_tau,
    )
    evidence = replace(
        evidence,
        example=replace(
            evidence.example,
            reference_answer_groups=((replay.evaluation_candidate,),),
        ),
    )
    prompt_evidence = (evidence.original,) + probe_views
    target_events = tuple(
        CallEvent(
            event_kind=EventKind.TARGET_GENERATION,
            model_id=metadata.models.target_model_id,
            latency_seconds=0.1,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            metadata={
                "deployment_ref": metadata.models.target_deployment_ref,
                "prompt_id": item.view.view_id,
                "sample_index": str(sample_index),
                "seed_ref": metadata.generation_seed_ref,
                "finish_reason": "stop",
            },
        )
        for item in prompt_evidence
        for sample_index in range(len(item.view.answers))
    )
    within_view_events = tuple(
        CallEvent(
            event_kind=EventKind.NLI_FORWARD,
            model_id=metadata.models.nli_model_id,
            latency_seconds=0.1,
            metadata={
                "view_id": item.view.view_id,
                "nli_stage": "within_view",
                "premise_index": str(left),
                "hypothesis_index": str(right),
                "checkpoint_ref": metadata.models.nli_checkpoint_ref,
            },
        )
        for item in prompt_evidence
        for left in range(len(item.view.answers))
        for right in range(len(item.view.answers))
        if left != right
    )
    representative_events = tuple(
        CallEvent(
            event_kind=EventKind.NLI_FORWARD,
            model_id=metadata.models.nli_model_id,
            latency_seconds=0.1,
            metadata={
                "example_id": sample_id,
                "nli_stage": "representative_answers",
                "premise_index": str(left),
                "hypothesis_index": str(right),
                "checkpoint_ref": metadata.models.nli_checkpoint_ref,
            },
        )
        for left in range(representative_count)
        for right in range(representative_count)
        if left != right
    )
    paraphrase_events = tuple(
        CallEvent(
            event_kind=EventKind.PARAPHRASE_GENERATION,
            model_id=metadata.models.auxiliary_model_id,
            latency_seconds=0.1,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            metadata={
                "deployment_ref": metadata.models.auxiliary_deployment_ref,
                "example_id": sample_id,
                "role": candidate.role.value,
                "seed_ref": metadata.paraphrase_seed_ref,
                "finish_reason": "stop",
            },
        )
        for candidate in candidates
    )
    equivalence_events = tuple(
        CallEvent(
            event_kind=EventKind.EQUIVALENCE_JUDGE,
            model_id=metadata.models.auxiliary_model_id,
            latency_seconds=0.1,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            metadata={
                "deployment_ref": metadata.models.auxiliary_deployment_ref,
                "example_id": sample_id,
                "role": item.role.value,
                "seed_ref": metadata.equivalence_judge_seed_ref,
                "decision": (
                    "EQUIVALENT" if item.equivalent else "NOT_EQUIVALENT"
                ),
            },
        )
        for item in verified or ()
    )
    return ExampleRunRecord(
        run_identity=RunIdentity(
            run_id=metadata.run_id,
            dataset=metadata.dataset,
            split=metadata.split.value,
            variant=metadata.variant,
            models=metadata.models,
            generation_seed_ref=metadata.generation_seed_ref,
            paraphrase_seed_ref=metadata.paraphrase_seed_ref,
            equivalence_judge_seed_ref=metadata.equivalence_judge_seed_ref,
            correctness_judge_seed_ref=metadata.correctness_judge_seed_ref,
            uncertainty_threshold=metadata.selected_tau,
            calibration_fingerprint=metadata.calibration_fingerprint,
            manifest_fingerprint=metadata.manifest_fingerprint,
            controlled_config_fingerprint=metadata.controlled_config_fingerprint,
        ),
        evidence_sources=EvidenceSources(
            target=EvidenceOrigin.COLLECTED,
            nli=EvidenceOrigin.COLLECTED,
            paraphrase=EvidenceOrigin.COLLECTED,
        ),
        evidence=evidence,
        replay=replay,
        correctness_provenance=CorrectnessProvenance(
            candidate_answer=replay.evaluation_candidate,
            question=evidence.example.question,
            reference_answer_groups=evidence.example.reference_answer_groups,
            judge_model_id=metadata.models.auxiliary_model_id,
            judge_deployment_ref=metadata.models.auxiliary_deployment_ref,
            judge_seed_ref=metadata.correctness_judge_seed_ref,
        ),
        correctness=CorrectnessResult(
            correct=True,
            exact_match=grouped_exact_match(
                replay.evaluation_candidate,
                evidence.example.reference_answer_groups,
            )
            is not None,
            matched_group_index=grouped_exact_match(
                replay.evaluation_candidate,
                evidence.example.reference_answer_groups,
            ),
            judge_called=False,
            judge_cache_hit=False,
        ),
        real_events=(
            target_events
            + within_view_events
            + representative_events
            + paraphrase_events
            + equivalence_events
        ),
    )


def test_snapshot_round_trip_preserves_private_evidence(tmp_path) -> None:
    metadata = make_metadata()
    record = make_abstained_record()

    checkpoint = write_run_snapshot(
        tmp_path,
        metadata,
        (record,),
        status=RunStatus.IN_PROGRESS,
    )
    loaded = load_run_snapshot(tmp_path)

    assert loaded.metadata == metadata
    assert loaded.checkpoint == checkpoint
    assert loaded.records[0].sample_id == "q-0"
    assert loaded.records[0].evidence.original.directional_labels == (
        record.evidence.original.directional_labels
    )


def test_public_abstention_does_not_expose_private_candidate() -> None:
    record = make_abstained_record()

    public = public_output_record(record)

    assert public["output"] == {"kind": "abstain", "answer": None}
    assert record.replay.evaluation_candidate not in str(public)


def test_snapshot_rejects_record_from_a_different_generation_seed(tmp_path) -> None:
    record = make_abstained_record()
    mismatched = replace(
        record,
        evidence=replace(
            record.evidence,
            provenance=replace(
                record.evidence.provenance,
                generation_seed_ref="different-generation-seed-ref",
            ),
        ),
    )

    with pytest.raises(ArtifactValidationError):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_different_correctness_seed(tmp_path) -> None:
    record = make_abstained_record()
    mismatched = replace(
        record,
        correctness_provenance=replace(
            record.correctness_provenance,
            judge_seed_ref="different-correctness-seed-ref",
        ),
    )

    with pytest.raises(ArtifactValidationError):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_record_rejects_correctness_for_a_different_candidate() -> None:
    record = make_abstained_record()

    with pytest.raises(ScientificDefinitionError):
        replace(
            record,
            correctness_provenance=replace(
                record.correctness_provenance,
                candidate_answer="Different candidate",
            ),
        )


def test_snapshot_rejects_real_event_for_cached_evidence(tmp_path) -> None:
    record = make_abstained_record()
    mismatched = replace(
        record,
        real_events=(
            CallEvent(
                event_kind=EventKind.TARGET_GENERATION,
                model_id=record.run_identity.models.target_model_id,
                latency_seconds=0.1,
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            ),
        ),
    )

    with pytest.raises(ArtifactValidationError):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_incomplete_collected_event_set(tmp_path) -> None:
    record = make_collected_record()
    incomplete = replace(record, real_events=record.real_events[:-1])

    with pytest.raises(ArtifactValidationError, match="NLI events"):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (incomplete,),
            status=RunStatus.IN_PROGRESS,
        )


def test_collected_stage1_snapshot_round_trip(tmp_path) -> None:
    record = make_collected_record()

    write_run_snapshot(
        tmp_path,
        make_metadata(),
        (record,),
        status=RunStatus.IN_PROGRESS,
    )
    loaded = load_run_snapshot(tmp_path)

    assert loaded.records == (record,)


def test_metadata_rejects_tau_outside_bound_calibration() -> None:
    metadata = make_metadata()

    with pytest.raises(ArtifactValidationError, match="bound calibration"):
        replace(metadata, selected_tau=0.2)


def test_snapshot_rejects_incorrect_target_event_deployment(tmp_path) -> None:
    record = make_collected_record()
    event_index = next(
        index
        for index, event in enumerate(record.real_events)
        if event.event_kind == EventKind.TARGET_GENERATION
    )
    events = list(record.real_events)
    event = events[event_index]
    events[event_index] = replace(
        event,
        metadata={**event.metadata, "deployment_ref": "different-target-ref"},
    )

    with pytest.raises(ArtifactValidationError, match="target events"):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (replace(record, real_events=tuple(events)),),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_incorrect_target_event_seed(tmp_path) -> None:
    record = make_collected_record()
    event_index = next(
        index
        for index, event in enumerate(record.real_events)
        if event.event_kind == EventKind.TARGET_GENERATION
    )
    events = list(record.real_events)
    event = events[event_index]
    events[event_index] = replace(
        event,
        metadata={**event.metadata, "seed_ref": "different-generation-seed-ref"},
    )

    with pytest.raises(ArtifactValidationError, match="target events"):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (replace(record, real_events=tuple(events)),),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_incorrect_nli_event_checkpoint(tmp_path) -> None:
    record = make_collected_record()
    event_index = next(
        index
        for index, event in enumerate(record.real_events)
        if event.event_kind == EventKind.NLI_FORWARD
    )
    events = list(record.real_events)
    event = events[event_index]
    events[event_index] = replace(
        event,
        metadata={**event.metadata, "checkpoint_ref": "different-checkpoint-ref"},
    )

    with pytest.raises(ArtifactValidationError, match="NLI events"):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (replace(record, real_events=tuple(events)),),
            status=RunStatus.IN_PROGRESS,
        )


@pytest.mark.parametrize(
    "variant",
    (
        MethodVariant.VARIANT_A,
        MethodVariant.VARIANT_B,
        MethodVariant.VARIANT_B_NO_EQUIV,
    ),
)
def test_probing_variant_collected_snapshot_round_trip(
    tmp_path,
    variant: MethodVariant,
) -> None:
    record = make_probing_collected_record(variant=variant)

    write_run_snapshot(
        tmp_path,
        make_metadata(variant),
        (record,),
        status=RunStatus.IN_PROGRESS,
    )
    loaded = load_run_snapshot(tmp_path)

    assert loaded.records == (record,)
    equivalence_events = tuple(
        event
        for event in record.real_events
        if event.event_kind == EventKind.EQUIVALENCE_JUDGE
    )
    if variant == MethodVariant.VARIANT_B_NO_EQUIV:
        assert record.evidence.shared_paraphrases is not None
        assert record.evidence.shared_paraphrases.verified is None
        assert equivalence_events == ()
    else:
        assert len(equivalence_events) == len(tuple(ParaphraseRole))


@pytest.mark.parametrize(
    "variant",
    (MethodVariant.VARIANT_A, MethodVariant.VARIANT_B),
)
def test_filtered_probe_snapshot_round_trip(
    tmp_path,
    variant: MethodVariant,
) -> None:
    retained_roles = frozenset(
        (ParaphraseRole.LAYPERSON, ParaphraseRole.STUDENT)
    )
    record = make_probing_collected_record(
        variant=variant,
        equivalent_roles=retained_roles,
    )

    write_run_snapshot(
        tmp_path,
        make_metadata(variant),
        (record,),
        status=RunStatus.IN_PROGRESS,
    )
    loaded = load_run_snapshot(tmp_path)

    assert loaded.records == (record,)
    assert record.evidence.probe is not None
    assert tuple(item.role for item in record.evidence.probe.prompt_views) == (
        ParaphraseRole.LAYPERSON,
        ParaphraseRole.STUDENT,
    )


def test_variant_b_early_exit_collected_snapshot_round_trip(tmp_path) -> None:
    record = make_variant_b_early_exit_collected_record()

    write_run_snapshot(
        tmp_path,
        make_metadata(MethodVariant.VARIANT_B),
        (record,),
        status=RunStatus.IN_PROGRESS,
    )
    loaded = load_run_snapshot(tmp_path)

    assert loaded.records == (record,)
    assert record.replay.decision.kind == DecisionKind.EARLY_EXIT
    assert not record.replay.entered_paraphrase_probe


def test_snapshot_rejects_different_paraphrase_seed(tmp_path) -> None:
    record = make_probing_collected_record()
    shared = record.evidence.shared_paraphrases
    assert shared is not None
    mismatched = replace(
        record,
        evidence=replace(
            record.evidence,
            shared_paraphrases=replace(
                shared,
                provenance=replace(
                    shared.provenance,
                    paraphrase_seed_ref="different-paraphrase-seed-ref",
                ),
            ),
        ),
    )

    with pytest.raises(ArtifactValidationError, match="paraphrase provenance"):
        write_run_snapshot(
            tmp_path,
            make_metadata(MethodVariant.VARIANT_A),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_different_equivalence_seed(tmp_path) -> None:
    record = make_probing_collected_record()
    shared = record.evidence.shared_paraphrases
    assert shared is not None
    mismatched = replace(
        record,
        evidence=replace(
            record.evidence,
            shared_paraphrases=replace(
                shared,
                provenance=replace(
                    shared.provenance,
                    equivalence_seed_ref="different-equivalence-seed-ref",
                ),
            ),
        ),
    )

    with pytest.raises(ArtifactValidationError, match="equivalence provenance"):
        write_run_snapshot(
            tmp_path,
            make_metadata(MethodVariant.VARIANT_A),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_incorrect_equivalence_event(tmp_path) -> None:
    record = make_probing_collected_record()
    event_index = next(
        index
        for index, event in enumerate(record.real_events)
        if event.event_kind == EventKind.EQUIVALENCE_JUDGE
    )
    event = record.real_events[event_index]
    events = list(record.real_events)
    events[event_index] = replace(
        event,
        metadata={**event.metadata, "decision": "NOT_EQUIVALENT"},
    )
    mismatched = replace(record, real_events=tuple(events))

    with pytest.raises(ArtifactValidationError, match="equivalence events"):
        write_run_snapshot(
            tmp_path,
            make_metadata(MethodVariant.VARIANT_A),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_incorrect_paraphrase_event_seed(tmp_path) -> None:
    record = make_probing_collected_record()
    event_index = next(
        index
        for index, event in enumerate(record.real_events)
        if event.event_kind == EventKind.PARAPHRASE_GENERATION
    )
    event = record.real_events[event_index]
    events = list(record.real_events)
    events[event_index] = replace(
        event,
        metadata={**event.metadata, "seed_ref": "different-seed-ref"},
    )
    mismatched = replace(record, real_events=tuple(events))

    with pytest.raises(ArtifactValidationError, match="paraphrase events"):
        write_run_snapshot(
            tmp_path,
            make_metadata(MethodVariant.VARIANT_A),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_incorrect_auxiliary_event_deployment(tmp_path) -> None:
    record = make_probing_collected_record()
    event_index = next(
        index
        for index, event in enumerate(record.real_events)
        if event.event_kind == EventKind.PARAPHRASE_GENERATION
    )
    event = record.real_events[event_index]
    events = list(record.real_events)
    events[event_index] = replace(
        event,
        metadata={
            **event.metadata,
            "deployment_ref": "different-auxiliary-deployment-ref",
        },
    )
    mismatched = replace(record, real_events=tuple(events))

    with pytest.raises(ArtifactValidationError, match="paraphrase events"):
        write_run_snapshot(
            tmp_path,
            make_metadata(MethodVariant.VARIANT_A),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_missing_correctness_event(tmp_path) -> None:
    record = make_abstained_record()
    mismatched = replace(
        record,
        correctness_provenance=replace(
            record.correctness_provenance,
            cache_fingerprint=None,
        ),
        correctness=CorrectnessResult(
            correct=False,
            exact_match=False,
            matched_group_index=None,
            judge_called=True,
            judge_cache_hit=False,
        ),
    )

    with pytest.raises(ArtifactValidationError):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_incorrect_exact_match_group(tmp_path) -> None:
    record = make_abstained_record()
    candidate = record.replay.evaluation_candidate
    mismatched = replace(
        record,
        evidence=replace(
            record.evidence,
            example=replace(
                record.evidence.example,
                reference_answer_groups=((candidate,),),
            ),
        ),
        correctness_provenance=replace(
            record.correctness_provenance,
            reference_answer_groups=((candidate,),),
            cache_fingerprint=None,
        ),
        correctness=CorrectnessResult(
            correct=True,
            exact_match=True,
            matched_group_index=1,
            judge_called=False,
            judge_cache_hit=False,
        ),
    )

    with pytest.raises(ArtifactValidationError, match="Exact Match"):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_incorrect_judge_event_decision(tmp_path) -> None:
    record = make_abstained_record()
    mismatched = replace(
        record,
        correctness_provenance=replace(
            record.correctness_provenance,
            cache_fingerprint=None,
        ),
        correctness=CorrectnessResult(
            correct=False,
            exact_match=False,
            matched_group_index=None,
            judge_called=True,
            judge_cache_hit=False,
        ),
        real_events=(
            CallEvent(
                event_kind=EventKind.CORRECTNESS_JUDGE,
                model_id=record.run_identity.models.auxiliary_model_id,
                latency_seconds=0.1,
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                metadata={
                    "example_id": record.sample_id,
                    "decision": "CORRECT",
                    "deployment_ref": (
                        record.run_identity.models.auxiliary_deployment_ref
                    ),
                    "seed_ref": (
                        record.run_identity.correctness_judge_seed_ref
                    ),
                },
            ),
        ),
    )

    with pytest.raises(ArtifactValidationError, match="decision"):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_cache_hit_without_fingerprint(tmp_path) -> None:
    record = make_abstained_record()
    mismatched = replace(
        record,
        correctness_provenance=replace(
            record.correctness_provenance,
            cache_fingerprint=None,
        ),
    )

    with pytest.raises(ArtifactValidationError, match="cache route"):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (mismatched,),
            status=RunStatus.IN_PROGRESS,
        )


def test_snapshot_rejects_conflicting_correctness_cache_labels(tmp_path) -> None:
    first = make_abstained_record("q-0", correct=True)
    second = make_abstained_record("q-1", correct=False)
    conflicting = replace(
        second,
        evidence=replace(
            second.evidence,
            example=replace(
                second.evidence.example,
                question=first.evidence.example.question,
            ),
            original=replace(
                second.evidence.original,
                view=replace(
                    second.evidence.original.view,
                    prompt=first.evidence.example.question,
                ),
            ),
        ),
        correctness_provenance=replace(
            second.correctness_provenance,
            question=first.correctness_provenance.question,
        ),
    )

    with pytest.raises(ArtifactValidationError, match="conflicting labels"):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (first, conflicting),
            status=RunStatus.IN_PROGRESS,
        )


def test_complete_snapshot_rejects_missing_examples(tmp_path) -> None:
    with pytest.raises(ArtifactValidationError):
        write_run_snapshot(
            tmp_path,
            make_metadata(),
            (make_abstained_record(),),
            status=RunStatus.COMPLETE,
        )


def test_snapshot_rejects_examples_changed_after_checkpoint(tmp_path) -> None:
    write_run_snapshot(
        tmp_path,
        make_metadata(),
        (make_abstained_record(),),
        status=RunStatus.IN_PROGRESS,
    )
    generation_directory = next(
        path
        for path in (tmp_path / GENERATIONS_DIRECTORY).iterdir()
        if not path.name.startswith(".pending-")
    )
    examples_path = generation_directory / EXAMPLES_FILE
    examples_path.write_text(
        examples_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError):
        load_run_snapshot(tmp_path)


def test_interrupted_generation_preserves_previous_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    metadata = make_metadata()
    first_record = make_abstained_record("q-0")
    write_run_snapshot(
        tmp_path,
        metadata,
        (first_record,),
        status=RunStatus.IN_PROGRESS,
    )
    original_atomic_write = artifacts_module._atomic_write_text

    def interrupt_pointer_write(path, text):
        if path.name == CURRENT_FILE:
            raise OSError("simulated interruption")
        original_atomic_write(path, text)

    monkeypatch.setattr(
        artifacts_module,
        "_atomic_write_text",
        interrupt_pointer_write,
    )
    with pytest.raises(OSError, match="simulated interruption"):
        write_run_snapshot(
            tmp_path,
            metadata,
            (first_record, make_abstained_record("q-1")),
            status=RunStatus.IN_PROGRESS,
        )

    loaded = load_run_snapshot(tmp_path)

    assert tuple(record.sample_id for record in loaded.records) == ("q-0",)


def test_complete_snapshot_binds_aggregate(tmp_path) -> None:
    metadata = make_metadata()
    records = tuple(
        make_abstained_record(sample_id, correct=index % 2 == 0)
        for index, sample_id in enumerate(metadata.expected_sample_ids)
    )

    write_run_snapshot(
        tmp_path,
        metadata,
        records,
        status=RunStatus.COMPLETE,
    )
    loaded = load_run_snapshot(tmp_path)

    assert loaded.aggregate is not None
    assert loaded.aggregate.example_count == len(records)
    assert loaded.aggregate.run_identity == records[0].run_identity
