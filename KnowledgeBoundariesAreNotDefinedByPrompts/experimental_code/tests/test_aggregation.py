

from qprobe.aggregation import (
    EvidenceOrigin,
    EvidenceSources,
    ExampleRunRecord,
    RunIdentity,
    aggregate_run,
)
from qprobe.backends import TokenUsage
from qprobe.config import NLI_MODEL_ID, TARGET_MODEL_IDS, ModelRunConfig
from qprobe.correctness import CorrectnessProvenance, CorrectnessResult
from qprobe.datasets import DatasetExample
from qprobe.decision import MethodVariant
from qprobe.evidence import (
    ExampleEvidence,
    PromptEvidence,
    TargetEvidenceProvenance,
)
from qprobe.errors import ScientificDefinitionError
from qprobe.pipeline import replay_decision
from qprobe.semantic import analyze_prompt_semantics
from qprobe.telemetry import CallEvent, EventKind
from qprobe.views import PromptView


def make_record(
    sample_id: str,
    answer: str,
    *,
    correct: bool,
) -> ExampleRunRecord:
    models = ModelRunConfig(
        target_model_id=TARGET_MODEL_IDS[0],
        target_deployment_ref="controlled-target-ref",
        nli_checkpoint_ref="controlled-nli-checkpoint-ref",
        auxiliary_model_id="controlled-judge-model",
        auxiliary_deployment_ref="controlled-judge-ref",
    )
    labels = ((None,),)
    evidence = ExampleEvidence(
        example=DatasetExample(
            dataset="triviaqa",
            sample_id=sample_id,
            question=f"Question {sample_id}?",
            reference_answer_groups=((answer,),),
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
                prompt=f"Question {sample_id}?",
                answers=(answer,),
                semantics=analyze_prompt_semantics(labels),
            ),
            directional_labels=labels,
        ),
    )
    replay = replay_decision(
        evidence,
        variant=MethodVariant.VARIANT_A,
        uncertainty_threshold=0.1,
    )
    return ExampleRunRecord(
        run_identity=RunIdentity(
            run_id="aggregate-test-run",
            dataset="triviaqa",
            split="test",
            variant=MethodVariant.VARIANT_A,
            models=models,
            generation_seed_ref="generation-seed-ref",
            paraphrase_seed_ref="paraphrase-seed-ref",
            equivalence_judge_seed_ref="equivalence-seed-ref",
            correctness_judge_seed_ref="correctness-seed-ref",
            uncertainty_threshold=0.1,
            calibration_fingerprint="f" * 64,
            manifest_fingerprint="a" * 64,
            controlled_config_fingerprint="b" * 64,
        ),
        evidence_sources=EvidenceSources(
            target=EvidenceOrigin.COLLECTED,
            nli=EvidenceOrigin.COLLECTED,
            paraphrase=None,
        ),
        evidence=evidence,
        replay=replay,
        correctness_provenance=CorrectnessProvenance(
            candidate_answer=replay.evaluation_candidate,
            question=evidence.example.question,
            reference_answer_groups=evidence.example.reference_answer_groups,
            judge_model_id=models.auxiliary_model_id,
            judge_deployment_ref=models.auxiliary_deployment_ref,
            judge_seed_ref="correctness-seed-ref",
        ),
        correctness=CorrectnessResult(
            correct=correct,
            exact_match=correct,
            matched_group_index=0 if correct else None,
            judge_called=not correct,
            judge_cache_hit=False,
        ),
        real_events=(
            CallEvent(
                event_kind=EventKind.TARGET_GENERATION,
                model_id=TARGET_MODEL_IDS[0],
                latency_seconds=0.2,
                usage=TokenUsage(input_tokens=1, output_tokens=2),
            ),
        ),
    )


def test_run_aggregation_separates_policy_counts_and_real_events() -> None:
    records = (
        make_record("q-1", "A", correct=True),
        make_record("q-2", "B", correct=False),
    )

    aggregate = aggregate_run(records)
    target_events = next(
        item
        for item in aggregate.real_event_aggregates
        if item.event_kind == EventKind.TARGET_GENERATION
    )

    assert aggregate.example_count == 2
    assert aggregate.coverage == 1.0
    assert aggregate.answered_accuracy == 0.5
    assert aggregate.auroc == 0.5
    assert aggregate.recovery is None
    assert aggregate.stable_wrong_rate == 0.5
    assert aggregate.mean_counterfactual_method_calls == 1.0
    assert target_events.event_count == 2
    assert target_events.total_tokens == 6
    assert target_events.total_serial_latency_seconds == 0.4


def test_run_aggregation_rejects_mixed_run_identities() -> None:
    first = make_record("q-1", "A", correct=True)
    second = make_record("q-2", "B", correct=False)
    mixed = ExampleRunRecord(
        run_identity=RunIdentity(
            run_id="aggregate-test-run",
            dataset="triviaqa",
            split="test",
            variant=MethodVariant.VARIANT_A,
            models=second.run_identity.models,
            generation_seed_ref="different-seed-ref",
            paraphrase_seed_ref="paraphrase-seed-ref",
            equivalence_judge_seed_ref="equivalence-seed-ref",
            correctness_judge_seed_ref="correctness-seed-ref",
            uncertainty_threshold=0.1,
            calibration_fingerprint="f" * 64,
            manifest_fingerprint="a" * 64,
            controlled_config_fingerprint="b" * 64,
        ),
        evidence_sources=second.evidence_sources,
        evidence=second.evidence,
        replay=second.replay,
        correctness_provenance=second.correctness_provenance,
        correctness=second.correctness,
        real_events=second.real_events,
    )

    try:
        aggregate_run((first, mixed))
    except ScientificDefinitionError as error:
        assert "scientific run identity" in str(error)
    else:
        raise AssertionError("Mixed run identities must be rejected")
