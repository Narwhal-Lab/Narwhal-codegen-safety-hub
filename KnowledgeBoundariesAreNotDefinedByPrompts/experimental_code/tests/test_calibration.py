

from dataclasses import replace

import pytest

from qprobe.backends import NLILabel
from qprobe.calibration import (
    CalibrationExample,
    CandidateCorrectness,
    SeedCalibrationEvidence,
    TauCalibrationScore,
    calibrate_shared_tau,
    select_calibration_candidate,
)
from qprobe.config import NLI_MODEL_ID, TARGET_MODEL_IDS, TAU_CANDIDATES
from qprobe.datasets import DatasetExample
from qprobe.errors import ScientificDefinitionError
from qprobe.evidence import (
    ExampleEvidence,
    ParaphraseEvidenceProvenance,
    ProbeEvidence,
    PromptEvidence,
    SharedParaphraseEvidence,
    TargetEvidenceProvenance,
)
from qprobe.paraphrasing import RoleParaphrase, VerifiedParaphrase
from qprobe.prompts import ParaphraseRole
from qprobe.semantic import analyze_prompt_semantics
from qprobe.views import PromptView


def make_mutual_entailment_labels(size: int):
    return tuple(
        tuple(
            None if left == right else NLILabel.ENTAILMENT
            for right in range(size)
        )
        for left in range(size)
    )


def make_shared_evidence(
    sample_id: str,
    question: str,
) -> SharedParaphraseEvidence:
    candidates = tuple(
        RoleParaphrase(role=role, prompt=f"{role.value} {question}")
        for role in ParaphraseRole
    )
    verified = tuple(
        VerifiedParaphrase(
            role=candidate.role,
            prompt=candidate.prompt,
            equivalent=False,
        )
        for candidate in candidates
    )
    return SharedParaphraseEvidence(
        example_id=sample_id,
        original_question=question,
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


def make_seed_evidence(
    seed_ref: str,
    shared_by_sample: tuple[SharedParaphraseEvidence, ...],
) -> SeedCalibrationEvidence:
    labels = make_mutual_entailment_labels(10)
    semantics = analyze_prompt_semantics(labels)
    examples = []
    for index, shared in enumerate(shared_by_sample):
        answer = f"Answer {index}"
        dataset_example = DatasetExample(
            dataset="triviaqa",
            sample_id=shared.example_id,
            question=shared.original_question,
            reference_answer_groups=((answer,),),
            source_split="validation",
        )
        original = PromptEvidence(
            role=None,
            view=PromptView(
                view_id=f"{shared.example_id}:original",
                prompt=shared.original_question,
                answers=(answer,) * 10,
                semantics=semantics,
            ),
            directional_labels=labels,
        )
        evidence = ExampleEvidence(
            example=dataset_example,
            provenance=TargetEvidenceProvenance(
                target_model_id=TARGET_MODEL_IDS[0],
                target_deployment_ref="controlled-target-ref",
                generation_seed_ref=seed_ref,
                nli_model_id=NLI_MODEL_ID,
                nli_checkpoint_ref="controlled-nli-checkpoint-ref",
            ),
            original=original,
            shared_paraphrases=shared,
            probe=ProbeEvidence(prompt_views=(), representative_labels=()),
        )
        examples.append(
            CalibrationExample(
                evidence=evidence,
                candidate_correctness=(
                    CandidateCorrectness(answer=answer, correct=index % 2 == 0),
                ),
            )
        )
    return SeedCalibrationEvidence(
        generation_seed_ref=seed_ref,
        examples=tuple(examples),
    )


def make_score(
    tau: float,
    *,
    mean_auroc: float,
    mean_augrc: float,
    mean_method_calls: float,
) -> TauCalibrationScore:
    return TauCalibrationScore(
        tau=tau,
        seed_aurocs=(mean_auroc,) * 3,
        seed_augrcs=(mean_augrc,) * 3,
        seed_mean_method_calls=(mean_method_calls,) * 3,
    )


@pytest.fixture(scope="module")
def calibration_seeds() -> tuple[SeedCalibrationEvidence, ...]:
    shared_by_sample = tuple(
        make_shared_evidence(f"q-{index}", f"Question {index}?")
        for index in range(200)
    )
    return tuple(
        make_seed_evidence(f"generation-seed-{index}", shared_by_sample)
        for index in range(3)
    )


def test_candidate_selection_applies_all_confirmed_tie_breaks() -> None:
    scores = []
    for tau in TAU_CANDIDATES:
        metrics = (0.1, 0.5, 30.0)
        if tau == 0.1:
            metrics = (0.8, 0.5, 30.0)
        elif tau == 0.2:
            metrics = (0.9, 0.4, 30.0)
        elif tau == 0.3:
            metrics = (0.9, 0.3, 20.0)
        elif tau in {0.4, 0.5}:
            metrics = (0.9, 0.3, 10.0)
        scores.append(
            make_score(
                tau,
                mean_auroc=metrics[0],
                mean_augrc=metrics[1],
                mean_method_calls=metrics[2],
            )
        )

    selected = select_calibration_candidate(tuple(scores))

    assert selected.tau == 0.4


def test_calibration_replays_twenty_candidates_over_three_seeds(
    calibration_seeds: tuple[SeedCalibrationEvidence, ...],
) -> None:
    result = calibrate_shared_tau(calibration_seeds)

    assert result.target_model_id == TARGET_MODEL_IDS[0]
    assert result.dataset == "triviaqa"
    assert result.generation_seed_refs == tuple(
        f"generation-seed-{index}" for index in range(3)
    )
    assert tuple(score.tau for score in result.candidate_scores) == TAU_CANDIDATES
    assert result.selected_tau in TAU_CANDIDATES


def test_seed_evidence_rejects_the_wrong_dev_count(
    calibration_seeds: tuple[SeedCalibrationEvidence, ...],
) -> None:
    seed = calibration_seeds[0]

    with pytest.raises(ScientificDefinitionError):
        SeedCalibrationEvidence(
            generation_seed_ref=seed.generation_seed_ref,
            examples=seed.examples[:-1],
        )


def test_calibration_rejects_duplicate_seed_references(
    calibration_seeds: tuple[SeedCalibrationEvidence, ...],
) -> None:
    with pytest.raises(ScientificDefinitionError):
        calibrate_shared_tau(
            (calibration_seeds[0], calibration_seeds[0], calibration_seeds[2])
        )


def test_calibration_rejects_different_sample_order(
    calibration_seeds: tuple[SeedCalibrationEvidence, ...],
) -> None:
    seed = calibration_seeds[1]
    reordered = SeedCalibrationEvidence(
        generation_seed_ref=seed.generation_seed_ref,
        examples=(seed.examples[1], seed.examples[0], *seed.examples[2:]),
    )

    with pytest.raises(ScientificDefinitionError):
        calibrate_shared_tau(
            (calibration_seeds[0], reordered, calibration_seeds[2])
        )


def test_calibration_rejects_different_shared_paraphrase_evidence(
    calibration_seeds: tuple[SeedCalibrationEvidence, ...],
) -> None:
    seed = calibration_seeds[1]
    item = seed.examples[0]
    shared = item.evidence.shared_paraphrases
    assert shared is not None
    different_shared = replace(
        shared,
        provenance=replace(
            shared.provenance,
            paraphrase_seed_ref="different-paraphrase-seed-ref",
        ),
    )
    different_item = replace(
        item,
        evidence=replace(item.evidence, shared_paraphrases=different_shared),
    )
    different_seed = SeedCalibrationEvidence(
        generation_seed_ref=seed.generation_seed_ref,
        examples=(different_item, *seed.examples[1:]),
    )

    with pytest.raises(ScientificDefinitionError):
        calibrate_shared_tau(
            (calibration_seeds[0], different_seed, calibration_seeds[2])
        )


def test_calibration_rejects_conflicting_candidate_correctness(
    calibration_seeds: tuple[SeedCalibrationEvidence, ...],
) -> None:
    seed = calibration_seeds[1]
    item = seed.examples[0]
    candidate = item.candidate_correctness[0]
    conflicting_item = replace(
        item,
        candidate_correctness=(
            replace(candidate, correct=not candidate.correct),
        ),
    )
    conflicting_seed = SeedCalibrationEvidence(
        generation_seed_ref=seed.generation_seed_ref,
        examples=(conflicting_item, *seed.examples[1:]),
    )

    with pytest.raises(ScientificDefinitionError):
        calibrate_shared_tau(
            (calibration_seeds[0], conflicting_seed, calibration_seeds[2])
        )
