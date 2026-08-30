

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from qprobe.config import TAU_CANDIDATES
from qprobe.decision import MethodVariant
from qprobe.errors import ScientificDefinitionError
from qprobe.evidence import ExampleEvidence
from qprobe.metrics import ScoredExample, augrc, auroc
from qprobe.pipeline import replay_decision


DEV_EXAMPLE_COUNT = 200
GENERATION_SEED_COUNT = 3


def _require_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ScientificDefinitionError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CandidateCorrectness:


    answer: str
    correct: bool

    def __post_init__(self) -> None:
        _require_non_empty_string(self.answer, "answer")
        if not isinstance(self.correct, bool):
            raise ScientificDefinitionError("correct must be boolean")


@dataclass(frozen=True, slots=True)
class CalibrationExample:


    evidence: ExampleEvidence
    candidate_correctness: tuple[CandidateCorrectness, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ExampleEvidence):
            raise ScientificDefinitionError("evidence must be ExampleEvidence")
        if self.evidence.shared_paraphrases is None or self.evidence.probe is None:
            raise ScientificDefinitionError(
                "Calibration requires complete paraphrase and probe evidence"
            )
        if len(self.evidence.original.view.answers) != 10:
            raise ScientificDefinitionError(
                "Calibration evidence must contain ten target samples per view"
            )
        if not isinstance(self.candidate_correctness, tuple) or not all(
            isinstance(item, CandidateCorrectness)
            for item in self.candidate_correctness
        ):
            raise ScientificDefinitionError(
                "candidate_correctness must contain CandidateCorrectness records"
            )
        answers = tuple(item.answer for item in self.candidate_correctness)
        if len(set(answers)) != len(answers):
            raise ScientificDefinitionError(
                "Candidate correctness answers must be unique"
            )
        required_roles = self.evidence.shared_paraphrases.roles_for_variant(
            MethodVariant.VARIANT_A
        )
        views_by_role = {
            item.role: item.view for item in self.evidence.probe.prompt_views
        }
        missing_roles = tuple(
            role for role in required_roles if role not in views_by_role
        )
        if missing_roles:
            raise ScientificDefinitionError(
                "Calibration probe evidence is missing retained Variant A roles"
            )
        required_answers = {self.evidence.original.view.representative_answer}
        required_answers.update(
            views_by_role[role].representative_answer for role in required_roles
        )
        if set(answers) != required_answers:
            raise ScientificDefinitionError(
                "Candidate correctness must cover exactly the Variant A candidates"
            )

    def correctness_for(self, answer: str) -> bool:


        for item in self.candidate_correctness:
            if item.answer == answer:
                return item.correct
        raise ScientificDefinitionError(
            "Replay selected a candidate without cached correctness"
        )


@dataclass(frozen=True, slots=True)
class SeedCalibrationEvidence:


    generation_seed_ref: str
    examples: tuple[CalibrationExample, ...]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.generation_seed_ref, "generation_seed_ref")
        if (
            not isinstance(self.examples, tuple)
            or len(self.examples) != DEV_EXAMPLE_COUNT
        ):
            raise ScientificDefinitionError(
                f"Calibration requires exactly {DEV_EXAMPLE_COUNT} dev examples"
            )
        if not all(isinstance(item, CalibrationExample) for item in self.examples):
            raise ScientificDefinitionError(
                "examples must contain CalibrationExample records"
            )
        sample_ids = tuple(item.evidence.example.sample_id for item in self.examples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ScientificDefinitionError("Calibration sample IDs must be unique")
        if any(
            item.evidence.provenance.generation_seed_ref
            != self.generation_seed_ref
            for item in self.examples
        ):
            raise ScientificDefinitionError(
                "Evidence generation seed does not match its calibration seed"
            )
        if len({item.evidence.example.dataset for item in self.examples}) != 1:
            raise ScientificDefinitionError(
                "One calibration seed must contain exactly one dataset"
            )
        target_model_ids = {
            item.evidence.provenance.target_model_id for item in self.examples
        }
        if len(target_model_ids) != 1:
            raise ScientificDefinitionError(
                "One calibration seed must contain exactly one target model"
            )


@dataclass(frozen=True, slots=True)
class TauCalibrationScore:


    tau: float
    seed_aurocs: tuple[float, float, float]
    seed_augrcs: tuple[float, float, float]
    seed_mean_method_calls: tuple[float, float, float]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tau, float)
            or not math.isfinite(self.tau)
            or self.tau not in TAU_CANDIDATES
        ):
            raise ScientificDefinitionError("tau must be a confirmed candidate")
        metric_groups = (
            self.seed_aurocs,
            self.seed_augrcs,
            self.seed_mean_method_calls,
        )
        if any(
            not isinstance(group, tuple)
            or len(group) != GENERATION_SEED_COUNT
            for group in metric_groups
        ):
            raise ScientificDefinitionError(
                "Each calibration metric must contain exactly three seed values"
            )
        values = self.seed_aurocs + self.seed_augrcs + self.seed_mean_method_calls
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in values
        ):
            raise ScientificDefinitionError("Calibration metrics must be finite")
        if any(not 0.0 <= value <= 1.0 for value in self.seed_aurocs):
            raise ScientificDefinitionError("Seed AUROC values must be in [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in self.seed_augrcs):
            raise ScientificDefinitionError("Seed AUGRC values must be in [0, 1]")
        if any(value < 0 for value in self.seed_mean_method_calls):
            raise ScientificDefinitionError("Mean method calls cannot be negative")

    @property
    def mean_auroc(self) -> float:
        return sum(self.seed_aurocs) / GENERATION_SEED_COUNT

    @property
    def mean_augrc(self) -> float:
        return sum(self.seed_augrcs) / GENERATION_SEED_COUNT

    @property
    def mean_method_calls(self) -> float:
        return sum(self.seed_mean_method_calls) / GENERATION_SEED_COUNT


@dataclass(frozen=True, slots=True)
class CalibrationResult:


    target_model_id: str
    dataset: str
    generation_seed_refs: tuple[str, str, str]
    selected_tau: float
    candidate_scores: tuple[TauCalibrationScore, ...]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.target_model_id, "target_model_id")
        _require_non_empty_string(self.dataset, "dataset")
        valid_seed_refs = (
            isinstance(self.generation_seed_refs, tuple)
            and len(self.generation_seed_refs) == GENERATION_SEED_COUNT
            and all(
                isinstance(seed_ref, str) and bool(seed_ref.strip())
                for seed_ref in self.generation_seed_refs
            )
        )
        if not valid_seed_refs:
            raise ScientificDefinitionError(
                "generation_seed_refs must contain three non-empty references"
            )
        if len(set(self.generation_seed_refs)) != GENERATION_SEED_COUNT:
            raise ScientificDefinitionError("Generation seed references must be unique")
        if not isinstance(self.candidate_scores, tuple) or not all(
            isinstance(score, TauCalibrationScore)
            for score in self.candidate_scores
        ):
            raise ScientificDefinitionError(
                "candidate_scores must contain TauCalibrationScore records"
            )
        if tuple(score.tau for score in self.candidate_scores) != TAU_CANDIDATES:
            raise ScientificDefinitionError(
                "candidate_scores must contain all tau candidates in fixed order"
            )
        if (
            not isinstance(self.selected_tau, float)
            or self.selected_tau not in TAU_CANDIDATES
        ):
            raise ScientificDefinitionError("selected_tau must be a confirmed candidate")
        selected = select_calibration_candidate(self.candidate_scores)
        if self.selected_tau != selected.tau:
            raise ScientificDefinitionError(
                "selected_tau does not match the calibration scores"
            )


def select_calibration_candidate(
    candidate_scores: Sequence[TauCalibrationScore],
) -> TauCalibrationScore:


    scores = tuple(candidate_scores)
    if not all(isinstance(score, TauCalibrationScore) for score in scores):
        raise ScientificDefinitionError(
            "candidate_scores must contain TauCalibrationScore records"
        )
    if tuple(score.tau for score in scores) != TAU_CANDIDATES:
        raise ScientificDefinitionError(
            "Calibration scores must contain all tau candidates in fixed order"
        )
    return min(
        scores,
        key=lambda score: (
            -score.mean_auroc,
            score.mean_augrc,
            score.mean_method_calls,
            score.tau,
        ),
    )


def _validate_seed_alignment(
    seeds: tuple[SeedCalibrationEvidence, ...],
) -> tuple[str, str]:
    if len(seeds) != GENERATION_SEED_COUNT:
        raise ScientificDefinitionError(
            f"Calibration requires exactly {GENERATION_SEED_COUNT} generation seeds"
        )
    if not all(isinstance(seed, SeedCalibrationEvidence) for seed in seeds):
        raise ScientificDefinitionError(
            "seeds must contain SeedCalibrationEvidence records"
        )
    seed_refs = tuple(seed.generation_seed_ref for seed in seeds)
    if len(set(seed_refs)) != GENERATION_SEED_COUNT:
        raise ScientificDefinitionError("Generation seed references must be unique")

    baseline = seeds[0]
    baseline_examples = tuple(item.evidence for item in baseline.examples)
    dataset = baseline_examples[0].example.dataset
    target_model_id = baseline_examples[0].provenance.target_model_id
    target_identity = (
        baseline_examples[0].provenance.target_model_id,
        baseline_examples[0].provenance.target_deployment_ref,
        baseline_examples[0].provenance.nli_model_id,
        baseline_examples[0].provenance.nli_checkpoint_ref,
    )
    correctness_by_candidate: dict[tuple[str, str], bool] = {}
    for seed_index, seed in enumerate(seeds):
        for example_index, item in enumerate(seed.examples):
            evidence = item.evidence
            baseline_evidence = baseline_examples[example_index]
            if seed_index > 0 and evidence.example != baseline_evidence.example:
                raise ScientificDefinitionError(
                    "Calibration seeds must use the same ordered dev examples"
                )
            if (
                seed_index > 0
                and evidence.shared_paraphrases
                != baseline_evidence.shared_paraphrases
            ):
                raise ScientificDefinitionError(
                    "Calibration seeds must reuse the same paraphrase evidence"
                )
            identity = (
                evidence.provenance.target_model_id,
                evidence.provenance.target_deployment_ref,
                evidence.provenance.nli_model_id,
                evidence.provenance.nli_checkpoint_ref,
            )
            if identity != target_identity:
                raise ScientificDefinitionError(
                    "Calibration seeds must share target and NLI provenance"
                )
            for candidate in item.candidate_correctness:
                key = (evidence.example.sample_id, candidate.answer)
                previous = correctness_by_candidate.setdefault(
                    key, candidate.correct
                )
                if previous != candidate.correct:
                    raise ScientificDefinitionError(
                        "Candidate correctness must agree across generation seeds"
                    )
    return target_model_id, dataset


def _score_seed(
    seed: SeedCalibrationEvidence,
    tau: float,
) -> tuple[float, float, float]:
    scored_examples: list[ScoredExample] = []
    total_method_calls = 0
    for item in seed.examples:
        replay = replay_decision(
            item.evidence,
            variant=MethodVariant.VARIANT_A,
            uncertainty_threshold=tau,
        )
        scored_examples.append(
            ScoredExample(
                sample_id=item.evidence.example.sample_id,
                correct=item.correctness_for(replay.evaluation_candidate),
                confidence=replay.confidence,
            )
        )
        total_method_calls += replay.policy_calls.total_method_calls
    return (
        auroc(scored_examples),
        augrc(scored_examples),
        total_method_calls / len(seed.examples),
    )


def calibrate_shared_tau(
    seeds: Sequence[SeedCalibrationEvidence],
) -> CalibrationResult:


    seed_tuple = tuple(seeds)
    target_model_id, dataset = _validate_seed_alignment(seed_tuple)
    scores: list[TauCalibrationScore] = []
    for tau in TAU_CANDIDATES:
        seed_metrics = tuple(_score_seed(seed, tau) for seed in seed_tuple)
        scores.append(
            TauCalibrationScore(
                tau=tau,
                seed_aurocs=tuple(item[0] for item in seed_metrics),
                seed_augrcs=tuple(item[1] for item in seed_metrics),
                seed_mean_method_calls=tuple(item[2] for item in seed_metrics),
            )
        )
    candidate_scores = tuple(scores)
    selected = select_calibration_candidate(candidate_scores)
    return CalibrationResult(
        target_model_id=target_model_id,
        dataset=dataset,
        generation_seed_refs=tuple(seed.generation_seed_ref for seed in seed_tuple),
        selected_tau=selected.tau,
        candidate_scores=candidate_scores,
    )
