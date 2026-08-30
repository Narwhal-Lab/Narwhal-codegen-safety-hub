

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from qprobe.errors import ScientificDefinitionError


@dataclass(frozen=True, slots=True)
class ScoredExample:


    sample_id: str
    correct: bool
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ScientificDefinitionError("sample_id must be a non-empty string")
        if not isinstance(self.correct, bool):
            raise ScientificDefinitionError("correct must be boolean")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
        ):
            raise ScientificDefinitionError("confidence must be finite")


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    coverage: float
    risk: float

    def __post_init__(self) -> None:
        values = (self.coverage, self.risk)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in values
        ):
            raise ScientificDefinitionError(
                "Risk-coverage values must be finite and in [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class MeanStandardDeviation:
    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        values = (self.mean, self.standard_deviation)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in values
        ) or self.standard_deviation < 0:
            raise ScientificDefinitionError(
                "Aggregate mean and standard deviation must be finite"
            )


@dataclass(frozen=True, slots=True)
class ProbeOutcome:


    answered: bool
    correct: bool

    def __post_init__(self) -> None:
        if not isinstance(self.answered, bool) or not isinstance(self.correct, bool):
            raise ScientificDefinitionError("Probe outcome fields must be boolean")


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    probed_count: int
    correct_recovery_count: int
    wrong_recovery_count: int
    correct_recovery_rate: float
    wrong_recovery_rate: float

    def __post_init__(self) -> None:
        counts = (
            self.probed_count,
            self.correct_recovery_count,
            self.wrong_recovery_count,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ) or self.probed_count == 0:
            raise ScientificDefinitionError("Recovery counts must be valid integers")
        if self.correct_recovery_count + self.wrong_recovery_count > self.probed_count:
            raise ScientificDefinitionError("Recovery counts exceed probed examples")
        expected_rates = (
            self.correct_recovery_count / self.probed_count,
            self.wrong_recovery_count / self.probed_count,
        )
        actual_rates = (self.correct_recovery_rate, self.wrong_recovery_rate)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in actual_rates
        ) or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(actual_rates, expected_rates)
        ):
            raise ScientificDefinitionError("Recovery rates do not match counts")


@dataclass(frozen=True, slots=True)
class AcceptedAnswer:


    correct: bool
    low_uncertainty: bool

    def __post_init__(self) -> None:
        if not isinstance(self.correct, bool) or not isinstance(
            self.low_uncertainty, bool
        ):
            raise ScientificDefinitionError("Accepted-answer fields must be boolean")


@dataclass(frozen=True, slots=True)
class AnswerOutcome:


    sample_id: str
    answered: bool
    correct: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ScientificDefinitionError("sample_id must be a non-empty string")
        if not isinstance(self.answered, bool) or not isinstance(self.correct, bool):
            raise ScientificDefinitionError("Answer outcome fields must be boolean")


@dataclass(frozen=True, slots=True)
class CorrectToWrongGain:
    new_correct_answers: int
    new_wrong_answers: int
    ratio: float | None

    def __post_init__(self) -> None:
        counts = (self.new_correct_answers, self.new_wrong_answers)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise ScientificDefinitionError("Gain counts must be non-negative integers")
        if self.new_wrong_answers == 0:
            if self.ratio is not None:
                raise ScientificDefinitionError("Zero wrong gain requires an N/A ratio")
            return
        expected = self.new_correct_answers / self.new_wrong_answers
        if (
            not isinstance(self.ratio, (int, float))
            or isinstance(self.ratio, bool)
            or not math.isfinite(self.ratio)
            or not math.isclose(self.ratio, expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ScientificDefinitionError("Gain ratio does not match gain counts")


def _validate_scored_examples(examples: Sequence[ScoredExample]) -> None:
    if not examples:
        raise ScientificDefinitionError("At least one scored example is required")
    if not all(isinstance(example, ScoredExample) for example in examples):
        raise ScientificDefinitionError("All inputs must be ScoredExample records")
    sample_ids = tuple(example.sample_id for example in examples)
    if len(set(sample_ids)) != len(sample_ids):
        raise ScientificDefinitionError("Scored example IDs must be unique")


def auroc(examples: Sequence[ScoredExample]) -> float:


    _validate_scored_examples(examples)
    positive_confidences = tuple(
        example.confidence for example in examples if example.correct
    )
    negative_confidences = tuple(
        example.confidence for example in examples if not example.correct
    )
    if not positive_confidences or not negative_confidences:
        raise ScientificDefinitionError(
            "AUROC is undefined without both correct and incorrect examples"
        )
    credit = 0.0
    for positive in positive_confidences:
        for negative in negative_confidences:
            if positive > negative:
                credit += 1.0
            elif positive == negative:
                credit += 0.5
    return credit / (len(positive_confidences) * len(negative_confidences))


def risk_coverage_curve(
    examples: Sequence[ScoredExample],
) -> tuple[RiskCoveragePoint, ...]:


    _validate_scored_examples(examples)
    confidence_groups: dict[float, list[ScoredExample]] = {}
    for example in examples:
        confidence_groups.setdefault(float(example.confidence), []).append(example)

    total = len(examples)
    answered = 0
    errors = 0
    points = [RiskCoveragePoint(coverage=0.0, risk=0.0)]
    for confidence in sorted(confidence_groups, reverse=True):
        group = confidence_groups[confidence]
        answered += len(group)
        errors += sum(not example.correct for example in group)
        points.append(
            RiskCoveragePoint(
                coverage=answered / total,
                risk=errors / total,
            )
        )
    return tuple(points)


def augrc(examples: Sequence[ScoredExample]) -> float:


    points = risk_coverage_curve(examples)
    area = 0.0
    for left, right in zip(points, points[1:]):
        width = right.coverage - left.coverage
        area += width * (left.risk + right.risk) / 2.0
    return area


def recovery_metrics(outcomes: Sequence[ProbeOutcome]) -> RecoveryMetrics:


    if not outcomes:
        raise ScientificDefinitionError("At least one probed outcome is required")
    if not all(isinstance(outcome, ProbeOutcome) for outcome in outcomes):
        raise ScientificDefinitionError("All outcomes must be ProbeOutcome records")
    correct_count = sum(outcome.answered and outcome.correct for outcome in outcomes)
    wrong_count = sum(outcome.answered and not outcome.correct for outcome in outcomes)
    denominator = len(outcomes)
    return RecoveryMetrics(
        probed_count=denominator,
        correct_recovery_count=correct_count,
        wrong_recovery_count=wrong_count,
        correct_recovery_rate=correct_count / denominator,
        wrong_recovery_rate=wrong_count / denominator,
    )


def stable_wrong_rate(accepted_answers: Sequence[AcceptedAnswer]) -> float | None:


    if not all(
        isinstance(answer, AcceptedAnswer) for answer in accepted_answers
    ):
        raise ScientificDefinitionError(
            "All accepted answers must be AcceptedAnswer records"
        )
    if not accepted_answers:
        return None
    stable_wrong = sum(
        answer.low_uncertainty and not answer.correct for answer in accepted_answers
    )
    return stable_wrong / len(accepted_answers)


def correct_to_wrong_gain(
    stage1_outcomes: Sequence[AnswerOutcome],
    variant_outcomes: Sequence[AnswerOutcome],
) -> CorrectToWrongGain:


    if not stage1_outcomes or not variant_outcomes:
        raise ScientificDefinitionError(
            "Correct-to-Wrong Gain requires non-empty paired outcomes"
        )
    if not all(isinstance(item, AnswerOutcome) for item in stage1_outcomes):
        raise ScientificDefinitionError("Stage 1 outcomes have an invalid record")
    if not all(isinstance(item, AnswerOutcome) for item in variant_outcomes):
        raise ScientificDefinitionError("Variant outcomes have an invalid record")
    stage1_by_id = {item.sample_id: item for item in stage1_outcomes}
    variant_by_id = {item.sample_id: item for item in variant_outcomes}
    if len(stage1_by_id) != len(stage1_outcomes) or len(variant_by_id) != len(
        variant_outcomes
    ):
        raise ScientificDefinitionError("Answer outcome IDs must be unique")
    if stage1_by_id.keys() != variant_by_id.keys():
        raise ScientificDefinitionError("Stage 1 and variant IDs must match")

    new_correct = 0
    new_wrong = 0
    for sample_id, variant in variant_by_id.items():
        stage1 = stage1_by_id[sample_id]
        if stage1.answered and not variant.answered:
            raise ScientificDefinitionError(
                "A variant cannot remove a Stage 1 accepted answer"
            )
        if stage1.answered and variant.correct != stage1.correct:
            raise ScientificDefinitionError(
                "A variant cannot change Stage 1 accepted-answer correctness"
            )
        if variant.answered and not stage1.answered:
            if variant.correct:
                new_correct += 1
            else:
                new_wrong += 1
    return CorrectToWrongGain(
        new_correct_answers=new_correct,
        new_wrong_answers=new_wrong,
        ratio=None if new_wrong == 0 else new_correct / new_wrong,
    )


def aggregate_three_seeds(values: Sequence[float]) -> MeanStandardDeviation:


    if len(values) != 3:
        raise ScientificDefinitionError("Exactly three seed values are required")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        for value in values
    ):
        raise ScientificDefinitionError("Seed values must be finite numbers")
    mean = sum(values) / 3
    variance = sum((value - mean) ** 2 for value in values) / 2
    return MeanStandardDeviation(
        mean=mean,
        standard_deviation=math.sqrt(variance),
    )
