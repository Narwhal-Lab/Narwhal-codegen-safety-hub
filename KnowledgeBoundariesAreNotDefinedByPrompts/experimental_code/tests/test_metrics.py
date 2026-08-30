

import math

import pytest

from qprobe.errors import ScientificDefinitionError
from qprobe.metrics import (
    AcceptedAnswer,
    AnswerOutcome,
    ProbeOutcome,
    ScoredExample,
    aggregate_three_seeds,
    augrc,
    auroc,
    correct_to_wrong_gain,
    recovery_metrics,
    stable_wrong_rate,
)


def test_auroc_gives_half_credit_to_confidence_ties() -> None:
    examples = (
        ScoredExample("correct", True, 0.0),
        ScoredExample("wrong", False, 0.0),
    )

    assert auroc(examples) == 0.5


def test_auroc_fails_when_only_one_class_is_present() -> None:
    with pytest.raises(ScientificDefinitionError):
        auroc((ScoredExample("correct", True, 0.5),))


def test_augrc_is_invariant_to_order_within_confidence_ties() -> None:
    examples = (
        ScoredExample("a", True, 0.8),
        ScoredExample("b", False, 0.8),
        ScoredExample("c", True, 0.2),
    )

    assert augrc(examples) == augrc(tuple(reversed(examples)))


def test_augrc_uses_generalized_risk() -> None:
    examples = (
        ScoredExample("high-wrong", False, 0.8),
        ScoredExample("low-correct", True, 0.2),
    )

    assert augrc(examples) == 0.375


def test_recovery_metrics_use_only_probed_examples() -> None:
    metrics = recovery_metrics(
        (
            ProbeOutcome(answered=True, correct=True),
            ProbeOutcome(answered=True, correct=False),
            ProbeOutcome(answered=False, correct=True),
        )
    )

    assert metrics.probed_count == 3
    assert metrics.correct_recovery_rate == 1 / 3
    assert metrics.wrong_recovery_rate == 1 / 3


def test_stable_wrong_rate_uses_accepted_answers_as_denominator() -> None:
    rate = stable_wrong_rate(
        (
            AcceptedAnswer(correct=True, low_uncertainty=True),
            AcceptedAnswer(correct=False, low_uncertainty=True),
            AcceptedAnswer(correct=False, low_uncertainty=False),
        )
    )

    assert rate == 1 / 3
    assert stable_wrong_rate(()) is None


def test_correct_to_wrong_gain_reports_na_for_zero_wrong_gain() -> None:
    stage1 = (
        AnswerOutcome("a", answered=False, correct=True),
        AnswerOutcome("b", answered=False, correct=False),
    )
    variant = (
        AnswerOutcome("a", answered=True, correct=True),
        AnswerOutcome("b", answered=False, correct=False),
    )

    gain = correct_to_wrong_gain(stage1, variant)

    assert gain.new_correct_answers == 1
    assert gain.new_wrong_answers == 0
    assert gain.ratio is None


def test_correct_to_wrong_gain_rejects_changed_stage1_answers() -> None:
    stage1 = (AnswerOutcome("a", answered=True, correct=True),)
    variant = (AnswerOutcome("a", answered=False, correct=True),)

    with pytest.raises(ScientificDefinitionError):
        correct_to_wrong_gain(stage1, variant)


def test_three_seed_aggregation_uses_sample_standard_deviation() -> None:
    aggregate = aggregate_three_seeds((1.0, 2.0, 3.0))

    assert aggregate.mean == 2.0
    assert math.isclose(aggregate.standard_deviation, 1.0)
