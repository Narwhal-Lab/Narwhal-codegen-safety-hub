

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from qprobe.errors import ConfigurationError, ScientificDefinitionError
from qprobe.semantic import (
    DirectionalLabels,
    dominant_early_exit,
    mutual_entailment_components,
    validate_directional_labels,
)
from qprobe.views import PromptView


class MethodVariant(str, Enum):
    STAGE1_ONLY = "stage1_only"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    VARIANT_B_NO_EQUIV = "variant_b_no_equiv"


class DecisionKind(str, Enum):
    DIRECT_ACCEPT = "direct_accept"
    EARLY_EXIT = "early_exit"
    RECOVERED = "recovered"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class MethodDecision:
    kind: DecisionKind
    answer: str | None
    selected_view_id: str | None
    reason: str
    supporting_view_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        accepted = self.kind != DecisionKind.ABSTAIN
        if accepted and (self.answer is None or self.selected_view_id is None):
            raise ScientificDefinitionError("Accepted decisions require an answer view")
        if not accepted and (self.answer is not None or self.selected_view_id is not None):
            raise ScientificDefinitionError("Abstention cannot contain an answer view")


def _validate_uncertainty_threshold(uncertainty_threshold: float) -> None:
    if (
        not isinstance(uncertainty_threshold, (int, float))
        or isinstance(uncertainty_threshold, bool)
        or not math.isfinite(uncertainty_threshold)
        or uncertainty_threshold < 0
    ):
        raise ConfigurationError(
            "uncertainty_threshold must be a finite non-negative number"
        )


def decide_original_view(
    original: PromptView,
    *,
    variant: MethodVariant,
    uncertainty_threshold: float,
    dominant_threshold: float = 0.5,
    competitor_threshold: float = 0.10,
) -> MethodDecision | None:


    _validate_uncertainty_threshold(uncertainty_threshold)
    if original.semantics.uncertainty < uncertainty_threshold:
        return MethodDecision(
            kind=DecisionKind.DIRECT_ACCEPT,
            answer=original.representative_answer,
            selected_view_id=original.view_id,
            reason="original_uncertainty_below_threshold",
            supporting_view_ids=(original.view_id,),
        )
    if variant == MethodVariant.STAGE1_ONLY:
        return MethodDecision(
            kind=DecisionKind.ABSTAIN,
            answer=None,
            selected_view_id=None,
            reason="stage1_uncertainty_not_below_threshold",
        )
    if variant in {MethodVariant.VARIANT_B, MethodVariant.VARIANT_B_NO_EQUIV}:
        if dominant_early_exit(
            original.semantics,
            dominant_threshold=dominant_threshold,
            competitor_threshold=competitor_threshold,
        ):
            return MethodDecision(
                kind=DecisionKind.EARLY_EXIT,
                answer=original.representative_answer,
                selected_view_id=original.view_id,
                reason="dominant_mode_without_competing_mode",
                supporting_view_ids=(original.view_id,),
            )
    return None


def decide_recovery(
    views: Sequence[PromptView],
    representative_labels: DirectionalLabels,
    *,
    uncertainty_threshold: float,
    minimum_support: int = 2,
) -> MethodDecision:


    _validate_uncertainty_threshold(uncertainty_threshold)
    if (
        not isinstance(minimum_support, int)
        or isinstance(minimum_support, bool)
        or minimum_support <= 0
    ):
        raise ConfigurationError("minimum_support must be a positive integer")
    if len(views) != validate_directional_labels(representative_labels):
        raise ScientificDefinitionError(
            "View count does not match representative-answer NLI labels"
        )
    stable_indices = tuple(
        index
        for index, view in enumerate(views)
        if view.semantics.uncertainty < uncertainty_threshold
    )
    if len(stable_indices) < minimum_support:
        return MethodDecision(
            kind=DecisionKind.ABSTAIN,
            answer=None,
            selected_view_id=None,
            reason="insufficient_low_uncertainty_views",
        )

    stable_labels = tuple(
        tuple(
            None if left == right else representative_labels[source_left][source_right]
            for right, source_right in enumerate(stable_indices)
        )
        for left, source_left in enumerate(stable_indices)
    )
    components = mutual_entailment_components(stable_labels)
    qualifying = tuple(
        component for component in components if len(component) >= minimum_support
    )
    if not qualifying:
        return MethodDecision(
            kind=DecisionKind.ABSTAIN,
            answer=None,
            selected_view_id=None,
            reason="no_answer_group_reached_minimum_support",
        )
    if len(qualifying) > 1:
        return MethodDecision(
            kind=DecisionKind.ABSTAIN,
            answer=None,
            selected_view_id=None,
            reason="multiple_competing_stable_answer_groups",
        )

    supporting_global_indices = tuple(
        stable_indices[local_index] for local_index in qualifying[0]
    )
    selected_index = min(
        supporting_global_indices,
        key=lambda index: (views[index].semantics.uncertainty, index),
    )
    selected = views[selected_index]
    return MethodDecision(
        kind=DecisionKind.RECOVERED,
        answer=selected.representative_answer,
        selected_view_id=selected.view_id,
        reason="unique_answer_group_reached_minimum_support",
        supporting_view_ids=tuple(
            views[index].view_id for index in supporting_global_indices
        ),
    )
