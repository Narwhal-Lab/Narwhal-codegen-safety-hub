

import numpy as np

from qprobe.backends import NLILabel
from qprobe.semantic import (
    analyze_prompt_semantics,
    build_heat_kernel,
    build_weight_matrix,
    dominant_early_exit,
)


def labels_for_two_equivalent_answers():
    return (
        (None, NLILabel.ENTAILMENT),
        (NLILabel.ENTAILMENT, None),
    )


def test_directional_weights_are_averaged_symmetrically() -> None:
    labels = (
        (None, NLILabel.ENTAILMENT),
        (NLILabel.NEUTRAL, None),
    )
    weights = build_weight_matrix(labels)

    assert np.array_equal(weights, np.array([[0.0, 0.75], [0.75, 0.0]]))


def test_heat_kernel_is_trace_normalized_and_psd() -> None:
    weights = build_weight_matrix(labels_for_two_equivalent_answers())
    _, kernel = build_heat_kernel(weights)

    assert np.isclose(np.trace(kernel), 1.0)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-12


def test_mutual_entailment_forms_one_mode() -> None:
    semantics = analyze_prompt_semantics(labels_for_two_equivalent_answers())

    assert len(semantics.modes) == 1
    assert semantics.dominant_strength == 1.0
    assert semantics.competitor_strength == 0.0
    assert dominant_early_exit(semantics)
