

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from qprobe.backends import NLILabel
from qprobe.errors import ConfigurationError, ScientificDefinitionError


FloatMatrix = NDArray[np.float64]
DirectionalLabels = Sequence[Sequence[NLILabel | None]]

LABEL_WEIGHTS: dict[NLILabel, float] = {
    NLILabel.ENTAILMENT: 1.0,
    NLILabel.NEUTRAL: 0.5,
    NLILabel.CONTRADICTION: 0.0,
}


@dataclass(frozen=True, slots=True)
class SemanticMode:


    member_indices: tuple[int, ...]
    representative_index: int

    @property
    def size(self) -> int:
        return len(self.member_indices)


@dataclass(frozen=True, slots=True)
class PromptSemantics:


    weight_matrix: FloatMatrix
    laplacian: FloatMatrix
    normalized_kernel: FloatMatrix
    uncertainty: float
    modes: tuple[SemanticMode, ...]
    dominant_mode_index: int
    dominant_strength: float
    competitor_strength: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PromptSemantics):
            return NotImplemented
        return (
            np.array_equal(self.weight_matrix, other.weight_matrix)
            and np.array_equal(self.laplacian, other.laplacian)
            and np.array_equal(self.normalized_kernel, other.normalized_kernel)
            and self.uncertainty == other.uncertainty
            and self.modes == other.modes
            and self.dominant_mode_index == other.dominant_mode_index
            and self.dominant_strength == other.dominant_strength
            and self.competitor_strength == other.competitor_strength
        )

    @property
    def dominant_mode(self) -> SemanticMode:
        return self.modes[self.dominant_mode_index]

    @property
    def representative_index(self) -> int:
        return self.dominant_mode.representative_index


def validate_directional_labels(labels: DirectionalLabels) -> int:


    size = len(labels)
    if size == 0:
        raise ScientificDefinitionError("At least one answer is required")
    if any(len(row) != size for row in labels):
        raise ScientificDefinitionError("Directional NLI labels must be square")
    for row_index, row in enumerate(labels):
        for column_index, label in enumerate(row):
            if row_index == column_index:
                if label is not None:
                    raise ScientificDefinitionError("NLI diagonal must be None")
            elif not isinstance(label, NLILabel):
                raise ScientificDefinitionError(
                    f"Missing NLI label at ({row_index}, {column_index})"
                )
    return size


def build_weight_matrix(labels: DirectionalLabels) -> FloatMatrix:


    size = validate_directional_labels(labels)
    weights = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            forward = labels[left][right]
            backward = labels[right][left]
            if forward is None or backward is None:
                raise ScientificDefinitionError("Off-diagonal NLI label is missing")
            weight = (LABEL_WEIGHTS[forward] + LABEL_WEIGHTS[backward]) / 2.0
            weights[left, right] = weight
            weights[right, left] = weight
    return weights


def build_heat_kernel(
    weights: FloatMatrix, *, heat_time: float = 0.3
) -> tuple[FloatMatrix, FloatMatrix]:


    if heat_time <= 0:
        raise ConfigurationError("heat_time must be positive")
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ScientificDefinitionError("Weight matrix must be non-empty and square")
    if not np.all(np.isfinite(matrix)):
        raise ScientificDefinitionError("Weight matrix contains non-finite values")
    if np.any(matrix < 0):
        raise ScientificDefinitionError("Weight matrix cannot contain negative values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ScientificDefinitionError("Weight matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ScientificDefinitionError("Weight matrix diagonal must be zero")

    degrees = np.sum(matrix, axis=1)
    laplacian = np.diag(degrees) - matrix
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    heat_eigenvalues = np.exp(-heat_time * np.clip(eigenvalues, 0.0, None))
    kernel = (eigenvectors * heat_eigenvalues) @ eigenvectors.T
    kernel = (kernel + kernel.T) / 2.0
    trace = float(np.trace(kernel))
    if not math.isfinite(trace) or trace <= 0:
        raise ScientificDefinitionError("Heat kernel has an invalid trace")
    normalized = kernel / trace
    return laplacian, normalized


def kernel_language_entropy(normalized_kernel: FloatMatrix) -> float:


    kernel = np.asarray(normalized_kernel, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(kernel)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ScientificDefinitionError("Semantic kernel is not PSD")
    clipped = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(clipped))
    if not math.isfinite(total) or total <= 0:
        raise ScientificDefinitionError("Semantic kernel has no positive spectrum")
    probabilities = clipped / total
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def mutual_entailment_components(
    labels: DirectionalLabels,
) -> tuple[tuple[int, ...], ...]:


    size = validate_directional_labels(labels)
    adjacency: list[list[int]] = [[] for _ in range(size)]
    for left in range(size):
        for right in range(left + 1, size):
            if (
                labels[left][right] == NLILabel.ENTAILMENT
                and labels[right][left] == NLILabel.ENTAILMENT
            ):
                adjacency[left].append(right)
                adjacency[right].append(left)

    components: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for start in range(size):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        members: list[int] = []
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(members)))
    return tuple(sorted(components, key=lambda item: (-len(item), item[0])))


def select_mode_representative(
    member_indices: Sequence[int], normalized_kernel: FloatMatrix
) -> int:


    if not member_indices:
        raise ScientificDefinitionError("A semantic mode cannot be empty")
    members = tuple(sorted(member_indices))
    best_index = members[0]
    best_score = -math.inf
    for candidate in members:
        score = float(np.mean(normalized_kernel[candidate, members]))
        if score > best_score + 1e-15:
            best_index = candidate
            best_score = score
    return best_index


def analyze_prompt_semantics(
    labels: DirectionalLabels, *, heat_time: float = 0.3
) -> PromptSemantics:


    weights = build_weight_matrix(labels)
    laplacian, kernel = build_heat_kernel(weights, heat_time=heat_time)
    uncertainty = kernel_language_entropy(kernel)
    components = mutual_entailment_components(labels)
    modes = tuple(
        SemanticMode(
            member_indices=component,
            representative_index=select_mode_representative(component, kernel),
        )
        for component in components
    )
    sample_count = len(labels)
    dominant_strength = modes[0].size / sample_count
    competitor_strength = modes[1].size / sample_count if len(modes) > 1 else 0.0
    return PromptSemantics(
        weight_matrix=weights,
        laplacian=laplacian,
        normalized_kernel=kernel,
        uncertainty=uncertainty,
        modes=modes,
        dominant_mode_index=0,
        dominant_strength=dominant_strength,
        competitor_strength=competitor_strength,
    )


def dominant_early_exit(
    semantics: PromptSemantics,
    *,
    dominant_threshold: float = 0.5,
    competitor_threshold: float = 0.10,
) -> bool:


    if not 0 <= dominant_threshold <= 1:
        raise ConfigurationError("dominant_threshold must be in [0, 1]")
    if not 0 <= competitor_threshold <= 1:
        raise ConfigurationError("competitor_threshold must be in [0, 1]")
    return (
        semantics.dominant_strength >= dominant_threshold
        and semantics.competitor_strength <= competitor_threshold
    )
