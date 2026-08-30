

from __future__ import annotations

from typing import Mapping, Sequence

from qprobe.backends import NLIBackend, NLILabel, NLIRequest
from qprobe.errors import ConfigurationError, ModelCallError
from qprobe.telemetry import CallEvent, EventKind, TraceRecorder


def classify_all_directions(
    answers: Sequence[str],
    *,
    backend: NLIBackend,
    model_id: str,
    checkpoint_ref: str | None = None,
    recorder: TraceRecorder | None = None,
    metadata: Mapping[str, str] | None = None,
) -> tuple[tuple[NLILabel | None, ...], ...]:


    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
        raise ConfigurationError("answers must be a sequence of strings")
    if not answers:
        raise ConfigurationError("At least one answer is required")
    if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ConfigurationError("Answers must be non-empty")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ConfigurationError("model_id must be a non-empty string")
    if checkpoint_ref is not None and (
        not isinstance(checkpoint_ref, str) or not checkpoint_ref.strip()
    ):
        raise ConfigurationError("checkpoint_ref must be non-empty or None")
    if len(answers) == 1:
        return ((None,),)

    indexed_requests: list[tuple[int, int, NLIRequest]] = []
    for left in range(len(answers)):
        for right in range(left + 1, len(answers)):
            indexed_requests.append(
                (
                    left,
                    right,
                    NLIRequest(
                        premise=answers[left],
                        hypothesis=answers[right],
                        model_id=model_id,
                    ),
                )
            )
            indexed_requests.append(
                (
                    right,
                    left,
                    NLIRequest(
                        premise=answers[right],
                        hypothesis=answers[left],
                        model_id=model_id,
                    ),
                )
            )

    responses = backend.classify_many(
        tuple(request for _, _, request in indexed_requests)
    )
    if len(responses) != len(indexed_requests):
        raise ModelCallError("NLI backend returned the wrong response count")

    matrix: list[list[NLILabel | None]] = [
        [None for _ in answers] for _ in answers
    ]
    for (left, right, request), response in zip(indexed_requests, responses):
        if response.model_id != request.model_id:
            raise ModelCallError("NLI backend returned an unexpected model_id")
        matrix[left][right] = response.label
        if recorder is not None:
            recorder.record(
                CallEvent(
                    event_kind=EventKind.NLI_FORWARD,
                    model_id=response.model_id,
                    latency_seconds=response.latency_seconds,
                    metadata={
                        **(metadata or {}),
                        **(
                            {"checkpoint_ref": checkpoint_ref}
                            if checkpoint_ref is not None
                            else {}
                        ),
                        "premise_index": str(left),
                        "hypothesis_index": str(right),
                    },
                )
            )
    return tuple(tuple(row) for row in matrix)
