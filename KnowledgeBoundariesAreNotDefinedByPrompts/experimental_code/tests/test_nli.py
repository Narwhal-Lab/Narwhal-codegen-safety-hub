

from qprobe.backends import NLILabel, NLIResponse
from qprobe.nli import classify_all_directions
from qprobe.telemetry import EventKind, TraceRecorder


class RejectCallsBackend:
    def classify_many(self, requests):
        raise AssertionError("A singleton answer set must not call the NLI backend")


class FixedNLIBackend:
    def classify_many(self, requests):
        return tuple(
            NLIResponse(
                label=NLILabel.ENTAILMENT,
                scores={
                    NLILabel.ENTAILMENT: 1.0,
                    NLILabel.NEUTRAL: 0.0,
                    NLILabel.CONTRADICTION: 0.0,
                },
                model_id=request.model_id,
                latency_seconds=0.1,
            )
            for request in requests
        )


def test_singleton_answer_set_requires_no_nli_forward_event() -> None:
    labels = classify_all_directions(
        ("only answer",),
        backend=RejectCallsBackend(),
        model_id="microsoft/deberta-large-mnli",
    )

    assert labels == ((None,),)


def test_nli_events_record_the_controlled_checkpoint() -> None:
    recorder = TraceRecorder()

    labels = classify_all_directions(
        ("A", "B"),
        backend=FixedNLIBackend(),
        model_id="microsoft/deberta-large-mnli",
        checkpoint_ref="controlled-nli-checkpoint-ref",
        recorder=recorder,
        metadata={"view_id": "q-1:original", "nli_stage": "within_view"},
    )

    assert labels[0][1] == NLILabel.ENTAILMENT
    assert recorder.count(EventKind.NLI_FORWARD) == 2
    assert all(
        event.metadata["checkpoint_ref"] == "controlled-nli-checkpoint-ref"
        for event in recorder.events
    )
