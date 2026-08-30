

from qprobe.backends import GenerationResponse, TokenUsage
from qprobe.generation import SamplingPlan, sample_prompt
from qprobe.telemetry import EventKind, TraceRecorder


class WhitespaceGenerationBackend:
    def generate_many(self, requests):
        return tuple(
            GenerationResponse(
                text="  Alpha  Beta\n",
                model_id=request.model_id,
                finish_reason="stop",
                usage=TokenUsage(input_tokens=1, output_tokens=2),
                latency_seconds=0.0,
            )
            for request in requests
        )


def test_sample_prompt_strips_only_boundary_whitespace() -> None:
    recorder = TraceRecorder()
    outputs = sample_prompt(
        "Question?",
        prompt_id="q-1:original",
        plan=SamplingPlan(
            model_id="target-model",
            deployment_ref="controlled-target-ref",
            base_seed=1,
            seed_ref="generation-seed-ref",
            sample_count=2,
        ),
        backend=WhitespaceGenerationBackend(),
        recorder=recorder,
    )

    assert outputs == ("Alpha  Beta", "Alpha  Beta")
    assert recorder.count(EventKind.TARGET_GENERATION) == 2
    assert all(
        event.metadata["deployment_ref"] == "controlled-target-ref"
        and event.metadata["seed_ref"] == "generation-seed-ref"
        for event in recorder.events
    )
