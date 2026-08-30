

import pytest

from qprobe.backends import GenerationRequest, ModelCallError, GenerationResponse, TokenUsage


def test_generation_request_uses_paper_defaults() -> None:
    request = GenerationRequest(
        prompt="Who founded Apple?",
        model_id="example-model",
        seed=123,
        seed_ref="private-seed-ref",
    )

    assert request.temperature == 1.0
    assert request.max_new_tokens == 100


def test_generation_request_allows_deterministic_judges() -> None:
    request = GenerationRequest(
        prompt="Judge this answer.",
        model_id="controlled-judge-model",
        seed=123,
        seed_ref="private-seed-ref",
        temperature=0.0,
        max_new_tokens=16,
    )

    assert request.temperature == 0.0


def test_empty_generation_is_rejected() -> None:
    with pytest.raises(ModelCallError):
        GenerationResponse(
            text=" ",
            model_id="example-model",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=4, output_tokens=0),
            latency_seconds=0.1,
        )
