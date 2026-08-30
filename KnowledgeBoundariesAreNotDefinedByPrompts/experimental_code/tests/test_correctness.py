

import pytest

from qprobe.backends import GenerationResponse, TokenUsage
from qprobe.correctness import (
    evaluate_correctness,
    grouped_exact_match,
    normalize_exact_match_text,
)
from qprobe.errors import DataValidationError
from qprobe.paraphrasing import AuxiliaryCallPlan
from qprobe.telemetry import EventKind, TraceRecorder


class FixedJudgeBackend:
    def __init__(self, label: str = "CORRECT") -> None:
        self.label = label
        self.call_count = 0

    def generate(self, request):
        self.call_count += 1
        return GenerationResponse(
            text=self.label,
            model_id=request.model_id,
            finish_reason="stop",
            usage=TokenUsage(input_tokens=20, output_tokens=1),
            latency_seconds=0.2,
        )


def make_judge_plan(
    model_id: str = "controlled-claude-model-id",
    deployment_ref: str = "controlled-claude-deployment-ref",
    seed_ref: str = "controlled-judge-seed-ref",
) -> AuxiliaryCallPlan:
    return AuxiliaryCallPlan(
        model_id=model_id,
        deployment_ref=deployment_ref,
        base_seed=7,
        seed_ref=seed_ref,
        temperature=0.0,
        max_new_tokens=16,
    )


def test_exact_match_normalization_follows_confirmed_order() -> None:
    assert normalize_exact_match_text("  The Café, Inc.! ") == "café inc"


def test_exact_match_uses_only_single_component_groups() -> None:
    groups = (("Steve Jobs", "Steve Wozniak"), ("Jobs",))

    assert grouped_exact_match("Steve Jobs Steve Wozniak", groups) is None
    assert grouped_exact_match("jobs", groups) == 1


def test_exact_match_skips_the_correctness_judge() -> None:
    backend = FixedJudgeBackend()
    result = evaluate_correctness(
        example_id="q-1",
        question="Who co-founded Apple?",
        reference_answer_groups=(("Steve Jobs",),),
        candidate_answer="Steve Jobs",
        plan=make_judge_plan(),
        backend=backend,
    )

    assert result.correct
    assert result.exact_match
    assert not result.judge_called
    assert backend.call_count == 0


def test_correctness_judgment_is_cached_by_structured_input() -> None:
    backend = FixedJudgeBackend()
    recorder = TraceRecorder()
    cache = {}
    arguments = {
        "example_id": "q-2",
        "question": "Who co-founded Apple?",
        "reference_answer_groups": (("Steve Jobs", "Steve Wozniak"),),
        "candidate_answer": "Jobs and Wozniak",
        "plan": make_judge_plan(),
        "backend": backend,
        "recorder": recorder,
        "cache": cache,
    }

    first = evaluate_correctness(**arguments)
    second = evaluate_correctness(**arguments)

    assert first.judge_called
    assert second.judge_cache_hit
    assert backend.call_count == 1
    assert recorder.count(EventKind.CORRECTNESS_JUDGE) == 1


def test_exact_match_path_still_validates_question() -> None:
    with pytest.raises(DataValidationError):
        evaluate_correctness(
            example_id="q-3",
            question=" ",
            reference_answer_groups=(("Answer",),),
            candidate_answer="Answer",
            plan=make_judge_plan(),
            backend=FixedJudgeBackend(),
        )


def test_correctness_cache_is_scoped_to_the_judge_model() -> None:
    backend = FixedJudgeBackend()
    cache = {}
    arguments = {
        "example_id": "q-4",
        "question": "Who co-founded Apple?",
        "reference_answer_groups": (("Steve Jobs", "Steve Wozniak"),),
        "candidate_answer": "Jobs and Wozniak",
        "backend": backend,
        "cache": cache,
    }

    evaluate_correctness(
        **arguments,
        plan=make_judge_plan("controlled-claude-model-version-1"),
    )
    evaluate_correctness(
        **arguments,
        plan=make_judge_plan("controlled-claude-model-version-2"),
    )

    assert backend.call_count == 2


def test_correctness_cache_is_scoped_to_the_judge_seed() -> None:
    backend = FixedJudgeBackend()
    cache = {}
    arguments = {
        "example_id": "q-5",
        "question": "Who co-founded Apple?",
        "reference_answer_groups": (("Steve Jobs", "Steve Wozniak"),),
        "candidate_answer": "Jobs and Wozniak",
        "backend": backend,
        "cache": cache,
    }

    evaluate_correctness(
        **arguments,
        plan=make_judge_plan(seed_ref="controlled-judge-seed-ref-1"),
    )
    evaluate_correctness(
        **arguments,
        plan=make_judge_plan(seed_ref="controlled-judge-seed-ref-2"),
    )

    assert backend.call_count == 2


def test_correctness_cache_is_scoped_to_the_judge_deployment() -> None:
    backend = FixedJudgeBackend()
    cache = {}
    arguments = {
        "example_id": "q-6",
        "question": "Who co-founded Apple?",
        "reference_answer_groups": (("Steve Jobs", "Steve Wozniak"),),
        "candidate_answer": "Jobs and Wozniak",
        "backend": backend,
        "cache": cache,
    }

    evaluate_correctness(
        **arguments,
        plan=make_judge_plan(deployment_ref="controlled-deployment-ref-1"),
    )
    evaluate_correctness(
        **arguments,
        plan=make_judge_plan(deployment_ref="controlled-deployment-ref-2"),
    )

    assert backend.call_count == 2
