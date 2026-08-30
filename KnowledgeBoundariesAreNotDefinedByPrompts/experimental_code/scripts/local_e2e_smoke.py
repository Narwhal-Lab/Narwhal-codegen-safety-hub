from __future__ import annotations

import json
import os
from pathlib import Path

from qprobe.config import ModelRunConfig
from qprobe.correctness import evaluate_correctness
from qprobe.datasets import DatasetExample
from qprobe.decision import MethodVariant
from qprobe.evidence import ROLE_ORDER
from qprobe.paraphrasing import AuxiliaryCallPlan
from qprobe.pipeline import PipelineBackends, PipelinePlans, execute_example
from qprobe.generation import SamplingPlan
from qprobe.telemetry import EventKind, TraceRecorder
from qprobe.transformers_backends import (
    TransformersGenerationBackend,
    TransformersNLIBackend,
)


TARGET_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
AUXILIARY_ID = "Qwen/Qwen2.5-7B-Instruct"
NLI_ID = "microsoft/deberta-large-mnli"


def main() -> None:
    target_path = Path(os.environ["QPROBE_TARGET_MODEL_PATH"])
    auxiliary_path = Path(os.environ["QPROBE_AUXILIARY_MODEL_PATH"])
    nli_path = Path(os.environ["QPROBE_NLI_MODEL_PATH"])
    target_device = os.environ.get("QPROBE_TARGET_DEVICE", "cuda:0")
    auxiliary_device = os.environ.get("QPROBE_AUXILIARY_DEVICE", "cuda:1")
    nli_device = os.environ.get("QPROBE_NLI_DEVICE", "cuda:0")
    example = DatasetExample(
        dataset="triviaqa",
        sample_id="local-e2e-apple-founder",
        question="Who founded Apple?",
        reference_answer_groups=(("Steve Jobs",), ("Steve Wozniak",), ("Ronald Wayne",)),
        source_split="validation",
    )
    target = TransformersGenerationBackend(
        model_id=TARGET_ID,
        model_path=target_path,
        device=target_device,
    )
    auxiliary = TransformersGenerationBackend(
        model_id=AUXILIARY_ID,
        model_path=auxiliary_path,
        device=auxiliary_device,
    )
    nli = TransformersNLIBackend(
        model_id=NLI_ID,
        model_path=nli_path,
        device=nli_device,
    )
    models = ModelRunConfig(
        target_model_id=TARGET_ID,
        target_deployment_ref="lab3-local:target-checkpoint",
        nli_checkpoint_ref="lab3-local:nli-checkpoint",
        auxiliary_model_id=AUXILIARY_ID,
        auxiliary_deployment_ref="lab3-local:auxiliary-checkpoint",
    )
    plans = PipelinePlans(
        models=models,
        target_sampling=SamplingPlan(
            model_id=TARGET_ID,
            deployment_ref=models.target_deployment_ref,
            base_seed=17,
            seed_ref="local-e2e-target-seed",
        ),
        paraphrase=AuxiliaryCallPlan(
            model_id=AUXILIARY_ID,
            deployment_ref=models.auxiliary_deployment_ref,
            base_seed=19,
            seed_ref="local-e2e-paraphrase-seed",
            temperature=1.0,
            max_new_tokens=100,
        ),
        equivalence_judge=AuxiliaryCallPlan(
            model_id=AUXILIARY_ID,
            deployment_ref=models.auxiliary_deployment_ref,
            base_seed=23,
            seed_ref="local-e2e-equivalence-seed",
            temperature=0.0,
            max_new_tokens=16,
        ),
    )
    recorder = TraceRecorder()
    evidence, replay = execute_example(
        example,
        variant=MethodVariant.VARIANT_A,
        uncertainty_threshold=0.0,
        plans=plans,
        backends=PipelineBackends(
            target_generation=target,
            nli=nli,
            paraphrase_generation=auxiliary,
            equivalence_judge=auxiliary,
        ),
        recorder=recorder,
    )
    correctness_plan = AuxiliaryCallPlan(
        model_id=AUXILIARY_ID,
        deployment_ref=models.auxiliary_deployment_ref,
        base_seed=29,
        seed_ref="local-e2e-correctness-seed",
        temperature=0.0,
        max_new_tokens=16,
    )
    correctness = evaluate_correctness(
        example_id=example.sample_id,
        question=example.question,
        reference_answer_groups=example.reference_answer_groups,
        candidate_answer=replay.evaluation_candidate,
        plan=correctness_plan,
        backend=auxiliary,
        recorder=recorder,
    )
    counts = {kind.value: recorder.count(kind) for kind in EventKind}
    if evidence.shared_paraphrases is None or evidence.probe is None:
        raise RuntimeError("Smoke did not collect complete Variant A evidence")
    if len(evidence.shared_paraphrases.candidates) != len(ROLE_ORDER):
        raise RuntimeError("Smoke did not collect all role paraphrases")
    print(json.dumps({
        "sample_id": example.sample_id,
        "variant": MethodVariant.VARIANT_A.value,
        "decision": replay.decision.kind.value,
        "evaluation_candidate": replay.evaluation_candidate,
        "correctness": correctness.correct,
        "judge_called": correctness.judge_called,
        "event_counts": counts,
        "total_tokens": recorder.total_tokens,
        "serial_latency_seconds": recorder.total_serial_latency_seconds,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
