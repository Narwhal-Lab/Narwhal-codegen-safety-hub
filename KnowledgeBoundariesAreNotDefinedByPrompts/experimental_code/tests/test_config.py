

from pathlib import Path

import pytest

from qprobe.config import (
    TARGET_MODEL_IDS,
    GenerationSeed,
    ManifestBuildConfig,
    ModelRunConfig,
    PrivateRunConfig,
    PublicMethodConfig,
    RunPaths,
    SamplingSeed,
)
from qprobe.errors import ConfigurationError


def make_private_config() -> PrivateRunConfig:
    return PrivateRunConfig(
        dataset="triviaqa",
        paths=RunPaths(
            source_data=Path("private/triviaqa.jsonl"),
            manifest=Path("private/triviaqa-manifest.json"),
            output_directory=Path("private/results"),
        ),
        generation_seeds=(
            GenerationSeed(11, "generation-ref-1"),
            GenerationSeed(22, "generation-ref-2"),
            GenerationSeed(33, "generation-ref-3"),
        ),
        models=ModelRunConfig(
            target_model_id=TARGET_MODEL_IDS[0],
            target_deployment_ref="controlled-target-ref",
            nli_checkpoint_ref="controlled-nli-checkpoint-ref",
            auxiliary_model_id="controlled-claude-model-id",
            auxiliary_deployment_ref="controlled-claude-deployment-ref",
        ),
    )


def test_confirmed_method_configuration_is_fixed() -> None:
    config = PublicMethodConfig()

    assert config.sample_count == 10
    assert config.tau_candidates == tuple(index / 10 for index in range(1, 21))

    with pytest.raises(ConfigurationError):
        PublicMethodConfig(sample_count=5)


def test_private_configuration_requires_three_distinct_generation_seeds() -> None:
    valid = make_private_config()

    with pytest.raises(ConfigurationError):
        PrivateRunConfig(
            dataset=valid.dataset,
            paths=valid.paths,
            generation_seeds=valid.generation_seeds[:2],
            models=valid.models,
        )


def test_manifest_build_configuration_owns_sampling_secret() -> None:
    config = ManifestBuildConfig(
        dataset="triviaqa",
        source_data=Path("private/triviaqa.jsonl"),
        manifest=Path("private/triviaqa-manifest.json"),
        sampling_seed=SamplingSeed(
            secret=b"private-sampling-secret",
            seed_ref="sampling-ref",
        ),
    )

    assert config.sampling_seed.seed_ref == "sampling-ref"


def test_invalid_seed_types_raise_configuration_errors() -> None:
    with pytest.raises(ConfigurationError):
        SamplingSeed(secret="not-bytes", seed_ref="sampling-ref")

    with pytest.raises(ConfigurationError):
        GenerationSeed(value="11", seed_ref="generation-ref")


def test_private_configuration_accepts_confirmed_boundaries() -> None:
    config = make_private_config()

    assert config.dataset == "triviaqa"
    assert config.models.target_model_id == TARGET_MODEL_IDS[0]
