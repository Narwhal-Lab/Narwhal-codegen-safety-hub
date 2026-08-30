

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from qprobe.datasets import SUPPORTED_DATASETS
from qprobe.errors import ConfigurationError


TARGET_MODEL_IDS = (
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
)
NLI_MODEL_ID = "microsoft/deberta-large-mnli"
TAU_CANDIDATES = tuple(index / 10 for index in range(1, 21))


def _require_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} must be a non-empty string")


def _require_path(value: object, field_name: str) -> None:
    if not isinstance(value, Path):
        raise ConfigurationError(f"{field_name} must be a pathlib.Path value")


@dataclass(frozen=True, slots=True)
class PublicMethodConfig:


    sample_count: int = 10
    target_temperature: float = 1.0
    target_max_new_tokens: int = 100
    paraphrase_temperature: float = 1.0
    paraphrase_max_new_tokens: int = 100
    judge_temperature: float = 0.0
    judge_max_new_tokens: int = 16
    heat_time: float = 0.3
    minimum_support: int = 2
    dominant_threshold: float = 0.5
    competitor_threshold: float = 0.10
    tau_candidates: tuple[float, ...] = TAU_CANDIDATES

    def __post_init__(self) -> None:
        configured_values = (
            self.sample_count,
            self.target_temperature,
            self.target_max_new_tokens,
            self.paraphrase_temperature,
            self.paraphrase_max_new_tokens,
            self.judge_temperature,
            self.judge_max_new_tokens,
            self.heat_time,
            self.minimum_support,
            self.dominant_threshold,
            self.competitor_threshold,
            self.tau_candidates,
        )
        confirmed_values = (
            10,
            1.0,
            100,
            1.0,
            100,
            0.0,
            16,
            0.3,
            2,
            0.5,
            0.10,
            TAU_CANDIDATES,
        )
        if configured_values != confirmed_values:
            raise ConfigurationError(
                "Public method settings must match the confirmed experiment settings"
            )


@dataclass(frozen=True, slots=True)
class GenerationSeed:


    value: int
    seed_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise ConfigurationError("generation seed value must be an integer")
        if self.value < 0:
            raise ConfigurationError("generation seed value cannot be negative")
        _require_non_empty_string(self.seed_ref, "generation seed_ref")


@dataclass(frozen=True, slots=True)
class SamplingSeed:


    secret: bytes
    seed_ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.secret, bytes) or not self.secret:
            raise ConfigurationError("sampling secret must be non-empty bytes")
        if len(self.secret) > 64:
            raise ConfigurationError("sampling secret cannot exceed 64 bytes")
        _require_non_empty_string(self.seed_ref, "sampling seed_ref")


@dataclass(frozen=True, slots=True)
class ManifestBuildConfig:


    dataset: str
    source_data: Path
    manifest: Path
    sampling_seed: SamplingSeed

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or self.dataset not in SUPPORTED_DATASETS:
            raise ConfigurationError(f"Unsupported dataset: {self.dataset!r}")
        _require_path(self.source_data, "source_data")
        _require_path(self.manifest, "manifest")
        if self.source_data == self.manifest:
            raise ConfigurationError("Source data and manifest paths must be distinct")
        if not isinstance(self.sampling_seed, SamplingSeed):
            raise ConfigurationError("sampling_seed must be a SamplingSeed")


@dataclass(frozen=True, slots=True)
class RunPaths:


    source_data: Path
    manifest: Path
    output_directory: Path

    def __post_init__(self) -> None:
        paths = (self.source_data, self.manifest, self.output_directory)
        for field_name, path in zip(
            ("source_data", "manifest", "output_directory"), paths
        ):
            _require_path(path, field_name)
        if len(set(paths)) != len(paths):
            raise ConfigurationError("Run paths must refer to distinct locations")


@dataclass(frozen=True, slots=True)
class ModelRunConfig:


    target_model_id: str
    target_deployment_ref: str
    nli_checkpoint_ref: str
    auxiliary_model_id: str
    auxiliary_deployment_ref: str
    nli_model_id: str = NLI_MODEL_ID

    def __post_init__(self) -> None:
        _require_non_empty_string(self.target_model_id, "target_model_id")
        if self.target_model_id not in TARGET_MODEL_IDS:
            raise ConfigurationError(
                f"Unsupported target model: {self.target_model_id!r}"
            )
        if self.nli_model_id != NLI_MODEL_ID:
            raise ConfigurationError(f"NLI model must be {NLI_MODEL_ID!r}")
        controlled_values = (
            ("target_deployment_ref", self.target_deployment_ref),
            ("nli_checkpoint_ref", self.nli_checkpoint_ref),
            ("auxiliary_model_id", self.auxiliary_model_id),
            ("auxiliary_deployment_ref", self.auxiliary_deployment_ref),
        )
        for field_name, value in controlled_values:
            _require_non_empty_string(value, field_name)


@dataclass(frozen=True, slots=True)
class PrivateRunConfig:


    dataset: str
    paths: RunPaths
    generation_seeds: tuple[GenerationSeed, ...]
    models: ModelRunConfig
    method: PublicMethodConfig = field(default_factory=PublicMethodConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or self.dataset not in SUPPORTED_DATASETS:
            raise ConfigurationError(f"Unsupported dataset: {self.dataset!r}")
        if not isinstance(self.paths, RunPaths):
            raise ConfigurationError("paths must be a RunPaths instance")
        if not isinstance(self.models, ModelRunConfig):
            raise ConfigurationError("models must be a ModelRunConfig instance")
        if not isinstance(self.method, PublicMethodConfig):
            raise ConfigurationError("method must be a PublicMethodConfig instance")
        if not isinstance(self.generation_seeds, tuple) or not all(
            isinstance(seed, GenerationSeed) for seed in self.generation_seeds
        ):
            raise ConfigurationError(
                "generation_seeds must be a tuple of GenerationSeed records"
            )
        if len(self.generation_seeds) != 3:
            raise ConfigurationError("Exactly three generation seeds are required")
        values = tuple(seed.value for seed in self.generation_seeds)
        references = tuple(seed.seed_ref for seed in self.generation_seeds)
        if len(set(values)) != len(values):
            raise ConfigurationError("Generation seed values must be distinct")
        if len(set(references)) != len(references):
            raise ConfigurationError("Generation seed references must be distinct")
