

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np

from qprobe.aggregation import (
    EvidenceOrigin,
    EvidenceSources,
    EventAggregate,
    ExampleRunRecord,
    RunIdentity,
    RunAggregate,
    aggregate_run,
)
from qprobe.backends import NLILabel, TokenUsage
from qprobe.calibration import (
    CalibrationExample,
    CalibrationResult,
    CandidateCorrectness,
    SeedCalibrationEvidence,
    TauCalibrationScore,
)
from qprobe.config import (
    ModelRunConfig,
    PublicMethodConfig,
    TAU_CANDIDATES,
)
from qprobe.correctness import (
    CorrectnessProvenance,
    CorrectnessResult,
    grouped_exact_match,
)
from qprobe.datasets import DatasetExample, SUPPORTED_DATASETS
from qprobe.decision import DecisionKind, MethodDecision, MethodVariant
from qprobe.errors import ArtifactValidationError, QProbeError
from qprobe.evidence import (
    ExampleEvidence,
    MethodOutput,
    OutputKind,
    ParaphraseEvidenceProvenance,
    PolicyCallCounts,
    ProbeEvidence,
    PromptEvidence,
    ReplayResult,
    SharedParaphraseEvidence,
    TargetEvidenceProvenance,
)
from qprobe.metrics import (
    CorrectToWrongGain,
    MeanStandardDeviation,
    RecoveryMetrics,
    RiskCoveragePoint,
)
from qprobe.paraphrasing import RoleParaphrase, VerifiedParaphrase
from qprobe.pipeline import replay_decision
from qprobe.prompts import ParaphraseRole
from qprobe.semantic import PromptSemantics, SemanticMode
from qprobe.telemetry import CallEvent, EventKind
from qprobe.views import PromptView


SCHEMA_VERSION = "qprobe-artifact-v5"
METADATA_FILE = "metadata.json"
EXAMPLES_FILE = "examples.jsonl"
CHECKPOINT_FILE = "checkpoint.json"
AGGREGATE_FILE = "aggregate.json"
CURRENT_FILE = "current.json"
GENERATIONS_DIRECTORY = "generations"
_SHA256_LENGTH = 64


class RunSplit(str, Enum):
    DEV = "dev"
    TEST = "test"


class RunStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


class ResultOrigin(str, Enum):
    RECONSTRUCTED_RUN = "reconstructed_run"


@dataclass(frozen=True, slots=True)
class RunMetadata:


    run_id: str
    dataset: str
    split: RunSplit
    variant: MethodVariant
    models: ModelRunConfig
    method: PublicMethodConfig
    generation_seed_ref: str
    paraphrase_seed_ref: str
    equivalence_judge_seed_ref: str
    correctness_judge_seed_ref: str
    calibration: BoundCalibrationResult
    selected_tau: float
    manifest_fingerprint: str
    controlled_config_fingerprint: str
    expected_sample_ids: tuple[str, ...]
    result_origin: ResultOrigin = ResultOrigin.RECONSTRUCTED_RUN

    def __post_init__(self) -> None:
        identity_values = (
            self.run_id,
            self.generation_seed_ref,
            self.paraphrase_seed_ref,
            self.equivalence_judge_seed_ref,
            self.correctness_judge_seed_ref,
        )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in identity_values
        ):
            raise ArtifactValidationError(
                "Run identity fields must be non-empty strings"
            )
        fingerprints = (
            self.manifest_fingerprint,
            self.controlled_config_fingerprint,
        )
        if any(not _is_sha256(value) for value in fingerprints):
            raise ArtifactValidationError(
                "Run fingerprints must be SHA-256 hex strings"
            )
        if (
            not isinstance(self.dataset, str)
            or self.dataset not in SUPPORTED_DATASETS
        ):
            raise ArtifactValidationError(f"Unsupported dataset: {self.dataset!r}")
        if not isinstance(self.split, RunSplit):
            raise ArtifactValidationError("split must be a RunSplit")
        if not isinstance(self.variant, MethodVariant):
            raise ArtifactValidationError("variant must be a MethodVariant")
        if not isinstance(self.models, ModelRunConfig):
            raise ArtifactValidationError("models must be ModelRunConfig")
        if not isinstance(self.method, PublicMethodConfig):
            raise ArtifactValidationError("method must be PublicMethodConfig")
        if not isinstance(self.calibration, BoundCalibrationResult):
            raise ArtifactValidationError(
                "calibration must be a BoundCalibrationResult"
            )
        calibration = self.calibration
        if (
            calibration.manifest_fingerprint != self.manifest_fingerprint
            or calibration.controlled_config_fingerprint
            != self.controlled_config_fingerprint
        ):
            raise ArtifactValidationError(
                "Calibration bindings must match run metadata"
            )
        if (
            calibration.result.target_model_id != self.models.target_model_id
            or calibration.result.dataset != self.dataset
            or self.generation_seed_ref
            not in calibration.result.generation_seed_refs
        ):
            raise ArtifactValidationError(
                "Calibration identity must match the controlled run"
            )
        if (
            not isinstance(self.selected_tau, float)
            or self.selected_tau not in TAU_CANDIDATES
        ):
            raise ArtifactValidationError(
                "selected_tau must be a confirmed candidate"
            )
        if self.selected_tau != calibration.result.selected_tau:
            raise ArtifactValidationError(
                "selected_tau must match the bound calibration result"
            )
        if not isinstance(self.expected_sample_ids, tuple) or not all(
            isinstance(sample_id, str) and sample_id.strip()
            for sample_id in self.expected_sample_ids
        ):
            raise ArtifactValidationError(
                "expected_sample_ids must contain non-empty strings"
            )
        expected_count = 200 if self.split == RunSplit.DEV else 1000
        if len(self.expected_sample_ids) != expected_count:
            raise ArtifactValidationError(
                f"{self.split.value} runs require exactly {expected_count} sample IDs"
            )
        if len(set(self.expected_sample_ids)) != len(self.expected_sample_ids):
            raise ArtifactValidationError("Expected sample IDs must be unique")
        if (
            not isinstance(self.result_origin, ResultOrigin)
            or self.result_origin != ResultOrigin.RECONSTRUCTED_RUN
        ):
            raise ArtifactValidationError(
                "New artifacts cannot claim historical run identity"
            )

    @property
    def calibration_fingerprint(self) -> str:


        return _sha256(_canonical_json(_encode(self.calibration)))


@dataclass(frozen=True, slots=True)
class RunCheckpoint:


    run_fingerprint: str
    metadata_digest: str
    examples_digest: str
    aggregate_digest: str | None
    completed_sample_ids: tuple[str, ...]
    status: RunStatus

    def __post_init__(self) -> None:
        digests = (
            self.run_fingerprint,
            self.metadata_digest,
            self.examples_digest,
        )
        if any(not _is_sha256(value) for value in digests):
            raise ArtifactValidationError(
                "Checkpoint fingerprints must be SHA-256 hex strings"
            )
        if self.aggregate_digest is not None and not _is_sha256(
            self.aggregate_digest
        ):
            raise ArtifactValidationError("Aggregate digest must be SHA-256 or None")
        if not isinstance(self.completed_sample_ids, tuple) or not all(
            isinstance(sample_id, str) and sample_id.strip()
            for sample_id in self.completed_sample_ids
        ):
            raise ArtifactValidationError(
                "completed_sample_ids must contain non-empty strings"
            )
        if len(set(self.completed_sample_ids)) != len(self.completed_sample_ids):
            raise ArtifactValidationError("Completed sample IDs must be unique")
        if not isinstance(self.status, RunStatus):
            raise ArtifactValidationError("status must be a RunStatus")
        if (self.status == RunStatus.COMPLETE) != (
            self.aggregate_digest is not None
        ):
            raise ArtifactValidationError(
                "Only complete snapshots contain an aggregate digest"
            )


@dataclass(frozen=True, slots=True)
class SnapshotPointer:


    generation_id: str
    checkpoint_digest: str

    def __post_init__(self) -> None:
        if not _is_sha256(self.generation_id) or not _is_sha256(
            self.checkpoint_digest
        ):
            raise ArtifactValidationError(
                "Snapshot pointer values must be SHA-256 hex strings"
            )
        if self.generation_id != self.checkpoint_digest:
            raise ArtifactValidationError(
                "Snapshot generation must equal its checkpoint digest"
            )


@dataclass(frozen=True, slots=True)
class BoundCalibrationResult:


    manifest_fingerprint: str
    controlled_config_fingerprint: str
    result: CalibrationResult

    def __post_init__(self) -> None:
        if not _is_sha256(self.manifest_fingerprint) or not _is_sha256(
            self.controlled_config_fingerprint
        ):
            raise ArtifactValidationError(
                "Calibration bindings must be SHA-256 fingerprints"
            )
        if not isinstance(self.result, CalibrationResult):
            raise ArtifactValidationError("result must be CalibrationResult")


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    metadata: RunMetadata
    records: tuple[ExampleRunRecord, ...]
    checkpoint: RunCheckpoint
    aggregate: RunAggregate | None


_DATACLASS_TYPES = (
    ModelRunConfig,
    PublicMethodConfig,
    DatasetExample,
    SemanticMode,
    PromptSemantics,
    PromptView,
    RoleParaphrase,
    VerifiedParaphrase,
    TargetEvidenceProvenance,
    ParaphraseEvidenceProvenance,
    PromptEvidence,
    SharedParaphraseEvidence,
    ProbeEvidence,
    ExampleEvidence,
    MethodDecision,
    MethodOutput,
    PolicyCallCounts,
    ReplayResult,
    CorrectnessProvenance,
    CorrectnessResult,
    TokenUsage,
    CallEvent,
    EvidenceSources,
    RunIdentity,
    ExampleRunRecord,
    EventAggregate,
    RunAggregate,
    RiskCoveragePoint,
    MeanStandardDeviation,
    RecoveryMetrics,
    CorrectToWrongGain,
    CandidateCorrectness,
    CalibrationExample,
    SeedCalibrationEvidence,
    TauCalibrationScore,
    CalibrationResult,
    RunMetadata,
    RunCheckpoint,
    SnapshotPointer,
    BoundCalibrationResult,
)
_DATACLASS_REGISTRY = {item.__name__: item for item in _DATACLASS_TYPES}
_ENUM_TYPES = (
    NLILabel,
    ParaphraseRole,
    MethodVariant,
    DecisionKind,
    OutputKind,
    EventKind,
    EvidenceOrigin,
    RunSplit,
    RunStatus,
    ResultOrigin,
)
_ENUM_REGISTRY = {item.__name__: item for item in _ENUM_TYPES}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _encode(value: object) -> object:
    if isinstance(value, Enum):
        return {
            "$kind": "enum",
            "type": type(value).__name__,
            "value": value.value,
        }
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactValidationError("Artifact floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype != np.float64 or value.ndim != 2:
            raise ArtifactValidationError(
                "Only two-dimensional float64 matrices can be persisted"
            )
        if not np.all(np.isfinite(value)):
            raise ArtifactValidationError("Artifact matrices must be finite")
        return {
            "$kind": "ndarray",
            "dtype": "float64",
            "values": value.tolist(),
        }
    if is_dataclass(value) and type(value) in _DATACLASS_TYPES:
        return {
            "$kind": "dataclass",
            "type": type(value).__name__,
            "fields": {
                field.name: _encode(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"$kind": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ArtifactValidationError("Artifact mapping keys must be strings")
        return {
            "$kind": "mapping",
            "items": {key: _encode(item) for key, item in value.items()},
        }
    raise ArtifactValidationError(
        f"Unsupported artifact value type: {type(value).__name__}"
    )


def _decode(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactValidationError("Artifact floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        raise ArtifactValidationError("Artifact values must be JSON-compatible")
    kind = value.get("$kind")
    if kind == "tuple":
        if set(value) != {"$kind", "items"} or not isinstance(value["items"], list):
            raise ArtifactValidationError("Invalid tuple artifact")
        return tuple(_decode(item) for item in value["items"])
    if kind == "mapping":
        if set(value) != {"$kind", "items"} or not isinstance(value["items"], dict):
            raise ArtifactValidationError("Invalid mapping artifact")
        return {key: _decode(item) for key, item in value["items"].items()}
    if kind == "enum":
        if set(value) != {"$kind", "type", "value"}:
            raise ArtifactValidationError("Invalid enum artifact")
        enum_type = _ENUM_REGISTRY.get(value["type"])
        if enum_type is None:
            raise ArtifactValidationError(f"Unknown enum type: {value['type']!r}")
        try:
            return enum_type(value["value"])
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError("Invalid enum value") from error
    if kind == "ndarray":
        if set(value) != {"$kind", "dtype", "values"}:
            raise ArtifactValidationError("Invalid ndarray artifact")
        if value["dtype"] != "float64":
            raise ArtifactValidationError("Unsupported ndarray dtype")
        try:
            matrix = np.asarray(value["values"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError("Invalid ndarray values") from error
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ArtifactValidationError(
                "Artifact ndarray must be a finite matrix"
            )
        return matrix
    if kind == "dataclass":
        if set(value) != {"$kind", "type", "fields"} or not isinstance(
            value["fields"], dict
        ):
            raise ArtifactValidationError("Invalid dataclass artifact")
        record_type = _DATACLASS_REGISTRY.get(value["type"])
        if record_type is None:
            raise ArtifactValidationError(
                f"Unknown dataclass type: {value['type']!r}"
            )
        decoded_fields = {
            key: _decode(item) for key, item in value["fields"].items()
        }
        try:
            return record_type(**decoded_fields)
        except (TypeError, ValueError, QProbeError) as error:
            raise ArtifactValidationError(
                f"Invalid {record_type.__name__} artifact"
            ) from error
    raise ArtifactValidationError("Artifact object has an unknown kind")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _versioned_payload(value: object) -> dict[str, object]:
    return {"schema_version": SCHEMA_VERSION, "payload": _encode(value)}


def _decode_versioned_payload(value: object) -> object:
    if not isinstance(value, dict) or set(value) != {"schema_version", "payload"}:
        raise ArtifactValidationError("Invalid versioned artifact envelope")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ArtifactValidationError("Artifact schema version is incompatible")
    return _decode(value["payload"])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            destination.write(text)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _json_text(value: object) -> str:
    return _canonical_json(_versioned_payload(value)) + "\n"


def _jsonl_text(values: Sequence[object]) -> str:
    return "".join(
        _canonical_json(_versioned_payload(value)) + "\n" for value in values
    )


def _run_fingerprint(metadata: RunMetadata) -> str:
    return _sha256(_canonical_json(_encode(metadata)))


def _run_identity(metadata: RunMetadata) -> RunIdentity:
    return RunIdentity(
        run_id=metadata.run_id,
        dataset=metadata.dataset,
        split=metadata.split.value,
        variant=metadata.variant,
        models=metadata.models,
        generation_seed_ref=metadata.generation_seed_ref,
        paraphrase_seed_ref=metadata.paraphrase_seed_ref,
        equivalence_judge_seed_ref=metadata.equivalence_judge_seed_ref,
        correctness_judge_seed_ref=metadata.correctness_judge_seed_ref,
        uncertainty_threshold=metadata.selected_tau,
        calibration_fingerprint=metadata.calibration_fingerprint,
        manifest_fingerprint=metadata.manifest_fingerprint,
        controlled_config_fingerprint=metadata.controlled_config_fingerprint,
    )


def _validate_snapshot_records(
    metadata: RunMetadata,
    records: tuple[ExampleRunRecord, ...],
    status: RunStatus,
) -> tuple[str, ...]:
    if not isinstance(status, RunStatus):
        raise ArtifactValidationError("status must be a RunStatus")
    if not all(isinstance(record, ExampleRunRecord) for record in records):
        raise ArtifactValidationError(
            "Snapshot records must contain ExampleRunRecord values"
        )
    completed_ids = tuple(record.sample_id for record in records)
    if completed_ids != metadata.expected_sample_ids[: len(completed_ids)]:
        raise ArtifactValidationError(
            "Completed records must be an ordered prefix of expected sample IDs"
        )
    if status == RunStatus.COMPLETE and completed_ids != metadata.expected_sample_ids:
        raise ArtifactValidationError(
            "A complete snapshot must contain every expected sample"
        )
    if len(completed_ids) > len(metadata.expected_sample_ids):
        raise ArtifactValidationError("Snapshot contains too many completed records")
    for record in records:
        _validate_record_identity(metadata, record)
    correctness_by_key: dict[tuple[object, ...], bool] = {}
    for record in records:
        provenance = record.correctness_provenance
        key = (
            provenance.judge_model_id,
            provenance.judge_deployment_ref,
            provenance.judge_seed_ref,
            provenance.question,
            provenance.reference_answer_groups,
            provenance.candidate_answer,
        )
        previous = correctness_by_key.setdefault(key, record.correctness.correct)
        if previous != record.correctness.correct:
            raise ArtifactValidationError(
                "Identical correctness inputs cannot have conflicting labels"
            )
    return completed_ids


def _require_event_keys(
    *,
    source: EvidenceOrigin | None,
    actual: tuple[tuple[object, ...], ...],
    expected: tuple[tuple[object, ...], ...],
    component: str,
) -> None:
    if source is None:
        if actual:
            raise ArtifactValidationError(
                f"Absent {component} evidence cannot contain real events"
            )
        return
    if source == EvidenceOrigin.CACHED:
        if actual:
            raise ArtifactValidationError(
                f"Cached {component} evidence cannot contain real events"
            )
        return
    if Counter(actual) != Counter(expected):
        raise ArtifactValidationError(
            f"Collected {component} events do not match stored evidence"
        )


def _validate_evidence_events(record: ExampleRunRecord) -> None:
    evidence = record.evidence
    prompt_evidence = (evidence.original,)
    if evidence.probe is not None:
        prompt_evidence += evidence.probe.prompt_views

    target_expected = tuple(
        (
            item.view.view_id,
            str(sample_index),
            evidence.provenance.target_deployment_ref,
            evidence.provenance.generation_seed_ref,
        )
        for item in prompt_evidence
        for sample_index in range(len(item.view.answers))
    )
    target_actual = tuple(
        (
            event.metadata.get("prompt_id"),
            event.metadata.get("sample_index"),
            event.metadata.get("deployment_ref"),
            event.metadata.get("seed_ref"),
        )
        for event in record.real_events
        if event.event_kind == EventKind.TARGET_GENERATION
    )
    _require_event_keys(
        source=record.evidence_sources.target,
        actual=target_actual,
        expected=target_expected,
        component="target",
    )

    nli_expected = tuple(
        (
            "within_view",
            item.view.view_id,
            str(left),
            str(right),
            evidence.provenance.nli_checkpoint_ref,
        )
        for item in prompt_evidence
        for left in range(len(item.view.answers))
        for right in range(len(item.view.answers))
        if left != right
    )
    if evidence.probe is not None:
        representative_count = len(evidence.probe.representative_labels)
        nli_expected += tuple(
            (
                "representative_answers",
                record.sample_id,
                str(left),
                str(right),
                evidence.provenance.nli_checkpoint_ref,
            )
            for left in range(representative_count)
            for right in range(representative_count)
            if left != right
        )
    nli_actual = tuple(
        (
            event.metadata.get("nli_stage"),
            (
                event.metadata.get("view_id")
                if event.metadata.get("nli_stage") == "within_view"
                else event.metadata.get("example_id")
            ),
            event.metadata.get("premise_index"),
            event.metadata.get("hypothesis_index"),
            event.metadata.get("checkpoint_ref"),
        )
        for event in record.real_events
        if event.event_kind == EventKind.NLI_FORWARD
    )
    _require_event_keys(
        source=record.evidence_sources.nli,
        actual=nli_actual,
        expected=nli_expected,
        component="NLI",
    )

    shared = evidence.shared_paraphrases
    paraphrase_expected = (
        ()
        if shared is None
        else tuple(
            (
                record.sample_id,
                candidate.role.value,
                shared.provenance.paraphrase_deployment_ref,
                shared.provenance.paraphrase_seed_ref,
            )
            for candidate in shared.candidates
        )
    )
    paraphrase_actual = tuple(
        (
            event.metadata.get("example_id"),
            event.metadata.get("role"),
            event.metadata.get("deployment_ref"),
            event.metadata.get("seed_ref"),
        )
        for event in record.real_events
        if event.event_kind == EventKind.PARAPHRASE_GENERATION
    )
    _require_event_keys(
        source=record.evidence_sources.paraphrase,
        actual=paraphrase_actual,
        expected=paraphrase_expected,
        component="paraphrase",
    )

    equivalence_expected = (
        ()
        if shared is None or shared.verified is None
        else tuple(
            (
                record.sample_id,
                item.role.value,
                shared.provenance.equivalence_deployment_ref,
                shared.provenance.equivalence_seed_ref,
                "EQUIVALENT" if item.equivalent else "NOT_EQUIVALENT",
            )
            for item in shared.verified
        )
    )
    equivalence_actual = tuple(
        (
            event.metadata.get("example_id"),
            event.metadata.get("role"),
            event.metadata.get("deployment_ref"),
            event.metadata.get("seed_ref"),
            event.metadata.get("decision"),
        )
        for event in record.real_events
        if event.event_kind == EventKind.EQUIVALENCE_JUDGE
    )
    equivalence_source = (
        record.evidence_sources.paraphrase
        if equivalence_expected
        else None
    )
    _require_event_keys(
        source=equivalence_source,
        actual=equivalence_actual,
        expected=equivalence_expected,
        component="equivalence",
    )


def _validate_record_identity(
    metadata: RunMetadata,
    record: ExampleRunRecord,
) -> None:
    evidence = record.evidence
    provenance = evidence.provenance
    if record.run_identity != _run_identity(metadata):
        raise ArtifactValidationError(
            "Record scientific identity does not match run metadata"
        )
    if evidence.example.dataset != metadata.dataset:
        raise ArtifactValidationError("Record dataset does not match run metadata")
    target_identity = (
        provenance.target_model_id,
        provenance.target_deployment_ref,
        provenance.nli_model_id,
        provenance.nli_checkpoint_ref,
    )
    expected_target_identity = (
        metadata.models.target_model_id,
        metadata.models.target_deployment_ref,
        metadata.models.nli_model_id,
        metadata.models.nli_checkpoint_ref,
    )
    if target_identity != expected_target_identity:
        raise ArtifactValidationError(
            "Record target or NLI provenance does not match run metadata"
        )
    if provenance.generation_seed_ref != metadata.generation_seed_ref:
        raise ArtifactValidationError(
            "Record generation seed does not match run metadata"
        )
    if len(evidence.original.view.answers) != metadata.method.sample_count:
        raise ArtifactValidationError(
            "Record sample count does not match the public method configuration"
        )
    shared = evidence.shared_paraphrases
    if shared is not None:
        paraphrase_identity = (
            shared.provenance.paraphrase_model_id,
            shared.provenance.paraphrase_deployment_ref,
            shared.provenance.paraphrase_seed_ref,
        )
        expected_auxiliary_identity = (
            metadata.models.auxiliary_model_id,
            metadata.models.auxiliary_deployment_ref,
            metadata.paraphrase_seed_ref,
        )
        if paraphrase_identity != expected_auxiliary_identity:
            raise ArtifactValidationError(
                "Record paraphrase provenance does not match run metadata"
            )
        if shared.provenance.has_equivalence:
            equivalence_identity = (
                shared.provenance.equivalence_model_id,
                shared.provenance.equivalence_deployment_ref,
                shared.provenance.equivalence_seed_ref,
            )
            expected_equivalence_identity = (
                metadata.models.auxiliary_model_id,
                metadata.models.auxiliary_deployment_ref,
                metadata.equivalence_judge_seed_ref,
            )
            if equivalence_identity != expected_equivalence_identity:
                raise ArtifactValidationError(
                    "Record equivalence provenance does not match run metadata"
                )
    try:
        expected_replay = replay_decision(
            evidence,
            variant=metadata.variant,
            uncertainty_threshold=metadata.selected_tau,
        )
    except QProbeError as error:
        raise ArtifactValidationError(
            "Record evidence cannot reproduce its declared policy"
        ) from error
    if record.replay != expected_replay:
        raise ArtifactValidationError(
            "Stored replay does not match evidence and run metadata"
        )
    correctness_provenance = record.correctness_provenance
    if (
        correctness_provenance.question != evidence.example.question
        or correctness_provenance.reference_answer_groups
        != evidence.example.reference_answer_groups
    ):
        raise ArtifactValidationError(
            "Correctness inputs do not match the dataset example"
        )
    expected_correctness_identity = (
        metadata.models.auxiliary_model_id,
        metadata.models.auxiliary_deployment_ref,
        metadata.correctness_judge_seed_ref,
    )
    actual_correctness_identity = (
        correctness_provenance.judge_model_id,
        correctness_provenance.judge_deployment_ref,
        correctness_provenance.judge_seed_ref,
    )
    if actual_correctness_identity != expected_correctness_identity:
        raise ArtifactValidationError(
            "Correctness provenance does not match run metadata"
        )
    expected_match = grouped_exact_match(
        correctness_provenance.candidate_answer,
        correctness_provenance.reference_answer_groups,
    )
    if record.correctness.exact_match:
        if record.correctness.matched_group_index != expected_match:
            raise ArtifactValidationError(
                "Exact Match result does not match correctness inputs"
            )
    elif expected_match is not None:
        raise ArtifactValidationError(
            "Correctness result bypassed an available Exact Match"
        )
    if record.correctness.judge_cache_hit != (
        correctness_provenance.cache_fingerprint is not None
    ):
        raise ArtifactValidationError(
            "Correctness cache route requires its artifact fingerprint"
        )
    expected_models = {
        EventKind.TARGET_GENERATION: metadata.models.target_model_id,
        EventKind.NLI_FORWARD: metadata.models.nli_model_id,
        EventKind.PARAPHRASE_GENERATION: metadata.models.auxiliary_model_id,
        EventKind.EQUIVALENCE_JUDGE: metadata.models.auxiliary_model_id,
        EventKind.CORRECTNESS_JUDGE: metadata.models.auxiliary_model_id,
    }
    for event in record.real_events:
        if not isinstance(event.event_kind, EventKind):
            raise ArtifactValidationError("Real event has an invalid event kind")
        if event.model_id != expected_models[event.event_kind]:
            raise ArtifactValidationError(
                "Real event model does not match run metadata"
            )
    if (shared is None) != (record.evidence_sources.paraphrase is None):
        raise ArtifactValidationError(
            "Paraphrase evidence and its source must be present together"
        )
    _validate_evidence_events(record)
    correctness_events = tuple(
        event
        for event in record.real_events
        if event.event_kind == EventKind.CORRECTNESS_JUDGE
    )
    expected_correctness_events = 1 if record.correctness.judge_called else 0
    if len(correctness_events) != expected_correctness_events:
        raise ArtifactValidationError(
            "Correctness route does not match observed Judge events"
        )
    for event in correctness_events:
        if event.metadata.get("example_id") != record.sample_id:
            raise ArtifactValidationError(
                "Correctness event example does not match the record"
            )
        if event.metadata.get("seed_ref") != metadata.correctness_judge_seed_ref:
            raise ArtifactValidationError(
                "Correctness event seed does not match run metadata"
            )
        if event.metadata.get("deployment_ref") != (
            metadata.models.auxiliary_deployment_ref
        ):
            raise ArtifactValidationError(
                "Correctness event deployment does not match run metadata"
            )
        expected_decision = (
            "CORRECT" if record.correctness.correct else "INCORRECT"
        )
        if event.metadata.get("decision") != expected_decision:
            raise ArtifactValidationError(
                "Correctness event decision does not match its result"
            )


def write_run_snapshot(
    directory: Path,
    metadata: RunMetadata,
    records: Sequence[ExampleRunRecord],
    *,
    status: RunStatus,
) -> RunCheckpoint:


    if not isinstance(directory, Path):
        raise ArtifactValidationError("directory must be a pathlib.Path")
    if not isinstance(metadata, RunMetadata):
        raise ArtifactValidationError("metadata must be RunMetadata")
    record_tuple = tuple(records)
    completed_ids = _validate_snapshot_records(metadata, record_tuple, status)
    metadata_text = _json_text(metadata)
    examples_text = _jsonl_text(record_tuple)
    aggregate = aggregate_run(record_tuple) if status == RunStatus.COMPLETE else None
    aggregate_text = _json_text(aggregate) if aggregate is not None else None
    checkpoint = RunCheckpoint(
        run_fingerprint=_run_fingerprint(metadata),
        metadata_digest=_sha256(metadata_text),
        examples_digest=_sha256(examples_text),
        aggregate_digest=(
            _sha256(aggregate_text) if aggregate_text is not None else None
        ),
        completed_sample_ids=completed_ids,
        status=status,
    )
    checkpoint_text = _json_text(checkpoint)
    generation_id = _sha256(checkpoint_text)
    generations_directory = directory / GENERATIONS_DIRECTORY
    generations_directory.mkdir(parents=True, exist_ok=True)
    generation_directory = generations_directory / generation_id
    staging_directory = Path(
        tempfile.mkdtemp(prefix=".pending-", dir=generations_directory)
    )
    try:
        _atomic_write_text(staging_directory / METADATA_FILE, metadata_text)
        _atomic_write_text(staging_directory / EXAMPLES_FILE, examples_text)
        if aggregate_text is not None:
            _atomic_write_text(staging_directory / AGGREGATE_FILE, aggregate_text)
        _atomic_write_text(staging_directory / CHECKPOINT_FILE, checkpoint_text)
        if generation_directory.exists():
            expected_files = {
                METADATA_FILE: metadata_text,
                EXAMPLES_FILE: examples_text,
                CHECKPOINT_FILE: checkpoint_text,
            }
            if aggregate_text is not None:
                expected_files[AGGREGATE_FILE] = aggregate_text
            if any(
                _read_text(generation_directory / name) != expected_text
                for name, expected_text in expected_files.items()
            ):
                raise ArtifactValidationError(
                    "Existing snapshot generation failed content validation"
                )
        else:
            os.replace(staging_directory, generation_directory)
    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory)
    pointer = SnapshotPointer(
        generation_id=generation_id,
        checkpoint_digest=_sha256(checkpoint_text),
    )
    _atomic_write_text(directory / CURRENT_FILE, _json_text(pointer))
    return checkpoint


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ArtifactValidationError(f"Cannot read artifact: {path.name}") from error


def _parse_json(text: str, artifact_name: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ArtifactValidationError(
            f"Invalid JSON in {artifact_name}: {error.msg}"
        ) from error


def _parse_jsonl(text: str) -> tuple[object, ...]:
    if not text:
        return ()
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ArtifactValidationError("Examples JSONL contains a blank line")
    return tuple(
        _decode_versioned_payload(_parse_json(line, EXAMPLES_FILE)) for line in lines
    )


def load_run_snapshot(directory: Path) -> RunSnapshot:


    if not isinstance(directory, Path):
        raise ArtifactValidationError("directory must be a pathlib.Path")
    pointer = _decode_versioned_payload(
        _parse_json(_read_text(directory / CURRENT_FILE), CURRENT_FILE)
    )
    if not isinstance(pointer, SnapshotPointer):
        raise ArtifactValidationError("Current snapshot pointer has the wrong type")
    generation_directory = (
        directory / GENERATIONS_DIRECTORY / pointer.generation_id
    )
    metadata_text = _read_text(generation_directory / METADATA_FILE)
    examples_text = _read_text(generation_directory / EXAMPLES_FILE)
    checkpoint_text = _read_text(generation_directory / CHECKPOINT_FILE)
    if pointer.checkpoint_digest != _sha256(checkpoint_text):
        raise ArtifactValidationError("Snapshot pointer digest mismatch")
    metadata = _decode_versioned_payload(
        _parse_json(metadata_text, METADATA_FILE)
    )
    checkpoint = _decode_versioned_payload(
        _parse_json(checkpoint_text, CHECKPOINT_FILE)
    )
    records = _parse_jsonl(examples_text)
    if not isinstance(metadata, RunMetadata):
        raise ArtifactValidationError("Metadata artifact has the wrong record type")
    if not isinstance(checkpoint, RunCheckpoint):
        raise ArtifactValidationError("Checkpoint artifact has the wrong record type")
    if not all(isinstance(record, ExampleRunRecord) for record in records):
        raise ArtifactValidationError("Examples artifact has the wrong record type")
    record_tuple = tuple(records)
    completed_ids = _validate_snapshot_records(
        metadata, record_tuple, checkpoint.status
    )
    if checkpoint.run_fingerprint != _run_fingerprint(metadata):
        raise ArtifactValidationError("Run fingerprint does not match metadata")
    if checkpoint.metadata_digest != _sha256(metadata_text):
        raise ArtifactValidationError("Metadata digest mismatch")
    if checkpoint.examples_digest != _sha256(examples_text):
        raise ArtifactValidationError("Examples digest mismatch")
    if checkpoint.completed_sample_ids != completed_ids:
        raise ArtifactValidationError("Checkpoint progress does not match examples")
    aggregate: RunAggregate | None = None
    if checkpoint.aggregate_digest is not None:
        aggregate_text = _read_text(generation_directory / AGGREGATE_FILE)
        if checkpoint.aggregate_digest != _sha256(aggregate_text):
            raise ArtifactValidationError("Aggregate digest mismatch")
        decoded_aggregate = _decode_versioned_payload(
            _parse_json(aggregate_text, AGGREGATE_FILE)
        )
        if not isinstance(decoded_aggregate, RunAggregate):
            raise ArtifactValidationError("Aggregate artifact has the wrong type")
        expected_aggregate = aggregate_run(record_tuple)
        if decoded_aggregate != expected_aggregate:
            raise ArtifactValidationError(
                "Aggregate does not match the completed records"
            )
        aggregate = decoded_aggregate
    return RunSnapshot(metadata, record_tuple, checkpoint, aggregate)


def write_versioned_json(path: Path, value: object) -> None:


    if not isinstance(path, Path):
        raise ArtifactValidationError("path must be a pathlib.Path")
    if not is_dataclass(value) or type(value) not in _DATACLASS_TYPES:
        raise ArtifactValidationError("value must be a registered artifact record")
    if isinstance(value, (CalibrationResult, RunAggregate)):
        raise ArtifactValidationError(
            "Aggregate and calibration results require identity bindings"
        )
    _atomic_write_text(path, _json_text(value))


def read_versioned_json(path: Path, expected_type: type[object]) -> object:


    if not isinstance(path, Path):
        raise ArtifactValidationError("path must be a pathlib.Path")
    if expected_type not in _DATACLASS_TYPES:
        raise ArtifactValidationError("expected_type must be a registered record type")
    if expected_type in (CalibrationResult, RunAggregate):
        raise ArtifactValidationError(
            "Aggregate and calibration results require identity bindings"
        )
    value = _decode_versioned_payload(_parse_json(_read_text(path), path.name))
    if not isinstance(value, expected_type):
        raise ArtifactValidationError(
            f"Expected {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def public_output_record(record: ExampleRunRecord) -> dict[str, object]:


    if not isinstance(record, ExampleRunRecord):
        raise ArtifactValidationError("record must be ExampleRunRecord")
    return {
        "sample_id": record.sample_id,
        "output": {
            "kind": record.replay.output.kind.value,
            "answer": record.replay.output.answer,
        },
    }
