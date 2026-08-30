

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from qprobe.config import ModelRunConfig
from qprobe.correctness import CorrectnessProvenance, CorrectnessResult
from qprobe.decision import MethodVariant
from qprobe.errors import ScientificDefinitionError
from qprobe.evidence import ExampleEvidence, OutputKind, ReplayResult
from qprobe.metrics import (
    AcceptedAnswer,
    ProbeOutcome,
    RecoveryMetrics,
    ScoredExample,
    augrc,
    auroc,
    recovery_metrics,
    stable_wrong_rate,
)
from qprobe.telemetry import CallEvent, EventKind


class EvidenceOrigin(str, Enum):
    COLLECTED = "collected"
    CACHED = "cached"


@dataclass(frozen=True, slots=True)
class EvidenceSources:


    target: EvidenceOrigin
    nli: EvidenceOrigin
    paraphrase: EvidenceOrigin | None
    target_cache_fingerprint: str | None = None
    nli_cache_fingerprint: str | None = None
    paraphrase_cache_fingerprint: str | None = None

    def __post_init__(self) -> None:
        origins = (self.target, self.nli)
        if not all(isinstance(origin, EvidenceOrigin) for origin in origins):
            raise ScientificDefinitionError("Evidence origins must be declared")
        if self.paraphrase is not None and not isinstance(
            self.paraphrase, EvidenceOrigin
        ):
            raise ScientificDefinitionError("Paraphrase origin must be declared")
        source_bindings = (
            (self.target, self.target_cache_fingerprint),
            (self.nli, self.nli_cache_fingerprint),
            (self.paraphrase, self.paraphrase_cache_fingerprint),
        )
        for origin, fingerprint in source_bindings:
            if fingerprint is not None and not (
                isinstance(fingerprint, str)
                and len(fingerprint) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in fingerprint
                )
            ):
                raise ScientificDefinitionError(
                    "Cache fingerprints must be SHA-256 hex strings"
                )
            if (origin == EvidenceOrigin.CACHED) != (fingerprint is not None):
                raise ScientificDefinitionError(
                    "Each cached evidence source requires its own fingerprint"
                )


@dataclass(frozen=True, slots=True)
class RunIdentity:


    run_id: str
    dataset: str
    split: str
    variant: MethodVariant
    models: ModelRunConfig
    generation_seed_ref: str
    paraphrase_seed_ref: str
    equivalence_judge_seed_ref: str
    correctness_judge_seed_ref: str
    uncertainty_threshold: float
    calibration_fingerprint: str
    manifest_fingerprint: str
    controlled_config_fingerprint: str

    def __post_init__(self) -> None:
        identity_values = (self.run_id, self.dataset)
        if any(
            not isinstance(value, str) or not value.strip()
            for value in identity_values
        ):
            raise ScientificDefinitionError("Run identity fields must be non-empty")
        if self.split not in {"dev", "test"}:
            raise ScientificDefinitionError("Run split must be dev or test")
        if not isinstance(self.variant, MethodVariant):
            raise ScientificDefinitionError("Run variant must be a MethodVariant")
        if not isinstance(self.models, ModelRunConfig):
            raise ScientificDefinitionError("Run models must be ModelRunConfig")
        seed_refs = (
            self.generation_seed_ref,
            self.paraphrase_seed_ref,
            self.equivalence_judge_seed_ref,
            self.correctness_judge_seed_ref,
        )
        if any(not isinstance(value, str) or not value.strip() for value in seed_refs):
            raise ScientificDefinitionError("Run seed references must be non-empty")
        if (
            not isinstance(self.uncertainty_threshold, (int, float))
            or isinstance(self.uncertainty_threshold, bool)
            or not math.isfinite(self.uncertainty_threshold)
            or self.uncertainty_threshold < 0
        ):
            raise ScientificDefinitionError(
                "Run uncertainty threshold must be finite and non-negative"
            )
        fingerprints = (
            self.calibration_fingerprint,
            self.manifest_fingerprint,
            self.controlled_config_fingerprint,
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef" for character in value
            )
            for value in fingerprints
        ):
            raise ScientificDefinitionError(
                "Run fingerprints must be SHA-256 hex strings"
            )


@dataclass(frozen=True, slots=True)
class ExampleRunRecord:


    run_identity: RunIdentity
    evidence_sources: EvidenceSources
    evidence: ExampleEvidence
    replay: ReplayResult
    correctness_provenance: CorrectnessProvenance
    correctness: CorrectnessResult
    real_events: tuple[CallEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_identity, RunIdentity):
            raise ScientificDefinitionError("run_identity must be RunIdentity")
        if not isinstance(self.evidence_sources, EvidenceSources):
            raise ScientificDefinitionError("evidence_sources must be EvidenceSources")
        if not isinstance(self.evidence, ExampleEvidence):
            raise ScientificDefinitionError("evidence must be ExampleEvidence")
        if not isinstance(self.replay, ReplayResult):
            raise ScientificDefinitionError("replay must be ReplayResult")
        if not isinstance(self.correctness, CorrectnessResult):
            raise ScientificDefinitionError(
                "correctness must be CorrectnessResult"
            )
        if not isinstance(self.correctness_provenance, CorrectnessProvenance):
            raise ScientificDefinitionError(
                "correctness_provenance must be CorrectnessProvenance"
            )
        if self.correctness_provenance.candidate_answer != (
            self.replay.evaluation_candidate
        ):
            raise ScientificDefinitionError(
                "Correctness candidate must match the replay candidate"
            )
        if not isinstance(self.correctness.correct, bool):
            raise ScientificDefinitionError("Correctness result must be boolean")
        if not isinstance(self.real_events, tuple) or not all(
            isinstance(event, CallEvent) for event in self.real_events
        ):
            raise ScientificDefinitionError(
                "real_events must contain observed CallEvent records"
            )
        output = self.replay.output
        decision = self.replay.decision
        if output.kind == OutputKind.ANSWER and output.answer != decision.answer:
            raise ScientificDefinitionError(
                "Formal output must match the accepted decision"
            )
        if output.kind == OutputKind.ABSTAIN and decision.answer is not None:
            raise ScientificDefinitionError(
                "Formal abstention cannot contain a decision answer"
            )

    @property
    def sample_id(self) -> str:
        return self.evidence.example.sample_id


@dataclass(frozen=True, slots=True)
class EventAggregate:


    event_kind: EventKind
    event_count: int
    total_tokens: int
    total_serial_latency_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.event_kind, EventKind):
            raise ScientificDefinitionError("event_kind must be an EventKind")
        counts = (self.event_count, self.total_tokens)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise ScientificDefinitionError("Event counts must be non-negative integers")
        if (
            not isinstance(self.total_serial_latency_seconds, (int, float))
            or isinstance(self.total_serial_latency_seconds, bool)
            or not math.isfinite(self.total_serial_latency_seconds)
            or self.total_serial_latency_seconds < 0
        ):
            raise ScientificDefinitionError(
                "Event latency must be a finite non-negative number"
            )


@dataclass(frozen=True, slots=True)
class RunAggregate:


    run_identity: RunIdentity
    example_count: int
    answered_count: int
    abstained_count: int
    coverage: float
    answered_accuracy: float | None
    auroc: float
    augrc: float
    recovery: RecoveryMetrics | None
    stable_wrong_rate: float | None
    mean_counterfactual_method_calls: float
    real_event_aggregates: tuple[EventAggregate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_identity, RunIdentity):
            raise ScientificDefinitionError("run_identity must be RunIdentity")
        counts = (self.example_count, self.answered_count, self.abstained_count)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise ScientificDefinitionError("Run counts must be non-negative integers")
        if self.example_count == 0:
            raise ScientificDefinitionError("Run aggregate cannot be empty")
        if self.answered_count + self.abstained_count != self.example_count:
            raise ScientificDefinitionError("Run answer counts do not sum correctly")
        numeric_rates = (self.coverage, self.auroc, self.augrc)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in numeric_rates
        ):
            raise ScientificDefinitionError("Run rates must be finite values in [0, 1]")
        expected_coverage = self.answered_count / self.example_count
        if not math.isclose(self.coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
            raise ScientificDefinitionError("Coverage does not match run counts")
        optional_rates = (self.answered_accuracy, self.stable_wrong_rate)
        if any(
            value is not None
            and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            )
            for value in optional_rates
        ):
            raise ScientificDefinitionError(
                "Optional run rates must be None or finite values in [0, 1]"
            )
        if (self.answered_count == 0) != (self.answered_accuracy is None):
            raise ScientificDefinitionError(
                "Answered accuracy must be present exactly when answers exist"
            )
        if (self.answered_count == 0) != (self.stable_wrong_rate is None):
            raise ScientificDefinitionError(
                "Stable-wrong rate must be present exactly when answers exist"
            )
        if self.recovery is not None and not isinstance(
            self.recovery, RecoveryMetrics
        ):
            raise ScientificDefinitionError("recovery must be RecoveryMetrics or None")
        if (
            not isinstance(self.mean_counterfactual_method_calls, (int, float))
            or isinstance(self.mean_counterfactual_method_calls, bool)
            or not math.isfinite(self.mean_counterfactual_method_calls)
            or self.mean_counterfactual_method_calls < 0
        ):
            raise ScientificDefinitionError(
                "Mean counterfactual calls must be finite and non-negative"
            )
        if (
            not isinstance(self.real_event_aggregates, tuple)
            or not all(
                isinstance(item, EventAggregate)
                for item in self.real_event_aggregates
            )
            or tuple(
                item.event_kind for item in self.real_event_aggregates
            )
            != tuple(EventKind)
        ):
            raise ScientificDefinitionError(
                "Real event aggregates must contain every event kind in fixed order"
            )


def _aggregate_events(records: Sequence[ExampleRunRecord]) -> tuple[EventAggregate, ...]:
    aggregates: list[EventAggregate] = []
    for event_kind in EventKind:
        events = tuple(
            event
            for record in records
            for event in record.real_events
            if event.event_kind == event_kind
        )
        aggregates.append(
            EventAggregate(
                event_kind=event_kind,
                event_count=len(events),
                total_tokens=sum(
                    event.usage.total_tokens
                    for event in events
                    if event.usage is not None
                ),
                total_serial_latency_seconds=sum(
                    event.latency_seconds for event in events
                ),
            )
        )
    return tuple(aggregates)


def aggregate_run(
    records: Sequence[ExampleRunRecord],
) -> RunAggregate:


    if not records or not all(
        isinstance(record, ExampleRunRecord) for record in records
    ):
        raise ScientificDefinitionError(
            "records must contain at least one ExampleRunRecord"
        )
    sample_ids = tuple(record.sample_id for record in records)
    if len(set(sample_ids)) != len(sample_ids):
        raise ScientificDefinitionError("Run record sample IDs must be unique")
    run_identity = records[0].run_identity
    if any(record.run_identity != run_identity for record in records[1:]):
        raise ScientificDefinitionError(
            "Aggregate records must share one scientific run identity"
        )

    scored = tuple(
        ScoredExample(
            sample_id=record.sample_id,
            correct=record.correctness.correct,
            confidence=record.replay.confidence,
        )
        for record in records
    )
    accepted = tuple(
        record
        for record in records
        if record.replay.output.kind == OutputKind.ANSWER
    )
    probed = tuple(record for record in records if record.replay.entered_paraphrase_probe)
    accepted_for_stability = tuple(
        AcceptedAnswer(
            correct=record.correctness.correct,
            low_uncertainty=(
                record.replay.u_final < run_identity.uncertainty_threshold
            ),
        )
        for record in accepted
    )
    recovery = (
        recovery_metrics(
            tuple(
                ProbeOutcome(
                    answered=record.replay.output.kind == OutputKind.ANSWER,
                    correct=record.correctness.correct,
                )
                for record in probed
            )
        )
        if probed
        else None
    )
    answered_count = len(accepted)
    example_count = len(records)
    return RunAggregate(
        run_identity=run_identity,
        example_count=example_count,
        answered_count=answered_count,
        abstained_count=example_count - answered_count,
        coverage=answered_count / example_count,
        answered_accuracy=(
            None
            if not accepted
            else sum(record.correctness.correct for record in accepted)
            / answered_count
        ),
        auroc=auroc(scored),
        augrc=augrc(scored),
        recovery=recovery,
        stable_wrong_rate=stable_wrong_rate(accepted_for_stability),
        mean_counterfactual_method_calls=(
            sum(record.replay.policy_calls.total_method_calls for record in records)
            / example_count
        ),
        real_event_aggregates=_aggregate_events(records),
    )
