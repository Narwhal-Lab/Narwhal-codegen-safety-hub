

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Protocol, Sequence

from qprobe.errors import DataValidationError


SUPPORTED_DATASETS = frozenset({"triviaqa", "natural_questions", "popqa"})
EXPECTED_SOURCE_SPLITS = {
    "triviaqa": "validation",
    "natural_questions": "validation",
    "popqa": "full",
}
MANIFEST_SCHEMA_VERSION = "qprobe-manifest-v1"

ReferenceAnswerGroup = tuple[str, ...]
ReferenceAnswerGroups = tuple[ReferenceAnswerGroup, ...]


def canonicalize_reference_answer_groups(
    groups: Sequence[Sequence[str]],
) -> ReferenceAnswerGroups:


    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise DataValidationError("Reference answer groups must be a sequence")
    canonical_groups: list[ReferenceAnswerGroup] = []
    seen_groups: set[ReferenceAnswerGroup] = set()
    for group in groups:
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise DataValidationError(
                "Each reference answer group must be a component sequence"
            )
        components: list[str] = []
        seen_components: set[str] = set()
        for component in group:
            if not isinstance(component, str):
                raise DataValidationError("Reference components must be strings")
            if not component.strip():
                raise DataValidationError("Reference components must be non-empty")
            if component not in seen_components:
                components.append(component)
                seen_components.add(component)
        if not components:
            raise DataValidationError("Reference answer groups must be non-empty")
        canonical_group = tuple(components)
        if canonical_group not in seen_groups:
            canonical_groups.append(canonical_group)
            seen_groups.add(canonical_group)
    if not canonical_groups:
        raise DataValidationError("At least one reference answer group is required")
    return tuple(canonical_groups)


@dataclass(frozen=True, slots=True)
class DatasetExample:


    dataset: str
    sample_id: str
    question: str
    reference_answer_groups: ReferenceAnswerGroups
    source_split: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        string_fields = {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "question": self.question,
            "source_split": self.source_split,
        }
        if any(not isinstance(value, str) for value in string_fields.values()):
            raise DataValidationError("Dataset identity fields must be strings")
        if self.dataset not in SUPPORTED_DATASETS:
            raise DataValidationError(f"Unsupported dataset: {self.dataset!r}")
        if not self.sample_id.strip():
            raise DataValidationError("sample_id must be non-empty")
        if not self.question.strip():
            raise DataValidationError(f"Question is empty for {self.sample_id!r}")
        if not self.reference_answer_groups:
            raise DataValidationError(
                f"reference_answer_groups is empty for {self.sample_id!r}"
            )
        canonical_groups = canonicalize_reference_answer_groups(
            self.reference_answer_groups
        )
        if canonical_groups != self.reference_answer_groups:
            raise DataValidationError(
                f"reference_answer_groups is not canonical for {self.sample_id!r}"
            )
        if not isinstance(self.metadata, Mapping):
            raise DataValidationError("metadata must be a mapping")
        expected_split = EXPECTED_SOURCE_SPLITS[self.dataset]
        if self.source_split != expected_split:
            raise DataValidationError(
                f"Expected split {expected_split!r} for {self.dataset!r}, "
                f"got {self.source_split!r}"
            )


@dataclass(frozen=True, slots=True)
class DatasetManifest:


    dataset: str
    source_split: str
    dev_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    sampling_seed_ref: str
    source_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str):
            raise DataValidationError("Manifest dataset must be a string")
        if self.dataset not in SUPPORTED_DATASETS:
            raise DataValidationError(f"Unsupported dataset: {self.dataset!r}")
        if not isinstance(self.source_split, str):
            raise DataValidationError("Manifest source_split must be a string")
        if self.source_split != EXPECTED_SOURCE_SPLITS[self.dataset]:
            raise DataValidationError("Manifest uses an unexpected source split")
        if not isinstance(self.dev_ids, tuple) or not all(
            isinstance(sample_id, str) and sample_id.strip()
            for sample_id in self.dev_ids
        ):
            raise DataValidationError(
                "Manifest dev_ids must be a tuple of non-empty strings"
            )
        if not isinstance(self.test_ids, tuple) or not all(
            isinstance(sample_id, str) and sample_id.strip()
            for sample_id in self.test_ids
        ):
            raise DataValidationError(
                "Manifest test_ids must be a tuple of non-empty strings"
            )
        if len(self.dev_ids) != 200:
            raise DataValidationError("Manifest must contain exactly 200 dev IDs")
        if len(self.test_ids) != 1000:
            raise DataValidationError("Manifest must contain exactly 1000 test IDs")
        all_ids = self.dev_ids + self.test_ids
        if len(set(all_ids)) != len(all_ids):
            raise DataValidationError("Manifest IDs must be unique and disjoint")
        string_fields = {
            "sampling_seed_ref": self.sampling_seed_ref,
            "source_fingerprint": self.source_fingerprint,
        }
        if any(
            not isinstance(value, str) or not value.strip()
            for value in string_fields.values()
        ):
            raise DataValidationError(
                "Manifest seed reference and source fingerprint must be non-empty strings"
            )

    @property
    def fingerprint(self) -> str:


        payload = {
            "dataset": self.dataset,
            "source_split": self.source_split,
            "dev_ids": self.dev_ids,
            "test_ids": self.test_ids,
            "sampling_seed_ref": self.sampling_seed_ref,
            "source_fingerprint": self.source_fingerprint,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:


    if not isinstance(path, Path):
        raise DataValidationError("Manifest path must be a pathlib.Path")
    if not isinstance(manifest, DatasetManifest):
        raise DataValidationError("manifest must be a DatasetManifest")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": manifest.dataset,
        "source_split": manifest.source_split,
        "dev_ids": list(manifest.dev_ids),
        "test_ids": list(manifest.test_ids),
        "sampling_seed_ref": manifest.sampling_seed_ref,
        "source_fingerprint": manifest.source_fingerprint,
        "manifest_fingerprint": manifest.fingerprint,
    }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
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
    except OSError as error:
        raise DataValidationError(f"Cannot write manifest: {path}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_dataset_manifest(path: Path) -> DatasetManifest:


    if not isinstance(path, Path):
        raise DataValidationError("Manifest path must be a pathlib.Path")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DataValidationError(f"Cannot read manifest: {path}") from error
    except json.JSONDecodeError as error:
        raise DataValidationError(f"Invalid manifest JSON: {error.msg}") from error
    expected_fields = {
        "schema_version",
        "dataset",
        "source_split",
        "dev_ids",
        "test_ids",
        "sampling_seed_ref",
        "source_fingerprint",
        "manifest_fingerprint",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise DataValidationError("Manifest fields do not match the public schema")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise DataValidationError("Manifest schema version is incompatible")
    if not isinstance(payload["dev_ids"], list) or not isinstance(
        payload["test_ids"], list
    ):
        raise DataValidationError("Manifest IDs must be JSON arrays")
    manifest = DatasetManifest(
        dataset=payload["dataset"],
        source_split=payload["source_split"],
        dev_ids=tuple(payload["dev_ids"]),
        test_ids=tuple(payload["test_ids"]),
        sampling_seed_ref=payload["sampling_seed_ref"],
        source_fingerprint=payload["source_fingerprint"],
    )
    if payload["manifest_fingerprint"] != manifest.fingerprint:
        raise DataValidationError("Manifest fingerprint mismatch")
    return manifest


class DatasetAdapter(Protocol):


    dataset: str
    source_split: str

    def load(self, path: Path) -> tuple[DatasetExample, ...]:
        pass


class PreparedJsonlAdapter:


    dataset: str
    source_split: str

    def __init__(self, dataset: str, source_split: str) -> None:
        if dataset not in SUPPORTED_DATASETS:
            raise DataValidationError(f"Unsupported dataset: {dataset!r}")
        if source_split != EXPECTED_SOURCE_SPLITS[dataset]:
            raise DataValidationError("Adapter uses an unexpected source split")
        self.dataset = dataset
        self.source_split = source_split

    def load(self, path: Path) -> tuple[DatasetExample, ...]:
        examples: list[DatasetExample] = []
        seen_ids: set[str] = set()
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise DataValidationError(f"Blank JSONL line at {line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataValidationError(
                        f"Invalid JSON at line {line_number}: {error.msg}"
                    ) from error
                example = self._convert(record, line_number)
                if example.sample_id in seen_ids:
                    raise DataValidationError(
                        f"Duplicate sample_id {example.sample_id!r}"
                    )
                seen_ids.add(example.sample_id)
                examples.append(example)
        if len(examples) < 1200:
            raise DataValidationError(
                f"{self.dataset} requires at least 1200 eligible examples"
            )
        return tuple(examples)

    def _convert(self, record: object, line_number: int) -> DatasetExample:
        if not isinstance(record, dict):
            raise DataValidationError(f"Line {line_number} must be a JSON object")
        required = {
            "dataset",
            "sample_id",
            "question",
            "reference_answer_groups",
            "source_split",
        }
        missing = required.difference(record)
        if missing:
            raise DataValidationError(
                f"Line {line_number} is missing fields: {sorted(missing)}"
            )
        if record["dataset"] != self.dataset:
            raise DataValidationError(f"Unexpected dataset at line {line_number}")
        if record["source_split"] != self.source_split:
            raise DataValidationError(f"Unexpected split at line {line_number}")
        groups = record["reference_answer_groups"]
        if not isinstance(groups, list) or not all(
            isinstance(group, list) for group in groups
        ):
            raise DataValidationError(
                "reference_answer_groups must be a list of lists "
                f"at line {line_number}"
            )
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise DataValidationError(f"metadata must be an object at line {line_number}")
        sample_id = record["sample_id"]
        question = record["question"]
        if not isinstance(sample_id, str) or not isinstance(question, str):
            raise DataValidationError(
                f"sample_id and question must be strings at line {line_number}"
            )
        return DatasetExample(
            dataset=self.dataset,
            sample_id=sample_id,
            question=question,
            reference_answer_groups=canonicalize_reference_answer_groups(groups),
            source_split=self.source_split,
            metadata=metadata,
        )


TRIVIAQA_ADAPTER = PreparedJsonlAdapter("triviaqa", "validation")
NATURAL_QUESTIONS_ADAPTER = PreparedJsonlAdapter(
    "natural_questions", "validation"
)
POPQA_ADAPTER = PreparedJsonlAdapter("popqa", "full")


def build_manifest(
    examples: Sequence[DatasetExample],
    *,
    dataset: str,
    source_split: str,
    sampling_secret: bytes,
    sampling_seed_ref: str,
    source_fingerprint: str,
) -> DatasetManifest:


    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise DataValidationError(
            "examples must be a sequence of DatasetExample records"
        )
    if not all(isinstance(example, DatasetExample) for example in examples):
        raise DataValidationError("examples must contain only DatasetExample records")
    if not isinstance(dataset, str) or dataset not in SUPPORTED_DATASETS:
        raise DataValidationError(f"Unsupported dataset: {dataset!r}")
    if (
        not isinstance(source_split, str)
        or source_split != EXPECTED_SOURCE_SPLITS[dataset]
    ):
        raise DataValidationError("Manifest build uses an unexpected source split")
    if not isinstance(sampling_secret, bytes) or not sampling_secret:
        raise DataValidationError("sampling_secret must be non-empty bytes")
    if len(sampling_secret) > 64:
        raise DataValidationError("sampling_secret cannot exceed 64 bytes")
    if not isinstance(sampling_seed_ref, str) or not sampling_seed_ref.strip():
        raise DataValidationError("sampling_seed_ref must be a non-empty string")
    if not isinstance(source_fingerprint, str) or not source_fingerprint.strip():
        raise DataValidationError("source_fingerprint must be a non-empty string")
    eligible = [
        example
        for example in examples
        if example.dataset == dataset and example.source_split == source_split
    ]
    if len(eligible) < 1200:
        raise DataValidationError("At least 1200 eligible examples are required")
    ids = [example.sample_id for example in eligible]
    if len(ids) != len(set(ids)):
        raise DataValidationError("Eligible sample IDs must be unique")

    def rank(sample_id: str) -> bytes:
        digest = hashlib.blake2b(key=sampling_secret, digest_size=32)
        digest.update(dataset.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(sample_id.encode("utf-8"))
        return digest.digest()

    ranked_ids = sorted(ids, key=lambda item: (rank(item), item))
    return DatasetManifest(
        dataset=dataset,
        source_split=source_split,
        dev_ids=tuple(ranked_ids[:200]),
        test_ids=tuple(ranked_ids[200:1200]),
        sampling_seed_ref=sampling_seed_ref,
        source_fingerprint=source_fingerprint,
    )


def select_manifest_examples(
    examples: Iterable[DatasetExample], manifest: DatasetManifest
) -> tuple[tuple[DatasetExample, ...], tuple[DatasetExample, ...]]:


    if not isinstance(manifest, DatasetManifest):
        raise DataValidationError("manifest must be a DatasetManifest record")
    by_id: dict[str, DatasetExample] = {}
    for example in examples:
        if not isinstance(example, DatasetExample):
            raise DataValidationError(
                "examples must contain only DatasetExample records"
            )
        if example.dataset != manifest.dataset:
            continue
        if example.sample_id in by_id:
            raise DataValidationError(f"Duplicate sample_id {example.sample_id!r}")
        by_id[example.sample_id] = example
    requested = manifest.dev_ids + manifest.test_ids
    missing = [sample_id for sample_id in requested if sample_id not in by_id]
    if missing:
        raise DataValidationError(f"Manifest contains {len(missing)} unknown IDs")
    dev = tuple(by_id[sample_id] for sample_id in manifest.dev_ids)
    test = tuple(by_id[sample_id] for sample_id in manifest.test_ids)
    return dev, test
