

from dataclasses import replace
import json

import pytest

from qprobe.datasets import (
    TRIVIAQA_ADAPTER,
    DatasetExample,
    build_manifest,
    canonicalize_reference_answer_groups,
    load_dataset_manifest,
    select_manifest_examples,
    write_dataset_manifest,
)
from qprobe.errors import DataValidationError


def make_examples(count: int = 1201) -> tuple[DatasetExample, ...]:
    return tuple(
        DatasetExample(
            dataset="triviaqa",
            sample_id=f"q-{index:04d}",
            question=f"Question {index}?",
            reference_answer_groups=((f"Answer {index}",),),
            source_split="validation",
        )
        for index in range(count)
    )


def test_manifest_is_deterministic_and_disjoint() -> None:
    examples = make_examples()
    arguments = {
        "dataset": "triviaqa",
        "source_split": "validation",
        "sampling_secret": b"private-test-secret",
        "sampling_seed_ref": "seed-ref",
        "source_fingerprint": "source-fingerprint",
    }
    first = build_manifest(examples, **arguments)
    second = build_manifest(tuple(reversed(examples)), **arguments)

    assert first == second
    assert len(first.dev_ids) == 200
    assert len(first.test_ids) == 1000
    assert set(first.dev_ids).isdisjoint(first.test_ids)


def test_manifest_selection_preserves_manifest_order() -> None:
    examples = make_examples()
    manifest = build_manifest(
        examples,
        dataset="triviaqa",
        source_split="validation",
        sampling_secret=b"private-test-secret",
        sampling_seed_ref="seed-ref",
        source_fingerprint="source-fingerprint",
    )
    dev, test = select_manifest_examples(tuple(reversed(examples)), manifest)

    assert tuple(example.sample_id for example in dev) == manifest.dev_ids
    assert tuple(example.sample_id for example in test) == manifest.test_ids


def test_manifest_serialization_rejects_fingerprint_changes(tmp_path) -> None:
    manifest = build_manifest(
        make_examples(),
        dataset="triviaqa",
        source_split="validation",
        sampling_secret=b"private-test-secret",
        sampling_seed_ref="seed-ref",
        source_fingerprint="source-fingerprint",
    )
    path = tmp_path / "manifest.json"
    write_dataset_manifest(path, manifest)

    assert load_dataset_manifest(path) == manifest

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dev_ids"][0] = "changed-id"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="fingerprint"):
        load_dataset_manifest(path)


def test_manifest_errors_use_data_validation_error() -> None:
    examples = make_examples()
    manifest = build_manifest(
        examples,
        dataset="triviaqa",
        source_split="validation",
        sampling_secret=b"private-test-secret",
        sampling_seed_ref="seed-ref",
        source_fingerprint="source-fingerprint",
    )

    with pytest.raises(DataValidationError):
        replace(manifest, dev_ids=list(manifest.dev_ids))

    with pytest.raises(DataValidationError):
        build_manifest(
            examples,
            dataset="triviaqa",
            source_split="validation",
            sampling_secret="not-bytes",
            sampling_seed_ref="seed-ref",
            source_fingerprint="source-fingerprint",
        )


def test_reference_groups_preserve_components_and_remove_exact_duplicates() -> None:
    groups = canonicalize_reference_answer_groups(
        (("New York", "United States"), ("NYC",), ("NYC",), ("New York", "United States"))
    )

    assert groups == (("New York", "United States"), ("NYC",))


def test_reference_group_shape_errors_are_explicit() -> None:
    with pytest.raises(DataValidationError):
        canonicalize_reference_answer_groups(("flat answer",))


def test_flat_reference_answers_are_rejected(tmp_path) -> None:
    source = tmp_path / "triviaqa.jsonl"
    record = {
        "dataset": "triviaqa",
        "sample_id": "q-0001",
        "question": "Who founded Apple?",
        "reference_answer_groups": ["Steve Jobs", "Steve Wozniak"],
        "source_split": "validation",
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(DataValidationError):
        TRIVIAQA_ADAPTER.load(source)
