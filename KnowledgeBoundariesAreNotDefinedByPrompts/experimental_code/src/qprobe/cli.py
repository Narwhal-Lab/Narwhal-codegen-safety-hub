

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable, Sequence, cast

from qprobe.artifacts import (
    BoundCalibrationResult,
    load_run_snapshot,
    read_versioned_json,
    write_versioned_json,
)
from qprobe.calibration import SeedCalibrationEvidence, calibrate_shared_tau
from qprobe.datasets import (
    NATURAL_QUESTIONS_ADAPTER,
    POPQA_ADAPTER,
    TRIVIAQA_ADAPTER,
    PreparedJsonlAdapter,
    build_manifest,
    load_dataset_manifest,
    select_manifest_examples,
    write_dataset_manifest,
)
from qprobe.errors import ArtifactValidationError, QProbeError


_ADAPTERS: dict[str, PreparedJsonlAdapter] = {
    "triviaqa": TRIVIAQA_ADAPTER,
    "natural_questions": NATURAL_QUESTIONS_ADAPTER,
    "popqa": POPQA_ADAPTER,
}


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _write_json(path: Path, payload: object) -> None:
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
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _manifest_build(arguments: argparse.Namespace) -> int:
    adapter = _ADAPTERS[arguments.dataset]
    examples = adapter.load(arguments.prepared_jsonl)
    try:
        sampling_secret = arguments.sampling_secret_file.read_bytes()
    except OSError as error:
        raise QProbeError("Cannot read the sampling secret file") from error
    manifest = build_manifest(
        examples,
        dataset=adapter.dataset,
        source_split=adapter.source_split,
        sampling_secret=sampling_secret,
        sampling_seed_ref=arguments.sampling_seed_ref,
        source_fingerprint=arguments.source_fingerprint,
    )
    write_dataset_manifest(arguments.output, manifest)
    _print_json(
        {
            "dataset": manifest.dataset,
            "dev_count": len(manifest.dev_ids),
            "manifest_fingerprint": manifest.fingerprint,
            "output": str(arguments.output),
            "test_count": len(manifest.test_ids),
        }
    )
    return 0


def _manifest_validate(arguments: argparse.Namespace) -> int:
    manifest = load_dataset_manifest(arguments.manifest)
    adapter = _ADAPTERS[manifest.dataset]
    examples = adapter.load(arguments.prepared_jsonl)
    dev, test = select_manifest_examples(examples, manifest)
    _print_json(
        {
            "dataset": manifest.dataset,
            "dev_count": len(dev),
            "manifest_fingerprint": manifest.fingerprint,
            "status": "valid",
            "test_count": len(test),
        }
    )
    return 0


def _calibrate(arguments: argparse.Namespace) -> int:
    seeds = tuple(
        cast(
            SeedCalibrationEvidence,
            read_versioned_json(path, SeedCalibrationEvidence),
        )
        for path in arguments.seed_evidence
    )
    result = calibrate_shared_tau(seeds)
    bound = BoundCalibrationResult(
        manifest_fingerprint=arguments.manifest_fingerprint,
        controlled_config_fingerprint=arguments.controlled_config_fingerprint,
        result=result,
    )
    write_versioned_json(arguments.output, bound)
    _print_json(
        {
            "dataset": result.dataset,
            "output": str(arguments.output),
            "selected_tau": result.selected_tau,
            "target_model_id": result.target_model_id,
        }
    )
    return 0


def _aggregate(arguments: argparse.Namespace) -> int:
    snapshot = load_run_snapshot(arguments.snapshot)
    aggregate = snapshot.aggregate
    if aggregate is None:
        raise ArtifactValidationError(
            "Aggregation requires a complete validated snapshot"
        )
    summary = {
        "run_id": aggregate.run_identity.run_id,
        "dataset": aggregate.run_identity.dataset,
        "split": aggregate.run_identity.split,
        "variant": aggregate.run_identity.variant.value,
        "generation_seed_ref": aggregate.run_identity.generation_seed_ref,
        "example_count": aggregate.example_count,
        "answered_count": aggregate.answered_count,
        "abstained_count": aggregate.abstained_count,
        "coverage": aggregate.coverage,
        "answered_accuracy": aggregate.answered_accuracy,
        "auroc": aggregate.auroc,
        "augrc": aggregate.augrc,
        "stable_wrong_rate": aggregate.stable_wrong_rate,
        "mean_counterfactual_method_calls": (
            aggregate.mean_counterfactual_method_calls
        ),
        "recovery": (
            None
            if aggregate.recovery is None
            else {
                "probed_count": aggregate.recovery.probed_count,
                "correct_recovery_count": (
                    aggregate.recovery.correct_recovery_count
                ),
                "wrong_recovery_count": aggregate.recovery.wrong_recovery_count,
                "correct_recovery_rate": aggregate.recovery.correct_recovery_rate,
                "wrong_recovery_rate": aggregate.recovery.wrong_recovery_rate,
            }
        ),
        "real_events": [
            {
                "event_kind": item.event_kind.value,
                "event_count": item.event_count,
                "total_tokens": item.total_tokens,
                "total_serial_latency_seconds": (
                    item.total_serial_latency_seconds
                ),
            }
            for item in aggregate.real_event_aggregates
        ],
    }
    _write_json(arguments.output, summary)
    _print_json({"output": str(arguments.output), "status": "written"})
    return 0


def _artifact_validate(arguments: argparse.Namespace) -> int:
    snapshot = load_run_snapshot(arguments.snapshot)
    _print_json(
        {
            "completed_count": len(snapshot.records),
            "dataset": snapshot.metadata.dataset,
            "run_id": snapshot.metadata.run_id,
            "status": snapshot.checkpoint.status.value,
            "variant": snapshot.metadata.variant.value,
        }
    )
    return 0


def _test(arguments: argparse.Namespace) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(arguments.tests)],
        check=False,
    )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:


    parser = argparse.ArgumentParser(prog="qprobe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_build = subparsers.add_parser("manifest-build")
    manifest_build.add_argument("--dataset", choices=tuple(sorted(_ADAPTERS)), required=True)
    manifest_build.add_argument("--prepared-jsonl", type=Path, required=True)
    manifest_build.add_argument("--sampling-secret-file", type=Path, required=True)
    manifest_build.add_argument("--sampling-seed-ref", required=True)
    manifest_build.add_argument("--source-fingerprint", required=True)
    manifest_build.add_argument("--output", type=Path, required=True)
    manifest_build.set_defaults(handler=_manifest_build)

    manifest_validate = subparsers.add_parser("manifest-validate")
    manifest_validate.add_argument("--manifest", type=Path, required=True)
    manifest_validate.add_argument("--prepared-jsonl", type=Path, required=True)
    manifest_validate.set_defaults(handler=_manifest_validate)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument(
        "--seed-evidence",
        type=Path,
        nargs=3,
        metavar=("SEED_1", "SEED_2", "SEED_3"),
        required=True,
    )
    calibrate.add_argument("--manifest-fingerprint", required=True)
    calibrate.add_argument("--controlled-config-fingerprint", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.set_defaults(handler=_calibrate)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--snapshot", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(handler=_aggregate)

    artifact_validate = subparsers.add_parser("artifact-validate")
    artifact_validate.add_argument("--snapshot", type=Path, required=True)
    artifact_validate.set_defaults(handler=_artifact_validate)

    test = subparsers.add_parser("test")
    test.add_argument("--tests", type=Path, required=True)
    test.set_defaults(handler=_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:


    parser = build_parser()
    arguments = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = arguments.handler
    try:
        return handler(arguments)
    except (QProbeError, OSError, ValueError) as error:
        print(f"qprobe: {error}", file=sys.stderr)
        return 2
