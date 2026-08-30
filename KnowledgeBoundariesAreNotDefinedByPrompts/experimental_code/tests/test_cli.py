

import json
from pathlib import Path
from types import SimpleNamespace

from qprobe.cli import main
from qprobe.datasets import load_dataset_manifest


def _write_prepared_jsonl(path: Path) -> None:
    records = (
        {
            "dataset": "triviaqa",
            "sample_id": f"q-{index:04d}",
            "question": f"Question {index}?",
            "reference_answer_groups": [[f"Answer {index}"]],
            "source_split": "validation",
        }
        for index in range(1200)
    )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_manifest_commands_use_explicit_synthetic_inputs(tmp_path) -> None:
    prepared_jsonl = tmp_path / "prepared.jsonl"
    secret_file = tmp_path / "sampling.secret"
    manifest_path = tmp_path / "manifest.json"
    _write_prepared_jsonl(prepared_jsonl)
    secret_file.write_bytes(b"synthetic-private-secret")

    build_status = main(
        (
            "manifest-build",
            "--dataset",
            "triviaqa",
            "--prepared-jsonl",
            str(prepared_jsonl),
            "--sampling-secret-file",
            str(secret_file),
            "--sampling-seed-ref",
            "synthetic-seed-ref",
            "--source-fingerprint",
            "synthetic-source-fingerprint",
            "--output",
            str(manifest_path),
        )
    )
    validate_status = main(
        (
            "manifest-validate",
            "--manifest",
            str(manifest_path),
            "--prepared-jsonl",
            str(prepared_jsonl),
        )
    )

    assert build_status == 0
    assert validate_status == 0
    assert load_dataset_manifest(manifest_path).dataset == "triviaqa"


def test_test_command_uses_the_current_interpreter(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_run(command, *, check):
        observed["command"] = command
        observed["check"] = check
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("qprobe.cli.subprocess.run", fake_run)

    status = main(("test", "--tests", str(tmp_path)))

    assert status == 7
    assert observed["command"][1:3] == ["-m", "pytest"]
    assert observed["check"] is False
