"""Phase-1 contracts for experiment specifications and manifests."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from ggufscan.domain import ModelSpec, ResultStatus, RunSpec, SuiteSpec
from ggufscan.manifest import (
    create_manifest,
    create_run_spec,
    security_benchmark_result,
    write_manifest,
)
from ggufscan.parser import parse
from ggufscan.tests.base import TestResult as SecurityTestResult


def test_run_contracts_are_immutable():
    model = ModelSpec(path="model.gguf")
    suite = SuiteSpec(name="security", version="1")
    run = RunSpec(
        run_id="run-1",
        created_at="2026-08-17T00:00:00+00:00",
        model=model,
        suites=(suite,),
    )

    with pytest.raises(FrozenInstanceError):
        run.seed = 7


def test_create_run_spec_records_resolved_configuration(tiny_gguf):
    parsed = parse(str(tiny_gguf))
    instant = datetime(2026, 8, 17, tzinfo=timezone.utc)

    run = create_run_spec(
        parsed,
        requested_tests=["jailbreak"],
        quick=True,
        static_only=False,
        n_ctx=8192,
        n_gpu_layers=12,
        tensor_split=[0.4, 0.6],
        chat_format="chatml",
        max_tokens=512,
        temperature=0.1,
        seed=7,
        config_path="config.yml",
        created_at=instant,
    )

    assert run.model.architecture == "qwen"
    assert run.model.file_size_bytes == tiny_gguf.stat().st_size
    assert run.model.tensor_split == (0.4, 0.6)
    assert run.model.chat_format == "chatml"
    assert run.suites[0].profile == "quick"
    assert run.suites[0].tests == ("jailbreak",)


def test_unsupported_task_is_non_scoring_and_suite_is_partial():
    result = SecurityTestResult(
        name="backdoor",
        score=0.0,
        passed=False,
        threshold=0.95,
        status="unsupported",
        details={"skip_reason": "no embeddings"},
    )

    benchmark = security_benchmark_result([result], static_verdict="CLEAN")

    assert benchmark.status is ResultStatus.PARTIAL
    assert benchmark.tasks[0].score is None
    assert benchmark.score is None


def test_manifest_is_versioned_and_cannot_be_overwritten(tmp_path):
    run = RunSpec(
        run_id="run-1",
        created_at="2026-08-17T00:00:00+00:00",
        model=ModelSpec(path="model.gguf"),
        suites=(SuiteSpec(name="security", version="1"),),
    )
    manifest = create_manifest(run)
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)

    assert manifest.as_dict()["schema_version"] == 1
    with pytest.raises(FileExistsError):
        write_manifest(path, manifest)
