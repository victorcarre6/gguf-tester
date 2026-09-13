"""Build and persist immutable run manifests."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from ggufscan import __version__
from ggufscan.domain import (
    BenchmarkResult,
    ModelSpec,
    ResultStatus,
    RunManifest,
    RunSpec,
    SuiteSpec,
    TaskResult,
)
from ggufscan.parser import ParsedGGUF
from ggufscan.suites.security import (
    PROMPT_SET_VERSION,
    REFUSAL_CLASSIFIER_VERSION,
    SECURITY_SUITE_VERSION,
    THRESHOLD_SET_VERSION,
)
from ggufscan.tests.base import TestResult


def create_run_spec(
    parsed: ParsedGGUF,
    *,
    requested_tests: list[str],
    quick: bool,
    static_only: bool,
    n_ctx: int,
    n_gpu_layers: int,
    tensor_split: list[float] | None,
    chat_format: str | None,
    max_tokens: int,
    temperature: float,
    seed: int,
    config_path: str | None,
    created_at: datetime | None = None,
    sha256: str | None = None,
    run_id: str | None = None,
    additional_suites: tuple[SuiteSpec, ...] = (),
    repetitions: int = 1,
) -> RunSpec:
    """Create an immutable run specification from resolved scan settings."""

    instant = created_at or datetime.now(timezone.utc)
    timestamp = instant.strftime("%Y%m%dT%H%M%S.%fZ")
    resolved_run_id = run_id or f"{Path(parsed.filename).stem}-{timestamp}"
    quantization = parsed.metadata.get("general.file_type")
    from ggufscan.inference import detect_chat_format

    effective_chat_format = detect_chat_format(parsed, chat_format)
    model = ModelSpec(
        path=str(Path(parsed.file_path).resolve()),
        sha256=sha256,
        architecture=parsed.architecture,
        quantization=str(quantization) if quantization is not None else None,
        file_size_bytes=parsed.file_size,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tuple(tensor_split or ()),
        chat_format=effective_chat_format,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    profile = "static" if static_only else ("quick" if quick else "full")
    suite = SuiteSpec(
        name="security",
        version=SECURITY_SUITE_VERSION,
        tests=tuple(requested_tests),
        profile=profile,
    )

    return RunSpec(
        run_id=resolved_run_id,
        created_at=instant.isoformat(),
        model=model,
        suites=(suite, *additional_suites),
        seed=seed,
        repetitions=repetitions,
        config_path=config_path,
    )


def create_manifest(
    run: RunSpec,
    *,
    results: tuple[BenchmarkResult, ...] = (),
    artifacts: tuple[str, ...] = (),
) -> RunManifest:
    """Capture runner and host provenance around a resolved run."""

    return RunManifest(
        run=run,
        runner_name="ggufscan",
        runner_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        results=results,
        artifacts=artifacts,
    )


def security_benchmark_result(
    results: list[TestResult],
    *,
    static_verdict: str,
) -> BenchmarkResult:
    """Map legacy security results into the provider-independent contract."""

    tasks = tuple(_security_task_result(result) for result in results)
    if static_verdict == "DETONATE":
        status = ResultStatus.FAILED
    elif not tasks:
        status = ResultStatus.PARTIAL if static_verdict == "WARN" else ResultStatus.PASSED
    else:
        status = _aggregate_status(tasks, static_verdict=static_verdict)

    scored = [task.score for task in tasks if task.score is not None]
    score = sum(scored) / len(scored) if scored else None

    return BenchmarkResult(
        suite="security",
        version=SECURITY_SUITE_VERSION,
        status=status,
        score=score,
        tasks=tasks,
        metrics=(
            ("static_verdict", static_verdict),
            ("prompt_set_version", PROMPT_SET_VERSION),
            ("refusal_classifier_version", REFUSAL_CLASSIFIER_VERSION),
            ("threshold_set_version", THRESHOLD_SET_VERSION),
        ),
    )


def _security_task_result(result: TestResult) -> TaskResult:
    status = result.status or ResultStatus.INVALID
    if not isinstance(status, ResultStatus):
        status = ResultStatus.INVALID
    error = result.details.get("error") or result.details.get("skip_reason")
    non_scoring = {
        ResultStatus.SKIPPED,
        ResultStatus.UNSUPPORTED,
        ResultStatus.INFRA_ERROR,
        ResultStatus.INVALID,
    }
    score = None if status in non_scoring else result.score

    return TaskResult(
        task_id=result.name,
        status=status,
        score=score,
        duration_seconds=result.details.get("elapsed_seconds"),
        metrics=(("threshold", result.threshold),),
        error=str(error) if error else None,
    )


def _aggregate_status(
    tasks: tuple[TaskResult, ...],
    *,
    static_verdict: str,
) -> ResultStatus:
    terminal_failures = {ResultStatus.FAILED, ResultStatus.INFRA_ERROR, ResultStatus.INVALID}
    incomplete = {ResultStatus.SKIPPED, ResultStatus.UNSUPPORTED, ResultStatus.TIMEOUT}
    statuses = {task.status for task in tasks}
    if statuses & terminal_failures:
        return ResultStatus.FAILED
    if statuses & incomplete or static_verdict == "WARN":
        return ResultStatus.PARTIAL

    return ResultStatus.PASSED


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Write a manifest once, refusing to overwrite an existing run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False, default=str)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.write("\n")
