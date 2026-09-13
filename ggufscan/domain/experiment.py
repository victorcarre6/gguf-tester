"""Immutable experiment specifications and generic benchmark results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


RUN_MANIFEST_SCHEMA_VERSION = 1


class ResultStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"
    INVALID = "invalid"


@dataclass(frozen=True)
class ModelSpec:
    """Model identity and inference parameters requested for a run."""

    path: str
    provider: str = "llama_cpp"
    sha256: str | None = None
    architecture: str | None = None
    quantization: str | None = None
    file_size_bytes: int | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    tensor_split: tuple[float, ...] = ()
    chat_format: str | None = None
    max_tokens: int = 256
    temperature: float = 0.3


@dataclass(frozen=True)
class SuiteSpec:
    """A versioned suite selection within an experiment."""

    name: str
    version: str
    tests: tuple[str, ...] = ()
    profile: str = "full"


@dataclass(frozen=True)
class RunSpec:
    """Immutable requested experiment configuration."""

    run_id: str
    created_at: str
    model: ModelSpec
    suites: tuple[SuiteSpec, ...]
    seed: int = 42
    repetitions: int = 1
    config_path: str | None = None


@dataclass(frozen=True)
class TaskResult:
    """Provider-independent result for one benchmark task."""

    task_id: str
    status: ResultStatus
    score: float | None = None
    duration_seconds: float | None = None
    metrics: tuple[tuple[str, Any], ...] = ()
    error: str | None = None
    artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate result for one suite or benchmark."""

    suite: str
    version: str
    status: ResultStatus
    score: float | None = None
    tasks: tuple[TaskResult, ...] = ()
    metrics: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class RunManifest:
    """Immutable provenance record written once for an experiment."""

    run: RunSpec
    runner_name: str
    runner_version: str
    python_version: str
    platform: str
    results: tuple[BenchmarkResult, ...] = ()
    artifacts: tuple[str, ...] = ()
    schema_version: int = field(default=RUN_MANIFEST_SCHEMA_VERSION, init=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)
