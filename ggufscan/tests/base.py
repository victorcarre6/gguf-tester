"""TestCase ABC + dataclasses for dynamic scan results.

Two-phase execution to handle the llama.cpp constraint that chat completion and
embeddings cannot coexist on a single Llama instance:

  Phase 1 — collect(chat_handle): drive all chat completions, store raw data.
  Phase 2 — score(collected, embedding_handle): apply judge / embed / cluster.

Tests that don't need embeddings ignore the second handle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any, TYPE_CHECKING

import yaml

from ggufscan.domain import ResultStatus

if TYPE_CHECKING:
    from ggufscan.inference import ModelHandle
    from ggufscan.judge import Verdict


@dataclass
class Sample:
    prompt: str
    response: str
    verdict: "Verdict | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Collected:
    """Per-test scratch data collected during the chat-completion phase."""
    samples: list[Sample] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    name: str
    score: float
    passed: bool
    threshold: float
    samples: list[Sample] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    status: ResultStatus | str | None = None

    def __post_init__(self) -> None:
        """Normalize the legacy boolean and the explicit execution status."""

        if self.status is None:
            self.status = ResultStatus.PASSED if self.passed else ResultStatus.FAILED
        elif self.status == "error":
            self.status = ResultStatus.INFRA_ERROR
        elif isinstance(self.status, str):
            self.status = ResultStatus(self.status)

        if self.status not in (ResultStatus.PASSED, ResultStatus.FAILED):
            self.passed = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "status": self.status,
            "threshold": self.threshold,
            "n_samples": len(self.samples),
            "failed_samples": [
                {"prompt": s.prompt[:200], "response": s.response[:300], "metadata": s.metadata}
                for s in self.samples
                if s.verdict is not None and not s.verdict.refused
            ][:10],
            "details": self.details,
        }


class TestCase(ABC):
    name: str = ""
    prompts_yaml: str = ""
    threshold: float = 0.8
    needs_embedding: bool = False

    def __init__(self, threshold: float | None = None):
        if threshold is not None:
            self.threshold = threshold

    def load_prompts(self) -> Any:
        if not self.prompts_yaml:
            return None
        raw = files("ggufscan.data.prompts").joinpath(self.prompts_yaml).read_text(encoding="utf-8")
        return yaml.safe_load(raw)

    @abstractmethod
    def collect(self, chat_handle: "ModelHandle", quick: bool = False, **kwargs) -> Collected:
        ...

    @abstractmethod
    def score(self, collected: Collected, embedding_handle: "ModelHandle | None" = None) -> TestResult:
        ...
