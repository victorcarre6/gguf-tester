"""Orchestration for the existing static and dynamic security suite."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ggufscan.load_guard import LoadGuard
from ggufscan.parser import ParsedGGUF
from ggufscan.static_scan import StaticReport, StaticScanner
from ggufscan.suites.security import security_test_registry
from ggufscan.tests.base import Collected, TestResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaticExecution:
    """Static report plus the load decision required by the caller."""

    report: StaticReport
    verdict: str
    findings: tuple[dict[str, Any], ...]


class SecurityRunner:
    """Run the security suite without CLI or presentation concerns."""

    def run_static(self, parsed: ParsedGGUF, compute_sha256: bool = False) -> StaticExecution:
        """Inspect a GGUF and apply the pre-load guard."""

        report = StaticScanner(parsed).scan()
        guard = LoadGuard(parsed)
        verdict, findings = guard.run()
        report.load_guard_verdict = verdict
        report.load_guard_findings = findings
        if compute_sha256:
            report.file_sha256 = guard.compute_sha256()

        return StaticExecution(
            report=report,
            verdict=verdict,
            findings=tuple(findings),
        )

    def run_dynamic(
        self,
        parsed: ParsedGGUF,
        requested: list[str],
        *,
        n_ctx: int,
        n_gpu_layers: int,
        tensor_split: list[float] | None,
        chat_format: str | None,
        seed: int,
        quick: bool,
        max_tokens: int,
        temperature: float,
        on_result: Callable[[TestResult, Collected], None] | None = None,
    ) -> list[TestResult]:
        """Collect and score the selected tests in sequential model passes."""

        from ggufscan.inference import ModelHandle

        registry = security_test_registry()
        tests = {name: registry[name]() for name in requested}
        if not tests:
            return []

        handle_kwargs: dict[str, Any] = {
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "tensor_split": tensor_split,
            "chat_format": chat_format,
            "seed": seed,
        }
        needs_embedding = any(test.needs_embedding for test in tests.values())

        log.info("Pass 1/2: loading model for chat completion...")
        collected: dict[str, Collected] = {}
        with ModelHandle(parsed, embedding=False, **handle_kwargs) as chat:
            for name, test in tests.items():
                log.info("Collecting samples for %s...", name)
                collected[name] = test.collect(
                    chat,
                    quick=quick,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )

        embedding_handle = self._open_embedding_handle(
            parsed,
            handle_kwargs,
            needs_embedding=needs_embedding,
        )
        results: list[TestResult] = []
        try:
            for name, test in tests.items():
                log.info("Scoring %s...", name)
                embedder = embedding_handle if test.needs_embedding else None
                result = test.score(collected[name], embedding_handle=embedder)
                results.append(result)
                if on_result is not None:
                    on_result(result, collected[name])
        finally:
            if embedding_handle is not None:
                embedding_handle.__exit__(None, None, None)

        return results

    def _open_embedding_handle(
        self,
        parsed: ParsedGGUF,
        handle_kwargs: dict[str, Any],
        *,
        needs_embedding: bool,
    ):
        """Open the optional second model pass and degrade to unsupported."""

        if not needs_embedding:
            return None

        from ggufscan.inference import ModelHandle

        log.info("Pass 2/2: reloading model with embedding=True...")
        try:
            handle = ModelHandle(parsed, embedding=True, **handle_kwargs)
            handle.__enter__()
        except (ValueError, RuntimeError) as exc:
            log.warning("Embedding context failed (%s); related tests are unsupported.", exc)

            return None

        return handle
