"""Tests for the CLI-independent security orchestrator."""

from ggufscan.parser import parse
from ggufscan.runners import SecurityRunner


def test_static_runner_applies_load_guard(tiny_gguf):
    parsed = parse(str(tiny_gguf))

    execution = SecurityRunner().run_static(parsed)

    assert execution.report.filename == "tiny.gguf"
    assert execution.report.load_guard_verdict == execution.verdict
    assert execution.verdict in {"CLEAN", "WARN", "DETONATE"}
    assert isinstance(execution.findings, tuple)
