"""Small stderr presentation layer for interactive CLI runs.

Reports contain the details; terminal output deliberately stays compact and
dependency-free so the execution flow remains easy to follow and test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ggufscan import __version__
from ggufscan.ragas_eval import METRIC_KEYS

if TYPE_CHECKING:
    from ggufscan.static_scan import StaticReport
    from ggufscan.tests.base import Collected, TestResult


def _write(message: str = "") -> None:
    print(message, file=sys.stderr)


def print_header(model_path: Path, version: str = __version__) -> None:
    _write(f"ggufscan {version} — {model_path.name}")


def print_static_analysis(
    report: "StaticReport",
    model_path: Path | None = None,
    version: str = __version__,
) -> None:
    if model_path:
        print_header(model_path, version)

    structure = report.analysis.get("structure", {})
    name = structure.get("model_name", "Unknown")
    architecture = structure.get("architecture", "unknown")
    tensors = structure.get("tensor_count", 0)
    quantization = ", ".join(structure.get("quantization_types") or ["unknown"])
    _write(
        f"static: {name} | arch={architecture} | tensors={tensors} | "
        f"quant={quantization} | size={report.file_size_mb:.1f} MB"
    )


def print_load_guard(verdict: str, findings: list[dict]) -> None:
    _write(f"load guard: {verdict} ({len(findings)} finding(s))")
    for finding in findings:
        if finding.get("verdict") != "CLEAN":
            _write(f"  - {finding.get('check')}: {finding.get('detail')}")


def print_hf_checksum(detail: str, match: bool | None) -> None:
    status = "match" if match is True else "mismatch" if match is False else "unknown"
    _write(f"HF checksum [{status}]: {detail}")


def print_detonate(findings: list[dict]) -> None:
    _write("BLOCKED: load guard returned DETONATE")
    for finding in findings:
        if finding.get("verdict") == "DETONATE":
            _write(f"  - {finding.get('check')}: {finding.get('detail')}")


def print_test_result(result: "TestResult", collected: "Collected") -> None:
    count = len(collected.samples) or len(result.samples)
    _write(
        f"{result.name}: {result.status.value} | score={result.score:.1%} | "
        f"threshold={result.threshold:.1%} | samples={count}"
    )


def print_summary(results: list["TestResult"]) -> None:
    if not results:
        return

    passed = sum(result.passed for result in results)
    _write(f"security summary: {passed}/{len(results)} tests passed")


def print_ragas_eval(summary: dict, analysis: str) -> None:
    total = summary.get("n_records", 0)
    errors = summary.get("n_errors", 0)
    _write(f"local judge: records={total} errors={errors}")
    for key in METRIC_KEYS:
        value = summary.get(f"{key}_mean")
        rendered = f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"
        _write(f"  - {key}: {rendered}")
    if analysis:
        _write(f"  analysis: {analysis.strip()[:500]}")


def print_info(message: str) -> None:
    _write(message)


def print_warn(message: str) -> None:
    _write(f"WARNING: {message}")


def print_error(message: str) -> None:
    _write(f"ERROR: {message}")


def print_artifact(label: str, path: str) -> None:
    _write(f"{label}: {path}")


def print_nemoclaw(kind: str, path: Path) -> None:
    label = "NemoClaw blueprint" if kind == "blueprint" else "NemoClaw blocked report"
    _write(f"{label}: {path}")
