"""Regression tests for the versioned scan-result contract."""

from ggufscan.report import to_json
from ggufscan.tests.base import TestResult as Result


class _StaticResult:
    def as_dict(self):
        return {"filename": "model.gguf"}


def test_scan_json_contract_is_versioned():
    result = Result(
        name="example",
        score=0.5,
        passed=False,
        threshold=0.8,
    )

    payload = to_json(_StaticResult(), [result])

    assert payload["schema_version"] == 1
    assert payload["dynamic"][0]["status"] == "failed"


def test_non_executed_result_cannot_pass():
    result = Result(
        name="example",
        score=1.0,
        passed=True,
        threshold=0.8,
        status="skipped",
    )

    assert not result.passed
    assert result.as_dict()["status"] == "skipped"
