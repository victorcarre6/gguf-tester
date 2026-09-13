"""Deterministic and delegated graders for personal tasks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from ggufscan.domain import ResultStatus
from ggufscan.suites.personal.models import GraderSpec, PersonalTask
from ggufscan.utils import strip_think


@dataclass(frozen=True)
class Grade:
    status: ResultStatus
    score: float | None
    detail: str


class Sandbox(Protocol):
    def grade_command(self, task: PersonalTask, response: str) -> Grade: ...


Judge = Callable[[PersonalTask, str], Grade]


def grade_response(
    task: PersonalTask,
    response: str,
    *,
    sandbox: Sandbox | None = None,
    judge: Judge | None = None,
) -> Grade:
    """Apply a task grader without executing untrusted code on the host."""

    grader = task.grader
    clean = strip_think(response).strip()
    if grader.type == "exact":
        return _grade_exact(grader, clean)
    if grader.type == "regex":
        return _grade_regex(grader, clean)
    if grader.type == "json_schema":
        return _grade_json_schema(grader, clean)
    if grader.type == "command":
        if sandbox is None:
            return Grade(ResultStatus.UNSUPPORTED, None, "command grader requires a sandbox")
        return sandbox.grade_command(task, clean)
    if judge is None:
        return Grade(ResultStatus.UNSUPPORTED, None, "llm_judge grader requires a judge")

    return judge(task, clean)


def _grade_exact(grader: GraderSpec, response: str) -> Grade:
    accepted = {value.strip().casefold() for value in grader.accepted}
    passed = response.casefold() in accepted

    return _boolean_grade(passed, "exact match" if passed else "no exact match")


def _grade_regex(grader: GraderSpec, response: str) -> Grade:
    if not grader.pattern:
        return Grade(ResultStatus.INVALID, None, "regex grader has no pattern")
    passed = re.search(grader.pattern, response, re.IGNORECASE | re.DOTALL) is not None

    return _boolean_grade(passed, "regex matched" if passed else "regex did not match")


def _grade_json_schema(grader: GraderSpec, response: str) -> Grade:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        return Grade(ResultStatus.FAILED, 0.0, f"invalid JSON: {exc.msg}")

    schema = dict(grader.schema)
    passed, detail = _validate_json(value, schema)

    return _boolean_grade(passed, detail)


def _validate_json(value, schema: dict) -> tuple[bool, str]:
    expected_type = schema.get("type")
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "boolean": bool,
    }
    python_type = type_map.get(expected_type)
    if python_type is not None and not isinstance(value, python_type):
        return False, f"expected JSON type {expected_type}"
    if isinstance(value, dict):
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            return False, f"missing required keys: {', '.join(missing)}"

    return True, "JSON schema matched"


def _boolean_grade(passed: bool, detail: str) -> Grade:
    status = ResultStatus.PASSED if passed else ResultStatus.FAILED

    return Grade(status, 1.0 if passed else 0.0, detail)
