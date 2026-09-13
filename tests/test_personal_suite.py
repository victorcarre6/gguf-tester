"""Phase-4 personal dataset, grader, and runner tests."""

from pathlib import Path
from unittest.mock import patch

import pytest

from ggufscan.domain import ResultStatus
from ggufscan.parser import parse
from ggufscan.suites.personal.graders import grade_response
from ggufscan.suites.personal.models import GraderSpec, PersonalTask, load_personal_dataset
from ggufscan.suites.personal.runner import PersonalRunner


class _FakeLlama:
    def __init__(self, response="42", **kwargs):
        self.response = response

    def create_chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": self.response}}]}


def _write_dataset(path: Path) -> None:
    path.write_text(
        """\
schema_version: 1
name: personal-test
version: "1.2"
tasks:
  - id: answer
    type: generation
    prompt: Answer only 42.
    grader: {type: exact, accepted: ["42"]}
    tags: [smoke]
  - id: command
    type: command
    prompt: Fix the workspace.
    workspace: fixture
    grader: {type: command, command: [pytest, -q]}
    tags: [command]
""",
        encoding="utf-8",
    )


def test_repository_personal_dataset_has_between_20_and_50_tasks():
    path = Path("benchmarks/personal/v1/core.yml")

    dataset = load_personal_dataset(path)

    assert 20 <= len(dataset.tasks) <= 50
    assert dataset.name == "personal-core"
    command_task = next(task for task in dataset.tasks if task.type == "command")
    assert Path(command_task.workspace).is_dir()


def test_loader_rejects_duplicate_task_ids(tmp_path):
    path = tmp_path / "duplicate.yml"
    path.write_text(
        """\
schema_version: 1
tasks:
  - {id: same, type: generation, prompt: x, grader: {type: exact, accepted: [x]}}
  - {id: same, type: generation, prompt: y, grader: {type: exact, accepted: [y]}}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_personal_dataset(path)


def test_deterministic_graders():
    exact = PersonalTask("exact", "generation", "p", GraderSpec("exact", accepted=("yes",)))
    regex = PersonalTask("regex", "generation", "p", GraderSpec("regex", pattern=r"value=\d+"))
    structured = PersonalTask(
        "json",
        "generation",
        "p",
        GraderSpec("json_schema", schema=(("type", "object"), ("required", ["ok"]))),
    )

    assert grade_response(exact, "YES").status is ResultStatus.PASSED
    assert grade_response(regex, "value=12").score == 1.0
    assert grade_response(structured, '{"ok": true}').score == 1.0
    assert grade_response(structured, "not-json").score == 0.0


def test_command_and_judge_graders_require_delegates():
    command = PersonalTask("cmd", "command", "p", GraderSpec("command", command=("pytest",)))
    judged = PersonalTask("judge", "generation", "p", GraderSpec("llm_judge", rubric="good"))

    assert grade_response(command, "patch").status is ResultStatus.UNSUPPORTED
    assert grade_response(judged, "answer").status is ResultStatus.UNSUPPORTED


def test_personal_runner_filters_and_aggregates(tiny_gguf, tmp_path):
    dataset_path = tmp_path / "personal.yml"
    _write_dataset(dataset_path)
    parsed = parse(str(tiny_gguf))

    with patch("llama_cpp.Llama", _FakeLlama, create=True):
        result = PersonalRunner().run(
            parsed,
            dataset_path,
            n_ctx=1024,
            n_gpu_layers=0,
            tensor_split=None,
            chat_format=None,
            seed=42,
            max_tokens=32,
            temperature=0.0,
            tags={"smoke"},
            repetitions=2,
        )

    assert result.benchmark.status is ResultStatus.PASSED
    assert result.benchmark.score == 1.0
    assert len(result.records) == 2
    assert [record.repetition for record in result.records] == [1, 2]
