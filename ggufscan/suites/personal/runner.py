"""Runner for versioned personal benchmark datasets."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ggufscan.domain import BenchmarkResult, ResultStatus, TaskResult
from ggufscan.parser import ParsedGGUF
from ggufscan.suites.personal.graders import Judge, Sandbox, grade_response
from ggufscan.suites.personal.models import PersonalDataset, load_personal_dataset


@dataclass(frozen=True)
class PersonalRecord:
    task_id: str
    prompt: str
    response: str
    status: ResultStatus
    score: float | None
    detail: str
    duration_seconds: float
    repetition: int = 1


@dataclass(frozen=True)
class PersonalSuiteResult:
    dataset: PersonalDataset
    benchmark: BenchmarkResult
    records: tuple[PersonalRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": {
                "name": self.dataset.name,
                "version": self.dataset.version,
                "schema_version": self.dataset.schema_version,
            },
            "benchmark": asdict(self.benchmark),
            "records": [asdict(record) for record in self.records],
        }


class PersonalRunner:
    """Generate and grade personal tasks using a chat model handle."""

    def run(
        self,
        parsed: ParsedGGUF,
        dataset_path: Path,
        *,
        n_ctx: int,
        n_gpu_layers: int,
        tensor_split: list[float] | None,
        chat_format: str | None,
        seed: int,
        max_tokens: int,
        temperature: float,
        tags: set[str] | None = None,
        limit: int | None = None,
        repetitions: int = 1,
        sandbox: Sandbox | None = None,
        judge: Judge | None = None,
    ) -> PersonalSuiteResult:
        """Execute a filtered dataset and aggregate only scored tasks."""

        from ggufscan.inference import ModelHandle

        dataset = load_personal_dataset(dataset_path)
        tasks = [task for task in dataset.tasks if not tags or tags.intersection(task.tags)]
        if limit is not None:
            tasks = tasks[:limit]

        if repetitions < 1:
            raise ValueError("personal repetitions must be at least 1")

        records: list[PersonalRecord] = []
        task_results: list[TaskResult] = []
        handle_kwargs = {
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "tensor_split": tensor_split,
            "chat_format": chat_format,
            "seed": seed,
        }
        with ModelHandle(parsed, embedding=False, **handle_kwargs) as model:
            for repetition in range(1, repetitions + 1):
                for task in tasks:
                    response, grade, duration = _evaluate_task(
                        model,
                        task,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        sandbox=sandbox,
                        judge=judge,
                    )
                    result_id = task.id if repetitions == 1 else f"{task.id}#{repetition}"
                    records.append(PersonalRecord(
                        task_id=task.id,
                        prompt=task.prompt,
                        response=response,
                        status=grade.status,
                        score=grade.score,
                        detail=grade.detail,
                        duration_seconds=duration,
                        repetition=repetition,
                    ))
                    task_results.append(TaskResult(
                        task_id=result_id,
                        status=grade.status,
                        score=grade.score,
                        duration_seconds=duration,
                        error=(
                            grade.detail
                            if grade.status not in (ResultStatus.PASSED, ResultStatus.FAILED)
                            else None
                        ),
                    ))

        benchmark = _aggregate(dataset, task_results)

        return PersonalSuiteResult(dataset, benchmark, tuple(records))


def _evaluate_task(model, task, *, max_tokens, temperature, sandbox, judge):
    """Generate one response, unless its required grader is unavailable."""

    started = time.monotonic()
    missing_sandbox = task.grader.type == "command" and sandbox is None
    missing_judge = task.grader.type == "llm_judge" and judge is None
    if missing_sandbox or missing_judge:
        grade = grade_response(task, "", sandbox=sandbox, judge=judge)

        return "", grade, time.monotonic() - started

    messages = []
    if task.system_prompt:
        messages.append({"role": "system", "content": task.system_prompt})
    messages.append({"role": "user", "content": task.prompt})
    response = model.complete(
        messages,
        max_tokens=task.max_tokens or max_tokens,
        temperature=temperature,
    )
    grade = grade_response(task, response, sandbox=sandbox, judge=judge)

    return response, grade, time.monotonic() - started


def _aggregate(dataset: PersonalDataset, tasks: list[TaskResult]) -> BenchmarkResult:
    scored = [task.score for task in tasks if task.score is not None]
    score = sum(scored) / len(scored) if scored else None
    statuses = {task.status for task in tasks}
    if not tasks:
        status = ResultStatus.SKIPPED
    elif ResultStatus.FAILED in statuses:
        status = ResultStatus.FAILED
    elif statuses - {ResultStatus.PASSED}:
        status = ResultStatus.PARTIAL
    else:
        status = ResultStatus.PASSED

    score_std = statistics.pstdev(scored) if len(scored) > 1 else 0.0 if scored else None

    return BenchmarkResult(
        suite=dataset.name,
        version=dataset.version,
        status=status,
        score=score,
        tasks=tuple(tasks),
        metrics=(
            ("scored_tasks", len(scored)),
            ("total_tasks", len(tasks)),
            ("score_std", score_std),
        ),
    )


def write_personal_result(path: Path, result: PersonalSuiteResult) -> None:
    """Persist the detailed personal benchmark artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str)
    path.write_text(content + "\n", encoding="utf-8")
