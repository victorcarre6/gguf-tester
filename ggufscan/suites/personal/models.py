"""Strict task contracts for personal benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Literal

import yaml


PERSONAL_DATASET_SCHEMA_VERSION = 1
TaskType = Literal["generation", "command"]
GraderType = Literal["exact", "regex", "json_schema", "command", "llm_judge"]


@dataclass(frozen=True)
class GraderSpec:
    type: GraderType
    accepted: tuple[str, ...] = ()
    pattern: str | None = None
    schema: tuple[tuple[str, Any], ...] = ()
    command: tuple[str, ...] = ()
    rubric: str | None = None


@dataclass(frozen=True)
class PersonalTask:
    id: str
    type: TaskType
    prompt: str
    grader: GraderSpec
    system_prompt: str | None = None
    tags: tuple[str, ...] = ()
    workspace: str | None = None
    timeout_seconds: int = 120
    max_tokens: int | None = None


@dataclass(frozen=True)
class PersonalDataset:
    name: str
    version: str
    tasks: tuple[PersonalTask, ...]
    schema_version: int = field(default=PERSONAL_DATASET_SCHEMA_VERSION, init=False)


def load_personal_dataset(path: Path) -> PersonalDataset:
    """Load and validate a personal benchmark YAML file."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("personal dataset root must be a mapping")
    if raw.get("schema_version") != PERSONAL_DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported personal dataset schema: {raw.get('schema_version')!r}")

    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("personal dataset must contain a non-empty tasks list")

    tasks = tuple(_parse_task(item, source=path) for item in raw_tasks)
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("personal task ids must be unique")

    return PersonalDataset(
        name=str(raw.get("name") or path.stem),
        version=str(raw.get("version") or "1"),
        tasks=tasks,
    )


def _parse_task(raw: Any, *, source: Path) -> PersonalTask:
    if not isinstance(raw, dict):
        raise ValueError("each personal task must be a mapping")

    task_id = str(raw.get("id") or "").strip()
    task_type = raw.get("type")
    prompt = str(raw.get("prompt") or "").strip()
    if not task_id or task_type not in ("generation", "command") or not prompt:
        raise ValueError(f"invalid personal task in {source}: id, type and prompt are required")

    grader_raw = raw.get("grader")
    grader = _parse_grader(grader_raw, task_type=task_type)
    workspace = raw.get("workspace")
    if workspace is not None:
        workspace = str((source.parent / str(workspace)).resolve())

    return PersonalTask(
        id=task_id,
        type=task_type,
        prompt=prompt,
        grader=grader,
        system_prompt=raw.get("system_prompt"),
        tags=tuple(str(tag) for tag in raw.get("tags") or ()),
        workspace=workspace,
        timeout_seconds=int((raw.get("limits") or {}).get("timeout_seconds", 120)),
        max_tokens=raw.get("max_tokens"),
    )


def _parse_grader(raw: Any, *, task_type: TaskType) -> GraderSpec:
    if not isinstance(raw, dict):
        raise ValueError("task grader must be a mapping")

    grader_type = raw.get("type") or ("command" if task_type == "command" else None)
    if grader_type not in ("exact", "regex", "json_schema", "command", "llm_judge"):
        raise ValueError(f"unsupported grader type: {grader_type!r}")

    command = raw.get("command") or ()
    if isinstance(command, str):
        raise ValueError("command grader must use an argv list, not a shell string")
    schema = raw.get("schema") or {}
    if not isinstance(schema, dict):
        raise ValueError("json_schema grader schema must be a mapping")

    accepted = tuple(str(value) for value in raw.get("accepted") or ())
    pattern = str(raw["pattern"]) if raw.get("pattern") is not None else None
    rubric = str(raw["rubric"]) if raw.get("rubric") is not None else None
    if grader_type == "exact" and not accepted:
        raise ValueError("exact grader requires at least one accepted value")
    if grader_type == "regex":
        if not pattern:
            raise ValueError("regex grader requires a pattern")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex grader pattern: {exc}") from exc
    if grader_type == "json_schema" and not schema:
        raise ValueError("json_schema grader requires a schema")
    if grader_type == "command" and not command:
        raise ValueError("command grader requires a non-empty argv list")
    if grader_type == "llm_judge" and not rubric:
        raise ValueError("llm_judge grader requires a rubric")

    return GraderSpec(
        type=grader_type,
        accepted=accepted,
        pattern=pattern,
        schema=tuple((str(key), value) for key, value in schema.items()),
        command=tuple(str(part) for part in command),
        rubric=rubric,
    )
