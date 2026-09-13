"""Small helpers shared by CLI commands that write artifacts."""

from __future__ import annotations

from pathlib import Path


def resolve_output_path(raw: str, default_name: str) -> Path:
    """Interpret an existing directory or trailing slash as directory mode."""

    path = Path(raw).expanduser()
    if path.is_dir() or raw.endswith(("/", "\\")):
        return path / default_name

    return path


def ensure_output_parent(path: Path) -> None:
    """Create the parent directory and reject a parent that is a file."""

    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise ValueError(f"output parent is not a directory: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
