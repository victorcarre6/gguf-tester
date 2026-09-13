"""Shared text utilities + progress bar helper."""

from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager

import numpy as np

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove `<think>...</think>` blocks from reasoning-model output.

    Also handles the truncated case (`<think>...` with no closing tag) — common
    when `max_tokens` is exhausted during the reasoning phase. Leaves the
    visible answer (or empty string) intact.
    """
    if not text:
        return text
    stripped = _THINK_RE.sub("", text)
    return stripped.strip()


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity, or 0 when either vector has no direction."""

    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0

    return float(np.dot(first, second) / denominator)


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return a non-negative cosine distance."""

    return max(0.0, 1.0 - cosine_similarity(first, second))


def _fmt_secs(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


@contextmanager
def progress(total: int, title: str):
    """Yield a `step()` callable iterated `total` times.

    The implementation is intentionally dependency-free and writes one compact
    ticker to stderr.
    """
    start = time.monotonic()
    count = [0]

    def step(n: int = 1):
        count[0] += n
        elapsed = time.monotonic() - start
        eta = (elapsed / count[0]) * (total - count[0]) if count[0] else 0.0
        sys.stderr.write(
            f"\r[{_fmt_secs(elapsed)} eta {_fmt_secs(eta)}] {title} {count[0]}/{total}"
        )
        sys.stderr.flush()

    yield step
    sys.stderr.write("\n")
    sys.stderr.flush()
