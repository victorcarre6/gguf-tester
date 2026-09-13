"""Shared pytest fixtures."""

from __future__ import annotations

import struct
import sys
import types
from pathlib import Path

import pytest


# Inject a stub `llama_cpp` module so `mock.patch("llama_cpp.Llama")` succeeds
# without requiring the heavy native package in CI / dev shells.
if "llama_cpp" not in sys.modules:
    stub = types.ModuleType("llama_cpp")
    stub.Llama = type("Llama", (), {})  # placeholder; tests will patch this
    sys.modules["llama_cpp"] = stub


def _write_gguf_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


@pytest.fixture
def tiny_gguf(tmp_path: Path) -> Path:
    """Synthetic minimal GGUF file with one metadata kv and one tensor."""
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", 3))      # version
    buf.extend(struct.pack("<Q", 1))      # tensor_count
    buf.extend(struct.pack("<Q", 2))      # metadata_kv_count

    # metadata 1: general.architecture = "qwen" (string type=8)
    _write_gguf_string(buf, "general.architecture")
    buf.extend(struct.pack("<I", 8))      # value type STRING
    _write_gguf_string(buf, "qwen")

    # metadata 2: general.name = "tiny-test"
    _write_gguf_string(buf, "general.name")
    buf.extend(struct.pack("<I", 8))
    _write_gguf_string(buf, "tiny-test")

    # tensor 1
    _write_gguf_string(buf, "blk.0.attn_q.weight")
    buf.extend(struct.pack("<I", 2))                     # ndim
    buf.extend(struct.pack("<Q", 4))                     # shape[0]
    buf.extend(struct.pack("<Q", 4))                     # shape[1]
    buf.extend(struct.pack("<I", 1))                     # ggml_type = F16
    buf.extend(struct.pack("<Q", 0))                     # offset

    path = tmp_path / "tiny.gguf"
    path.write_bytes(bytes(buf))
    return path
