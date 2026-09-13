"""Parser unit tests."""

from __future__ import annotations

from pathlib import Path
import struct

import pytest

from ggufscan.parser import GGUFParser, parse


def test_parse_invalid_magic(tmp_path: Path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOTAGGUF___")
    with pytest.raises(ValueError, match="magic"):
        GGUFParser(str(bad)).parse()


def test_parse_minimal(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    assert parsed.filename == "tiny.gguf"
    assert parsed.file_size > 0
    assert parsed.metadata["general.architecture"] == "qwen"
    assert parsed.metadata["general.name"] == "tiny-test"
    assert "blk.0.attn_q.weight" in parsed.tensors_info
    t = parsed.tensors_info["blk.0.attn_q.weight"]
    assert t["shape"] == [4, 4]
    assert t["element_count"] == 16
    assert t["type_name"] == "F16"
    assert not t["is_quantized"]


def test_parsed_properties(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    assert parsed.architecture == "qwen"
    assert parsed.model_name == "tiny-test"
    assert parsed.total_size_mb > 0


def test_parse_rejects_unsupported_version(tmp_path: Path):
    path = tmp_path / "v1.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 1) + b"\0" * 16)

    with pytest.raises(ValueError, match="version"):
        parse(str(path))


def test_parse_rejects_truncated_header(tmp_path: Path):
    path = tmp_path / "truncated.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3))

    with pytest.raises(ValueError, match="end of GGUF header"):
        parse(str(path))
