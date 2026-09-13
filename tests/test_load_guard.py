"""Unit tests for ggufscan.load_guard.

All tests use ParsedGGUF constructed in-memory (no real model).
"""

from __future__ import annotations

import struct
from pathlib import Path
import pytest

from ggufscan.load_guard import LoadGuard, _MAX_REASONABLE_KV, _MAX_STRING_BYTES
from ggufscan.parser import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parsed(tmp_path: Path, extra_meta: dict | None = None):
    """Build a minimal valid ParsedGGUF with optional extra metadata."""
    from tests.conftest import _write_gguf_string

    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", 3))   # version

    base_meta = {
        "general.architecture": "qwen",
        "general.name": "test-model",
        "general.quantization_version": "2",
    }
    if extra_meta:
        base_meta.update(extra_meta)

    buf.extend(struct.pack("<Q", 1))                    # tensor_count
    buf.extend(struct.pack("<Q", len(base_meta)))       # metadata_kv_count

    for key, val in base_meta.items():
        _write_gguf_string(buf, key)
        buf.extend(struct.pack("<I", 8))                # STRING type
        _write_gguf_string(buf, str(val))

    # one tensor
    _write_gguf_string(buf, "blk.0.attn_q.weight")
    buf.extend(struct.pack("<I", 2))
    buf.extend(struct.pack("<Q", 4))
    buf.extend(struct.pack("<Q", 4))
    buf.extend(struct.pack("<I", 1))   # F16
    buf.extend(struct.pack("<Q", 0))

    p = tmp_path / "test.gguf"
    p.write_bytes(bytes(buf))
    return parse(str(p))


# ---------------------------------------------------------------------------
# SSTI / chat_template
# ---------------------------------------------------------------------------

class TestSSTI:
    def test_no_template_is_clean(self, tmp_path):
        parsed = _make_parsed(tmp_path)
        verdict, findings = LoadGuard(parsed).run()
        ssti = [f for f in findings if f["check"] == "ssti"]
        assert not ssti

    def test_safe_template_warns(self, tmp_path):
        """Template with {{ }} but no dangerous patterns → WARN."""
        parsed = _make_parsed(tmp_path, {
            "tokenizer.chat_template": "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        })
        verdict, findings = LoadGuard(parsed).run()
        ssti = [f for f in findings if f["check"] == "ssti"]
        assert ssti
        assert ssti[0]["verdict"] == "WARN"
        assert verdict in ("WARN", "DETONATE")

    @pytest.mark.parametrize("dangerous_pattern", [
        "__class__",
        "__globals__",
        "os.system",
        "subprocess",
        "eval(",
        "exec(",
        "__import__",
    ])
    def test_dangerous_template_detonates(self, tmp_path, dangerous_pattern):
        template = "{% for msg in messages %}{{ " + dangerous_pattern + " }}{% endfor %}"
        parsed = _make_parsed(tmp_path, {"tokenizer.chat_template": template})
        verdict, findings = LoadGuard(parsed).run()
        assert verdict == "DETONATE"
        ssti = [f for f in findings if f["check"] == "ssti"]
        assert ssti
        assert ssti[0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_clean_model_no_integrity_finding(self, tmp_path):
        parsed = _make_parsed(tmp_path)
        _, findings = LoadGuard(parsed).run()
        integrity = [f for f in findings if f["check"] == "integrity"]
        assert not integrity

    def test_zero_tensors_detonates(self, tmp_path):
        parsed = _make_parsed(tmp_path)
        parsed.tensors_info = {}          # simulate corrupted / zero tensors
        verdict, findings = LoadGuard(parsed).run()
        assert verdict == "DETONATE"
        integrity = [f for f in findings if f["check"] == "integrity"]
        assert any("zero tensors" in f["detail"] for f in integrity)

    def test_absurd_kv_count_warns(self, tmp_path):
        parsed = _make_parsed(tmp_path)
        # Inject more keys than the threshold
        for i in range(_MAX_REASONABLE_KV + 1):
            parsed.metadata[f"dummy.key.{i}"] = "v"
        verdict, findings = LoadGuard(parsed).run()
        assert verdict in ("WARN", "DETONATE")
        integrity = [f for f in findings if f["check"] == "integrity"]
        assert any("metadata_kv_count" in f["detail"] for f in integrity)

    def test_alloc_bomb_string_warns(self, tmp_path):
        parsed = _make_parsed(tmp_path)
        parsed.metadata["suspicious.blob"] = "x" * (_MAX_STRING_BYTES + 1)
        verdict, findings = LoadGuard(parsed).run()
        assert verdict in ("WARN", "DETONATE")
        integrity = [f for f in findings if f["check"] == "integrity"]
        assert any("alloc bomb" in f["detail"] for f in integrity)


# ---------------------------------------------------------------------------
# Quant version
# ---------------------------------------------------------------------------

class TestQuantVersion:
    def test_missing_quant_version_warns(self, tmp_path):
        parsed = _make_parsed(tmp_path, {"general.quantization_version": ""})
        # Override: remove the key entirely
        parsed.metadata.pop("general.quantization_version", None)
        verdict, findings = LoadGuard(parsed).run()
        qv = [f for f in findings if f["check"] == "quant_version"]
        assert qv
        assert qv[0]["verdict"] == "WARN"

    def test_present_quant_version_no_finding(self, tmp_path):
        parsed = _make_parsed(tmp_path)  # fixture sets quantization_version="2"
        _, findings = LoadGuard(parsed).run()
        qv = [f for f in findings if f["check"] == "quant_version"]
        assert not qv


# ---------------------------------------------------------------------------
# Verdict escalation
# ---------------------------------------------------------------------------

class TestVerdictEscalation:
    def test_clean_model_is_clean(self, tmp_path):
        """A well-formed model with no template and known quant version → CLEAN or WARN.

        WARN is acceptable because the resource check may warn if pynvml is
        absent or VRAM estimate is unavailable.  DETONATE must not fire.
        """
        parsed = _make_parsed(tmp_path)
        verdict, _ = LoadGuard(parsed).run()
        assert verdict != "DETONATE"

    def test_detonate_overrides_warn(self, tmp_path):
        """DETONATE from zero-tensors overrides WARN from missing quant_version."""
        parsed = _make_parsed(tmp_path)
        parsed.tensors_info = {}
        parsed.metadata.pop("general.quantization_version", None)
        verdict, _ = LoadGuard(parsed).run()
        assert verdict == "DETONATE"


# ---------------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------------

class TestSHA256:
    def test_sha256_computed(self, tmp_path):
        parsed = _make_parsed(tmp_path)
        lg = LoadGuard(parsed)
        lg.run()
        # SHA-256 is now lazy — must call compute_sha256() explicitly
        sha = lg.compute_sha256()
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_sha256_deterministic(self, tmp_path):
        parsed = _make_parsed(tmp_path)
        lg1 = LoadGuard(parsed)
        lg2 = LoadGuard(parsed)
        lg1.run()
        lg2.run()
        assert lg1.compute_sha256() == lg2.compute_sha256()
