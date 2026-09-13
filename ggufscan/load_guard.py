"""Pre-load attack-surface gate.

Runs entirely on the parsed GGUF metadata — no Llama() instantiation.
Must execute before any ModelHandle is entered.

Verdict escalation: CLEAN → WARN → DETONATE (never downgrades).
- CLEAN    : safe to load
- WARN     : load with caution; findings surfaced in report
- DETONATE : block load; CLI exits non-zero, dynamic phase skipped
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ggufscan.parser import GGML_TYPE_INFO

if TYPE_CHECKING:
    from ggufscan.parser import ParsedGGUF

log = logging.getLogger(__name__)

# Jinja2 / minja patterns that indicate active code-exec risk in chat_template.
# llama.cpp executes the embedded template via its minja engine on every
# inference call — these patterns are actual RCE vectors.
_SSTI_DANGEROUS = [
    "__class__", "__mro__", "__globals__", "__builtins__",
    "os.system", "os.popen", "os.execv", "subprocess",
    "eval(", "exec(", "compile(",
    "import os", "import sys", "import subprocess",
    "__import__",
]
# Jinja2 delimiters: present in most valid templates, but warrant manual review
# if they appear alongside unusual patterns.
_SSTI_SUSPICIOUS = ["{{", "{%", "{#"]

# Threshold above which metadata kv count is suspicious.
_MAX_REASONABLE_KV = 5_000
# Single string value longer than this is an alloc-bomb candidate (bytes).
_MAX_STRING_BYTES = 10_000_000  # 10 MB

# Derived from GGML_TYPE_INFO so new quant types added to parser.py are
# automatically picked up here without a second manual edit.
_BYTES_PER_ELEM: dict[str, float] = {
    v["name"]: v["bits"] / 8 for v in GGML_TYPE_INFO.values()
}
_DEFAULT_BYTES_PER_ELEM = 2.0  # F16 fallback for unknown types
_VRAM_OVERHEAD = 1.25  # 25% overhead for KV cache + runtime


@dataclass
class Finding:
    check: str
    severity: str   # "critical" | "high" | "medium" | "low" | "info"
    detail: str
    verdict: str    # "DETONATE" | "WARN" | "CLEAN"

    def as_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "detail": self.detail,
            "verdict": self.verdict,
        }


_VERDICT_ORDER = {"CLEAN": 0, "WARN": 1, "DETONATE": 2}


class LoadGuard:
    """Static pre-load security gate. Call run() before ModelHandle.__enter__."""

    def __init__(self, parsed: "ParsedGGUF"):
        self.parsed = parsed
        self.findings: list[Finding] = []
        self._verdict = "CLEAN"
        self.file_sha256 = ""

    def _escalate(self, verdict: str) -> None:
        if _VERDICT_ORDER.get(verdict, 0) > _VERDICT_ORDER.get(self._verdict, 0):
            self._verdict = verdict

    def _add(self, check: str, severity: str, detail: str, verdict: str) -> None:
        self.findings.append(Finding(check=check, severity=severity, detail=detail, verdict=verdict))
        self._escalate(verdict)

    # ------------------------------------------------------------------
    # Check: chat_template SSTI / RCE
    # ------------------------------------------------------------------
    def _check_ssti(self) -> None:
        template = self.parsed.metadata.get("tokenizer.chat_template", "")
        if not isinstance(template, str) or not template.strip():
            return

        has_jinja = any(d in template for d in _SSTI_SUSPICIOUS)
        if not has_jinja:
            return

        for pattern in _SSTI_DANGEROUS:
            if pattern in template:
                self._add(
                    check="ssti",
                    severity="critical",
                    detail=(
                        f"chat_template contains dangerous pattern {pattern!r} — "
                        "llama.cpp executes this template via minja on every call (RCE risk)"
                    ),
                    verdict="DETONATE",
                )
                log.warning("[load_guard] SSTI/RCE pattern in chat_template: %r", pattern)
                return  # one finding is enough for DETONATE

        # Template syntax present but no known-dangerous patterns → manual review
        self._add(
            check="ssti",
            severity="medium",
            detail=(
                "chat_template uses Jinja2/minja syntax — no known-dangerous patterns found, "
                "but template execution is a potential attack surface; inspect before loading"
            ),
            verdict="WARN",
        )

    # ------------------------------------------------------------------
    # Check: structural integrity (zero tensors, absurd metadata count)
    # ------------------------------------------------------------------
    def _check_integrity(self) -> None:
        n_tensors = len(self.parsed.tensors_info)
        if n_tensors == 0:
            self._add(
                check="integrity",
                severity="critical",
                detail="zero tensors detected — file is likely corrupted or truncated",
                verdict="DETONATE",
            )

        n_kv = len(self.parsed.metadata)
        if n_kv > _MAX_REASONABLE_KV:
            self._add(
                check="integrity",
                severity="high",
                detail=f"metadata_kv_count={n_kv:,} unusually large (threshold {_MAX_REASONABLE_KV:,})",
                verdict="WARN",
            )

        # Check for alloc-bomb string values.
        for key, value in self.parsed.metadata.items():
            if isinstance(value, str) and len(value.encode("utf-8")) > _MAX_STRING_BYTES:
                self._add(
                    check="integrity",
                    severity="high",
                    detail=f"metadata key {key!r} contains a string of {len(value):,} chars — potential alloc bomb",
                    verdict="WARN",
                )

    # ------------------------------------------------------------------
    # Check: VRAM / resource estimate
    # ------------------------------------------------------------------
    def _check_resource(self) -> None:
        tensors = self.parsed.tensors_info
        if not tensors:
            return

        # Weight bytes-per-elem by element count so large attention matrices
        # dominate over many tiny normalization tensors (previously used tensor
        # count as weight, which underestimated VRAM for high-F32-ratio models).
        total_elements = sum(t.get("element_count", 0) for t in tensors.values())
        weighted_bytes = sum(
            t.get("element_count", 0)
            * _BYTES_PER_ELEM.get(t.get("type_name", ""), _DEFAULT_BYTES_PER_ELEM)
            for t in tensors.values()
        )
        avg_bytes = weighted_bytes / max(total_elements, 1)
        estimated_gb = total_elements * avg_bytes * _VRAM_OVERHEAD / 1e9

        free_gb: float | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            free_gb = sum(
                pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).free
                for i in range(count)
            ) / 1e9
            pynvml.nvmlShutdown()
        except Exception:
            pass

        detail = f"estimated VRAM required: {estimated_gb:.1f} GB"
        if free_gb is not None:
            detail += f" / available: {free_gb:.1f} GB across all GPUs"
            if estimated_gb > free_gb:
                self._add(
                    check="resource",
                    severity="high",
                    detail=detail + " — model likely to OOM on this system",
                    verdict="WARN",
                )
                return

        self._add(check="resource", severity="info", detail=detail, verdict="CLEAN")

    # ------------------------------------------------------------------
    # Check: missing quantization_version (provenance)
    # ------------------------------------------------------------------
    def _check_quant_version(self) -> None:
        if not self.parsed.metadata.get("general.quantization_version"):
            self._add(
                check="quant_version",
                severity="low",
                detail=(
                    "general.quantization_version not set — "
                    "provenance of quantization is unknown (community re-quant?)"
                ),
                verdict="WARN",
            )

    # ------------------------------------------------------------------
    # Check: quantization consistency (tensor entropy proxy)
    #
    # Real weight-level entropy requires reading GBs of float data; that is
    # out of scope for a pre-load gate (parser drops offsets, no weights read).
    # As a cheap proxy we check quantization consistency across layers:
    #   - A model with wildly mixed quant types across same-layer tensors
    #     (e.g. some layers at Q4_K, others at F32) may have been surgically
    #     modified rather than uniformly re-quantized.
    #   - An abnormally high proportion of F32 tensors in an otherwise
    #     quantized model often indicates unintended weight un-quantization.
    # ------------------------------------------------------------------
    def _check_quant_consistency(self) -> None:
        tensors = self.parsed.tensors_info
        if not tensors:
            return

        type_counts: dict[str, int] = {}
        for t in tensors.values():
            type_counts[t.get("type_name", "Unknown")] = (
                type_counts.get(t.get("type_name", "Unknown"), 0) + 1
            )
        n_total = len(tensors)
        if n_total == 0:
            return

        # Flag if F32 tensors dominate a supposedly-quantized model.
        # Embedding / output layers legitimately stay F32, but if >30 % of
        # all tensors are F32 while the rest are quantized → suspicious.
        f32_count = type_counts.get("F32", 0)
        non_f32_quant = {
            k: v for k, v in type_counts.items()
            if k not in ("F32", "F16", "BF16", "Unknown")
        }
        if non_f32_quant and f32_count / n_total > 0.30:
            self._add(
                check="quant_consistency",
                severity="medium",
                detail=(
                    f"{f32_count}/{n_total} tensors ({f32_count/n_total:.0%}) are F32 "
                    f"alongside quantized types {list(non_f32_quant)[:3]} — "
                    "unusually high F32 ratio in a quantized model; may indicate "
                    "selective de-quantization"
                ),
                verdict="WARN",
            )
            return

        # Flag extreme type fragmentation: >4 distinct quant types across
        # weight tensors (embeddings excluded) in a single model is unusual.
        weight_types = {
            t.get("type_name")
            for name, t in tensors.items()
            if "embed" not in name.lower() and "output" not in name.lower()
        }
        if len(weight_types) > 4:
            self._add(
                check="quant_consistency",
                severity="low",
                detail=(
                    f"{len(weight_types)} distinct quantization types across weight "
                    f"tensors: {sorted(weight_types)} — unusual; inspect model provenance"
                ),
                verdict="WARN",
            )

    # ------------------------------------------------------------------
    # Check: per-layer quantization transitions (F32 island detection)
    #
    # Groups tensors by block index (blk.N.*) and checks whether any layer
    # consists entirely of high-precision (F32/F16/BF16) types while the
    # majority of layers are quantized.  A "F32 island" in an otherwise
    # quantized model is a known weight-injection pattern: an attacker
    # re-quantizes only targeted layers to F32 to insert modified values
    # while keeping surrounding layers quantized for plausibility.
    # ------------------------------------------------------------------
    def _check_quant_layers(self) -> None:
        tensors = self.parsed.tensors_info
        if not tensors:
            return

        high_precision = {"F32", "F16", "BF16"}

        layer_types: dict[int, set[str]] = {}
        for name, t in tensors.items():
            m = re.search(r'blk\.(\d+)\.', name)
            if m:
                idx = int(m.group(1))
                layer_types.setdefault(idx, set()).add(t.get("type_name", "Unknown"))

        if len(layer_types) < 3:
            return

        all_types = {t.get("type_name") for t in tensors.values()}
        quantized_global = all_types - high_precision - {"Unknown"}
        if not quantized_global:
            return  # pure high-precision model — no anomaly

        n_total = len(layer_types)
        n_quantized_layers = sum(
            1 for types in layer_types.values()
            if types - high_precision  # at least one quantized tensor in layer
        )

        # Only flag if majority of layers ARE quantized (otherwise could be legitimate)
        if n_quantized_layers <= n_total * 0.5:
            return

        f32_islands = sorted(
            idx for idx, types in layer_types.items()
            if not (types - high_precision)  # all tensors in layer are high-precision
        )
        if f32_islands:
            self._add(
                check="quant_layers",
                severity="medium",
                detail=(
                    f"{len(f32_islands)} layer(s) contain only F32/F16/BF16 tensors "
                    f"(blk indices {f32_islands[:5]}{'…' if len(f32_islands) > 5 else ''}) "
                    f"while {n_quantized_layers}/{n_total} layers are quantized — "
                    "selective de-quantization pattern; possible weight injection site"
                ),
                verdict="WARN",
            )

    # ------------------------------------------------------------------
    # File hash (SHA-256)
    # ------------------------------------------------------------------
    def _compute_sha256(self) -> str:
        h = hashlib.sha256()
        try:
            with open(self.parsed.file_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError as exc:
            log.warning("[load_guard] could not hash file: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> tuple[str, list[dict]]:
        """Run all security checks. Returns (verdict, findings_as_dicts).

        SHA-256 is NOT computed here — call compute_sha256() explicitly when
        needed (e.g. only when --hf-repo is passed) to avoid reading GB of data
        on every scan.
        """
        self._check_ssti()
        self._check_integrity()
        self._check_resource()
        self._check_quant_version()
        self._check_quant_consistency()
        self._check_quant_layers()
        return self._verdict, [f.as_dict() for f in self.findings]

    def compute_sha256(self) -> str:
        """Compute and cache SHA-256 of the model file. Call explicitly when needed."""
        if not self.file_sha256:
            self.file_sha256 = self._compute_sha256()
        return self.file_sha256
