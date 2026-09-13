"""Static structural analysis (informational baseline).

The OWASP LLM Top 10 pattern-matching checks from v1.0 have been removed —
they produced systematic false positives (ELN §1, ROADMAP §1bis) and
duplicated signals now produced correctly by the dynamic test battery and
`load_guard.py`. What remains: structural facts parsed from the GGUF binary.

Load-guard verdict and findings are injected by `cli.py` after `LoadGuard.run()`
and stored on the returned `StaticReport` before rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ggufscan.parser import ParsedGGUF


@dataclass
class StaticReport:
    filename: str
    file_size_mb: float
    metadata: dict[str, Any]
    tensors: dict[str, dict[str, Any]]
    analysis: dict[str, Any]
    # Populated by cli.py after LoadGuard.run()
    load_guard_verdict: str = "SKIPPED"
    load_guard_findings: list[dict[str, Any]] = field(default_factory=list)
    file_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_size_mb": self.file_size_mb,
            "metadata": self.metadata,
            "tensors": self.tensors,
            "analysis": self.analysis,
            "load_guard_verdict": self.load_guard_verdict,
            "load_guard_findings": self.load_guard_findings,
            "file_sha256": self.file_sha256,
        }


def _tensor_role(name: str) -> str:
    n = name.lower()
    if "attn_q" in n or ".q_proj" in n:
        return "attention_q"
    if "attn_k" in n or ".k_proj" in n:
        return "attention_k"
    if "attn_v" in n or ".v_proj" in n:
        return "attention_v"
    if "attn_output" in n or "o_proj" in n or ".attn.out" in n:
        return "attention_output"
    if "ffn_gate" in n or ".mlp.gate" in n:
        return "mlp_gate"
    if "ffn_up" in n or ".mlp.up" in n:
        return "mlp_up"
    if "ffn_down" in n or ".mlp.down" in n:
        return "mlp_down"
    if "embed" in n:
        return "embedding"
    if "output.weight" in n or "lm_head" in n:
        return "output_head"
    if "norm" in n:
        return "normalization"
    return "other"


class StaticScanner:
    """Structural facts extractor. OWASP pattern-matching removed (see ELN §1)."""

    def __init__(self, parsed: ParsedGGUF):
        self.parsed = parsed

    def scan(self) -> StaticReport:
        return StaticReport(
            filename=self.parsed.filename,
            file_size_mb=self.parsed.file_size_mb,
            metadata=self.parsed.metadata,
            tensors=self.parsed.tensors_info,
            analysis=self._analyze_structure(),
        )

    def _analyze_structure(self) -> dict[str, Any]:
        md = self.parsed.metadata
        arch = md.get("general.architecture", "unknown")
        tensors = self.parsed.tensors_info

        ctx_len = md.get(f"{arch}.context_length") or md.get("llama.context_length")
        n_layers = md.get(f"{arch}.block_count")
        n_heads = md.get(f"{arch}.attention.head_count")
        n_kv_heads = md.get(f"{arch}.attention.head_count_kv") or n_heads
        emb_length = md.get(f"{arch}.embedding_length")
        head_dim = md.get(f"{arch}.attention.key_length")
        if head_dim is None and n_heads and emb_length:
            head_dim = emb_length // n_heads

        return {
            "structure": {
                "architecture": arch,
                "model_name": md.get("general.name", "Unknown"),
                "author": md.get("general.author", ""),
                "license": md.get("general.license", ""),
                "file_type": md.get("general.file_type", "Unknown"),
                "tensor_count": len(tensors),
                "total_size_mb": sum(t["size_mb"] for t in tensors.values()),
                "quantization_types": sorted(set(t["type_name"] for t in tensors.values())),
                "quantization_version": md.get("general.quantization_version"),
                "context_length": ctx_len,
                "mean_bits_per_weight": self._mean_bits_per_weight(tensors),
                "tensor_roles": self._classify_tensor_roles(tensors),
                "kv_cache_gb": self._estimate_kv_cache(ctx_len, n_layers, n_kv_heads, head_dim),
            }
        }

    def _mean_bits_per_weight(self, tensors: dict) -> float | None:
        weight_tensors = {n: t for n, t in tensors.items() if "weight" in n}
        if not weight_tensors:
            weight_tensors = tensors
        if not weight_tensors:
            return None
        total_elem = sum(t.get("element_count", 0) for t in weight_tensors.values())
        if total_elem == 0:
            return None
        weighted_bits = sum(
            t.get("element_count", 0) * t.get("bits_per_element", 16.0)
            for t in weight_tensors.values()
        )
        return round(weighted_bits / total_elem, 2)

    def _classify_tensor_roles(self, tensors: dict) -> dict[str, int]:
        roles: dict[str, int] = {}
        for name in tensors:
            role = _tensor_role(name)
            roles[role] = roles.get(role, 0) + 1
        return dict(sorted(roles.items()))

    def _estimate_kv_cache(
        self,
        ctx_len: int | None,
        n_layers: int | None,
        n_kv_heads: int | None,
        head_dim: int | None,
    ) -> float | None:
        if not all([ctx_len, n_layers, n_kv_heads, head_dim]):
            return None
        # 2 (K+V) × layers × kv_heads × head_dim × context × F16 bytes
        gb = 2 * n_layers * n_kv_heads * head_dim * ctx_len * 2 / 1e9
        return round(gb, 3)
