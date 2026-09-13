"""HuggingFace official checksum comparison (LLM05 — Supply Chain).

Queries the HF Hub API to retrieve the server-side SHA-256 for a given
model file, then compares it to the locally computed hash.

Requires network access. Gracefully degrades when offline or when the
model is not on HuggingFace (community re-quants, local fine-tunes).

Usage:
    from ggufscan.hf_checksum import compare_to_hf
    result = compare_to_hf(local_sha256, repo_id="Qwen/Qwen3-14B-GGUF",
                           filename="qwen3-14b-q5_k_m.gguf")
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from ggufscan import __version__

log = logging.getLogger(__name__)

_HF_API_BASE = "https://huggingface.co/api"
_TIMEOUT_S = 10


@dataclass
class HFChecksumResult:
    local_sha256: str
    remote_sha256: str | None
    match: bool | None         # None when comparison was not possible
    source: str                # "hf_api" | "offline" | "not_found" | "error"
    detail: str


def query_hf_sha256(repo_id: str, filename: str) -> str | None:
    """Return the HF-reported SHA-256 for `repo_id/filename`, or None on failure."""
    url = f"{_HF_API_BASE}/models/{repo_id}/tree/main"
    user_agent = f"ggufscan/{__version__}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            import json
            tree = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        log.debug("HF API unreachable (%s): %s", url, exc)
        return None
    except Exception as exc:
        log.debug("HF API error: %s", exc)
        return None

    fn_lower = filename.lower()
    for entry in tree:
        if entry.get("path", "").lower() == fn_lower:
            lfs = entry.get("lfs") or {}
            sha = lfs.get("sha256") or entry.get("sha256") or entry.get("oid")
            return sha
    return None


def compare_to_hf(
    local_sha256: str,
    repo_id: str | None,
    filename: str | None,
) -> HFChecksumResult:
    """Compare local SHA-256 to HF's reported hash.

    Returns an HFChecksumResult with match=None when the comparison is
    not possible (no repo_id, offline, file not found on HF).
    """
    if not repo_id or not filename:
        return HFChecksumResult(
            local_sha256=local_sha256,
            remote_sha256=None,
            match=None,
            source="not_configured",
            detail="--hf-repo not provided; skipping supply-chain checksum check",
        )

    remote = query_hf_sha256(repo_id, filename)
    if remote is None:
        return HFChecksumResult(
            local_sha256=local_sha256,
            remote_sha256=None,
            match=None,
            source="offline",
            detail=f"HF API unreachable or file {filename!r} not found in {repo_id!r}",
        )

    match = local_sha256.lower() == remote.lower()
    return HFChecksumResult(
        local_sha256=local_sha256,
        remote_sha256=remote,
        match=match,
        source="hf_api",
        detail=(
            "SHA-256 matches HF registry ✓" if match
            else (
                f"SHA-256 MISMATCH — local={local_sha256[:16]}… "
                f"remote={remote[:16]}… — file may have been tampered with"
            )
        ),
    )
