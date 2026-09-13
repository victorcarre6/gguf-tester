"""NemoClaw deployment artifact generator.

Given a completed scan JSON dict:
- All dynamic tests passed  → write {model_name}_blueprint.yaml
- Any dynamic test failed   → write {model_name}_BLOCKED.md
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any


def derive_ollama_model(general_name: str, filename: str) -> str:
    """Derive Ollama model tag from general.name and filename stem.

    "Qwen3-14B" + "Qwen3-14B-Q5_K_M.gguf" -> "qwen3:14b-q5_k_m"
    """
    stem = Path(filename).stem
    quant_match = re.search(r'[-_]((?:IQ|Q)\d[_A-Z\d]*|F\d+)$', stem, re.IGNORECASE)
    quant_suffix = quant_match.group(1).lower() if quant_match else ""

    name = general_name.lower().replace(" ", "-")
    # "qwen3-14b" → "qwen3:14b"
    name = re.sub(r'-(\d+\.?\d*[bm])', r':\1', name, count=1)

    return f"{name}-{quant_suffix}" if quant_suffix else name


def _context_length(metadata: dict, architecture: str) -> int | None:
    val = metadata.get(f"{architecture}.context_length")
    if val is not None:
        return int(val)
    for k, v in metadata.items():
        if k.endswith(".context_length"):
            return int(v)
    return None


def generate(scan_data: dict, output_dir: Path) -> tuple[str, Path]:
    """Generate blueprint.yaml or BLOCKED.md from a scan JSON dict.

    Returns ("blueprint", path) or ("blocked", path).
    """
    static = scan_data["static"]
    dynamic = scan_data.get("dynamic", [])

    metadata = static.get("metadata", {})
    general_name = metadata.get("general.name", Path(static["filename"]).stem)
    safe_name = general_name.replace(" ", "-")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_passed = all(t["passed"] for t in dynamic) if dynamic else True

    if all_passed:
        return _write_blueprint(static, dynamic, metadata, general_name, safe_name, output_dir)
    return _write_blocked(static, dynamic, general_name, safe_name, output_dir)


def _write_blueprint(
    static: dict,
    dynamic: list[dict],
    metadata: dict,
    general_name: str,
    safe_name: str,
    output_dir: Path,
) -> tuple[str, Path]:
    import yaml

    architecture = metadata.get("general.architecture", "unknown")
    ctx = _context_length(metadata, architecture)
    ollama_model = derive_ollama_model(general_name, static["filename"])

    n_passed = sum(1 for t in dynamic if t["passed"])
    total = len(dynamic)

    blueprint: dict[str, Any] = {
        "agent": {
            "name": f"{general_name}-assistant",
            "model_router": {
                "endpoint": "http://nyx:11434",
                "default_model": ollama_model,
            },
        },
        "sandbox": {
            "landlock": True,
            "seccomp_profile": "strict",
            "network_policy": "baseline",
            "credential_mount": "vault",
        },
        "gateway": {
            "port": 8765,
            "channels": ["webhook"],
        },
        "memory": {
            "path": "/data/openclaw/memory",
        },
        "scan_metadata": {
            "scanner": "ggufscan",
            "scan_date": date.today().isoformat(),
            "model_file": static["filename"],
            "size_mb": static.get("file_size_mb"),
            "architecture": architecture,
            "context_length": ctx,
            "tests_passed": f"{n_passed}/{total}",
            "verdict": "APPROVED",
        },
    }

    out = output_dir / f"{safe_name}_blueprint.yaml"
    out.write_text(yaml.dump(blueprint, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return ("blueprint", out)


def _write_blocked(
    static: dict,
    dynamic: list[dict],
    general_name: str,
    safe_name: str,
    output_dir: Path,
) -> tuple[str, Path]:
    failing = [t for t in dynamic if not t["passed"]]
    today = date.today().isoformat()

    lines = [
        f"# NemoClaw Deployment BLOCKED — {general_name}",
        "",
        f"**Scan date:** {today}  ",
        f"**Verdict:** BLOCKED — {len(failing)}/{len(dynamic)} test(s) failed",
        "",
        "## Failing Tests",
        "",
        "| Test | Score | Threshold | Failed Samples |",
        "|------|-------|-----------|----------------|",
    ]
    for t in failing:
        n_failed = len(t.get("failed_samples", []))
        lines.append(
            f"| `{t['name']}` | {t['score']:.1%} | {t['threshold']:.1%} | {n_failed} |"
        )

    lines.append("")

    for t in failing:
        failed_samples = t.get("failed_samples", [])
        if not failed_samples:
            continue
        lines += [f"## {t['name']} — Failed Examples", ""]
        for s in failed_samples[:3]:
            prompt = str(s.get("prompt", "")).replace("\n", " ")[:200]
            response = str(s.get("response", "")).replace("\n", " ")[:200]
            lines += [
                f"**Prompt:** {prompt}",
                "",
                f"**Response:** {response}",
                "",
                "---",
                "",
            ]

    out = output_dir / f"{safe_name}_BLOCKED.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return ("blocked", out)
