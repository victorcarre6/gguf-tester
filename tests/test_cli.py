"""Unit tests for ggufscan.cli helpers."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

from ggufscan.cli import _resolve_judge_cfg, _resolve_out_path
from ggufscan.cli import main


# ---------------------------------------------------------------------------
# _resolve_judge_cfg
# ---------------------------------------------------------------------------

def _args(**kw):
    defaults = dict(
        judge_model=None, judge_n_ctx=None, judge_max_tokens=None,
        judge_temperature=None, judge_tensor_split=None,
    )
    defaults.update(kw)
    return Namespace(**defaults)


def test_resolve_judge_cfg_no_model_returns_none():
    path, params, source = _resolve_judge_cfg(_args(), {})
    assert path is None
    assert source is None
    assert params == {}


def test_resolve_judge_cfg_flag_takes_precedence(tmp_path):
    model = tmp_path / "judge.gguf"
    model.write_bytes(b"fake")
    cfg = {"judge": {"model": str(tmp_path / "other.gguf"), "n_ctx": 2048}}
    path, params, source = _resolve_judge_cfg(_args(judge_model=str(model)), cfg)
    assert source == "flag"
    assert path == model.expanduser()


def test_resolve_judge_cfg_from_config(tmp_path):
    model = tmp_path / "judge.gguf"
    model.write_bytes(b"fake")
    cfg = {"judge": {"model": str(model), "n_ctx": 8192, "max_tokens": 256}}
    path, params, source = _resolve_judge_cfg(_args(), cfg)
    assert source == "config"
    assert params["n_ctx"] == 8192
    assert params["max_tokens"] == 256


def test_resolve_judge_cfg_flag_overrides_config_params(tmp_path):
    model = tmp_path / "judge.gguf"
    model.write_bytes(b"fake")
    cfg = {"judge": {"model": str(model), "n_ctx": 2048, "temperature": 0.5}}
    path, params, source = _resolve_judge_cfg(
        _args(judge_model=str(model), judge_n_ctx=16384, judge_temperature=0.0), cfg
    )
    assert params["n_ctx"] == 16384       # flag wins
    assert params["temperature"] == 0.0   # flag wins


def test_resolve_judge_cfg_defaults_applied(tmp_path):
    model = tmp_path / "judge.gguf"
    model.write_bytes(b"fake")
    path, params, source = _resolve_judge_cfg(_args(judge_model=str(model)), {})
    assert params["n_ctx"] == 4096        # default
    assert params["max_tokens"] == 512    # default
    assert params["temperature"] == 0.0  # default


# ---------------------------------------------------------------------------
# _resolve_out_path
# ---------------------------------------------------------------------------

def test_resolve_out_path_file(tmp_path):
    result = _resolve_out_path(str(tmp_path / "report.md"), "default.md")
    assert result == tmp_path / "report.md"


def test_resolve_out_path_directory(tmp_path):
    result = _resolve_out_path(str(tmp_path) + "/", "default.md")
    assert result == tmp_path / "default.md"


def test_resolve_out_path_existing_dir(tmp_path):
    result = _resolve_out_path(str(tmp_path), "default.md")
    assert result == tmp_path / "default.md"


def test_static_cli_writes_phase_one_manifest(tiny_gguf, tmp_path):
    markdown = tmp_path / "report.md"
    json_report = tmp_path / "report.json"
    missing_config = tmp_path / "missing.yml"

    exit_code = main([
        str(tiny_gguf),
        "--static-only",
        "--output", str(markdown),
        "--json", str(json_report),
        "--config", str(missing_config),
    ])

    manifests = list(tmp_path.glob("tiny_static_*_manifest.json"))
    assert exit_code == 0
    assert markdown.is_file()
    assert json_report.is_file()
    assert len(manifests) == 1

    import json

    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["run"]["model"]["architecture"] == "qwen"
    assert manifest["run"]["suites"][0]["name"] == "security"
    assert ["static_verdict", "WARN"] in manifest["results"][0]["metrics"]


def test_cli_runs_personal_suite_and_adds_it_to_manifest(tiny_gguf, tmp_path):
    dataset = tmp_path / "personal.yml"
    dataset.write_text(
        """\
schema_version: 1
name: personal-smoke
version: "1"
tasks:
  - id: answer
    type: generation
    prompt: Answer 42.
    grader: {type: exact, accepted: ["42"]}
""",
        encoding="utf-8",
    )
    fake = type("FakeLlama", (), {
        "__init__": lambda self, **kwargs: None,
        "create_chat_completion": lambda self, messages, **kwargs: {
            "choices": [{"message": {"content": "42"}}]
        },
    })

    with patch("llama_cpp.Llama", fake, create=True):
        exit_code = main([
            str(tiny_gguf),
            "--tests", "",
            "--personal-dataset", str(dataset),
            "--output", str(tmp_path) + "/",
            "--json", str(tmp_path) + "/",
            "--personal-output", str(tmp_path) + "/",
            "--config", str(tmp_path / "missing.yml"),
        ])

    manifests = list(tmp_path.glob("tiny_scan_*_manifest.json"))
    personal_outputs = list(tmp_path.glob("tiny_scan_*_personal.json"))
    assert exit_code == 0
    assert len(manifests) == 1
    assert len(personal_outputs) == 1

    import json

    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert [suite["name"] for suite in manifest["run"]["suites"]] == [
        "security",
        "personal-smoke",
    ]
    assert manifest["results"][1]["score"] == 1.0
