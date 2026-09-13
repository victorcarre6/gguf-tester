"""Command-line entrypoint for ggufscan.

Orchestration strategy
----------------------
llama.cpp cannot do chat completion and embeddings on the same Llama instance.
We therefore run the scan in two passes when any selected test needs embeddings:

  Pass 1 — chat handle: every test calls `collect()` to produce raw samples.
  Pass 2 — embedding handle (if needed): tests that need vectors call `score()`
           with the embedding handle; others score in pass 1.

The model is loaded sequentially (chat unloaded before embedding), so peak VRAM
stays at one model instance.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ggufscan import tui
from ggufscan.artifacts import ensure_output_parent, resolve_output_path
from ggufscan.domain import BenchmarkResult, SuiteSpec
from ggufscan.judge_runner import resolve_judge_config, run_judge
from ggufscan.parser import parse
from ggufscan.report import to_evaluation_records, to_json, to_markdown
from ggufscan.runners import SecurityRunner
from ggufscan.suites.security import security_test_registry
from ggufscan.tests.base import TestResult

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "ggufscan.yml"

log = logging.getLogger("ggufscan")
TEST_REGISTRY = security_test_registry()
_resolve_judge_cfg = resolve_judge_config
_resolve_out_path = resolve_output_path


def _load_config(path: Path) -> dict:
    """Load a YAML mapping, or return an empty config when absent."""

    if not path.is_file():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")

    return data


def _parse_split(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x) for x in s.split(",")]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ggufscan",
        description="GGUF static + dynamic security scanner.",
    )
    p.add_argument("model_pos", nargs="?", metavar="MODEL",
                   help="Path to .gguf file")
    p.add_argument(
        "--tests",
        default="jailbreak,harmful_bias,backdoor,extraction,agency,determinism,output_safety,factuality",
        help="Comma-separated test names. Choices: " + ", ".join(TEST_REGISTRY),
    )
    p.add_argument("--static-only", action="store_true", help="Skip dynamic inference tests")
    p.add_argument("--quick", action="store_true", help="Subset of prompts per test (smoke run)")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--tensor-split", default=None,
                   help="Comma-separated floats (e.g. '0.5,0.5')")
    p.add_argument("--chat-format", default=None,
                   help="Override chat_format (default: auto-detect)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=-1,
                   help="Default: 256, auto-bumped to 1024 for thinking models")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--personal-dataset", default=None,
                   help="Versioned YAML dataset for the personal benchmark suite.")
    p.add_argument("--personal-tags", default=None,
                   help="Comma-separated personal-task tags; any matching tag is selected.")
    p.add_argument("--personal-limit", type=int, default=None,
                   help="Run only the first N selected personal tasks.")
    p.add_argument("--personal-repetitions", type=int, default=1,
                   help="Repeat each selected personal task N times (default: 1).")
    p.add_argument("--personal-output", default=None,
                   help="Detailed personal benchmark JSON output path (file or dir).")
    p.add_argument("--output", default=None, help="Markdown report output path (file or dir)")
    p.add_argument("--json", dest="json_out", default=None,
                   help="Full JSON report output path (file or dir)")
    p.add_argument("--ragas", dest="ragas_out", default=None,
                   help="Deprecated alias of --records.")
    p.add_argument("--records", dest="ragas_out",
                   help="Prompt/response/reference records (JSON file or directory).")

    p.add_argument("--config", default=str(_DEFAULT_CONFIG_PATH),
                   help="Path to YAML config (default: config/ggufscan.yml relative to package). "
                        "`judge.model` enables the local judge.")
    p.add_argument("--judge-model", default=None,
                   help="Path to judge .gguf (overrides config `judge.model`).")
    p.add_argument("--judge-n-ctx", type=int, default=None)
    p.add_argument("--judge-tensor-split", default=None)
    p.add_argument("--judge-max-tokens", type=int, default=None)
    p.add_argument("--judge-temperature", type=float, default=None)
    p.add_argument("--ragas-results", dest="ragas_results_out", default=None,
                   help="Deprecated alias of --judge-results.")
    p.add_argument("--judge-results", dest="ragas_results_out",
                   help="Enriched LLM-as-judge JSON output (per-record metrics).")
    p.add_argument("--skip-ragas", dest="skip_ragas_eval", action="store_true",
                   help="Deprecated alias of --skip-judge.")
    p.add_argument("--skip-judge", dest="skip_ragas_eval", action="store_true",
                   help="Skip the in-pipeline LLM-as-judge stage.")

    p.add_argument("--hf-repo", default=None,
                   help="HuggingFace repo ID (e.g. 'Qwen/Qwen3-14B-GGUF') to verify "
                        "the file's SHA-256 against the official registry (LLM05).")

    p.add_argument("--nemoclaw", action="store_true",
                   help="Generate NemoClaw blueprint.yaml (all pass) or BLOCKED.md (any fail)")
    p.add_argument("--from-json", dest="from_json", default=None, metavar="PATH",
                   help="Skip scan, read results from existing JSON file (use with --nemoclaw)")

    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _run_dynamic(parsed, args, requested: list[str]) -> list[TestResult]:
    runner = SecurityRunner()

    return runner.run_dynamic(
        parsed,
        requested,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        tensor_split=_parse_split(args.tensor_split),
        chat_format=args.chat_format,
        seed=args.seed,
        quick=args.quick,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        on_result=tui.print_test_result,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.from_json:
        from ggufscan import nemoclaw_blueprint
        json_path = Path(args.from_json).expanduser()
        if not json_path.is_file():
            tui.print_error(f"scan JSON not found: {json_path}")
            return 2
        scan_data = json.loads(json_path.read_text(encoding="utf-8"))
        if args.nemoclaw:
            kind, out = nemoclaw_blueprint.generate(scan_data, Path.cwd())
            tui.print_nemoclaw(kind, out)
        return 0

    if not args.model_pos:
        tui.print_error("usage: ggufscan MODEL [options]")
        return 2

    if args.static_only and args.personal_dataset:
        tui.print_error("--personal-dataset requires dynamic inference; remove --static-only")
        return 2

    model_path = Path(args.model_pos).expanduser()
    if not model_path.is_file():
        tui.print_error(f"model not found: {model_path}")
        return 2

    cfg = _load_config(Path(args.config).expanduser())

    log.info("Parsing %s", model_path)
    parsed = parse(str(model_path))

    security_runner = SecurityRunner()
    static_execution = security_runner.run_static(
        parsed,
        compute_sha256=bool(args.hf_repo),
    )
    static_report = static_execution.report
    tui.print_static_analysis(static_report, model_path=model_path)
    lg_verdict = static_execution.verdict
    lg_findings = list(static_execution.findings)
    log.info("Static scan complete (load_guard=%s)", lg_verdict)
    tui.print_load_guard(lg_verdict, lg_findings)

    if args.hf_repo and static_report.file_sha256:
        from ggufscan.hf_checksum import compare_to_hf
        hf_result = compare_to_hf(static_report.file_sha256, args.hf_repo, model_path.name)
        tui.print_hf_checksum(hf_result.detail, hf_result.match)
        static_report.load_guard_findings.append({
            "check": "hf_checksum",
            "severity": "critical" if hf_result.match is False else "info",
            "detail": hf_result.detail,
            "verdict": "WARN" if hf_result.match is False else "CLEAN",
        })
        if hf_result.match is False:
            static_report.load_guard_verdict = "WARN"

    if lg_verdict == "DETONATE":
        tui.print_detonate(lg_findings)
        stem = model_path.stem
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        md = to_markdown(static_report, [])
        out = _resolve_out_path(
            args.output or str(Path(__file__).parent.parent / "outputs") + "/",
            default_name=f"{stem}_blocked_{stamp}.md",
        )
        _mkdir_for(out)
        out.write_text(md, encoding="utf-8")
        tui.print_artifact("Markdown", str(out))
        _write_run_manifest(
            parsed,
            args,
            requested_tests=[],
            static_verdict=lg_verdict,
            dynamic_results=[],
            artifacts=[out],
            output_dir=out.parent,
            stem=f"{stem}_blocked_{stamp}",
            sha256=static_report.file_sha256 or None,
        )
        return 5

    if args.max_tokens == -1:
        from ggufscan.inference import is_thinking_model
        if is_thinking_model(parsed):
            args.max_tokens = 1024
            tui.print_info(
                f"thinking arch '{parsed.architecture}' detected → "
                f"max_tokens auto-bumped to {args.max_tokens}"
            )
        else:
            args.max_tokens = 256

    dynamic_results: list[TestResult] = []
    personal_result = None
    if not args.static_only:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            tui.print_error(
                "llama-cpp-python not installed. Use --static-only or install:\n"
                "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
            )
            return 3

        requested = [t.strip() for t in args.tests.split(",") if t.strip()]
        unknown = [t for t in requested if t not in TEST_REGISTRY]
        if unknown:
            tui.print_error(f"unknown test(s): {unknown}. Choices: {list(TEST_REGISTRY)}")
            return 4

        dynamic_results = _run_dynamic(parsed, args, requested)
        tui.print_summary(dynamic_results)

        if args.personal_dataset:
            from ggufscan.suites.personal.runner import PersonalRunner

            dataset_path = Path(args.personal_dataset).expanduser()
            if not dataset_path.is_file():
                tui.print_error(f"personal dataset not found: {dataset_path}")
                return 6
            personal_tags = {
                tag.strip() for tag in (args.personal_tags or "").split(",") if tag.strip()
            }
            try:
                personal_result = PersonalRunner().run(
                    parsed,
                    dataset_path,
                    n_ctx=args.n_ctx,
                    n_gpu_layers=args.n_gpu_layers,
                    tensor_split=_parse_split(args.tensor_split),
                    chat_format=args.chat_format,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    tags=personal_tags or None,
                    limit=args.personal_limit,
                    repetitions=args.personal_repetitions,
                )
            except ValueError as exc:
                tui.print_error(f"invalid personal dataset: {exc}")
                return 6

    stem = model_path.stem
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "static" if args.static_only else "scan"

    # Default all outputs to outputs/ when no explicit paths given
    _out_dir = Path(__file__).parent.parent / "outputs"
    _out_dir = str(_out_dir) + "/"
    out_md   = args.output   or _out_dir
    out_json = args.json_out or _out_dir
    out_ragas = args.ragas_out or (_out_dir if not args.static_only else None)
    artifact_paths: list[Path] = []

    if personal_result is not None:
        from ggufscan.suites.personal.runner import write_personal_result

        personal_raw = args.personal_output or _out_dir
        personal_path = _resolve_out_path(
            personal_raw,
            default_name=f"{stem}_{suffix}_{stamp}_personal.json",
        )
        write_personal_result(personal_path, personal_result)
        tui.print_artifact("Personal benchmark", str(personal_path))
        artifact_paths.append(personal_path)

    ragas_records: list[dict] = []
    if dynamic_results:
        ragas_records = to_evaluation_records(dynamic_results)

    if out_ragas and ragas_records:
        out = _resolve_out_path(out_ragas, default_name=f"{stem}_{suffix}_{stamp}_records.json")
        _mkdir_for(out)
        out.write_text(
            json.dumps(ragas_records, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
        )
        log.info("Evaluation records written to %s (%d records)", out, len(ragas_records))
        tui.print_artifact("Evaluation records", f"{out} ({len(ragas_records)} records)")
        artifact_paths.append(out)

    ragas_eval = run_judge(
        args,
        ragas_records,
        stem=stem,
        stamp=stamp,
        suffix=suffix,
        config=cfg,
        parse_split=_parse_split,
    )
    if ragas_eval:
        tui.print_ragas_eval(ragas_eval["summary"], ragas_eval["analysis"])

    md = to_markdown(static_report, dynamic_results, ragas_eval=ragas_eval)
    md_path = _resolve_out_path(out_md, default_name=f"{stem}_{suffix}_{stamp}.md")
    _mkdir_for(md_path)
    md_path.write_text(md, encoding="utf-8")
    log.info("Markdown report written to %s", md_path)
    tui.print_artifact("Markdown", str(md_path))

    data = to_json(static_report, dynamic_results)
    if ragas_eval:
        data["judge_eval"] = ragas_eval
    json_path = _resolve_out_path(out_json, default_name=f"{stem}_{suffix}_{stamp}.json")
    _mkdir_for(json_path)
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
    )
    log.info("JSON report written to %s", json_path)
    tui.print_artifact("JSON", str(json_path))

    if args.nemoclaw:
        from ggufscan import nemoclaw_blueprint
        scan_data = data
        out_dir = Path(out_json).expanduser().parent if args.json_out else Path(__file__).parent.parent / "outputs"
        kind, out = nemoclaw_blueprint.generate(scan_data, out_dir)
        tui.print_nemoclaw(kind, out)
        artifact_paths.append(out)

    artifact_paths.extend((md_path, json_path))
    requested = [] if args.static_only else [t.strip() for t in args.tests.split(",") if t.strip()]
    additional_results: tuple[BenchmarkResult, ...] = ()
    additional_suites: tuple[SuiteSpec, ...] = ()
    if personal_result is not None:
        additional_results = (personal_result.benchmark,)
        additional_suites = (SuiteSpec(
            name=personal_result.dataset.name,
            version=personal_result.dataset.version,
            tests=tuple(record.task_id for record in personal_result.records),
            profile="custom",
        ),)
    _write_run_manifest(
        parsed,
        args,
        requested_tests=requested,
        static_verdict=lg_verdict,
        dynamic_results=dynamic_results,
        artifacts=artifact_paths,
        output_dir=json_path.parent,
        stem=f"{stem}_{suffix}_{stamp}",
        sha256=static_report.file_sha256 or None,
        additional_results=additional_results,
        additional_suites=additional_suites,
        repetitions=args.personal_repetitions if personal_result is not None else 1,
    )

    return 0


def _write_run_manifest(
    parsed,
    args,
    *,
    requested_tests: list[str],
    static_verdict: str,
    dynamic_results: list[TestResult],
    artifacts: list[Path],
    output_dir: Path,
    stem: str,
    sha256: str | None,
    additional_results: tuple[BenchmarkResult, ...] = (),
    additional_suites: tuple[SuiteSpec, ...] = (),
    repetitions: int = 1,
) -> Path:
    """Build and write the immutable phase-1 run manifest."""

    from ggufscan.manifest import (
        create_manifest,
        create_run_spec,
        security_benchmark_result,
        write_manifest,
    )

    run = create_run_spec(
        parsed,
        requested_tests=requested_tests,
        quick=args.quick,
        static_only=args.static_only,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        tensor_split=_parse_split(args.tensor_split),
        chat_format=args.chat_format,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        config_path=str(Path(args.config).expanduser()),
        sha256=sha256,
        created_at=datetime.now(timezone.utc),
        run_id=stem,
        additional_suites=additional_suites,
        repetitions=repetitions,
    )
    benchmark = security_benchmark_result(
        dynamic_results,
        static_verdict=static_verdict,
    )
    artifact_names = tuple(str(path.resolve()) for path in artifacts)
    manifest = create_manifest(
        run,
        results=(benchmark, *additional_results),
        artifacts=artifact_names,
    )
    manifest_path = output_dir / f"{stem}_manifest.json"
    write_manifest(manifest_path, manifest)
    tui.print_artifact("Manifest", str(manifest_path))

    return manifest_path


def _mkdir_for(out: Path) -> None:
    try:
        ensure_output_parent(out)
    except ValueError:
        parent = out.parent
        tui.print_error(
            f"cannot write '{out.name}': '{parent}' exists as a file, not a directory.\n"
            f"Hint: add a trailing '/' to --output/--ragas/--json paths to force "
            f"directory mode, or use separate paths for each output type."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
