"""CLI entry-point for judging exported scan records with a local LLM.

Pipeline:
  1. Load records produced by `ggufscan --records ...`.
  2. Load the judge model (typically Qwen3-14B) via `ModelHandle`.
  3. Score each record (alignment / refusal_quality / harm / relevancy /
     faithfulness proxies) — strict JSON output from the judge.
  4. Aggregate stats, ask the same judge for an `# Analysis Summary`.
  5. Write `*_judge.json` and `*_judge.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ggufscan.parser import parse as parse_gguf
from ggufscan.artifacts import resolve_output_path
from ggufscan.ragas_eval import JudgeEvaluator, render_markdown

log = logging.getLogger("ggufscan-judge")


def _parse_split(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x) for x in s.split(",")]


_resolve_out_path = resolve_output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ggufscan-judge",
        description="Local LLM-as-judge evaluation of exported scan records.",
    )
    p.add_argument("--input", required=True, help="Path to *_records.json from a scan")
    p.add_argument("--judge-model", required=True, help="Path to judge .gguf (e.g. Qwen3-14B-Q5_K_M.gguf)")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--tensor-split", default=None, help="Comma-separated floats (e.g. '0.5,0.5')")
    p.add_argument("--chat-format", default=None, help="Override chat_format (default: auto-detect)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=None,
                   help="Enriched JSON output path (file or dir)")
    p.add_argument("--output-md", default=None,
                   help="Markdown report output path (file or dir)")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N records (smoke test)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2

    judge_path = Path(args.judge_model).expanduser()
    if not judge_path.is_file():
        print(f"ERROR: judge model not found: {judge_path}", file=sys.stderr)
        return 2

    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        print("ERROR: llama-cpp-python not installed. Install with:\n"
              "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python",
              file=sys.stderr)
        return 3

    records: list[dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print("ERROR: input is not a JSON list (expected schema: list of records)",
              file=sys.stderr)
        return 4
    if args.limit:
        records = records[: args.limit]
    log.info("Loaded %d records from %s", len(records), input_path)

    from ggufscan.inference import ModelHandle

    parsed = parse_gguf(str(judge_path))
    handle_kwargs: dict[str, Any] = {
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "tensor_split": _parse_split(args.tensor_split),
        "chat_format": args.chat_format,
        "seed": args.seed,
    }

    log.info("Loading judge %s ...", judge_path.name)
    with ModelHandle(parsed, embedding=False, **handle_kwargs) as judge:
        evaluator = JudgeEvaluator(
            judge=judge, max_tokens=args.max_tokens, temperature=args.temperature,
        )
        log.info("Scoring %d records ...", len(records))
        enriched = evaluator.evaluate_all(records)
        summary = evaluator.summarize(enriched)
        log.info("Generating analysis narrative ...")
        analysis = evaluator.generate_analysis(enriched, summary)

    stem = input_path.stem.removesuffix("_records").removesuffix("_ragas")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_json_name = f"{stem}_judge_{stamp}.json"
    default_md_name = f"{stem}_judge_{stamp}.md"

    json_out = _resolve_out_path(args.output_json, default_json_name) \
        if args.output_json else input_path.with_name(default_json_name)
    md_out = _resolve_out_path(args.output_md, default_md_name) \
        if args.output_md else input_path.with_name(default_md_name)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": str(input_path),
        "judge_model": str(judge_path),
        "summary": summary,
        "analysis": analysis,
        "records": enriched,
    }
    json_out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    md_out.write_text(
        render_markdown(enriched, summary, analysis, source=input_path.name),
        encoding="utf-8",
    )

    print(f"JSON results: {json_out}", file=sys.stderr)
    print(f"Markdown report: {md_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
