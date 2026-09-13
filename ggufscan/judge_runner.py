"""Configuration and execution of the optional local LLM judge."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from ggufscan.artifacts import ensure_output_parent, resolve_output_path
from ggufscan.inference import ModelHandle
from ggufscan.parser import parse
from ggufscan.ragas_eval import JudgeEvaluator

log = logging.getLogger(__name__)


def resolve_judge_config(args, config: dict) -> tuple[Path | None, dict, str | None]:
    """Resolve judge settings, with CLI values taking precedence over YAML."""

    judge_config = config.get("judge") or {}
    if args.judge_model:
        model_path = Path(args.judge_model).expanduser()
        source = "flag"
    elif judge_config.get("model"):
        model_path = Path(str(judge_config["model"])).expanduser()
        source = "config"
    else:
        return None, {}, None

    def choose(cli_value, key, default):
        return cli_value if cli_value is not None else judge_config.get(key, default)

    settings = {
        "n_ctx": choose(args.judge_n_ctx, "n_ctx", 4096),
        "max_tokens": choose(args.judge_max_tokens, "max_tokens", 512),
        "temperature": choose(args.judge_temperature, "temperature", 0.0),
        "tensor_split": choose(args.judge_tensor_split, "tensor_split", None),
    }

    return model_path, settings, source


def run_judge(
    args,
    records: list[dict],
    *,
    stem: str,
    stamp: str,
    suffix: str,
    config: dict,
    parse_split: Callable[[str | None], list[float] | None],
):
    """Run the configured judge and optionally persist its detailed result."""

    if args.skip_ragas_eval or not records:
        return None

    judge_path, settings, source = resolve_judge_config(args, config)
    if judge_path is None:
        return None
    if not judge_path.is_file():
        log.warning("Judge model not found (%s): %s", source, judge_path)
        return None

    tensor_split = _judge_tensor_split(settings["tensor_split"], args.tensor_split, parse_split)
    judge_model = parse(str(judge_path))
    with ModelHandle(
        judge_model,
        n_ctx=settings["n_ctx"],
        n_gpu_layers=args.n_gpu_layers,
        tensor_split=tensor_split,
        chat_format=args.chat_format,
        seed=args.seed,
    ) as judge:
        evaluator = JudgeEvaluator(
            judge=judge,
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        )
        enriched = evaluator.evaluate_all(records)
        summary = evaluator.summarize(enriched)
        analysis = evaluator.generate_analysis(enriched, summary)

    result = {
        "summary": summary,
        "analysis": analysis,
        "records": enriched,
        "judge_model": str(judge_path),
    }
    output_path = _judge_output_path(args, stem=stem, stamp=stamp, suffix=suffix)
    if output_path:
        ensure_output_parent(output_path)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    return result


def _judge_tensor_split(raw, fallback, parse_split):
    if isinstance(raw, list):
        return [float(value) for value in raw]
    if isinstance(raw, str):
        return parse_split(raw)

    return parse_split(fallback)


def _judge_output_path(args, *, stem: str, stamp: str, suffix: str) -> Path | None:
    filename = f"{stem}_{suffix}_{stamp}_judge.json"
    if args.ragas_results_out:
        return resolve_output_path(args.ragas_results_out, filename)
    if args.output:
        report_path = resolve_output_path(args.output, "_report")

        return report_path.parent / filename

    return None
