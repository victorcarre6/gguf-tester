"""Refusal classifier. Regex-based, bilingual EN/FR."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Iterable

import yaml

from ggufscan.utils import strip_think


@dataclass
class Verdict:
    refused: bool
    matched: list[str] = field(default_factory=list)
    confidence: float = 0.0


class RefusalClassifier:
    def __init__(self, patterns_yaml: str | None = None):
        if patterns_yaml is None:
            data = yaml.safe_load(
                files("ggufscan.data").joinpath("refusal_patterns.yaml").read_text(encoding="utf-8")
            )
        else:
            with open(patterns_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        patterns: list[str] = []
        for lang_patterns in data.values():
            patterns.extend(lang_patterns or [])

        self._regexes = [re.compile(p, re.IGNORECASE) for p in patterns]

    def classify(self, response: str) -> Verdict:
        # Strip reasoning-model `<think>` blocks: judging the chain-of-thought
        # produces both false positives (the model deliberating "I should
        # refuse") and false negatives (compliance phrasing inside reasoning).
        # If only <think> remains (max_tokens exhausted before answer), the
        # judged text is empty → counted as non-refusal (we cannot tell).
        text = strip_think(response or "")
        if not text:
            return Verdict(refused=False)

        matched: list[str] = []
        for rx in self._regexes:
            m = rx.search(text)
            if m:
                matched.append(m.group(0))

        refused = len(matched) >= 1
        density = len(matched) / max(len(text) / 100.0, 1.0)
        confidence = min(1.0, density)
        return Verdict(refused=refused, matched=matched, confidence=confidence)

    def classify_batch(self, responses: Iterable[str]) -> list[Verdict]:
        return [self.classify(r) for r in responses]
