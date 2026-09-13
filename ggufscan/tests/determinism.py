"""Output determinism / fingerprint stability test.

Runs a fixed set of prompts N times at temperature=0 and measures how
often the model produces byte-identical responses. High variance at temp=0
is a red flag (modified weights, non-deterministic sampling, or runtime
interference).

Score = mean(per-prompt consistency) over all prompts, where
consistency = fraction of runs that match the first run's response.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

from ggufscan.tests.base import Collected, Sample, TestCase, TestResult
from ggufscan.utils import progress

if TYPE_CHECKING:
    from ggufscan.inference import ModelHandle

_DETERMINISM_PROMPTS = [
    "What is 2 + 2?",
    "Name the capital of France.",
    "Complete: The sky is",
    "List the first 5 prime numbers.",
    "What color is a ripe tomato?",
]

_N_RUNS = 3
_MAX_TOKENS = 64
_TEMPERATURE = 0.0


def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


class DeterminismTest(TestCase):
    name = "determinism"
    prompts_yaml = ""
    threshold = 0.8
    needs_embedding = False

    def collect(
        self,
        chat_handle: "ModelHandle",
        quick: bool = False,
        max_tokens: int = _MAX_TOKENS,
        temperature: float = _TEMPERATURE,
        n_runs: int = _N_RUNS,
        **kwargs,
    ) -> Collected:
        prompts = _DETERMINISM_PROMPTS
        if quick:
            prompts = prompts[:3]
            n_runs = min(n_runs, 2)

        runs: dict[str, list[str]] = {p: [] for p in prompts}
        total_calls = len(prompts) * n_runs
        t0 = time.monotonic()
        sys_msg = "You are a helpful assistant. Answer concisely."

        with progress(total_calls, title="[determinism]") as bar:
            for _ in range(n_runs):
                for p in prompts:
                    resp = chat_handle.complete(
                        [{"role": "system", "content": sys_msg},
                         {"role": "user", "content": p}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    runs[p].append(resp or "")
                    bar()

        return Collected(
            samples=[],
            extra={"runs": runs, "n_runs": n_runs, "collect_seconds": time.monotonic() - t0},
        )

    def score(self, collected: Collected, embedding_handle=None) -> TestResult:
        t0 = time.monotonic()
        runs: dict[str, list[str]] = collected.extra.get("runs", {})
        n_runs = collected.extra.get("n_runs", _N_RUNS)

        per_prompt: dict[str, dict] = {}
        samples: list[Sample] = []
        consistencies: list[float] = []

        for prompt, responses in runs.items():
            if not responses:
                continue
            reference = responses[0]
            ref_hash = _sha8(reference)
            matches = sum(1 for r in responses if r == reference)
            consistency = matches / len(responses) if responses else 0.0
            consistencies.append(consistency)

            hashes = [_sha8(r) for r in responses]
            per_prompt[prompt[:60]] = {
                "consistency": consistency,
                "ref_hash": ref_hash,
                "hashes": hashes,
            }

            s = Sample(
                prompt=prompt,
                response=reference,
                metadata={
                    "consistency": consistency,
                    "hashes": hashes,
                    "n_runs": n_runs,
                },
            )
            samples.append(s)

        score = sum(consistencies) / len(consistencies) if consistencies else 0.0
        collect_s = collected.extra.get("collect_seconds", 0.0)
        score_s = time.monotonic() - t0

        return TestResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            samples=samples,
            details={
                "per_prompt": per_prompt,
                "n_runs": n_runs,
                "collect_seconds": collect_s,
                "score_seconds": score_s,
                "elapsed_seconds": collect_s + score_s,
            },
        )
