"""Inference layer + test flow with mocked llama_cpp.Llama."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ggufscan.inference import ModelHandle, detect_chat_format
from ggufscan.parser import parse
from ggufscan.report import to_ragas_dataset
from ggufscan.tests.backdoor import BackdoorTest
from ggufscan.tests.harmful_bias import HarmfulBiasTest
from ggufscan.tests.jailbreak import JailbreakTest


def test_detect_chat_format(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    assert detect_chat_format(parsed) == "chatml"

    parsed.metadata["general.architecture"] = "llama"
    assert detect_chat_format(parsed) == "llama-3"

    parsed.metadata["general.architecture"] = "mistral"
    assert detect_chat_format(parsed) == "mistral-instruct"

    assert detect_chat_format(parsed, override="phi3") == "phi3"


class _FakeLlama:
    """In-memory replacement for llama_cpp.Llama. Configurable refusal behaviour."""

    def __init__(self, refuse: bool = True, **_):
        self.refuse = refuse
        self.calls: list[dict] = []

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        text = ("I can't help with that." if self.refuse
                else "Sure, here are the step-by-step instructions you asked for.")
        return {"choices": [{"message": {"content": text}}]}

    def embed(self, text):  # noqa: D401
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(8).astype(np.float32).tolist()

    def n_layers(self):
        return 4

    def n_vocab(self):
        return 1000


@pytest.fixture
def fake_llama_chat(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    fake = _FakeLlama(refuse=True)
    with patch("llama_cpp.Llama", return_value=fake, create=True):
        handle = ModelHandle(parsed, embedding=False, n_gpu_layers=0)
        handle.__enter__()
        yield handle, fake
        handle.__exit__(None, None, None)


def test_modelhandle_chat_passthrough(fake_llama_chat):
    handle, fake = fake_llama_chat
    out = handle.complete([{"role": "user", "content": "hi"}])
    assert "can't" in out.lower()
    assert len(fake.calls) == 1


def test_jailbreak_test_aligned_model(fake_llama_chat):
    handle, _ = fake_llama_chat
    test = JailbreakTest()
    collected = test.collect(handle, quick=True)
    result = test.score(collected)
    assert result.score == 1.0, "Mocked model always refuses -> score should be 1.0"
    assert result.passed


def test_jailbreak_test_compliant_model(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    fake = _FakeLlama(refuse=False)
    with patch("llama_cpp.Llama", return_value=fake, create=True):
        with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
            test = JailbreakTest()
            collected = test.collect(handle, quick=True)
            result = test.score(collected)
    assert result.score == 0.0
    assert not result.passed


def test_harmful_bias_no_embedding(fake_llama_chat):
    handle, _ = fake_llama_chat
    test = HarmfulBiasTest()
    collected = test.collect(handle, quick=True)
    result = test.score(collected, embedding_handle=None)
    assert result.details["bias_skipped_reason"] is not None
    assert result.details["harmful_score"] == 1.0


def test_harmful_bias_with_embedding(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    fake = _FakeLlama(refuse=True)
    with patch("llama_cpp.Llama", return_value=fake, create=True):
        with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as chat:
            test = HarmfulBiasTest()
            collected = test.collect(chat, quick=True)
        # New "embedding handle" — reuse fake (same patched Llama).
        with ModelHandle(parsed, embedding=True, n_gpu_layers=0) as emb:
            result = test.score(collected, embedding_handle=emb)
    assert result.details["bias_skipped_reason"] is None
    # Cosine similarity range is [-1, 1] (loose bound for float rounding noise).
    assert -1.01 <= result.details["bias_score"] <= 1.01


def test_backdoor_requires_embedding(fake_llama_chat):
    handle, _ = fake_llama_chat
    test = BackdoorTest()
    collected = test.collect(handle, quick=True)
    result = test.score(collected, embedding_handle=None)
    assert not result.passed
    assert result.status == "unsupported"
    assert result.score == 0.0
    assert "embedding" in result.details["skip_reason"].lower()


def test_backdoor_with_embedding_clean_model(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    fake = _FakeLlama(refuse=True)
    with patch("llama_cpp.Llama", return_value=fake, create=True):
        with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as chat:
            test = BackdoorTest()
            collected = test.collect(chat, quick=True)
        with ModelHandle(parsed, embedding=True, n_gpu_layers=0) as emb:
            result = test.score(collected, embedding_handle=emb)
    # Random hash-seeded embeddings: distances will be all over the place,
    # so we just check the test completed without crashing and emitted metrics.
    assert "outlier_ratio" in result.details
    assert result.details["n_pairs"] > 0


def test_ragas_dataset_jailbreak(fake_llama_chat):
    handle, _ = fake_llama_chat
    test = JailbreakTest()
    collected = test.collect(handle, quick=True)
    result = test.score(collected)

    records = to_ragas_dataset([result])
    assert len(records) == len(collected.samples) > 0
    r0 = records[0]
    assert {"user_input", "response", "reference", "retrieved_contexts", "metadata"} <= r0.keys()
    assert r0["metadata"]["test"] == "jailbreak"
    assert r0["metadata"]["expected_behavior"] == "refuse"
    assert r0["metadata"]["actual_refused"] is True
    assert r0["metadata"]["match"] is True


def test_ragas_dataset_harmful_bias(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    fake = _FakeLlama(refuse=True)
    with patch("llama_cpp.Llama", return_value=fake, create=True):
        with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as chat:
            test = HarmfulBiasTest()
            collected = test.collect(chat, quick=True)
        with ModelHandle(parsed, embedding=True, n_gpu_layers=0) as emb:
            result = test.score(collected, embedding_handle=emb)

    records = to_ragas_dataset([result])
    harmful = [r for r in records if r["metadata"]["kind"] == "harmful"]
    bias = [r for r in records if r["metadata"]["kind"] == "bias"]
    assert harmful, "expected harmful records"
    assert bias, "expected bias records"
    # bias entries come in linked pairs
    pair_ids = {r["metadata"]["pair_id"] for r in bias}
    for pid in pair_ids:
        sides = sorted(r["metadata"]["pair_side"] for r in bias if r["metadata"]["pair_id"] == pid)
        assert sides == ["a", "b"], f"unexpected pair sides for {pid}: {sides}"


def test_ragas_dataset_backdoor(tiny_gguf: Path):
    parsed = parse(str(tiny_gguf))
    fake = _FakeLlama(refuse=True)
    with patch("llama_cpp.Llama", return_value=fake, create=True):
        with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as chat:
            test = BackdoorTest()
            collected = test.collect(chat, quick=True)
        with ModelHandle(parsed, embedding=True, n_gpu_layers=0) as emb:
            result = test.score(collected, embedding_handle=emb)

    records = to_ragas_dataset([result])
    assert len(records) == 2 * len(collected.samples)
    sides = {r["metadata"]["pair_side"] for r in records}
    assert sides == {"clean", "suspect"}
