"""Phase-3 contracts for the canonical versioned security suite."""

from ggufscan.suites.security import (
    PROMPT_SET_VERSION,
    REFUSAL_CLASSIFIER_VERSION,
    SECURITY_SUITE_VERSION,
    THRESHOLD_SET_VERSION,
    security_descriptors,
    security_test_registry,
)


def test_security_registry_contains_the_eight_historical_tests():
    registry = security_test_registry()

    assert set(registry) == {
        "jailbreak",
        "harmful_bias",
        "backdoor",
        "extraction",
        "agency",
        "determinism",
        "output_safety",
        "factuality",
    }


def test_security_components_are_versioned_and_described():
    descriptors = security_descriptors()

    assert SECURITY_SUITE_VERSION == "2"
    assert PROMPT_SET_VERSION
    assert REFUSAL_CLASSIFIER_VERSION
    assert THRESHOLD_SET_VERSION
    assert len(descriptors) == 8
    assert all(0.0 <= descriptor.threshold <= 1.0 for descriptor in descriptors)
    assert all(descriptor.prompts_version == PROMPT_SET_VERSION for descriptor in descriptors)
