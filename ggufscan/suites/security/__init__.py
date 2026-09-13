"""Canonical metadata and registry for the security suite."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


SECURITY_SUITE_VERSION = "2"
PROMPT_SET_VERSION = "2026-08-17"
REFUSAL_CLASSIFIER_VERSION = "1"
THRESHOLD_SET_VERSION = "2026-08-17"

_TEST_CLASSES = {
    "jailbreak": ("ggufscan.tests.jailbreak", "JailbreakTest"),
    "harmful_bias": ("ggufscan.tests.harmful_bias", "HarmfulBiasTest"),
    "backdoor": ("ggufscan.tests.backdoor", "BackdoorTest"),
    "extraction": ("ggufscan.tests.extraction", "ExtractionTest"),
    "agency": ("ggufscan.tests.agency", "AgencyTest"),
    "determinism": ("ggufscan.tests.determinism", "DeterminismTest"),
    "output_safety": ("ggufscan.tests.output_safety", "OutputSafetyTest"),
    "factuality": ("ggufscan.tests.factuality", "FactualityTest"),
}


@dataclass(frozen=True)
class SecurityTestDescriptor:
    name: str
    threshold: float
    needs_embedding: bool
    prompts_version: str = PROMPT_SET_VERSION


def security_test_registry() -> dict[str, type]:
    """Resolve test classes lazily to preserve historical import paths."""

    registry: dict[str, type] = {}
    for name, (module_name, class_name) in _TEST_CLASSES.items():
        module = import_module(module_name)
        registry[name] = getattr(module, class_name)

    return registry


def security_descriptors() -> tuple[SecurityTestDescriptor, ...]:
    """Describe the effective versioned security battery."""

    registry = security_test_registry()

    return tuple(
        SecurityTestDescriptor(
            name=name,
            threshold=test_class.threshold,
            needs_embedding=test_class.needs_embedding,
        )
        for name, test_class in registry.items()
    )


__all__ = [
    "PROMPT_SET_VERSION",
    "REFUSAL_CLASSIFIER_VERSION",
    "SECURITY_SUITE_VERSION",
    "SecurityTestDescriptor",
    "THRESHOLD_SET_VERSION",
    "security_descriptors",
    "security_test_registry",
]
