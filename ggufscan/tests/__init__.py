"""Dynamic test suite. Each test inherits from TestCase and emits a TestResult."""

from ggufscan.tests.agency import AgencyTest
from ggufscan.tests.backdoor import BackdoorTest
from ggufscan.tests.base import Sample, TestCase, TestResult
from ggufscan.tests.determinism import DeterminismTest
from ggufscan.tests.extraction import ExtractionTest
from ggufscan.tests.factuality import FactualityTest
from ggufscan.tests.harmful_bias import HarmfulBiasTest
from ggufscan.tests.jailbreak import JailbreakTest
from ggufscan.tests.output_safety import OutputSafetyTest
from ggufscan.suites.security import security_test_registry

# Historical import kept as a view of the canonical registry.
TEST_REGISTRY: dict[str, type[TestCase]] = security_test_registry()

__all__ = [
    "Sample", "TestCase", "TestResult",
    "JailbreakTest", "HarmfulBiasTest", "BackdoorTest",
    "ExtractionTest", "AgencyTest", "DeterminismTest",
    "OutputSafetyTest", "FactualityTest",
    "TEST_REGISTRY",
]
