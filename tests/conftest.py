"""Pytest fixtures for the Predictive Floor Heating tests."""

import sys
from pathlib import Path

import pytest

# Make the custom_components package importable as a top-level package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest_plugins = ["pytest_homeassistant_custom_component"]


@pytest.fixture
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable loading the custom integration in tests that request it."""
    yield
