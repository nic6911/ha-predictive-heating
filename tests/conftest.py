"""Pytest fixtures for the Predictive Floor Heating tests."""

import sys
from pathlib import Path

import pytest

# Make the custom_components package importable as a top-level package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest_plugins = ["pytest_homeassistant_custom_component"]

# On Windows the Proactor event loop builds its self-pipe from an AF_INET
# socketpair, which the Home Assistant test plugin's socket block rejects
# (it only allows Unix sockets). Linux CI is unaffected; there we keep the
# block. The block must be disabled before the plugin's configure hook runs.
if sys.platform == "win32":
    import pytest_socket

    pytest_socket.disable_socket = lambda *args, **kwargs: None  # noqa: ARG005


@pytest.fixture
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable loading the custom integration in tests that request it."""
    yield
