"""Shared pytest fixtures.

Two things matter here:

* Tests are split into ``unit`` (pure logic, no server) and ``live`` (needs the
  running app). Live tests are skipped automatically when the app is not
  reachable, so ``pytest`` is always safe to run.
* The project root is put on ``sys.path`` so the app modules import cleanly no
  matter which directory pytest is invoked from.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_URL = os.environ.get("RAG_TEST_URL", "http://127.0.0.1:7860").rstrip("/")


def app_is_up(url: str = APP_URL, timeout: float = 2.0) -> bool:
    """True when the Gradio app answers on the given URL."""
    try:
        with urllib.request.urlopen(f"{url}/", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def pytest_configure(config):
    """Register markers so --strict-markers does not reject them."""
    config.addinivalue_line("markers", "unit: pure logic, no running server")
    config.addinivalue_line("markers", "live: needs the running app")
    config.addinivalue_line("markers", "slow: real model inference")


@pytest.fixture(scope="session")
def app_url() -> str:
    """Base URL of the running app, or skip the test if it is not up."""
    if not app_is_up():
        pytest.skip(
            f"app not reachable at {APP_URL} - start it with start_rag.bat")
    return APP_URL


@pytest.fixture(scope="session")
def client(app_url):
    """A gradio_client bound to the running app."""
    from gradio_client import Client

    return Client(f"{app_url}/")