"""
Pytest configuration for the Ragas evaluation suite.

Checks that OPENAI_API_KEY is set before any test runs — avoids a confusing
OpenAI AuthenticationError mid-test when someone forgets to export the key.
"""
import os
import sys
from pathlib import Path

import pytest

# ensure meetmind_ai is importable when pytest is run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config: pytest.Config) -> None:
    """Fail fast with a clear message if the API key is missing."""
    # only enforce for the evaluation suite — other test directories don't need this
    test_paths = [str(p) for p in config.args]
    is_eval_run = any("evaluation" in p for p in test_paths) or not test_paths

    if is_eval_run and not os.getenv("OPENAI_API_KEY"):
        pytest.exit(
            "\n\nOPENAI_API_KEY is not set.\n"
            "Export it before running the eval suite:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "Or add it to backend/.env and source it.\n"
            "In CI, add it as a GitHub Actions secret named OPENAI_API_KEY.\n",
            returncode=1,
        )
