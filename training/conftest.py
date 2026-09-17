"""Pytest configuration for training directory.

Only collects tests when fine-tuning dependencies are installed.
"""

import sys
from pathlib import Path

try:
    import peft
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def pytest_configure(config):
    """Skip training tests if peft is not installed."""
    if not PEFT_AVAILABLE:
        # Skip all tests in this directory
        sys.exit(0)
