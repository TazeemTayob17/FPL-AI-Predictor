"""Adds the standalone scripts/ directory to sys.path so its pure helper functions are testable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
