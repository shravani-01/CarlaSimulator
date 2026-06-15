"""Pytest bootstrap: make `carla_perception` importable without relying on the
editable install (which can be shadowed by conda/pyenv on some setups).

This adds the package directory `perception_py/` to the import path before tests
collect, so `import carla_perception` always works from a clean checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "perception_py"))
