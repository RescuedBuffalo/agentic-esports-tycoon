"""Pytest fixtures for the ecosystem service tests.

The graph-export tests are pure (no Postgres dependency), so this
conftest is much smaller than the data_pipeline / packages-shared
twins. It only handles the sibling-import path so
``from graph_fixtures import ...`` keeps working under
``--import-mode=importlib``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Sibling-import path: pytest's ``--import-mode=importlib`` (set at the
# workspace root) skips the sys.path manipulation that ``prepend`` mode
# would otherwise do. Adding the test directory here keeps
# ``from graph_fixtures import ...`` working in the per-test files.
sys.path.insert(0, str(Path(__file__).resolve().parent))
