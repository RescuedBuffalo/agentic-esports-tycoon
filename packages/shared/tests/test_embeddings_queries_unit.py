"""Unit tests for the BUF-28 :func:`similar_players` argument guards.

The full helper exercises Postgres + pgvector — those tests live in
``test_embeddings_db.py`` (skipped without ``TEST_DATABASE_URL``).
The pure-Python guards belong here so a fresh clone runs them without
a database.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from esports_sim.embeddings import similar_players


def test_similar_players_rejects_non_positive_k() -> None:
    session: Any = MagicMock()
    with pytest.raises(ValueError, match="k must be positive"):
        similar_players(session, uuid.uuid4(), k=0)
    session.execute.assert_not_called()


def test_similar_players_rejects_negative_k() -> None:
    session: Any = MagicMock()
    with pytest.raises(ValueError, match="k must be positive"):
        similar_players(session, uuid.uuid4(), k=-3)
    session.execute.assert_not_called()
