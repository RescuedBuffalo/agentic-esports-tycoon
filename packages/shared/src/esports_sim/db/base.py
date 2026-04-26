"""SQLAlchemy declarative base.

All BUF-6 models inherit from :class:`Base` so Alembic and tests have one
metadata object to introspect.
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Naming convention pinned so Alembic-generated and hand-written migrations
# produce the same constraint names — otherwise rolling forward and back
# diverges. Pattern is the SQLAlchemy default plus a length-safe primary-key
# template.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by every model in :mod:`esports_sim.db`."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
