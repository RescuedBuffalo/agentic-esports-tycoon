"""Alembic migration environment.

Reads ``DATABASE_URL`` from the process environment (falling back to the URL
in ``alembic.ini``), wires the SQLAlchemy metadata from
:mod:`esports_sim.db`, and runs migrations synchronously — Alembic doesn't
need the async path even though services do.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

import esports_sim.db.models  # noqa: F401  side-effect import: registers all tables
from alembic import context
from esports_sim.db.base import Base
from sqlalchemy import engine_from_config, pool

# Alembic Config object provides access to values within the .ini file.
config = context.config

# Allow DATABASE_URL to override alembic.ini at runtime. Services run with
# their own URL, CI uses the service-container URL, devs use the
# docker-compose default.
db_url = os.getenv("DATABASE_URL")
if db_url:
    # Alembic uses the sync `psycopg` driver; if the URL was pasted from a
    # service config that uses asyncpg, swap it transparently so we don't
    # force every consumer to maintain two URLs.
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    config.set_main_option("sqlalchemy.url", db_url)

# Configure Python logging from the alembic.ini file.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing against a live DB."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Open a connection and run migrations transactionally."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Surface server-side defaults in autogenerate diffs.
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
