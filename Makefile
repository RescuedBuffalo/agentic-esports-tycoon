# Bootstrap targets for the agentic-esports-tycoon monorepo.
#
# `make dev` is the single entrypoint promised by BUF-5: it brings up the local
# data plane (Postgres + Qdrant) and syncs the uv workspace.

SHELL := /bin/bash

.PHONY: help dev up down sync migrate test lint format typecheck precommit ci clean

help:
	@echo "Targets:"
	@echo "  dev         Sync uv workspace + bring up Postgres & Qdrant."
	@echo "  up          Bring up the docker-compose data plane."
	@echo "  down        Stop & remove the data plane (volumes preserved)."
	@echo "  sync        Resolve and install the uv workspace (incl. dev group)."
	@echo "  migrate     Apply Alembic migrations against \$$DATABASE_URL."
	@echo "  test        Run pytest across all workspace members."
	@echo "  lint        Run ruff + black --check."
	@echo "  format      Apply ruff --fix and black."
	@echo "  typecheck   Run mypy."
	@echo "  precommit   Run all configured pre-commit hooks on every file."
	@echo "  ci          Lint + typecheck + test (mirrors CI)."
	@echo "  clean       Tear down volumes and caches."

dev: sync up
	@echo ""
	@echo "Dev stack ready."
	@echo "  Postgres: localhost:$${POSTGRES_PORT:-5432}"
	@echo "  Qdrant:   http://localhost:$${QDRANT_HTTP_PORT:-6333}"

up:
	# `--wait` blocks until each service's healthcheck reports healthy. Without
	# it, compose returns as soon as containers are started (not ready), and
	# follow-on commands race the Postgres/Qdrant startup. See
	# https://docs.docker.com/reference/cli/docker/compose/up/#options
	docker compose up -d --wait postgres qdrant

down:
	docker compose down

sync:
	uv sync --all-packages

migrate:
	# Run from packages/shared so alembic finds alembic.ini next door.
	# DATABASE_URL overrides the dev-compose default (see env.py).
	cd packages/shared && uv run alembic upgrade head

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run ruff check --fix .
	uv run black .

typecheck:
	uv run mypy

precommit:
	uv run pre-commit run --all-files

ci: lint typecheck test

clean:
	docker compose down -v
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache
