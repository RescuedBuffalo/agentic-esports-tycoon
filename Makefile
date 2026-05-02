# Bootstrap targets for the agentic-esports-tycoon monorepo.
#
# `make dev` is the single entrypoint promised by BUF-5: it brings up the local
# data plane (Postgres + Qdrant) and syncs the uv workspace.

SHELL := /bin/bash

.PHONY: help dev up down sync migrate test coverage lint format typecheck precommit ci clean

# Minimum line coverage the ``make coverage`` / ``make ci`` targets enforce.
# Picked to sit just under the suite's measured number so the ratchet is
# locked in without false-failing the next PR. Bump as coverage grows.
COVERAGE_FAIL_UNDER ?= 80

help:
	@echo "Targets:"
	@echo "  dev         Sync uv workspace + bring up Postgres & Qdrant."
	@echo "  up          Bring up the docker-compose data plane."
	@echo "  down        Stop & remove the data plane (volumes preserved)."
	@echo "  sync        Resolve and install the uv workspace (incl. dev group)."
	@echo "  migrate     Apply Alembic migrations against \$$DATABASE_URL."
	@echo "  test        Run pytest across all workspace members."
	@echo "  coverage    Run pytest with line+branch coverage; fail under \$$COVERAGE_FAIL_UNDER (currently $(COVERAGE_FAIL_UNDER))."
	@echo "  lint        Run ruff + black --check."
	@echo "  format      Apply ruff --fix and black."
	@echo "  typecheck   Run mypy."
	@echo "  precommit   Run all configured pre-commit hooks on every file."
	@echo "  ci          Lint + typecheck + coverage (mirrors CI)."
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

coverage:
	# Run the full suite under coverage and fail the build if the line
	# coverage drops below ``$$COVERAGE_FAIL_UNDER``. ``term-missing``
	# prints uncovered lines inline so a regression points straight at
	# the file that lost coverage. Configuration (source paths, omit
	# patterns, branch coverage) lives in pyproject.toml under
	# ``[tool.coverage.*]``.
	uv run pytest \
		--cov \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=$(COVERAGE_FAIL_UNDER)

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

ci: lint typecheck coverage

clean:
	docker compose down -v
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache
