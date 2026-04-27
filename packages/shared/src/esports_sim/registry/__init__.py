"""Experiment registry (BUF-69).

Public API::

    from esports_sim.registry import Registry, RunRecord, RunStatus

    registry = Registry()
    run_id = registry.register(
        kind="graph-snapshot",
        config_path="configs/graph/era_7.09.yaml",
        data_paths=["data/raw/era_7.09/"],
    )
    record = registry.get(run_id)
    # record.run_dir → runs/{run_id}/
    # record.config_snapshot → runs/{run_id}/config.yaml
    # record.artifact_path("checkpoint.pt") → runs/{run_id}/checkpoint.pt

    # Mark terminal:
    registry.finalize(run_id, status="completed", notes="trained 10M steps")

See :mod:`esports_sim.registry.db` for the storage details and
:mod:`esports_sim.registry.cli` for the ``nexus run`` subcommands.
"""

from esports_sim.registry.db import (
    Registry,
    RunRecord,
    RunStatus,
)
from esports_sim.registry.errors import (
    InvalidKindError,
    RegistryError,
    RunNotFoundError,
)
from esports_sim.registry.fingerprint import compute_fingerprint, hash_file

__all__ = [
    "InvalidKindError",
    "Registry",
    "RegistryError",
    "RunNotFoundError",
    "RunRecord",
    "RunStatus",
    "compute_fingerprint",
    "hash_file",
]
