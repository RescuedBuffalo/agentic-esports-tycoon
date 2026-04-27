"""Registry-related exceptions."""

from __future__ import annotations


class RegistryError(RuntimeError):
    """Base for everything raised by :mod:`esports_sim.registry`."""


class RunNotFoundError(RegistryError):
    """``registry.get(run_id)`` was called with an unknown id."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"No run with run_id={run_id!r} in the registry")


class InvalidKindError(RegistryError):
    """The ``kind`` argument failed validation.

    Kinds become directory names and CLI filter values, so we keep them
    URL-/filesystem-safe (lowercase letters, digits, ``-``, ``_``, ``.``).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(
            f"Invalid run kind: {kind!r}. Use lowercase letters, digits, "
            "and -/_/. only (no whitespace, no path separators)."
        )
