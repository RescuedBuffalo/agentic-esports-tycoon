"""One-shot seed scripts that bootstrap the canonical entity store.

A seed is *not* a connector: it runs once per environment to populate
the canonical entity table before any incremental scraper goes live.
Connectors then keep that store fresh on their normal cadence.

Public surface::

    seed_from_liquipedia      # BUF-8 entry point
    SeedManifest              # auditable record of one seed run
    DEFAULT_SEEDS_DIR         # ./seeds
"""

from data_pipeline.seeds.liquipedia import (
    DEFAULT_SEEDS_DIR,
    SeedManifest,
    seed_from_liquipedia,
)

__all__ = [
    "DEFAULT_SEEDS_DIR",
    "SeedManifest",
    "seed_from_liquipedia",
]
