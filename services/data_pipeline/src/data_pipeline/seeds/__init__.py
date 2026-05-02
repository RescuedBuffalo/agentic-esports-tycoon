"""One-shot seed scripts that bootstrap the canonical entity store.

A seed is *not* a connector: it runs once per environment to populate
the canonical entity table before any incremental scraper goes live.
Connectors then keep that store fresh on their normal cadence.

Public surface::

    seed_from_vlr_csv      # BUF-8 v2 entry point — bulk match history
    VlrSeedManifest        # auditable record of one VLR seed run
    DEFAULT_SEEDS_DIR      # ./seeds
"""

from data_pipeline.seeds.vlr import (
    DEFAULT_SEEDS_DIR,
    VlrSeedManifest,
    seed_from_vlr_csv,
)

__all__ = [
    "DEFAULT_SEEDS_DIR",
    "VlrSeedManifest",
    "seed_from_vlr_csv",
]
