"""Fixture builders for the graph-export tests (BUF-53).

Produces a deterministic three-era :class:`InMemoryDataSource` with:

* 12 players, 4 teams, 3 patches, 6 agents per era.
* A full set of ``plays_for`` / ``relates_to`` / ``sponsored_by`` /
  ``affects`` edges.

The data is synthetic but *realistic* in distribution: ACS values
roughly mirror VLR norms, region one-hots are mutually exclusive, and
era-over-era player counts shift slightly so the normalisers see
different fits per era.

Kept under ``tests/`` (not ``src/``) — these helpers exist solely to
exercise the production builder; pollution in the production package
would mislead callers about which fixture loaders are blessed.
"""

from __future__ import annotations

from dataclasses import dataclass

from ecosystem.graph.source import EdgeRow, InMemoryDataSource, NodeRow

# Era slugs match the ones in config/patch_eras.yaml so a future
# integration test can reuse the same fixture loader pointed at the
# real seeder.
ERA_SLUGS = ("e2024_01", "e2024_02", "e2024_03")

_REGIONS = ("americas", "emea", "pacific", "china")
_AGENT_ROLES = ("duelist", "controller", "initiator", "sentinel")


@dataclass(frozen=True, slots=True)
class _EraConfig:
    slug: str
    season_year: int
    ordinal: float
    duration_days: float
    matches_played: float
    agents_added: float
    maps_added: float
    meta_magnitude: float
    is_major_shift: bool


_ERAS = (
    _EraConfig("e2024_01", 2024, 0.0, 96.0, 1200.0, 1.0, 0.0, 0.4, False),
    _EraConfig("e2024_02", 2024, 1.0, 112.0, 1500.0, 1.0, 0.0, 0.7, True),
    _EraConfig("e2024_03", 2024, 2.0, 154.0, 1750.0, 1.0, 0.0, 0.3, False),
)


def build_three_era_source() -> InMemoryDataSource:
    """Return a fully-populated three-era :class:`InMemoryDataSource`."""
    src = InMemoryDataSource()
    for era_idx, era in enumerate(_ERAS):
        _seed_era(src, era_idx, era)
    return src


# ---- per-era seeding ------------------------------------------------------


def _seed_era(src: InMemoryDataSource, era_idx: int, era: _EraConfig) -> None:
    src.set_patch(
        era.slug,
        {
            "season_year": era.season_year,
            "era_ordinal": era.ordinal,
            "duration_days": era.duration_days,
            "matches_played": era.matches_played,
            "agents_added": era.agents_added,
            "maps_added": era.maps_added,
            "meta_magnitude": era.meta_magnitude,
            "is_major_shift": era.is_major_shift,
            "agg_match_acs": 220.0 + 4.0 * era_idx,
            "agg_match_kast": 0.71 + 0.01 * era_idx,
        },
    )

    # ---- players --------------------------------------------------------
    n_players = 12
    for p in range(n_players):
        region = _REGIONS[p % len(_REGIONS)]
        # Vary stats per era so the per-era normaliser fit differs;
        # the shape is "meaningful but constrained" — ACS in [180, 280],
        # KAST in [0.55, 0.85], etc.
        acs = 180.0 + (p * 7.0) + (era_idx * 5.0)
        kast = 0.55 + 0.01 * p + 0.01 * era_idx
        adr = 120.0 + 2.5 * p + era_idx
        hs_pct = 0.18 + 0.01 * (p % 5)
        rating = 0.85 + 0.02 * p + 0.01 * era_idx
        fb_rate = 0.10 + 0.01 * (p % 6)
        fd_rate = 0.10 + 0.005 * (p % 7)
        clutch_rate = 0.10 + 0.01 * (p % 5)
        multikill_rate = 0.05 + 0.005 * (p % 4)
        latent_skill = float(p) - 5.5  # negative + positive z
        role_fit = 0.4 + 0.04 * (p % 6)
        synergy_mag = 0.3 + 0.05 * (p % 5)
        age_days = 200.0 + p * 30.0
        tier_rank = 0.2 + 0.05 * (p % 6)

        src.add_node(
            era.slug,
            "player",
            NodeRow(
                id=f"player-{p:02d}",
                features={
                    "acs": acs,
                    "kast": kast,
                    "adr": adr,
                    "hs_pct": hs_pct,
                    "rating": rating,
                    "fb_rate": fb_rate,
                    "fd_rate": fd_rate,
                    "clutch_rate": clutch_rate,
                    "multikill_rate": multikill_rate,
                    "latent_skill": latent_skill,
                    "role_fit": role_fit,
                    "synergy_mag": synergy_mag,
                    "age_days": age_days,
                    "region_americas": 1.0 if region == "americas" else 0.0,
                    "region_emea": 1.0 if region == "emea" else 0.0,
                    "region_pacific": 1.0 if region == "pacific" else 0.0,
                    "region_china": 1.0 if region == "china" else 0.0,
                    "tier_rank": tier_rank,
                },
            ),
        )

    # ---- teams ----------------------------------------------------------
    n_teams = 4
    for t in range(n_teams):
        region = _REGIONS[t % len(_REGIONS)]
        src.add_node(
            era.slug,
            "team",
            NodeRow(
                id=f"team-{t:02d}",
                features={
                    "avg_acs": 210.0 + 4.0 * t + era_idx,
                    "avg_kast": 0.68 + 0.01 * t + 0.005 * era_idx,
                    "win_rate": 0.40 + 0.05 * t,
                    "map_win_rate": 0.45 + 0.04 * t,
                    "attack_rwr": 0.42 + 0.03 * t,
                    "defense_rwr": 0.50 + 0.02 * t,
                    "eco_efficiency": 0.30 + 0.05 * t,
                    "comeback_rate": 0.10 + 0.02 * t,
                    "strength_rating": float(t) - 1.5,
                    "style_aggression": 0.40 + 0.05 * t,
                    "style_utility": 0.50 + 0.04 * t,
                    "region_americas": 1.0 if region == "americas" else 0.0,
                    "region_emea": 1.0 if region == "emea" else 0.0,
                    "region_pacific": 1.0 if region == "pacific" else 0.0,
                    "region_china": 1.0 if region == "china" else 0.0,
                    "tier_rank": 0.3 + 0.1 * t,
                    "roster_age_days": 365.0 + 90.0 * t,
                },
            ),
        )

    # ---- agents ---------------------------------------------------------
    n_agents = 6
    for a in range(n_agents):
        role = _AGENT_ROLES[a % len(_AGENT_ROLES)]
        src.add_node(
            era.slug,
            "agent",
            NodeRow(
                id=f"agent-{a:02d}",
                features={
                    "avg_acs_when_picked": 200.0 + 6.0 * a + era_idx,
                    "avg_rating_when_picked": 0.95 + 0.02 * a,
                    "pick_rate": 0.05 + 0.04 * a,
                    "win_rate_when_picked": 0.40 + 0.04 * a,
                    "ban_rate": 0.01 + 0.005 * a,
                    "strength_score": 0.40 + 0.05 * a,
                    "synergy_index": 0.30 + 0.06 * a,
                    "role_duelist": 1.0 if role == "duelist" else 0.0,
                    "role_controller": 1.0 if role == "controller" else 0.0,
                    "role_initiator": 1.0 if role == "initiator" else 0.0,
                    "role_sentinel": 1.0 if role == "sentinel" else 0.0,
                    "age_days": 100.0 + 200.0 * a,
                },
            ),
        )

    # ---- plays_for ------------------------------------------------------
    # Spread 12 players across 4 teams, 3 per team. Some players move
    # team across eras to exercise edge filtering.
    for p in range(n_players):
        team_idx = (p + era_idx) % n_teams
        src.add_edge(
            era.slug,
            ("player", "plays_for", "team"),
            EdgeRow(
                src_id=f"player-{p:02d}",
                dst_id=f"team-{team_idx:02d}",
                attributes={
                    "tenure_days": 30.0 + 10.0 * p,
                    "role_slot": 0.2 * (p % 5),
                },
            ),
        )

    # ---- relates_to (team↔team) ----------------------------------------
    # A small ring: 0→1, 1→2, 2→3, 3→0.
    for t in range(n_teams):
        src.add_edge(
            era.slug,
            ("team", "relates_to", "team"),
            EdgeRow(
                src_id=f"team-{t:02d}",
                dst_id=f"team-{(t + 1) % n_teams:02d}",
                attributes={
                    "rivalry_strength": 0.3 + 0.1 * t,
                    "head_to_head_count": 5.0 + t * 2.0,
                },
            ),
        )

    # ---- sponsored_by (team→team) --------------------------------------
    # Each org sponsors the next-but-one; a thin pattern that still
    # produces non-empty edge_attr matrices.
    for t in range(n_teams - 1):
        src.add_edge(
            era.slug,
            ("team", "sponsored_by", "team"),
            EdgeRow(
                src_id=f"team-{t:02d}",
                dst_id=f"team-{(t + 2) % n_teams:02d}",
                attributes={
                    "annual_value_usd": 100_000.0 * (t + 1),
                    "tenure_days": 200.0 + 50.0 * t,
                },
            ),
        )

    # ---- affects (patch→agent) -----------------------------------------
    # The patch node touches every agent — a real era usually tweaks
    # ~half of them but the validator only checks structural soundness,
    # so the dense fixture is fine here.
    for a in range(n_agents):
        src.add_edge(
            era.slug,
            ("patch", "affects", "agent"),
            EdgeRow(
                src_id=era.slug,
                dst_id=f"agent-{a:02d}",
                attributes={
                    "change_magnitude": 0.05 + 0.1 * a,
                    "buff_direction": 1.0 if a % 2 == 0 else 0.0,
                },
            ),
        )


__all__ = ["ERA_SLUGS", "build_three_era_source"]
