"""Riot Games API connector for the BUF-9 ingestion framework (BUF-82).

Pulls official VALORANT match history and per-player perspective from the
Riot Games developer API. Riot is the highest-priority source in conflict
resolution (BUF-12) — every other source defers to it for canonical
identifiers.

Pipeline shape (one ``fetch`` round per pass):

    seed PUUIDs -> matchlist endpoint -> per-match GET -> yield one upstream
    payload per match -> validate -> transform -> one IngestionRecord per
    player on the match (10 for a complete game).

Each transformed record carries enough of the player's perspective on the
match for downstream stat extraction (the per-round stat table is a
follow-up ticket — see PR description). The full match JSON lands in
``raw_record.payload`` automatically via the runner.

Rate-limiting strategy
----------------------

The runner already gates outbound traffic via :class:`TokenBucket`. Riot's
production keys, however, enforce a *rolling* window that doesn't always
align with the bucket — and the limit is shared across endpoints. We
defend in depth:

1.  Conservative bucket default (~10 req/min steady state, burst 20),
    overridable via constructor for tier-2 or higher production keys.
2.  Wrapper around ``http_get`` that, on a 429 response, reads
    ``Retry-After`` and sleeps before raising
    :class:`TransientFetchError` — the runner skips the row without
    persisting raw, so the next scheduled run retries naturally.
3.  Same wrapper escalates 5xx to :class:`TransientFetchError`. Other
    response-shape errors raise :class:`SchemaDriftError` instead, so
    the offending payload is preserved for offline triage.

PUUID linkage
-------------

``platform_id`` is the player's Riot PUUID — globally unique, stable
across name changes, and the canonical identifier Riot itself uses.
``platform_name`` is the public ``gameName#tagLine`` Riot identity tag
(what shows up in-client). The resolver mints aliases at ``confidence
1.0`` for every yielded record because Riot is authoritative; downstream
sources that disagree get demoted, not the other way around.

Out of scope
------------

* ``player_match_stat`` table and per-round stat extraction. The schema
  doesn't have one yet; designing it (per-round granularity, agent
  picks, ability casts) is its own ticket. This connector emits the
  full match-from-this-player's-perspective payload so the downstream
  extractor can run against ``staging_record.payload`` once that ticket
  lands.
* Backfill orchestration. ``cadence`` is the daily-pull hint for the
  scheduler; bulk historical loads (e.g., back to patch 5.0 for known
  pro PUUIDs) are a manual run that supplies an old ``since``
  watermark.
* Multi-region routing. Defaults to the ``americas`` Riot regional
  routing host; ``region`` is a constructor knob for ``europe`` /
  ``ap`` / ``kr``. A single ingest pass targets one region — the
  scheduler can fan out per region later.
* Production rate-limit tuning. Ship a conservative bucket, expose
  ``rate_limit`` as a constructor argument so a tier-2 production key
  can dial it up.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Any

import structlog
from esports_sim.db.enums import EntityType, Platform
from structlog.stdlib import BoundLogger

from data_pipeline.connector import Connector, IngestionRecord, RateLimit
from data_pipeline.errors import SchemaDriftError, TransientFetchError

# Conservative defaults for Riot's production-key 100-req-per-2-min window.
# 20 burst / (20/120 = 0.167 rps) = ~10 rpm steady state, well inside the
# limit even after retry storms. Override via constructor in environments
# with a higher tier key.
_DEFAULT_RATE_LIMIT = RateLimit(capacity=20, refill_per_second=20.0 / 120.0)

# Keep ``cadence`` aligned with the spec: daily incremental pulls. The
# runner is single-shot — this is a hint to the outer scheduler, not a
# sleep-loop inside the connector.
_DEFAULT_CADENCE = timedelta(days=1)

# Default to ``americas`` because tier-1 VCT pros mostly live there and
# the ticket's seed list is americas-weighted. Multi-region routing is
# explicitly out of scope (see module docstring).
_DEFAULT_REGION = "americas"

# Riot caps Retry-After at minutes, not hours — but a malformed header
# could otherwise wedge the run for the entire pass. Cap defensively.
_MAX_RETRY_AFTER_SECONDS = 120.0

# Type alias for the injectable HTTP shim. Returning a structured response
# (status + headers + json) keeps the wrapper logic in one place; tests
# fake the same shape without spinning up a server.
HttpResponse = dict[str, Any]
HttpGet = Callable[[str, dict[str, Any]], HttpResponse]


def _default_http_get_factory() -> HttpGet:
    """Build a real-network ``http_get`` backed by ``httpx``.

    The Riot API key comes from ``RIOT_API_KEY``. We deliberately delay
    httpx import + key lookup to call time so unit tests that inject a
    fake never need either present.
    """

    def http_get(url: str, params: dict[str, Any]) -> HttpResponse:
        import httpx  # lazy: tests with an injected http_get never need it

        api_key = os.environ.get("RIOT_API_KEY")
        if not api_key:
            # Fail loudly — a connector that silently 401s every call
            # would burn its rate budget for nothing.
            raise RuntimeError(
                "RIOT_API_KEY not set; cannot make Riot API calls. "
                "Set it in the environment or inject a fake http_get for tests."
            )
        headers = {"X-Riot-Token": api_key}
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params, headers=headers)
        # Normalise to the shape ``_RiotHttpClient`` expects regardless
        # of status — the client decides how to interpret it.
        try:
            body: Any = response.json()
        except ValueError:
            body = None
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "json": body,
        }

    return http_get


class RiotConnector(Connector):
    """Riot Games API connector. One pass per region."""

    def __init__(
        self,
        *,
        seed_puuids: list[str],
        http_get: HttpGet | None = None,
        region: str = _DEFAULT_REGION,
        rate_limit: RateLimit = _DEFAULT_RATE_LIMIT,
        cadence: timedelta = _DEFAULT_CADENCE,
        sleeper: Callable[[float], None] = time.sleep,
        logger: BoundLogger | None = None,
    ) -> None:
        """Construct a Riot connector for ``region``.

        Parameters
        ----------
        seed_puuids:
            The PUUIDs to crawl on each pass. In tests this is a small
            list; in production, it's loaded from a config file (see
            ``out of scope`` in the module docstring — that loader is a
            follow-up ticket).
        http_get:
            Injected HTTP shim. Defaults to an ``httpx``-backed factory
            that reads the Riot API key from ``RIOT_API_KEY``. Tests
            override this with a fake to avoid hitting the network.
        region:
            Riot regional routing host (``americas``, ``europe``, ``ap``,
            ``kr``). One connector instance covers one region; fan out
            via the scheduler.
        rate_limit:
            Token-bucket parameters. Defaults to a conservative
            ~10 rpm / burst 20 — well inside the production-key
            100-req-per-2-min window. Override for higher-tier keys.
        cadence:
            Hint for the scheduler. Default daily.
        sleeper:
            Indirection so the 429 backoff path is testable without
            real wall-clock waits.
        logger:
            Optional structlog logger; defaults to a fresh BoundLogger.
        """
        if not seed_puuids:
            # An empty seed list isn't a runtime error — the connector
            # would just yield nothing — but it almost certainly signals
            # a misconfiguration. Fail loudly so it shows up in CI rather
            # than as a silent no-op pass.
            raise ValueError("RiotConnector requires at least one seed PUUID")

        self._seed_puuids = list(seed_puuids)
        self._http_get = http_get or _default_http_get_factory()
        self._region = region
        self._rate_limit = rate_limit
        self._cadence = cadence
        self._sleeper = sleeper
        self._log = logger or structlog.get_logger("data_pipeline.connectors.riot")

    # --- Connector metadata ------------------------------------------------

    @property
    def source_name(self) -> str:
        return "riot_api"

    @property
    def platform(self) -> Platform:
        return Platform.RIOT_API

    @property
    def entity_types(self) -> tuple[EntityType, ...]:
        return (EntityType.PLAYER,)

    @property
    def cadence(self) -> timedelta:
        return self._cadence

    @property
    def rate_limit(self) -> RateLimit:
        return self._rate_limit

    # --- Connector hooks ---------------------------------------------------

    def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
        """Yield one upstream payload per match newer than ``since``.

        For each seed PUUID, GET the matchlist, then for every match
        whose ``gameStartTimeMillis`` is strictly greater than ``since``
        GET the full match payload and yield::

            {
                "match_id": <riot match id>,
                "puuid_seed": <the seed PUUID we found this match through>,
                "match": <full Riot match response>,
            }

        ``puuid_seed`` is preserved so a re-run can re-attribute the
        match if the seed list changes; the canonical link is the
        match ID.
        """
        since_millis = int(since.timestamp() * 1000)

        for puuid in self._seed_puuids:
            matchlist = self._fetch_matchlist(puuid)
            history = matchlist.get("history") or []
            for match_ref in history:
                # ``gameStartTimeMillis`` is Riot's documented field; an
                # older fixture version may use a slightly different key,
                # so a missing one is treated as "include" rather than
                # silently dropping the row. Validation downstream will
                # catch a malformed match payload.
                start_millis = match_ref.get("gameStartTimeMillis")
                if start_millis is not None and start_millis <= since_millis:
                    continue

                match_id = match_ref.get("matchId")
                if not match_id:
                    # Skip without raising — a single broken matchlist
                    # entry shouldn't abort the entire seed.
                    self._log.warning(
                        "riot.matchlist_entry_missing_id",
                        code="MATCHLIST_DRIFT",
                        puuid_seed=puuid,
                    )
                    continue

                match_payload = self._fetch_match(match_id)
                yield {
                    "match_id": match_id,
                    "puuid_seed": puuid,
                    "match": match_payload,
                }

    def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Shape-check the match payload.

        We require the three top-level keys the transform reads
        (``matchInfo``, ``players``, ``roundResults``); anything else is
        considered drift from Riot's documented schema and gets routed
        through the runner's ``SCHEMA_DRIFT`` log path. We don't do
        deep validation — Riot's documented schema is large, and we'd
        rather catch surface-level breakage early than maintain a
        full mirror.
        """
        match = raw_payload.get("match")
        if not isinstance(match, dict):
            raise SchemaDriftError("payload missing 'match' object")

        for required in ("matchInfo", "players", "roundResults"):
            if required not in match:
                raise SchemaDriftError(f"match payload missing required key '{required}'")

        if not isinstance(match["players"], list):
            raise SchemaDriftError("match.players must be a list")

        return raw_payload

    def transform(self, validated_payload: dict[str, Any]) -> Iterable[IngestionRecord]:
        """Yield one :class:`IngestionRecord` per player on the match.

        ``platform_id`` is the player's PUUID (deterministic; the
        resolver's exact-alias lookup keys on it). ``platform_name`` is
        the Riot ``gameName#tagLine`` identity tag.

        ``payload`` carries enough of the player's perspective on the
        match for the downstream stat-extraction step but trims
        *other* players' detailed round data — we keep their PUUIDs
        only, so a future join can rehydrate from the raw row if
        needed.
        """
        match_id = validated_payload["match_id"]
        match = validated_payload["match"]
        match_info = match["matchInfo"]
        players: list[dict[str, Any]] = match["players"]
        rounds: list[dict[str, Any]] = match["roundResults"]

        # Reference list of every player's PUUID — kept on every
        # emitted record so the downstream extractor knows the full
        # roster without re-reading raw_record.
        roster_puuids = [p.get("puuid") for p in players if p.get("puuid")]

        for player in players:
            puuid = player.get("puuid")
            game_name = player.get("gameName")
            tag_line = player.get("tagLine")
            if not puuid or not game_name or tag_line is None:
                # A player block without identity fields is unusable for
                # alias linkage — skip rather than mint a junk record.
                # We don't raise SchemaDriftError because partial player
                # blocks (anonymised customs, dev accounts) are a known
                # quirk of the Riot API rather than a schema regression.
                self._log.warning(
                    "riot.player_missing_identity",
                    code="PLAYER_IDENTITY_DRIFT",
                    match_id=match_id,
                    has_puuid=bool(puuid),
                    has_game_name=bool(game_name),
                    has_tag_line=tag_line is not None,
                )
                continue

            slimmed_rounds = [
                _slim_round_for_player(round_data, puuid) for round_data in rounds
            ]

            yield IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id=puuid,
                platform_name=f"{game_name}#{tag_line}",
                payload={
                    "match_id": match_id,
                    "match_info": match_info,
                    "this_player": player,
                    "roster_puuids": roster_puuids,
                    "rounds": slimmed_rounds,
                },
            )

    # --- internal HTTP plumbing -------------------------------------------

    def _fetch_matchlist(self, puuid: str) -> dict[str, Any]:
        url = (
            f"https://{self._region}.api.riotgames.com"
            f"/val/match/v1/matchlists/by-puuid/{puuid}"
        )
        return self._get(url, {})

    def _fetch_match(self, match_id: str) -> dict[str, Any]:
        url = (
            f"https://{self._region}.api.riotgames.com"
            f"/val/match/v1/matches/{match_id}"
        )
        return self._get(url, {})

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """HTTP GET with Riot-aware error translation.

        * 200 — return parsed JSON.
        * 429 — log structured rate-limit warning, sleep ``Retry-After``
          (capped), then raise :class:`TransientFetchError` so the
          runner skips without persisting raw.
        * 5xx — raise :class:`TransientFetchError` (recoverable).
        * Any other non-200 — raise :class:`SchemaDriftError` so the
          payload is preserved for triage.
        * Malformed JSON body on a 200 — :class:`SchemaDriftError`.
        """
        try:
            response = self._http_get(url, params)
        except TransientFetchError:
            # Allow a fake/real http_get to short-circuit translation and
            # raise its own TransientFetchError directly (e.g., a network
            # timeout in the real httpx wrapper).
            raise
        except Exception as exc:
            # Unknown network failure — treat as transient. The runner
            # already differentiates ``TransientFetchError`` from
            # ``SchemaDriftError``; a connect-timeout shouldn't burn
            # the retry slot.
            raise TransientFetchError(f"http_get failed for {url}: {exc}") from exc

        status = response.get("status_code")
        headers = response.get("headers") or {}
        body = response.get("json")

        if status == 200:
            if not isinstance(body, dict):
                raise SchemaDriftError(
                    f"Riot {url} returned 200 but body was {type(body).__name__}, expected dict"
                )
            return body

        if status == 429:
            retry_after = _parse_retry_after(headers.get("Retry-After"))
            self._log.warning(
                "riot.rate_limited",
                code="RATE_LIMITED",
                retry_after_seconds=retry_after,
                endpoint=url,
            )
            if retry_after > 0:
                self._sleeper(retry_after)
            raise TransientFetchError(
                f"Riot 429 at {url}; retry-after={retry_after}s"
            )

        if status is not None and 500 <= status < 600:
            self._log.warning(
                "riot.server_error",
                code="UPSTREAM_5XX",
                status=status,
                endpoint=url,
            )
            raise TransientFetchError(f"Riot {status} at {url}")

        # 400/401/403/404 etc. — these are caller-side problems
        # (bad PUUID, expired key, missing match) and persisting the
        # raw row helps a maintainer diagnose. SchemaDriftError makes
        # the runner persist raw and continue.
        raise SchemaDriftError(
            f"Riot returned unexpected status {status} for {url}; body={body!r}"
        )


def _parse_retry_after(header_value: Any) -> float:
    """Parse a ``Retry-After`` header into a non-negative seconds float.

    Caps at :data:`_MAX_RETRY_AFTER_SECONDS` defensively — Riot rate
    limits are minutes at most, so a header demanding hours is far more
    likely to be malformed than legitimate. Returns ``0.0`` for
    missing/unparseable values rather than raising; the caller will
    still raise :class:`TransientFetchError` so the row is retried,
    and the bucket alone will gate the next attempt.
    """
    if header_value is None:
        return 0.0
    try:
        seconds = float(header_value)
    except (TypeError, ValueError):
        return 0.0
    if seconds <= 0:
        return 0.0
    return min(seconds, _MAX_RETRY_AFTER_SECONDS)


def _slim_round_for_player(round_data: dict[str, Any], puuid: str) -> dict[str, Any]:
    """Drop *other* players' detailed round data, keep this player's full block.

    Each round entry in Riot's response carries a ``playerStats`` list
    with one entry per player, plus a flat ``playerLocations`` /
    ``plantPlayerLocations`` blob. For a per-player ingestion record
    we only need the focal player's stats plus enough match-level
    context (round number, winning team, plant/defuse markers) for
    downstream extraction. We keep other players' PUUIDs as a roster
    reference but trim their per-round detail — the raw match JSON
    is preserved by the runner if a future job needs it.
    """
    slimmed: dict[str, Any] = {
        "roundNum": round_data.get("roundNum"),
        "roundResult": round_data.get("roundResult"),
        "winningTeam": round_data.get("winningTeam"),
        "bombPlanter": round_data.get("bombPlanter"),
        "bombDefuser": round_data.get("bombDefuser"),
        "plantRoundTime": round_data.get("plantRoundTime"),
        "defuseRoundTime": round_data.get("defuseRoundTime"),
    }

    player_stats: list[dict[str, Any]] = round_data.get("playerStats") or []
    focal_stats: dict[str, Any] | None = None
    other_puuids: list[str] = []
    for entry in player_stats:
        entry_puuid = entry.get("puuid")
        if entry_puuid == puuid:
            focal_stats = entry
        elif entry_puuid:
            other_puuids.append(entry_puuid)

    slimmed["this_player_stats"] = focal_stats
    slimmed["other_player_puuids"] = other_puuids
    return slimmed


__all__ = ["HttpGet", "HttpResponse", "RiotConnector"]
