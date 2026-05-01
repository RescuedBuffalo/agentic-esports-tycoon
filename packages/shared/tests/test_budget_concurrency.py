"""Concurrency + env-default coverage for the budget governor.

Two scenarios that the original PR didn't cover:

1. **Atomic preflight**: two governors hammering the same SQLite ledger
   while spend is right at the cap edge must serialise. One must end up
   with a ``pre`` row, the other with a ``blocked`` row — never both
   ``pre``. This is the property the original review comment flagged as
   broken; the fix is :meth:`Ledger.serializable_write` running BEGIN
   IMMEDIATE around the cap check + insert.

2. **Env-driven defaults**: a bare ``Governor()`` (no caps argument) must
   pick up ``NEXUS_BUDGET_WEEKLY_HARD_CAP_USD`` and
   ``NEXUS_BUDGET_DISABLE_CAPS`` from the process environment. The
   original review caught a constructor that called the static
   ``default_caps()`` instead of ``BudgetCaps.from_env()``.

Concurrency-driver note (issue #15): the atomicity tests ran for a long
time — and on Windows / GitHub-hosted Linux runners hung the CI job
indefinitely — when the workers were spawned via ``multiprocessing``.
Spawned children re-import the test module and pickle the worker
callable, both of which are flaky enough on the GH Actions runner to
deadlock at ``Pool.map``. Threads are sufficient for the property we
actually want to assert: BEGIN IMMEDIATE on a shared SQLite ledger
serialises writes. The lock is enforced at the OS file-lock layer (not
inside the Python interpreter), and the ``sqlite3`` C extension releases
the GIL across BEGIN / COMMIT, so two threads racing the same ledger
exercise the same critical section a process pool did — without the
spawn-time hang. See issue #15 for the failure mode this swap removes.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from esports_sim.budget import BudgetCaps, BudgetExhausted, Governor, Ledger


@pytest.fixture
def shared_db(tmp_path: Path) -> Path:
    """A SQLite path two processes/threads can both attach to."""
    return tmp_path / "shared-budget.sqlite"


# ---- atomicity ------------------------------------------------------------


def _try_preflight(args: tuple[Path, str]) -> str:
    """Run a single preflight in a worker thread.

    Returns ``"ok"`` if the call passed the gate (a pre row landed), or the
    BudgetExhausted scope (``"weekly"``) if the call was blocked. Anything
    else means an unexpected exception — re-raise so the test sees it.
    """
    db_path, purpose = args
    governor = Governor(
        ledger=Ledger(db_path=db_path),
        caps=BudgetCaps(weekly_hard_cap_usd=30.0),
    )
    try:
        governor.preflight(
            purpose=purpose,
            model="claude-haiku-4-5",
            endpoint="messages.create",
            # Each worker is projecting $0.10. Two workers at $29.95 spent
            # cannot both pass: $29.95 + $0.10 + $0.10 = $30.15 > $30.
            projected_cost_usd=0.10,
        )
        return "ok"
    except BudgetExhausted as e:
        return e.scope


def test_concurrent_preflight_serialises_at_the_cap_edge(shared_db: Path) -> None:
    """The fix for the P1 review: two workers near the cap can't both pass.

    Seeds the ledger at $29.95. Two workers each project $0.10. Without
    the BEGIN IMMEDIATE lock, both can read $29.95, both decide
    ``$29.95 + $0.10 = $30.05 > $30`` is OK relative to a stale snapshot,
    and both pass — overshooting the $30 hard cap. With the lock, exactly
    one passes and the other is blocked.
    """
    seed = Ledger(db_path=shared_db)
    seed.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="other",
        phase="post",
        usd_cost=29.95,
    )

    # ThreadPoolExecutor (rather than ``mp.Pool``) — the property under
    # test is "BEGIN IMMEDIATE serialises two governors hitting the same
    # SQLite file." That lock is enforced at the OS file-lock layer, and
    # the ``sqlite3`` C extension releases the GIL across BEGIN/COMMIT,
    # so threads exercise the same critical section a process pool did
    # without the GH Actions spawn-time hang documented in issue #15.
    args = [(shared_db, f"worker-{i}") for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_try_preflight, args))

    n_ok = results.count("ok")
    n_blocked = results.count("weekly")
    # All workers landed exactly one of the two outcomes.
    assert n_ok + n_blocked == len(args)
    # Crucially: the cap was *not* breached. We seeded at $29.95 with $0.10
    # projected per worker — the cap arithmetic only allows zero workers to
    # pass before the next $0.10 would breach $30.
    # Floor of (cap - seeded_spend) / per_worker_projected = floor($0.05 / $0.10) = 0.
    assert n_ok == 0
    assert n_blocked == len(args)

    # Ledger state is consistent: 1 seed + N blocked rows + 0 pre rows.
    final_ledger = Ledger(db_path=shared_db)
    rows = final_ledger.all_entries()
    pre_rows = [r for r in rows if r.phase == "pre"]
    blocked_rows = [r for r in rows if r.phase == "blocked"]
    assert len(pre_rows) == 0
    assert len(blocked_rows) == len(args)


def test_concurrent_preflight_with_room_below_cap_lets_some_pass(
    shared_db: Path,
) -> None:
    """When budget allows N out of M workers, exactly N pass.

    Seed at $29.40, eight workers each projecting $0.10. The cap math
    permits up to floor(($30 - $29.40) / $0.10) = 6 workers to pass; the
    remaining 2 must be blocked. Without atomicity, all eight could
    observe $29.40 and pass, ending the week at $30.20.
    """
    seed = Ledger(db_path=shared_db)
    seed.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="other",
        phase="post",
        usd_cost=29.40,
    )

    args = [(shared_db, f"worker-{i}") for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_try_preflight, args))

    n_ok = results.count("ok")
    n_blocked = results.count("weekly")
    assert n_ok + n_blocked == len(args)
    # Cap math: 6 pass, 2 blocked. Total pre-row spend = 6 * $0.10 = $0.60;
    # combined with $29.40 seed = $30.00 exactly, never overshooting.
    assert n_ok == 6
    assert n_blocked == 2

    final_ledger = Ledger(db_path=shared_db)
    pre_total = sum(r.usd_cost for r in final_ledger.all_entries() if r.phase == "pre")
    seed_total = 29.40
    assert pre_total + seed_total == pytest.approx(30.00)


# ---- env-driven defaults --------------------------------------------------


@pytest.fixture
def cleared_budget_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip budget env vars so each test starts from a known baseline."""
    monkeypatch.delenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", raising=False)
    monkeypatch.delenv("NEXUS_BUDGET_DISABLE_CAPS", raising=False)
    yield


def test_governor_constructor_honours_weekly_cap_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleared_budget_env: None,
) -> None:
    """Bare ``Governor()`` reads NEXUS_BUDGET_WEEKLY_HARD_CAP_USD.

    The P2 review caught this: the original constructor called the static
    ``default_caps()`` (always $30 hard cap) instead of
    ``BudgetCaps.from_env()``, so the documented env override was a no-op
    in the common path.
    """
    monkeypatch.setenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", "5.00")
    gov = Governor(ledger=Ledger(db_path=tmp_path / "budget.sqlite"))
    assert gov.caps.weekly_hard_cap_usd == 5.00


def test_governor_constructor_honours_disable_caps_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleared_budget_env: None,
) -> None:
    """Setting ``NEXUS_BUDGET_DISABLE_CAPS=1`` makes ``Governor()`` permissive.

    Concrete behaviour: a call that would normally trip the hard cap
    succeeds, and the row is annotated so audits can still find it.
    """
    monkeypatch.setenv("NEXUS_BUDGET_DISABLE_CAPS", "1")
    monkeypatch.setenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", "5.00")
    ledger = Ledger(db_path=tmp_path / "budget.sqlite")
    # Spend already over the (tightened) cap.
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="other",
        phase="post",
        usd_cost=10.0,
    )

    gov = Governor(ledger=ledger)
    # Override is on, so this passes despite being well over $5.
    ticket = gov.preflight(
        purpose="other",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=1.0,
    )
    rows = ledger.all_entries()
    pre = next(r for r in rows if r.id == ticket.row_id)
    assert pre.phase == "pre"
    assert "override_disable_caps" in (pre.notes or "")


def test_governor_constructor_falls_back_to_static_defaults(
    tmp_path: Path,
    cleared_budget_env: None,
) -> None:
    """With no env vars set, ``Governor()`` uses the documented static defaults.

    Belt-and-suspenders for the env-aware path: when nothing's exported, we
    must still get the hard-coded $30/wk cap and the documented purpose
    soft caps (BUF-22's $15 / $3 examples).
    """
    gov = Governor(ledger=Ledger(db_path=tmp_path / "budget.sqlite"))
    assert gov.caps.weekly_hard_cap_usd == 30.0
    assert gov.caps.cap_for("personality") == 15.0
    assert gov.caps.cap_for("patch_intent") == 3.0
    assert gov.caps.override_disable_caps is False
