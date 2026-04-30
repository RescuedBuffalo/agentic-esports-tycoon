"""Tests for the BUF-9 token-bucket rate limiter.

These run without a database — the limiter has no DB dependency. The
load test uses a fake clock so it's both deterministic and fast,
sidestepping the real-time flakiness that wall-clock load tests usually
suffer on shared CI runners.
"""

from __future__ import annotations

import threading
import time

import pytest
from data_pipeline import RateLimit, TokenBucket


class FakeClock:
    """Monotonic clock under explicit caller control.

    ``advance`` is the only way time moves; ``time()`` records every read
    so tests can assert the bucket consults the clock the expected
    number of times. ``sleep`` advances the clock instead of suspending
    — turning real waits into virtual waits keeps the load test's wall
    time in the millisecond range.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.read_count = 0

    def time(self) -> float:
        self.read_count += 1
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        # Allow zero-or-negative sleeps as no-ops, mirroring time.sleep.
        if seconds > 0:
            self.advance(seconds)


def _bucket(
    *,
    capacity: int = 5,
    refill_per_second: float = 1.0,
    clock: FakeClock | None = None,
) -> tuple[TokenBucket, FakeClock]:
    clock = clock or FakeClock()
    bucket = TokenBucket(
        capacity=capacity,
        refill_per_second=refill_per_second,
        clock=clock.time,
        sleeper=clock.sleep,
    )
    return bucket, clock


# --- construction validation ------------------------------------------------


def test_capacity_below_one_rejected() -> None:
    with pytest.raises(ValueError):
        TokenBucket(capacity=0, refill_per_second=1.0)


def test_refill_non_positive_rejected() -> None:
    with pytest.raises(ValueError):
        TokenBucket(capacity=5, refill_per_second=0.0)
    with pytest.raises(ValueError):
        TokenBucket(capacity=5, refill_per_second=-1.0)


def test_rate_limit_dataclass_validates_inputs() -> None:
    with pytest.raises(ValueError):
        RateLimit(capacity=0, refill_per_second=1.0)
    with pytest.raises(ValueError):
        RateLimit(capacity=1, refill_per_second=0.0)


# --- basic acquire / try_acquire -------------------------------------------


def test_initial_burst_uses_full_capacity_without_waiting() -> None:
    bucket, clock = _bucket(capacity=5)
    for _ in range(5):
        bucket.acquire()
    # No advance() call needed — capacity tokens were already in the bucket
    # at t=0, so acquire never had to sleep.
    assert clock.now == 0.0


def test_acquire_after_burst_waits_for_refill() -> None:
    bucket, clock = _bucket(capacity=2, refill_per_second=10.0)  # 0.1s/token
    bucket.acquire()
    bucket.acquire()
    bucket.acquire()  # forces a sleep
    # Must have advanced at least 0.1s; allow a tiny epsilon for arithmetic.
    assert clock.now >= 0.1 - 1e-9


def test_try_acquire_returns_false_when_empty() -> None:
    bucket, clock = _bucket(capacity=1, refill_per_second=1.0)
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False
    clock.advance(1.0)
    assert bucket.try_acquire() is True


# --- the BUF-9 acceptance load test ----------------------------------------


def test_load_does_not_exceed_capacity_plus_refill() -> None:
    """BUF-9 acceptance: "exceeds by zero under load test".

    We hammer the bucket with 1000 acquires under capacity=5 / refill=10
    and assert the simulated wall-clock matches the theoretical minimum
    duration: ``(N - capacity) / refill``. The bucket cannot, by
    construction, dispense more than this — and the test fails if a
    bug ever lets it.
    """
    bucket, clock = _bucket(capacity=5, refill_per_second=10.0)
    n_calls = 1000

    for _ in range(n_calls):
        bucket.acquire()

    # Theoretical floor: capacity tokens are free, the rest pay 1/refill.
    floor = (n_calls - 5) / 10.0
    assert clock.now >= floor - 1e-9
    # And the bucket should not have *over*shot by more than one tick's
    # worth of refill — the limiter waits exactly long enough, no more.
    assert clock.now <= floor + 1.0 / 10.0 + 1e-9


def test_concurrent_acquire_respects_total_budget() -> None:
    """Multiple threads racing the bucket cannot collectively bypass it.

    Real wall-clock test (not the fake clock — concurrency wants real
    threads) on a tight budget so the assertion stays fast: 12 threads
    each grab 5 tokens out of a bucket sized for 10 burst + 50/sec
    refill. Total = 60 tokens, theoretical minimum elapsed wait =
    ``(60 - 10) / 50 = 1.0s``. Allow generous slack on CI.
    """
    bucket = TokenBucket(capacity=10, refill_per_second=50.0)
    n_threads = 12
    per_thread = 5

    def worker() -> None:
        for _ in range(per_thread):
            bucket.acquire()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start

    total = n_threads * per_thread
    floor = (total - 10) / 50.0
    # Lower bound: bucket must not have under-waited.
    assert (
        elapsed >= floor - 0.05
    ), f"acquired {total} tokens in {elapsed:.3f}s (floor {floor:.3f}s); limiter is leaking"


def test_idle_period_caps_at_capacity_not_unbounded() -> None:
    """Sitting idle for hours must not grant a million-token burst.

    A naive implementation that just multiplies elapsed * refill would
    let a connector sleep through cadence then blast upstream. The
    bucket caps at ``capacity``.
    """
    bucket, clock = _bucket(capacity=3, refill_per_second=1.0)
    clock.advance(10_000.0)  # idle a long time
    # Three free acquires (capacity), then must wait.
    bucket.acquire()
    bucket.acquire()
    bucket.acquire()
    pre = clock.now
    bucket.acquire()
    assert clock.now - pre >= 1.0 - 1e-9


def test_from_rate_limit_constructor() -> None:
    rl = RateLimit(capacity=4, refill_per_second=2.0)
    bucket = TokenBucket.from_rate_limit(rl)
    # Spot-check by exhausting capacity and seeing try_acquire fail.
    for _ in range(4):
        assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False


# --- refund (Codex P2 — token returns on EOS probe) ----------------------


def test_refund_returns_token_to_bucket() -> None:
    """A refunded token is immediately available for try_acquire."""
    bucket, _ = _bucket(capacity=1, refill_per_second=1.0)
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False  # exhausted
    bucket.refund()
    assert bucket.try_acquire() is True


def test_refund_caps_at_capacity() -> None:
    """Stray refund without a prior acquire can't overflow the bucket.

    Without the cap, calling ``refund`` repeatedly would let the bucket
    grow unbounded — turning rate-limiting into a no-op for any future
    workload that depends on the declared burst size.
    """
    bucket, _ = _bucket(capacity=2, refill_per_second=1.0)
    for _ in range(10):
        bucket.refund()
    # Only ``capacity`` tokens are ever spendable in one burst.
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is True
    assert bucket.try_acquire() is False
