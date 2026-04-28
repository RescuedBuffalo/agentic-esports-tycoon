"""Hot-path token-bucket rate limiter for ingestion connectors (BUF-9).

One :class:`TokenBucket` per source. The runner calls :meth:`acquire`
before each upstream request; ``acquire`` blocks until a token is
available, then deducts one. The implementation is intentionally tiny:
synchronous, ``threading.Lock``-guarded, ``time.monotonic``-based —
async ingestion paths will need an async variant later, but we don't
prematurely abstract.

Acceptance test target: "exceeds by zero under load test". The bucket
guarantees that the *number of requests permitted in a window* never
exceeds ``capacity + refill_per_second * window_seconds``, regardless of
contention. We allow a small clock-jitter epsilon when validating that
in tests so a slow CI runner doesn't false-positive.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from data_pipeline.connector import RateLimit

# Indirection so tests can substitute a fake clock without monkey-patching
# the time module wholesale. The runner default is ``time.monotonic`` —
# wall-clock changes would otherwise let a buggy NTP step "refund" tokens.
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


class TokenBucket:
    """Thread-safe token bucket with pluggable clock.

    Constructed once per source; the runner gates each ``fetch``-loop
    iteration through :meth:`acquire`. Tokens accumulate at
    ``refill_per_second`` up to ``capacity``; an empty bucket forces the
    caller to wait the precise interval until the next token is owed.
    """

    def __init__(
        self,
        *,
        capacity: int,
        refill_per_second: float,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be > 0")
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._clock = clock
        self._sleeper = sleeper
        self._tokens: float = float(capacity)
        self._last_refill = clock()
        # A single lock is enough — the critical section is microseconds
        # long. ``threading.Lock`` keeps the bucket fair across worker
        # threads without taking a dependency on asyncio yet.
        self._lock = threading.Lock()

    @classmethod
    def from_rate_limit(
        cls,
        rate_limit: RateLimit,
        *,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> TokenBucket:
        return cls(
            capacity=rate_limit.capacity,
            refill_per_second=rate_limit.refill_per_second,
            clock=clock,
            sleeper=sleeper,
        )

    def _refill(self) -> None:
        # Caller must hold ``self._lock``.
        now = self._clock()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        added = elapsed * self._refill_per_second
        self._tokens = min(self._capacity, self._tokens + added)
        self._last_refill = now

    def acquire(self) -> None:
        """Block until a token is available, then deduct one.

        We compute the wait outside the lock — sleeping under contention
        would serialise everything onto whichever thread happens to win
        the lock first, even though the wait itself doesn't need the
        lock held.
        """
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # ``self._tokens`` is in [0, 1); we need ``1 - self._tokens``
                # more tokens, each takes ``1 / refill_per_second`` seconds.
                missing = 1.0 - self._tokens
                wait = missing / self._refill_per_second
            self._sleeper(wait)

    def try_acquire(self) -> bool:
        """Non-blocking variant — return True if a token was available.

        Useful for tests and metrics paths; the runner uses :meth:`acquire`
        because the policy is "wait, don't drop".
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


__all__ = ["TokenBucket"]
