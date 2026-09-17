"""grok_fleet.router — worker selection + circuit breaker + rate limiting.

Implements the FROZEN public signatures from ``grok_fleet.interfaces`` for
``RateLimiter`` and ``Router``. Pure stdlib (threading only). No Hermes, no live
network, no live model calls — a model call only ever crosses the boundary via
the injected ``Caller`` alias elsewhere in the package; this module never
constructs one.

Design
------
RateLimiter
    A thin wrapper over ``threading.BoundedSemaphore`` capping the number of
    simultaneous in-flight model calls. ``acquire`` honors an optional timeout
    (never blocks forever when one is given), and an optional ``on_backoff``
    hook fires when a 429 / rate-limit signal is reported so callers can sleep
    before retrying. It is also a context manager.

Router
    Holds the registered fleet (registration order == priority), a per-model
    circuit breaker, and a shared RateLimiter. ``pick_worker`` returns the
    highest-priority worker whose breaker permits a call, or ``None`` when every
    worker is unavailable (so the caller PARKs instead of looping forever).

Circuit breaker (per model)
    States: CLOSED (normal), OPEN (too many recent failures — skip the model),
    HALF_OPEN (cool-down elapsed — allow ONE trial call to probe recovery).
    A breaker opens on either of two independent triggers:
      1. consecutive failures >= failure_threshold, OR
      2. rolling error-rate over the recent window >= error_rate_threshold
         (once the window has at least ``error_rate_min_samples`` samples).
    After ``cooldown_seconds`` an OPEN breaker moves to HALF_OPEN and offers a
    single trial; a success on that trial closes it, a failure re-opens it.

Thread safety
    Every mutation of breaker/limiter state is guarded by a lock so the router
    is safe to share across worker threads.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional

from .types import ModelSpec

log = logging.getLogger(__name__)

# --- tuning constants (named, not magic) ------------------------------------
DEFAULT_MAX_CONCURRENT = 4
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 30.0
DEFAULT_ERROR_RATE_THRESHOLD = 0.5
DEFAULT_ERROR_RATE_WINDOW = 10
DEFAULT_ERROR_RATE_MIN_SAMPLES = 4

__all__ = ["RateLimiter", "Router", "BreakerState"]


class BreakerState(Enum):
    """Circuit-breaker state for a single model."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RateLimiter:
    """A bounded-concurrency semaphore wrapper for model calls.

    Caps the number of simultaneous in-flight model calls at ``max_concurrent``.
    Backed by a ``threading.BoundedSemaphore`` so an over-release is caught
    rather than silently inflating the pool. Optionally fires ``on_backoff`` when
    a 429 / rate-limit signal is reported via :meth:`backoff`.
    """

    def __init__(
        self,
        max_concurrent: int,
        *,
        on_backoff: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Create a limiter allowing ``max_concurrent`` simultaneous holders.

        Backed by a threading.BoundedSemaphore. ``max_concurrent`` must be >= 1.
        ``on_backoff`` is an optional hook invoked with a delay (seconds) when a
        429 / rate-limit is reported through :meth:`backoff`; it lets a caller
        sleep or reschedule without this module ever sleeping on its own.
        """
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be an int >= 1")
        self._max_concurrent = max_concurrent
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self._on_backoff = on_backoff

    @property
    def max_concurrent(self) -> int:
        """The configured concurrency cap."""
        return self._max_concurrent

    def acquire(self, *, timeout: Optional[float] = None) -> bool:
        """Acquire a slot, blocking up to ``timeout`` seconds.

        Returns True on success, False if the timeout elapsed first. Never
        blocks forever when a timeout is given. A None timeout blocks until a
        slot frees (the context-manager path).
        """
        try:
            if timeout is None:
                return bool(self._sem.acquire())
            # threading.Semaphore.acquire(blocking, timeout): timeout requires
            # blocking=True and a non-negative value.
            safe_timeout = timeout if timeout >= 0 else 0.0
            return bool(self._sem.acquire(True, safe_timeout))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("RateLimiter.acquire failed: %s", exc)
            return False

    def release(self) -> None:
        """Release a previously acquired slot back to the pool."""
        try:
            self._sem.release()
        except ValueError as exc:
            # BoundedSemaphore raises ValueError on over-release. Log, swallow —
            # over-releasing must not crash a worker thread.
            log.warning("RateLimiter.release over-released: %s", exc)

    def backoff(self, delay: float) -> None:
        """Report a 429 / rate-limit; invoke the ``on_backoff`` hook if set.

        This module never sleeps itself. The hook decides what to do with the
        suggested ``delay`` (sleep, jittered retry, metric). Errors from the hook
        are logged and swallowed so a bad hook can't take down a worker.
        """
        if self._on_backoff is None:
            return
        try:
            self._on_backoff(float(delay))
        except Exception as exc:
            log.warning("RateLimiter on_backoff hook raised: %s", exc)

    def __enter__(self) -> "RateLimiter":
        """Context-manager acquire (blocking, no timeout)."""
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Context-manager release."""
        self.release()


class _Breaker:
    """Per-model circuit breaker. Internal; not part of the frozen surface."""

    __slots__ = (
        "state",
        "consecutive_failures",
        "opened_at",
        "_recent",
    )

    def __init__(self) -> None:
        self.state: BreakerState = BreakerState.CLOSED
        self.consecutive_failures: int = 0
        self.opened_at: Optional[float] = None
        # rolling window of recent outcomes: True == failure, False == success
        self._recent: Deque[bool] = deque(maxlen=DEFAULT_ERROR_RATE_WINDOW)


class Router:
    """Picks a worker model and opens a per-model circuit breaker on failures.

    Holds the registered fleet, a per-model failure count / open-circuit state,
    and a shared RateLimiter. Workers are tried in registration order; a model
    whose breaker is open is skipped until it cools down, at which point one
    half-open trial call is allowed to probe recovery.
    """

    def __init__(
        self,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        error_rate_threshold: float = DEFAULT_ERROR_RATE_THRESHOLD,
        error_rate_min_samples: int = DEFAULT_ERROR_RATE_MIN_SAMPLES,
        on_backoff: Optional[Callable[[float], None]] = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty router.

        max_concurrent         — bound passed to the internal RateLimiter.
        failure_threshold      — consecutive failures before a model's circuit
                                 opens.
        cooldown_seconds       — how long an OPEN breaker stays open before it
                                 offers a single HALF_OPEN trial.
        error_rate_threshold   — rolling error-rate (failures / window) at or
                                 above which the breaker also opens, once the
                                 window has ``error_rate_min_samples`` samples.
        error_rate_min_samples — minimum samples before the error-rate trigger
                                 is considered (avoids opening on 1/1).
        on_backoff             — optional 429 hook forwarded to the RateLimiter.
        time_fn                — monotonic clock injection point (tests pass a
                                 fake so cooldown is deterministic).
        """
        if not isinstance(failure_threshold, int) or failure_threshold < 1:
            raise ValueError("failure_threshold must be an int >= 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        if not 0.0 < error_rate_threshold <= 1.0:
            raise ValueError("error_rate_threshold must be in (0.0, 1.0]")
        if not isinstance(error_rate_min_samples, int) or error_rate_min_samples < 1:
            raise ValueError("error_rate_min_samples must be an int >= 1")

        self._failure_threshold = failure_threshold
        self._cooldown_seconds = float(cooldown_seconds)
        self._error_rate_threshold = float(error_rate_threshold)
        self._error_rate_min_samples = error_rate_min_samples
        self._time_fn = time_fn

        self._lock = threading.RLock()
        # registration order preserved => priority order
        self._order: List[str] = []
        self._specs: Dict[str, ModelSpec] = {}
        self._breakers: Dict[str, _Breaker] = {}

        self.rate_limiter = RateLimiter(max_concurrent, on_backoff=on_backoff)

    # -- registration --------------------------------------------------------

    def register(self, spec: ModelSpec) -> None:
        """Add a ModelSpec to the fleet.

        Order matters: earlier = higher priority. Re-registering the same id
        replaces the spec but keeps its original priority slot and resets no
        breaker state that already exists (idempotent id, updated record).
        """
        if not isinstance(spec, ModelSpec):
            raise TypeError("register expects a ModelSpec")
        with self._lock:
            if spec.id not in self._specs:
                self._order.append(spec.id)
                self._breakers[spec.id] = _Breaker()
            self._specs[spec.id] = spec

    # -- selection -----------------------------------------------------------

    def pick_worker(self) -> Optional[ModelSpec]:
        """Return the highest-priority available worker, or None if all open.

        Skips models whose role is not 'worker' and any worker whose circuit is
        currently open (cool-down not yet elapsed). When an OPEN breaker's
        cool-down HAS elapsed it transitions to HALF_OPEN and that worker is
        eligible for a single trial. Returns None when no worker is available so
        the caller can PARK rather than loop forever.
        """
        with self._lock:
            for model_id in self._order:
                spec = self._specs.get(model_id)
                if spec is None or spec.role != "worker":
                    continue
                breaker = self._breakers[model_id]
                if self._is_callable(breaker):
                    return spec
            return None

    def _is_callable(self, breaker: _Breaker) -> bool:
        """Whether a breaker currently permits a call.

        CLOSED and HALF_OPEN permit calls. An OPEN breaker whose cooldown has
        elapsed is promoted to HALF_OPEN (permits the one trial); otherwise it
        stays OPEN and is skipped. Caller holds the lock.
        """
        if breaker.state is BreakerState.CLOSED:
            return True
        if breaker.state is BreakerState.HALF_OPEN:
            return True
        # OPEN: check cooldown
        opened_at = breaker.opened_at
        if opened_at is None:
            # inconsistent state; treat as recoverable
            breaker.state = BreakerState.HALF_OPEN
            return True
        elapsed = self._time_fn() - opened_at
        if elapsed >= self._cooldown_seconds:
            breaker.state = BreakerState.HALF_OPEN
            log.info("breaker cooldown elapsed -> HALF_OPEN")
            return True
        return False

    # -- outcome recording ---------------------------------------------------

    def on_success(self, model_id: str) -> None:
        """Record a success: reset the failure count and close the circuit."""
        with self._lock:
            breaker = self._breakers.get(model_id)
            if breaker is None:
                log.warning("on_success for unregistered model_id=%r", model_id)
                return
            breaker.consecutive_failures = 0
            breaker.opened_at = None
            breaker._recent.append(False)
            if breaker.state is not BreakerState.CLOSED:
                log.info("breaker for %s closing after success", model_id)
            breaker.state = BreakerState.CLOSED

    def on_failure(self, model_id: str) -> None:
        """Record a failure: increment the count and open the circuit at threshold.

        Opens on either trigger:
          * consecutive_failures >= failure_threshold, OR
          * rolling error-rate >= error_rate_threshold (once the window has
            >= error_rate_min_samples samples).
        A failure while HALF_OPEN immediately re-opens the breaker (the trial
        did not recover the model).
        """
        with self._lock:
            breaker = self._breakers.get(model_id)
            if breaker is None:
                log.warning("on_failure for unregistered model_id=%r", model_id)
                return
            breaker.consecutive_failures += 1
            breaker._recent.append(True)

            if breaker.state is BreakerState.HALF_OPEN:
                self._open(model_id, breaker, reason="half-open trial failed")
                return

            if breaker.consecutive_failures >= self._failure_threshold:
                self._open(
                    model_id,
                    breaker,
                    reason="consecutive failures %d>=%d"
                    % (breaker.consecutive_failures, self._failure_threshold),
                )
                return

            if self._error_rate_tripped(breaker):
                self._open(model_id, breaker, reason="error-rate threshold")

    def _error_rate_tripped(self, breaker: _Breaker) -> bool:
        """True when the rolling error-rate meets the trip threshold."""
        samples = len(breaker._recent)
        if samples < self._error_rate_min_samples:
            return False
        failures = sum(1 for is_fail in breaker._recent if is_fail)
        rate = failures / samples
        return rate >= self._error_rate_threshold

    def _open(self, model_id: str, breaker: _Breaker, *, reason: str) -> None:
        """Transition a breaker to OPEN and stamp the open time. Lock held."""
        breaker.state = BreakerState.OPEN
        breaker.opened_at = self._time_fn()
        log.warning("breaker OPEN for %s (%s)", model_id, reason)

    # -- introspection (test + ops visibility, not part of frozen surface) ---

    def breaker_state(self, model_id: str) -> Optional[BreakerState]:
        """Current breaker state for ``model_id``, or None if unregistered.

        Read-only observation. Does NOT trigger a cooldown->half-open promotion
        (that only happens through :meth:`pick_worker`) so callers can inspect
        the true stored state.
        """
        with self._lock:
            breaker = self._breakers.get(model_id)
            return breaker.state if breaker is not None else None

    def registered_ids(self) -> List[str]:
        """Registered model ids in priority (registration) order."""
        with self._lock:
            return list(self._order)
