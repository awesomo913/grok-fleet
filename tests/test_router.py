"""Tests for grok_fleet.router — worker selection, circuit breaker, rate limiter.

Pure stdlib + pytest. No network, no model calls. A fake monotonic clock is
injected into the Router so cooldown transitions are deterministic (no sleeps).
"""
from __future__ import annotations

import threading
import time

import pytest

from grok_fleet.router import (
    DEFAULT_ERROR_RATE_MIN_SAMPLES,
    BreakerState,
    RateLimiter,
    Router,
)
from grok_fleet.types import ModelSpec


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class FakeClock:
    """Deterministic monotonic clock for cooldown tests."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _worker(model_id: str, is_local: bool = False) -> ModelSpec:
    return ModelSpec(id=model_id, role="worker", is_local=is_local)


def _make_router(**kwargs) -> Router:
    defaults = dict(
        max_concurrent=4,
        failure_threshold=3,
        cooldown_seconds=30.0,
        time_fn=FakeClock(),
    )
    defaults.update(kwargs)
    return Router(**defaults)


# --------------------------------------------------------------------------- #
# registration + priority
# --------------------------------------------------------------------------- #


def test_registration_order_is_priority():
    # Arrange
    r = _make_router()
    r.register(_worker("grok-4.20-reasoning"))
    r.register(_worker("gemini-2.5-flash"))
    r.register(_worker("deepseek-v4-pro"))
    # Act
    picked = r.pick_worker()
    # Assert — primary (first registered) wins
    assert picked is not None
    assert picked.id == "grok-4.20-reasoning"
    assert r.registered_ids() == [
        "grok-4.20-reasoning",
        "gemini-2.5-flash",
        "deepseek-v4-pro",
    ]


def test_non_worker_roles_are_skipped():
    # Arrange
    r = _make_router()
    r.register(ModelSpec(id="reviewer-fast", role="reviewer_fast"))
    r.register(ModelSpec(id="reviewer-trusted", role="reviewer_trusted", is_local=True))
    r.register(_worker("grok-primary"))
    # Act
    picked = r.pick_worker()
    # Assert — only the worker is eligible
    assert picked is not None
    assert picked.id == "grok-primary"


def test_empty_fleet_returns_none():
    r = _make_router()
    assert r.pick_worker() is None


def test_re_register_same_id_keeps_priority_slot():
    r = _make_router()
    r.register(_worker("grok"))
    r.register(_worker("gemini"))
    # re-register grok with an updated record (still worker)
    r.register(ModelSpec(id="grok", role="worker", is_local=True))
    assert r.registered_ids() == ["grok", "gemini"]
    assert r.pick_worker().id == "grok"


# --------------------------------------------------------------------------- #
# circuit breaker — trips and recovers (the required test)
# --------------------------------------------------------------------------- #


def test_breaker_trips_on_consecutive_failures_and_falls_through():
    # Arrange
    clock = FakeClock()
    r = _make_router(failure_threshold=3, time_fn=clock)
    r.register(_worker("grok-primary"))
    r.register(_worker("gemini-backup"))

    # Act — trip the primary with 3 consecutive failures
    for _ in range(3):
        r.on_failure("grok-primary")

    # Assert — primary breaker OPEN, router falls through to the backup
    assert r.breaker_state("grok-primary") is BreakerState.OPEN
    picked = r.pick_worker()
    assert picked is not None
    assert picked.id == "gemini-backup"


def test_breaker_recovers_via_half_open_then_close():
    # Arrange
    clock = FakeClock()
    r = _make_router(failure_threshold=2, cooldown_seconds=30.0, time_fn=clock)
    r.register(_worker("grok-primary"))
    r.register(_worker("gemini-backup"))

    # trip it
    r.on_failure("grok-primary")
    r.on_failure("grok-primary")
    assert r.breaker_state("grok-primary") is BreakerState.OPEN
    # still open before cooldown -> backup chosen
    assert r.pick_worker().id == "gemini-backup"

    # Act — advance past cooldown; pick_worker promotes OPEN -> HALF_OPEN
    clock.advance(31.0)
    picked = r.pick_worker()
    # Assert — primary offered a half-open trial again (highest priority)
    assert picked.id == "grok-primary"
    assert r.breaker_state("grok-primary") is BreakerState.HALF_OPEN

    # a success on the trial closes the breaker fully
    r.on_success("grok-primary")
    assert r.breaker_state("grok-primary") is BreakerState.CLOSED
    assert r.pick_worker().id == "grok-primary"


def test_half_open_failure_reopens_immediately():
    # Arrange
    clock = FakeClock()
    r = _make_router(failure_threshold=2, cooldown_seconds=10.0, time_fn=clock)
    r.register(_worker("grok-primary"))
    r.register(_worker("gemini-backup"))
    r.on_failure("grok-primary")
    r.on_failure("grok-primary")
    clock.advance(11.0)
    assert r.pick_worker().id == "grok-primary"  # -> HALF_OPEN
    assert r.breaker_state("grok-primary") is BreakerState.HALF_OPEN

    # Act — the trial call fails
    r.on_failure("grok-primary")

    # Assert — re-opened, backup chosen again
    assert r.breaker_state("grok-primary") is BreakerState.OPEN
    assert r.pick_worker().id == "gemini-backup"


def test_success_resets_consecutive_failure_count():
    # error_rate_threshold set to 1.0 so ONLY the consecutive-streak path can
    # trip the breaker — this test is isolating the streak-reset behavior.
    r = _make_router(failure_threshold=3, error_rate_threshold=1.0)
    r.register(_worker("grok"))
    r.on_failure("grok")
    r.on_failure("grok")
    r.on_success("grok")  # resets streak
    r.on_failure("grok")
    r.on_failure("grok")
    # only 2 consecutive since the reset -> still closed (streak path)
    assert r.breaker_state("grok") is BreakerState.CLOSED
    assert r.pick_worker().id == "grok"


def test_all_workers_open_returns_none():
    clock = FakeClock()
    r = _make_router(failure_threshold=1, cooldown_seconds=60.0, time_fn=clock)
    r.register(_worker("a"))
    r.register(_worker("b"))
    r.on_failure("a")
    r.on_failure("b")
    assert r.breaker_state("a") is BreakerState.OPEN
    assert r.breaker_state("b") is BreakerState.OPEN
    # both open, cooldown not elapsed -> None so caller PARKs
    assert r.pick_worker() is None


# --------------------------------------------------------------------------- #
# circuit breaker — error-rate trigger
# --------------------------------------------------------------------------- #


def test_error_rate_trigger_opens_without_consecutive_streak():
    # failure_threshold high so only the error-rate path can trip it
    r = _make_router(
        failure_threshold=99,
        error_rate_threshold=0.5,
        error_rate_min_samples=4,
    )
    r.register(_worker("grok"))
    # The breaker is only ever evaluated on a FAILURE (a success can never open
    # a circuit). Interleave so the consecutive count stays at 1 but the rolling
    # error-rate reaches 50% at the moment of a failure.
    r.on_success("grok")
    r.on_failure("grok")
    r.on_success("grok")
    r.on_failure("grok")  # window: S,F,S,F -> 2/4 = 0.5, evaluated on failure
    assert r.breaker_state("grok") is BreakerState.OPEN


def test_error_rate_needs_min_samples():
    r = _make_router(
        failure_threshold=99,
        error_rate_threshold=0.5,
        error_rate_min_samples=DEFAULT_ERROR_RATE_MIN_SAMPLES,
    )
    r.register(_worker("grok"))
    # 1 failure = 100% error-rate but below min-samples -> must NOT trip
    r.on_failure("grok")
    assert r.breaker_state("grok") is BreakerState.CLOSED


# --------------------------------------------------------------------------- #
# unregistered model ids are safe no-ops
# --------------------------------------------------------------------------- #


def test_outcome_calls_on_unregistered_id_are_safe():
    r = _make_router()
    # should log a warning and NOT raise
    r.on_success("nope")
    r.on_failure("nope")
    assert r.breaker_state("nope") is None


# --------------------------------------------------------------------------- #
# constructor validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failure_threshold": 0},
        {"cooldown_seconds": -1},
        {"error_rate_threshold": 0.0},
        {"error_rate_threshold": 1.5},
        {"error_rate_min_samples": 0},
    ],
)
def test_router_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        _make_router(**kwargs)


# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #


def test_rate_limiter_requires_positive_bound():
    with pytest.raises(ValueError):
        RateLimiter(0)
    with pytest.raises(ValueError):
        RateLimiter(-3)


def test_rate_limiter_caps_concurrency():
    limiter = RateLimiter(2)
    assert limiter.acquire(timeout=1.0) is True
    assert limiter.acquire(timeout=1.0) is True
    # third acquire must fail fast (cap is 2)
    assert limiter.acquire(timeout=0.05) is False
    limiter.release()
    # a slot freed -> next acquire succeeds
    assert limiter.acquire(timeout=1.0) is True
    limiter.release()
    limiter.release()


def test_rate_limiter_context_manager_roundtrips():
    limiter = RateLimiter(1)
    with limiter:
        # inside the block the only slot is taken
        assert limiter.acquire(timeout=0.05) is False
    # released on exit -> acquirable again
    assert limiter.acquire(timeout=0.5) is True
    limiter.release()


def test_rate_limiter_over_release_is_swallowed():
    limiter = RateLimiter(1)
    # releasing more than acquired must not raise (BoundedSemaphore guard)
    limiter.release()  # over-release, logged + swallowed
    # limiter still usable
    assert limiter.acquire(timeout=0.5) is True
    limiter.release()


def test_rate_limiter_backoff_hook_fires():
    seen = []

    def hook(delay: float) -> None:
        seen.append(delay)

    limiter = RateLimiter(2, on_backoff=hook)
    limiter.backoff(1.5)
    assert seen == [1.5]


def test_rate_limiter_backoff_no_hook_is_noop():
    limiter = RateLimiter(2)  # no hook
    limiter.backoff(2.0)  # must not raise


def test_rate_limiter_bad_hook_is_swallowed():
    def boom(delay: float) -> None:
        raise RuntimeError("hook exploded")

    limiter = RateLimiter(1, on_backoff=boom)
    limiter.backoff(1.0)  # error logged + swallowed, no raise


def test_router_exposes_shared_rate_limiter():
    r = _make_router(max_concurrent=1)
    assert isinstance(r.rate_limiter, RateLimiter)
    assert r.rate_limiter.max_concurrent == 1


# --------------------------------------------------------------------------- #
# thread-safety smoke: concurrent outcome recording doesn't corrupt state
# --------------------------------------------------------------------------- #


def test_concurrent_failures_thread_safe():
    r = _make_router(failure_threshold=1000)  # high so it won't open mid-run
    r.register(_worker("grok"))
    errors = []

    def hammer():
        try:
            for _ in range(500):
                r.on_failure("grok")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # still a valid, callable state
    assert r.breaker_state("grok") in (BreakerState.CLOSED, BreakerState.OPEN)
