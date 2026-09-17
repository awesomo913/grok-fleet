"""grok_fleet.interfaces — FROZEN public signatures for every implementer.

These are STUBS. Each has full type hints + a docstring + `raise
NotImplementedError`. Implementers fill the bodies in their own modules
(contract.py, verify.py, antisim.py, rubric.py, review.py, harness.py,
router.py, queue.py) against EXACTLY these signatures. Do not change a name,
parameter, or return type here without coordinating — this file is the contract
between parallel workers.

The signatures are grouped by their target module. When an implementer creates
e.g. verify.py, they should mirror these signatures there (this file may import
and re-expose them later, but the frozen shape lives here).
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Callable, Dict, List, Optional, Tuple

from .types import (
    AcceptanceCriterion,
    Artifact,
    Caller,
    HarnessOutcome,
    ModelSpec,
    ReviewResult,
    TaskContract,
    TaskType,
    VerifyResult,
)

# ===========================================================================
# contract.py  — Gate 1: Task Contract
# ===========================================================================


def build_contract(
    task_id: str,
    deliverable: str,
    criteria: List[AcceptanceCriterion],
    task_type: TaskType,
) -> TaskContract:
    """Assemble a validated TaskContract (Gate 1).

    Validates that task_id and deliverable are non-empty, task_type is one of
    TASK_TYPES, and criteria is a non-empty list of AcceptanceCriterion whose
    kinds are all in CRITERION_KINDS. Raises ValueError on any violation so a
    malformed contract can never enter the harness.
    """
    raise NotImplementedError


def parse_criteria(spec: object) -> List[AcceptanceCriterion]:
    """Parse a machine/JSON-ish spec into a list of AcceptanceCriterion.

    Accepts a list of dicts (each with keys kind/target and optional extra) or
    an already-parsed JSON string, and returns typed criteria. Raises ValueError
    on unknown kinds or missing required fields. This is the boundary that turns
    external task descriptions into checkable criteria.
    """
    raise NotImplementedError


# ===========================================================================
# verify.py  — Gate 2: Proof-of-Work, independently verified (anti-bluff heart)
# ===========================================================================

#: A runner maps one criterion + the artifacts to an observed result. The
#: default runner dispatches on criterion.kind to the helpers below; tests may
#: inject a fake runner for hermetic runs.
Runner = Callable[[AcceptanceCriterion, List[Artifact]], VerifyResult]


def verify(
    contract: TaskContract,
    artifacts: List[Artifact],
    *,
    runner: Optional[Runner] = None,
) -> List[VerifyResult]:
    """RE-RUN every acceptance criterion ourselves and record what we observe.

    For each criterion in `contract.criteria`, dispatch to `runner` (defaults to
    the built-in dispatcher over run_pytest / file_exists / cmd_zero_exit /
    json_schema / min_citations / regex_present). A criterion the harness cannot
    itself observe as passing yields VerifyResult(observed=False). Also stamps
    the relevant Artifact.observed flags. The model's own success claim is never
    trusted — only what this function observes counts.
    """
    raise NotImplementedError


def run_pytest(path: str, *, timeout: int = 300) -> Tuple[bool, str]:
    """Run pytest against `path` in a subprocess and observe the real result.

    Returns (passed, detail) where passed is True only on exit code 0 and detail
    is a trimmed summary of stdout/stderr (pass/fail counts, first failure).
    Never raises on test failure — a failing suite is a normal False result. May
    raise only on genuinely broken invocation (missing interpreter).
    """
    raise NotImplementedError


def file_exists(path: str) -> Tuple[bool, str]:
    """Stat `path` and report whether it exists.

    Returns (exists, detail) where detail notes type/size when present or the
    missing path when absent. Pure observation, no side effects.
    """
    raise NotImplementedError


def cmd_zero_exit(cmd: str, *, cwd: Optional[str] = None, timeout: int = 120) -> Tuple[bool, str]:
    """Run a shell command and report whether it exited 0.

    Returns (ok, detail) where ok is True iff the process exited with code 0 and
    detail carries the exit code plus trimmed combined output. Enforces a
    timeout; a timeout counts as ok=False, not an exception.
    """
    raise NotImplementedError


def count_citations(text: str) -> int:
    """Count resolvable citations in `text` for the min_citations criterion.

    A citation is a concrete, checkable reference (URL, DOI, or bracketed
    numeric ref backed by a source line) — NOT a bare claim. Returns the integer
    count so the verifier can compare against extra["min"].
    """
    raise NotImplementedError


# ===========================================================================
# antisim.py  — Gate 4: Anti-Simulation detector
# ===========================================================================


def scan(text: str, artifacts: List[Artifact]) -> List[str]:
    """Flag hedge/simulation language not backed by a concrete artifact (Gate 4).

    Scans `text` for sim/hedge markers ("I would", "this should", "simulating",
    "in a real scenario", "assuming", "hypothetically", "pretend", etc.). A
    marker is only a violation when there is no observed artifact substantiating
    the surrounding claim. Returns a list of human-readable flag strings; an
    empty list means the text is grounded. Non-empty flags => NOT_DONE upstream.
    """
    raise NotImplementedError


# ===========================================================================
# rubric.py  — Gate 5: Quality-First gate
# ===========================================================================


def score(task_type: TaskType, artifacts: List[Artifact], review_reasons: List[str]) -> float:
    """Compute a 0.0..1.0 quality score for the deliverable (Gate 5).

    Blends artifact signals (presence, observed-ness, substance) with the
    weight of reviewer reasons for `task_type`. Returns a float in [0.0, 1.0].
    Deterministic given identical inputs so the gate is reproducible.
    """
    raise NotImplementedError


def threshold(task_type: TaskType) -> float:
    """Return the minimum passing quality score for `task_type`.

    A score below this triggers a bounded REVISE (<=2), then escalation, then
    PARK. Values are per-type constants (e.g. code stricter than content).
    """
    raise NotImplementedError


# ===========================================================================
# review.py  — Gate 3: Real adversarial review
# ===========================================================================


def review(
    contract: TaskContract,
    artifacts: List[Artifact],
    *,
    fast_caller: Caller,
    trusted_caller: Optional[Caller] = None,
    escalate: bool = False,
) -> List[ReviewResult]:
    """Run refute-first adversarial review over the ACTUAL artifacts (Gate 3).

    Always runs the fast free-tier reviewer via `fast_caller`, handing it the
    real artifact contents (never the model's own summary). When `escalate` is
    True (fast tier raised a flag, or a tie needs breaking) and `trusted_caller`
    is provided, also runs the trusted local reviewer and includes its
    ReviewResult. Callers are injected; this function never contacts a real API
    itself. Returns one ReviewResult per reviewer that ran.
    """
    raise NotImplementedError


# ===========================================================================
# harness.py  — orchestrator wrapping all 5 gates
# ===========================================================================

#: produce() runs the worker and returns (claimed_summary, artifacts). It is the
#: only thing that "does the work"; the harness never trusts its summary — it
#: verifies the artifacts. Injected so tests supply deterministic producers.
Produce = Callable[[], Tuple[str, List[Artifact]]]

#: The rubric gate is injected as a (score_fn, threshold_fn) pair.
RubricFns = Tuple[
    Callable[[TaskType, List[Artifact], List[str]], float],
    Callable[[TaskType], float],
]


def enforce(
    contract: TaskContract,
    produce: Produce,
    *,
    verify_fn: Callable[..., List[VerifyResult]],
    antisim_fn: Callable[[str, List[Artifact]], List[str]],
    review_fn: Callable[..., List[ReviewResult]],
    rubric_fns: RubricFns,
    max_revise: int = 2,
) -> HarnessOutcome:
    """Orchestrate the 5 gates in order and return a full HarnessOutcome.

    Order per attempt:
      Gate 1  contract already built (validated on entry).
      Gate 2  call produce(); run verify_fn on the artifacts. Any unobserved
              criterion => this attempt is NOT_DONE.
      Gate 4  run antisim_fn on the claimed summary; any unbacked-sim flag =>
              NOT_DONE.
      Gate 3  run review_fn over the real artifacts; a REJECT verdict, or a
              REVISE, feeds the loop decision.
      Gate 5  score with rubric_fns; below threshold => REVISE.
    A NOT_DONE / REVISE / below-threshold result triggers a bounded revision
    (up to max_revise). Exhausting revisions escalates the review tier once,
    then PARKs for a human. All gates passing => ACCEPTED. Every decision is
    appended to outcome.trail. Dependencies are injected so the harness is
    testable without any real model or network.
    """
    raise NotImplementedError


# ===========================================================================
# router.py  — worker selection + circuit breaker + rate limiting
# ===========================================================================


class RateLimiter:
    """A bounded-concurrency semaphore wrapper for model calls."""

    def __init__(self, max_concurrent: int) -> None:
        """Create a limiter allowing `max_concurrent` simultaneous holders.

        Backed by a threading.Semaphore. `max_concurrent` must be >= 1.
        """
        raise NotImplementedError

    def acquire(self, *, timeout: Optional[float] = None) -> bool:
        """Acquire a slot, blocking up to `timeout` seconds.

        Returns True on success, False if the timeout elapsed first. Never
        blocks forever when a timeout is given.
        """
        raise NotImplementedError

    def release(self) -> None:
        """Release a previously acquired slot back to the pool."""
        raise NotImplementedError

    def __enter__(self) -> "RateLimiter":
        """Context-manager acquire (blocking, no timeout)."""
        raise NotImplementedError

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Context-manager release."""
        raise NotImplementedError


class Router:
    """Picks a worker model, and opens a per-model circuit breaker on failures.

    Holds the registered fleet, a per-model failure count / open-circuit state,
    and a shared RateLimiter. Workers are tried in registration order; a model
    whose breaker is open is skipped until it cools down.
    """

    def __init__(self, *, max_concurrent: int = 4, failure_threshold: int = 3) -> None:
        """Create an empty router.

        max_concurrent    — bound passed to the internal RateLimiter.
        failure_threshold — consecutive failures before a model's circuit opens.
        """
        raise NotImplementedError

    def register(self, spec: ModelSpec) -> None:
        """Add a ModelSpec to the fleet. Order matters: earlier = higher priority."""
        raise NotImplementedError

    def pick_worker(self) -> Optional[ModelSpec]:
        """Return the highest-priority available worker, or None if all open.

        Skips models whose role is not 'worker' and any worker whose circuit is
        currently open. Returns None when no worker is available so the caller
        can PARK rather than loop forever.
        """
        raise NotImplementedError

    def on_success(self, model_id: str) -> None:
        """Record a success: reset the model's failure count and close its circuit."""
        raise NotImplementedError

    def on_failure(self, model_id: str) -> None:
        """Record a failure: increment the count and open the circuit at threshold."""
        raise NotImplementedError


# ===========================================================================
# queue.py  — durable job queue (sqlite)
# ===========================================================================


class JobQueue:
    """A crash-safe SQLite-backed job queue with claim/complete/park + watchdog.

    Jobs move through states: queued -> claimed -> (done | parked). A claim
    stamps a lease/heartbeat; the watchdog requeues jobs whose lease expired
    (worker died mid-run) so no task is silently lost.
    """

    def __init__(self, db_path: str, *, lease_seconds: int = 300) -> None:
        """Open/create the SQLite queue at `db_path` and ensure the schema.

        lease_seconds — how long a claim is valid before the watchdog may
        requeue it. Uses stdlib sqlite3 only; no ORM.
        """
        raise NotImplementedError

    def enqueue(self, task_id: str, payload: str) -> None:
        """Insert a new queued job. `payload` is opaque JSON-ish text.

        Idempotent on task_id: re-enqueuing an existing id is a no-op (or a
        documented conflict), never a duplicate active job.
        """
        raise NotImplementedError

    def claim(self, worker_id: str) -> Optional[Tuple[str, str]]:
        """Atomically claim the oldest queued job for `worker_id`.

        Returns (task_id, payload) and stamps a lease, or None if the queue is
        empty. The claim is atomic so two workers never take the same job.
        """
        raise NotImplementedError

    def complete(self, task_id: str, result: str) -> None:
        """Mark a claimed job done and store its result text."""
        raise NotImplementedError

    def park(self, task_id: str, reason: str) -> None:
        """Mark a job parked for a human, recording why (the harness trail)."""
        raise NotImplementedError

    def watchdog_requeue(self) -> int:
        """Requeue every claimed job whose lease has expired. Returns the count.

        This is the crash-recovery mechanism: a worker that died mid-job leaves
        its claim to expire, and the next watchdog pass returns it to 'queued'.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        raise NotImplementedError
