"""grok_fleet — Grok Quality Fleet (GQF).

Pure core package (Python 3.9+, stdlib + pytest only). NO Hermes imports, NO
live network, NO live model calls. Model calls cross the boundary ONLY through
the injected `Caller` alias.

Public surface = the frozen shared types (types.py) + the frozen public
signatures (interfaces.py). Implementers fill interface bodies in their own
modules against these frozen shapes.

The 5 gates the harness orchestrates in order:
  1 Task Contract         — explicit deliverable + machine-checkable criteria.
  2 Proof-of-Work         — harness RE-RUNS the objective check itself.
  3 Adversarial Review    — reviewers get the ACTUAL artifacts, refute-first.
  4 Anti-Simulation       — unbacked hedge/sim language => NOT_DONE.
  5 Quality-First         — rubric must clear threshold; else bounded REVISE.
Outcome: ACCEPTED or PARKED (surfaced to a human with the full trail).
"""
from __future__ import annotations

__version__ = "0.1.0"

# --- Frozen shared types -----------------------------------------------------
from .types import (
    CRITERION_KINDS,
    MODEL_ROLES,
    TASK_TYPES,
    AcceptanceCriterion,
    Artifact,
    Caller,
    CriterionKind,
    HarnessOutcome,
    ModelRole,
    ModelSpec,
    OutcomeStatus,
    ReviewResult,
    TaskContract,
    TaskType,
    Verdict,
    VerifyResult,
)

# --- Frozen public signatures (stubs until implemented) ----------------------
from .interfaces import (
    JobQueue,
    Produce,
    RateLimiter,
    Router,
    RubricFns,
    Runner,
    build_contract,
    cmd_zero_exit,
    count_citations,
    enforce,
    file_exists,
    parse_criteria,
    review,
    run_pytest,
    scan,
    score,
    threshold,
    verify,
)

__all__ = [
    "__version__",
    # literal-set constants
    "CRITERION_KINDS",
    "TASK_TYPES",
    "MODEL_ROLES",
    "CriterionKind",
    "TaskType",
    "ModelRole",
    # enums
    "Verdict",
    "OutcomeStatus",
    # value objects
    "AcceptanceCriterion",
    "TaskContract",
    "Artifact",
    "VerifyResult",
    "ReviewResult",
    "ModelSpec",
    "HarnessOutcome",
    # type aliases
    "Caller",
    "Runner",
    "Produce",
    "RubricFns",
    # contract (Gate 1)
    "build_contract",
    "parse_criteria",
    # verify (Gate 2)
    "verify",
    "run_pytest",
    "file_exists",
    "cmd_zero_exit",
    "count_citations",
    # antisim (Gate 4)
    "scan",
    # rubric (Gate 5)
    "score",
    "threshold",
    # review (Gate 3)
    "review",
    # harness
    "enforce",
    # router
    "Router",
    "RateLimiter",
    # queue
    "JobQueue",
]
