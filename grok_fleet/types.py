"""grok_fleet.types — FROZEN shared vocabulary for the Grok Quality Fleet.

This module is the single source of truth for every dataclass, enum, and type
alias that crosses a module boundary. Implementers of contract / verify / review
/ antisim / rubric / harness / router / queue MUST import from here and MUST NOT
redefine these shapes locally.

Design rules honored here:
- Python 3.9+ only. Uses `from __future__ import annotations` so that the
  PEP 604 `X | Y` and builtin-generic hints (`list[...]`, `dict[...]`) are legal
  as *strings* under 3.9. Do not evaluate these annotations at runtime on 3.9.
- stdlib ONLY. No third-party imports. No network. No live model calls.
- Value objects are frozen where sensible (results, specs, criteria). The
  running accumulator `HarnessOutcome` is intentionally mutable so the harness
  can append to its trail/lists as gates execute.
- A model call is represented ONLY by the injected `Caller` alias. Nothing in
  this package may reach a real API; tests pass fakes.

Literal string-set contracts (kept as `Literal` for machine-checkability):
- AcceptanceCriterion.kind : one of CRITERION_KINDS
- TaskContract.task_type    : one of TASK_TYPES
- ModelSpec.role            : one of MODEL_ROLES
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:  # Literal is stdlib on 3.8+, but guard defensively.
    from typing import Literal
except ImportError:  # pragma: no cover - 3.9+ always has it
    Literal = None  # type: ignore[assignment]

__all__ = [
    # literal-set string constants
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
]

# ---------------------------------------------------------------------------
# Literal string-set contracts (frozen vocabularies)
# ---------------------------------------------------------------------------

#: Every acceptance-criterion kind the verifier knows how to RE-RUN itself.
CRITERION_KINDS = (
    "pytest",
    "file_exists",
    "cmd_zero_exit",
    "json_schema",
    "min_citations",
    "regex_present",
)

#: Task categories. Rubric thresholds and default criteria key off this.
TASK_TYPES = ("code", "research", "content")

#: The three roles a model can hold in the fleet.
MODEL_ROLES = ("worker", "reviewer_fast", "reviewer_trusted")

# Literal aliases for static checkers. At runtime these are plain typing objects.
CriterionKind = Literal[
    "pytest",
    "file_exists",
    "cmd_zero_exit",
    "json_schema",
    "min_citations",
    "regex_present",
]
TaskType = Literal["code", "research", "content"]
ModelRole = Literal["worker", "reviewer_fast", "reviewer_trusted"]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Verdict(Enum):
    """A reviewer's call on the artifacts.

    APPROVE — artifacts satisfy the contract; ship.
    REVISE  — fixable gaps; send back for a bounded revision.
    REJECT  — fundamentally fails; do not revise blindly, escalate/park.
    """

    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class OutcomeStatus(Enum):
    """Terminal (or interim) status of a harness run.

    ACCEPTED — all 5 gates passed; deliverable is trustworthy.
    PARKED   — quality bar not met after bounded revisions; surfaced to a human
               with the full trail.
    NOT_DONE — a hard gate failed in a way that is *not* a quality nuance: an
               unverifiable claim (gate 2), an unbacked simulation (gate 4).
               Used as an interim signal inside a revision loop, and as a
               terminal status when produce() cannot yield observable work.
    """

    ACCEPTED = "accepted"
    PARKED = "parked"
    NOT_DONE = "not_done"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceCriterion:
    """One machine-checkable success condition.

    kind   — a member of CRITERION_KINDS; selects which verifier runner fires.
    target — the primary argument the runner acts on. Its meaning depends on
             kind:
               pytest         -> path to a test file/dir to run
               file_exists    -> filesystem path that must exist
               cmd_zero_exit  -> shell command that must exit 0
               json_schema    -> the artifact ref whose content must validate
               min_citations  -> the artifact ref whose text must contain >= N
                                 resolvable citations
               regex_present  -> the artifact ref whose text must match a regex
    extra  — kind-specific knobs, e.g.:
               json_schema    -> {"schema": {...}}
               min_citations  -> {"min": 3}
               regex_present  -> {"pattern": r"..."}
               cmd_zero_exit  -> {"cwd": "...", "timeout": 60}
             Kept as a plain dict so criteria round-trip cleanly to/from JSON.
    """

    kind: CriterionKind
    target: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskContract:
    """Gate 1: the explicit deliverable + its acceptance criteria.

    task_id     — stable unique id (also the queue/job key).
    deliverable — plain-language statement of what "done" produces.
    criteria    — the checkable conditions; ALL must be observed to pass gate 2.
    task_type   — a member of TASK_TYPES; drives rubric threshold + defaults.
    """

    task_id: str
    deliverable: str
    criteria: List[AcceptanceCriterion]
    task_type: TaskType


@dataclass
class Artifact:
    """A concrete piece of produced work the gates operate on.

    kind     — free-form label, e.g. "file", "text", "patch", "report",
               "test_output", "citation_list".
    ref      — a locator: a filesystem path, a URL, or a synthetic id. May be
               empty for pure-inline artifacts.
    content  — the inline body when applicable (source text, report prose).
               May be empty when the artifact lives entirely at `ref`.
    observed — tri-state proof flag set by the VERIFIER, never by the model:
                 None  -> not yet checked
                 True  -> harness independently observed it
                 False -> harness looked and it was not there / did not pass
             Mutable by design: the verifier stamps it in place.
    """

    kind: str
    ref: str = ""
    content: str = ""
    observed: Optional[bool] = None


@dataclass(frozen=True)
class VerifyResult:
    """Gate 2 output for a single criterion — the anti-bluff record.

    criterion — the AcceptanceCriterion that was re-run.
    observed  — True ONLY if the harness itself observed success. A model's
                claim that it passed is irrelevant here.
    detail    — human-readable evidence/error (test summary, stat result,
                citation count, mismatch reason).
    """

    criterion: AcceptanceCriterion
    observed: bool
    detail: str


@dataclass(frozen=True)
class ReviewResult:
    """Gate 3 output from one reviewer pass.

    tier    — which reviewer produced this: "fast" (free every-output tier) or
              "trusted" (local tier fired on flag/tie). Free-form but these two
              are canonical.
    model   — the model id that reviewed (from ModelSpec.id).
    verdict — a Verdict enum member.
    score   — reviewer's 0.0..1.0 quality score for this pass.
    reasons — refute-first bullet points; why it fails, or why it holds up.
    """

    tier: str
    model: str
    verdict: Verdict
    score: float
    reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelSpec:
    """Registration record for one fleet model.

    id       — provider model id, e.g. "grok-4.20-reasoning", "gemini-2.5-flash",
               "deepseek-v4-pro", or a local tag.
    role     — a member of MODEL_ROLES.
    is_local — True for locally-hosted trusted reviewers / last-resort workers.
               The router prefers cloud workers but escalates ties to a local
               trusted reviewer.
    """

    id: str
    role: ModelRole
    is_local: bool = False


@dataclass
class HarnessOutcome:
    """The full trail of a harness run. MUTABLE accumulator.

    The harness appends to `artifacts`, `verifications`, `reviews`, and `trail`
    as each gate executes, then sets the terminal `status`.

    status        — ACCEPTED | PARKED | NOT_DONE (OutcomeStatus).
    contract      — the TaskContract this run enforced.
    artifacts     — every Artifact produced across attempts (observed-stamped).
    verifications — every VerifyResult from gate 2 across attempts.
    reviews       — every ReviewResult from gate 3 across attempts.
    revise_count  — how many bounded REVISE loops were spent (0..max_revise).
    trail         — ordered plain-language log lines, one per gate decision, so
                    a human reading a PARKED outcome sees exactly what happened.
    """

    status: OutcomeStatus
    contract: TaskContract
    artifacts: List[Artifact] = field(default_factory=list)
    verifications: List[VerifyResult] = field(default_factory=list)
    reviews: List[ReviewResult] = field(default_factory=list)
    revise_count: int = 0
    trail: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Injected model-call boundary. Signature: (model_id, prompt) -> raw_text.
#: This is the ONLY way the package talks to a "model". It is always injected;
#: production wires a real client here, tests wire a deterministic fake. Nothing
#: in grok_fleet may construct a real network client itself.
Caller = Callable[[str, str], str]
