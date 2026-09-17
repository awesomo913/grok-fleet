"""grok_fleet.harness — the orchestrator that wraps all 5 gates in order.

This is the conductor of the Grok Quality Fleet. It takes ONE ``produce()``
call (the only thing that "does work") and drives its output through the five
gates, in order, per attempt:

  Gate 1  Task Contract      — validated on entry (built by contract.py). We log
                               it; a malformed contract never reaches us.
  Gate 2  Proof-of-Work      — call produce(); RE-RUN every acceptance criterion
                               via ``verify_fn``. Any criterion we cannot OBSERVE
                               as passing => this attempt is NOT_DONE. The model's
                               own success claim is irrelevant here.
  Gate 4  Anti-Simulation    — ``antisim_fn`` scans the claimed summary for
                               hedge/sim language not backed by an artifact. Any
                               flag => NOT_DONE.
  Gate 3  Adversarial Review — ``review_fn`` re-reviews the ACTUAL artifacts,
                               refute-first. A REJECT or REVISE verdict feeds the
                               loop decision.
  Gate 5  Quality-First      — score with ``rubric_fns``; below the per-type
                               threshold => REVISE.

A NOT_DONE / REVISE / below-threshold result triggers a bounded revision (up to
``max_revise``). When revisions are exhausted, the harness escalates the review
tier ONCE (re-running review with ``escalate=True``); if that still does not
yield an ACCEPTED result, the outcome is PARKED for a human — with the full,
ordered ``trail`` explaining exactly what happened at every gate.

Every dependency (verify / antisim / review / rubric) is INJECTED so this
module is fully testable with deterministic fakes — no real model, no network,
no live call ever originates here. The path is identical for a grok primary
worker and a free-backup worker; the harness is model-agnostic by construction.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from .types import (
    Artifact,
    HarnessOutcome,
    OutcomeStatus,
    ReviewResult,
    TaskContract,
    Verdict,
    VerifyResult,
)

# ---------------------------------------------------------------------------
# Injected-dependency aliases (mirror grok_fleet.interfaces, kept local so this
# module never has to import sibling implementation modules).
# ---------------------------------------------------------------------------

#: produce() runs the worker and returns (claimed_summary, artifacts). It is the
#: ONLY thing that does the work; the harness never trusts its summary — it
#: verifies the artifacts. Injected so tests supply deterministic producers.
Produce = Callable[[], Tuple[str, List[Artifact]]]

#: The rubric gate is injected as a (score_fn, threshold_fn) pair.
RubricFns = Tuple[
    Callable[..., float],
    Callable[..., float],
]

log = logging.getLogger(__name__)

__all__ = ["enforce", "Produce", "RubricFns"]


# ---------------------------------------------------------------------------
# Small internal record for the result of running one full gate sweep.
# ---------------------------------------------------------------------------


class _AttemptResult:
    """Outcome of a single attempt through gates 2->5.

    Not a public type — a plain internal carrier so the loop in ``enforce`` can
    branch on what happened without re-deriving it. Kept mutable and simple.
    """

    __slots__ = ("summary", "artifacts", "verifications", "reviews", "reason", "accepted", "hard_fail")

    def __init__(self) -> None:
        self.summary: str = ""
        self.artifacts: List[Artifact] = []
        self.verifications: List[VerifyResult] = []
        self.reviews: List[ReviewResult] = []
        #: plain-language reason this attempt did not pass ("" when it did).
        self.reason: str = ""
        #: True only when all four gates (2,4,3,5) passed this attempt.
        self.accepted: bool = False
        #: True when gate 2 or gate 4 failed — a hard NOT_DONE (not a quality
        #: nuance). Distinguished so the trail can say "bluff/sim" vs "quality".
        self.hard_fail: bool = False


# ---------------------------------------------------------------------------
# Gate helpers — each returns (passed, reason). Pure, no side effects beyond
# what the injected fn does. Kept tiny so enforce() reads like the gate list.
# ---------------------------------------------------------------------------


def _gate2_verify(
    contract: TaskContract,
    artifacts: List[Artifact],
    verify_fn: Callable[..., List[VerifyResult]],
) -> Tuple[bool, List[VerifyResult], str]:
    """Gate 2: re-run every criterion ourselves; ALL must be observed True.

    Returns (passed, verifications, reason). ``passed`` is True only when the
    verifier returned at least one result AND every result.observed is True.
    An empty verification list is treated as a failure (nothing was observed),
    so a producer that yields no checkable work can never sneak an ACCEPTED.
    """
    try:
        verifications = verify_fn(contract, artifacts)
    except Exception as exc:  # noqa: BLE001 — bind+log per workspace rules.
        log.warning("gate2 verify_fn raised for task %s: %s", contract.task_id, exc)
        return False, [], f"verify raised: {exc}"

    if not verifications:
        return False, [], "no criteria observed (empty verification set)"

    unobserved = [v for v in verifications if not v.observed]
    if unobserved:
        names = ", ".join(f"{v.criterion.kind}:{v.criterion.target}" for v in unobserved)
        return False, verifications, f"unobserved criteria: {names}"
    return True, verifications, ""


def _gate4_antisim(
    summary: str,
    artifacts: List[Artifact],
    antisim_fn: Callable[[str, List[Artifact]], List[str]],
) -> Tuple[bool, str]:
    """Gate 4: any unbacked hedge/sim flag on the claimed summary => fail.

    Returns (passed, reason). ``passed`` is True only when the scanner returns
    no flags. Flags are joined into the reason so the trail names them.
    """
    try:
        flags = antisim_fn(summary, artifacts)
    except Exception as exc:  # noqa: BLE001 — bind+log per workspace rules.
        log.warning("gate4 antisim_fn raised: %s", exc)
        return False, f"antisim raised: {exc}"

    if flags:
        return False, "simulation/hedge flags: " + "; ".join(flags)
    return True, ""


def _gate3_review(
    contract: TaskContract,
    artifacts: List[Artifact],
    review_fn: Callable[..., List[ReviewResult]],
    *,
    escalate: bool,
) -> Tuple[bool, List[ReviewResult], str]:
    """Gate 3: adversarial review of the ACTUAL artifacts.

    Returns (approved, reviews, reason). ``approved`` is True only when reviews
    were produced AND none carry a REJECT or REVISE verdict. A REJECT or REVISE
    verdict fails the gate and its reviewer reasons flow into the trail reason.
    An empty review set is a failure (no reviewer signal to trust).
    """
    try:
        reviews = review_fn(contract, artifacts, escalate=escalate)
    except Exception as exc:  # noqa: BLE001 — bind+log per workspace rules.
        log.warning("gate3 review_fn raised for task %s: %s", contract.task_id, exc)
        return False, [], f"review raised: {exc}"

    if not reviews:
        return False, [], "no reviewer verdicts produced"

    blocking = [r for r in reviews if r.verdict in (Verdict.REJECT, Verdict.REVISE)]
    if blocking:
        worst = Verdict.REJECT if any(r.verdict is Verdict.REJECT for r in blocking) else Verdict.REVISE
        why = "; ".join(reason for r in blocking for reason in (r.reasons or [f"{r.tier} {r.verdict.value}"]))
        return False, reviews, f"review {worst.value}: {why}"
    return True, reviews, ""


def _gate5_rubric(
    contract: TaskContract,
    artifacts: List[Artifact],
    reviews: List[ReviewResult],
    rubric_fns: RubricFns,
) -> Tuple[bool, float, float, str]:
    """Gate 5: quality score must clear the per-type threshold.

    Returns (passed, score_val, threshold_val, reason). Flattens every reviewer
    reason into the list handed to the scorer so the rubric can weigh review
    signal. ``passed`` is True only when score >= threshold.
    """
    score_fn, threshold_fn = rubric_fns
    review_reasons: List[str] = [reason for r in reviews for reason in r.reasons]
    try:
        score_val = float(score_fn(contract.task_type, artifacts, review_reasons))
        threshold_val = float(threshold_fn(contract.task_type))
    except Exception as exc:  # noqa: BLE001 — bind+log per workspace rules.
        log.warning("gate5 rubric raised for task %s: %s", contract.task_id, exc)
        return False, 0.0, 1.0, f"rubric raised: {exc}"

    if score_val < threshold_val:
        return False, score_val, threshold_val, (
            f"quality {score_val:.3f} below threshold {threshold_val:.3f}"
        )
    return True, score_val, threshold_val, ""


# ---------------------------------------------------------------------------
# One full attempt through gates 2 -> 4 -> 3 -> 5.
# ---------------------------------------------------------------------------


def _run_attempt(
    contract: TaskContract,
    produce: Produce,
    *,
    verify_fn: Callable[..., List[VerifyResult]],
    antisim_fn: Callable[[str, List[Artifact]], List[str]],
    review_fn: Callable[..., List[ReviewResult]],
    rubric_fns: RubricFns,
    escalate: bool,
    attempt_no: int,
    trail: List[str],
) -> _AttemptResult:
    """Run produce() once and push its output through gates 2,4,3,5 in order.

    Short-circuits on the first failing gate (there is no point reviewing work
    that was never observed, or scoring work a reviewer rejected). Appends one
    plain-language line to ``trail`` per gate decision. Returns an
    ``_AttemptResult`` the caller uses to decide ACCEPTED / revise / PARK.
    """
    res = _AttemptResult()

    # --- produce (the only "work") ---
    try:
        summary, artifacts = produce()
    except Exception as exc:  # noqa: BLE001 — bind+log per workspace rules.
        log.warning("produce() raised on attempt %d for task %s: %s", attempt_no, contract.task_id, exc)
        res.reason = f"produce raised: {exc}"
        res.hard_fail = True
        trail.append(f"attempt {attempt_no}: produce() raised -> NOT_DONE ({exc})")
        return res

    res.summary = summary or ""
    res.artifacts = list(artifacts or [])
    trail.append(
        f"attempt {attempt_no}: produced {len(res.artifacts)} artifact(s), "
        f"summary={len(res.summary)} chars"
    )

    # --- Gate 2: proof-of-work (anti-bluff heart) ---
    ok2, verifications, reason2 = _gate2_verify(contract, res.artifacts, verify_fn)
    res.verifications = verifications
    if not ok2:
        res.reason = reason2
        res.hard_fail = True
        trail.append(f"attempt {attempt_no}: GATE 2 (proof) FAIL -> NOT_DONE ({reason2})")
        return res
    trail.append(f"attempt {attempt_no}: GATE 2 (proof) pass — {len(verifications)} criteria observed")

    # --- Gate 4: anti-simulation ---
    ok4, reason4 = _gate4_antisim(res.summary, res.artifacts, antisim_fn)
    if not ok4:
        res.reason = reason4
        res.hard_fail = True
        trail.append(f"attempt {attempt_no}: GATE 4 (anti-sim) FAIL -> NOT_DONE ({reason4})")
        return res
    trail.append(f"attempt {attempt_no}: GATE 4 (anti-sim) pass — no unbacked hedge/sim language")

    # --- Gate 3: adversarial review ---
    ok3, reviews, reason3 = _gate3_review(
        contract, res.artifacts, review_fn, escalate=escalate
    )
    res.reviews = reviews
    tier_note = " (escalated)" if escalate else ""
    if not ok3:
        res.reason = reason3
        res.hard_fail = False  # review disapproval is a quality nuance, not a bluff
        trail.append(f"attempt {attempt_no}: GATE 3 (review){tier_note} FAIL ({reason3})")
        return res
    trail.append(
        f"attempt {attempt_no}: GATE 3 (review){tier_note} pass — "
        f"{len(reviews)} reviewer(s) APPROVE"
    )

    # --- Gate 5: quality-first threshold ---
    ok5, score_val, threshold_val, reason5 = _gate5_rubric(
        contract, res.artifacts, reviews, rubric_fns
    )
    if not ok5:
        res.reason = reason5
        res.hard_fail = False
        trail.append(f"attempt {attempt_no}: GATE 5 (quality) FAIL ({reason5})")
        return res
    trail.append(
        f"attempt {attempt_no}: GATE 5 (quality) pass — "
        f"score {score_val:.3f} >= threshold {threshold_val:.3f}"
    )

    res.accepted = True
    return res


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


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

    See the module docstring for the gate order. This function owns the loop:
    it runs one attempt, and on any NOT_DONE / REVISE / below-threshold result
    it spends a bounded revision (up to ``max_revise``). When revisions are
    exhausted it escalates the review tier ONCE (a final attempt with
    ``escalate=True``); if that still fails, it PARKs for a human. All gates
    passing => ACCEPTED. Every gate decision is appended to ``outcome.trail``.

    Args:
        contract: the already-validated Gate-1 TaskContract to enforce.
        produce: the injected worker; ``() -> (claimed_summary, artifacts)``.
        verify_fn: Gate 2 — ``(contract, artifacts) -> List[VerifyResult]``.
        antisim_fn: Gate 4 — ``(summary, artifacts) -> List[str]`` (flags).
        review_fn: Gate 3 — ``(contract, artifacts, *, escalate) -> List[ReviewResult]``.
        rubric_fns: Gate 5 — ``(score_fn, threshold_fn)``.
        max_revise: max bounded REVISE loops (>=0). Clamped to 0 if negative.

    Returns:
        A mutable HarnessOutcome accumulator whose ``status`` is ACCEPTED or
        PARKED, with all artifacts / verifications / reviews / trail recorded.
    """
    if max_revise < 0:
        log.warning("max_revise %d < 0 for task %s; clamping to 0", max_revise, contract.task_id)
        max_revise = 0

    outcome = HarnessOutcome(status=OutcomeStatus.NOT_DONE, contract=contract)
    outcome.trail.append(
        f"GATE 1 (contract) pass — task_id={contract.task_id}, type={contract.task_type}, "
        f"{len(contract.criteria)} criteria: {contract.deliverable!r}"
    )

    # attempt 0 = first pass; attempts 1..max_revise = bounded revisions;
    # a final escalated attempt fires once after revisions are exhausted.
    total_normal_attempts = max_revise + 1  # first pass + bounded revisions
    escalated_result: Optional[_AttemptResult] = None

    for attempt_no in range(total_normal_attempts):
        if attempt_no > 0:
            outcome.revise_count = attempt_no
            outcome.trail.append(
                f"REVISE {attempt_no}/{max_revise}: re-running produce() for a bounded revision"
            )

        result = _run_attempt(
            contract,
            produce,
            verify_fn=verify_fn,
            antisim_fn=antisim_fn,
            review_fn=review_fn,
            rubric_fns=rubric_fns,
            escalate=False,
            attempt_no=attempt_no,
            trail=outcome.trail,
        )
        _absorb(outcome, result)

        if result.accepted:
            outcome.status = OutcomeStatus.ACCEPTED
            outcome.trail.append(
                f"ACCEPTED after attempt {attempt_no} — all 5 gates passed."
            )
            return outcome

    # --- revisions exhausted: escalate the review tier ONCE ---
    outcome.trail.append(
        f"revisions exhausted ({max_revise} used); escalating review tier once."
    )
    escalated_result = _run_attempt(
        contract,
        produce,
        verify_fn=verify_fn,
        antisim_fn=antisim_fn,
        review_fn=review_fn,
        rubric_fns=rubric_fns,
        escalate=True,
        attempt_no=total_normal_attempts,
        trail=outcome.trail,
    )
    _absorb(outcome, escalated_result)

    if escalated_result.accepted:
        outcome.status = OutcomeStatus.ACCEPTED
        outcome.trail.append("ACCEPTED after escalated review — all 5 gates passed.")
        return outcome

    # --- nothing cleared the bar: PARK for a human with the full trail ---
    outcome.status = OutcomeStatus.PARKED
    last_reason = escalated_result.reason or "quality bar not met"
    outcome.trail.append(
        f"PARKED for human — bar not met after {max_revise} revision(s) + escalation. "
        f"Last blocker: {last_reason}"
    )
    return outcome


def _absorb(outcome: HarnessOutcome, result: _AttemptResult) -> None:
    """Fold one attempt's artifacts/verifications/reviews into the outcome trail.

    The HarnessOutcome accumulates EVERY attempt's evidence (not just the last)
    so a human reading a PARKED outcome sees the whole history. Idempotent per
    attempt — call once per attempt.
    """
    outcome.artifacts.extend(result.artifacts)
    outcome.verifications.extend(result.verifications)
    outcome.reviews.extend(result.reviews)
