"""Tests for grok_fleet.harness — the 5-gate orchestrator.

All dependencies (produce / verify / antisim / review / rubric) are injected as
deterministic fakes, so these tests never touch a real model or the network.
They exercise:
  - the full ACCEPT path (all 5 gates pass first try),
  - the BLUFF-PARK path (gate 2 never observes work => NOT_DONE => PARKED),
  - the REVISE-THEN-ACCEPT path (fails once, improves, then passes),
  - gate-4 anti-simulation failure,
  - gate-3 review REJECT/REVISE failure,
  - escalation firing exactly once after revisions are exhausted,
  - accept-via-escalation,
  - empty-verification hard fail,
  - produce() raising,
  - trail ordering + evidence accumulation.

AAA pattern (Arrange-Act-Assert) throughout.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pytest

from grok_fleet.harness import enforce
from grok_fleet.types import (
    AcceptanceCriterion,
    Artifact,
    HarnessOutcome,
    OutcomeStatus,
    ReviewResult,
    TaskContract,
    Verdict,
    VerifyResult,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _contract(task_type: str = "code") -> TaskContract:
    """A minimal valid contract with one pytest criterion."""
    crit = AcceptanceCriterion(kind="pytest", target="tests/test_thing.py")
    return TaskContract(
        task_id="T-1",
        deliverable="ship a working thing with a passing test",
        criteria=[crit],
        task_type=task_type,  # type: ignore[arg-type]
    )


def _artifact(observed: Optional[bool] = True) -> Artifact:
    return Artifact(kind="file", ref="thing.py", content="print('hi')", observed=observed)


# --- injectable fakes -------------------------------------------------------


def _produce_ok() -> Tuple[str, List[Artifact]]:
    """A grounded producer: real artifact, no hedge words."""
    return "Wrote thing.py and ran the test; it passes.", [_artifact(observed=True)]


def _verify_pass(contract: TaskContract, artifacts: List[Artifact]) -> List[VerifyResult]:
    """Every criterion observed True."""
    return [
        VerifyResult(criterion=c, observed=True, detail="1 passed")
        for c in contract.criteria
    ]


def _verify_fail(contract: TaskContract, artifacts: List[Artifact]) -> List[VerifyResult]:
    """Every criterion observed False — the bluff case."""
    return [
        VerifyResult(criterion=c, observed=False, detail="test not found / did not pass")
        for c in contract.criteria
    ]


def _antisim_clean(text: str, artifacts: List[Artifact]) -> List[str]:
    return []


def _antisim_flag(text: str, artifacts: List[Artifact]) -> List[str]:
    return ["'would' at 'I would run the test' not backed by an artifact"]


def _review_approve(
    contract: TaskContract, artifacts: List[Artifact], *, escalate: bool = False
) -> List[ReviewResult]:
    tier = "trusted" if escalate else "fast"
    return [ReviewResult(tier=tier, model="fake", verdict=Verdict.APPROVE, score=0.95, reasons=["holds up"])]


def _review_revise(
    contract: TaskContract, artifacts: List[Artifact], *, escalate: bool = False
) -> List[ReviewResult]:
    tier = "trusted" if escalate else "fast"
    return [ReviewResult(tier=tier, model="fake", verdict=Verdict.REVISE, score=0.4, reasons=["missing edge case"])]


def _review_reject(
    contract: TaskContract, artifacts: List[Artifact], *, escalate: bool = False
) -> List[ReviewResult]:
    tier = "trusted" if escalate else "fast"
    return [ReviewResult(tier=tier, model="fake", verdict=Verdict.REJECT, score=0.1, reasons=["fundamentally wrong"])]


def _rubric_high() -> Tuple[Callable[..., float], Callable[..., float]]:
    return (lambda tt, arts, reasons: 0.9), (lambda tt: 0.7)


def _rubric_low() -> Tuple[Callable[..., float], Callable[..., float]]:
    return (lambda tt, arts, reasons: 0.3), (lambda tt: 0.7)


# ===========================================================================
# 1. FULL ACCEPT PATH
# ===========================================================================


def test_full_accept_path_all_gates_pass_first_try():
    # Arrange
    contract = _contract()

    # Act
    outcome = enforce(
        contract,
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,
        rubric_fns=_rubric_high(),
    )

    # Assert
    assert isinstance(outcome, HarnessOutcome)
    assert outcome.status is OutcomeStatus.ACCEPTED
    assert outcome.revise_count == 0
    assert len(outcome.verifications) == 1 and outcome.verifications[0].observed is True
    assert len(outcome.reviews) == 1 and outcome.reviews[0].verdict is Verdict.APPROVE
    # trail is ordered gate1 -> gate2 -> gate4 -> gate3 -> gate5 -> ACCEPTED
    joined = "\n".join(outcome.trail)
    assert "GATE 1" in joined
    assert outcome.trail.index(next(t for t in outcome.trail if "GATE 2" in t)) < \
           outcome.trail.index(next(t for t in outcome.trail if "GATE 3" in t))
    assert outcome.trail[-1].startswith("ACCEPTED")


# ===========================================================================
# 2. BLUFF -> PARK PATH  (gate 2 never observes the claimed work)
# ===========================================================================


def test_bluff_park_path_unverifiable_work_parks():
    # Arrange: producer CLAIMS success but verify never observes it.
    contract = _contract()

    def _produce_bluff() -> Tuple[str, List[Artifact]]:
        return "All tests pass, everything works!", [_artifact(observed=None)]

    # Act
    outcome = enforce(
        contract,
        _produce_bluff,
        verify_fn=_verify_fail,          # <-- harness cannot OBSERVE success
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,       # would approve, but never reached
        rubric_fns=_rubric_high(),
        max_revise=2,
    )

    # Assert: bluff is caught at gate 2 every attempt -> PARKED, never ACCEPTED.
    assert outcome.status is OutcomeStatus.PARKED
    assert outcome.revise_count == 2  # spent both bounded revisions
    # gate 2 failed => review should never have run (short-circuit)
    assert outcome.reviews == []
    joined = "\n".join(outcome.trail)
    assert "GATE 2 (proof) FAIL" in joined
    assert "NOT_DONE" in joined
    assert outcome.trail[-1].startswith("PARKED")
    assert "escalat" in joined.lower()  # escalation was attempted once


def test_bluff_empty_verification_set_is_hard_fail():
    # Arrange: verifier returns NO results at all (nothing checkable observed).
    contract = _contract()

    def _verify_empty(c: TaskContract, a: List[Artifact]) -> List[VerifyResult]:
        return []

    # Act
    outcome = enforce(
        contract,
        _produce_ok,
        verify_fn=_verify_empty,
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,
        rubric_fns=_rubric_high(),
        max_revise=0,
    )

    # Assert: empty observation set can never ACCEPT.
    assert outcome.status is OutcomeStatus.PARKED
    assert "no criteria observed" in "\n".join(outcome.trail)


# ===========================================================================
# 3. REVISE -> ACCEPT PATH  (fails once on quality, improves, then accepts)
# ===========================================================================


def test_revise_then_accept_via_improving_quality():
    # Arrange: a stateful rubric that scores low the first time, high after.
    calls = {"n": 0}

    def _score_improving(tt, arts, reasons) -> float:
        calls["n"] += 1
        return 0.3 if calls["n"] == 1 else 0.9  # low first, high on revision

    rubric = (_score_improving, (lambda tt: 0.7))

    # Act
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,
        rubric_fns=rubric,
        max_revise=2,
    )

    # Assert: first attempt fails gate 5, second attempt (revise 1) accepts.
    assert outcome.status is OutcomeStatus.ACCEPTED
    assert outcome.revise_count == 1
    joined = "\n".join(outcome.trail)
    assert "GATE 5 (quality) FAIL" in joined
    assert "REVISE 1/2" in joined
    assert outcome.trail[-1].startswith("ACCEPTED")


def test_revise_then_accept_via_improving_verification():
    # Arrange: verify fails first attempt, passes on the revision.
    calls = {"n": 0}

    def _verify_improving(c: TaskContract, a: List[Artifact]) -> List[VerifyResult]:
        calls["n"] += 1
        observed = calls["n"] > 1
        return [VerifyResult(criterion=c.criteria[0], observed=observed, detail="x")]

    # Act
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_improving,
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,
        rubric_fns=_rubric_high(),
        max_revise=2,
    )

    # Assert
    assert outcome.status is OutcomeStatus.ACCEPTED
    assert outcome.revise_count == 1
    assert "GATE 2 (proof) FAIL" in "\n".join(outcome.trail)


# ===========================================================================
# 4. GATE 4 — anti-simulation
# ===========================================================================


def test_gate4_unbacked_simulation_language_blocks():
    # Arrange: verify passes, but the summary hedges ("I would run the test").
    contract = _contract()

    # Act
    outcome = enforce(
        contract,
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_flag,   # <-- flags hedge language
        review_fn=_review_approve,
        rubric_fns=_rubric_high(),
        max_revise=1,
    )

    # Assert: gate 4 is a hard NOT_DONE; review never runs; ends PARKED.
    assert outcome.status is OutcomeStatus.PARKED
    assert outcome.reviews == []
    joined = "\n".join(outcome.trail)
    assert "GATE 4 (anti-sim) FAIL" in joined
    assert "NOT_DONE" in joined


# ===========================================================================
# 5. GATE 3 — adversarial review verdicts
# ===========================================================================


def test_gate3_reject_parks_after_escalation():
    # Arrange: reviewer REJECTs on every pass, including escalation.
    contract = _contract()

    # Act
    outcome = enforce(
        contract,
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_reject,
        rubric_fns=_rubric_high(),
        max_revise=1,
    )

    # Assert
    assert outcome.status is OutcomeStatus.PARKED
    joined = "\n".join(outcome.trail)
    assert "GATE 3 (review)" in joined and "reject" in joined.lower()
    # gate 5 never reached because review blocked
    assert "GATE 5" not in joined


def test_gate3_revise_then_review_approves_on_escalation():
    # Arrange: fast tier REVISEs on normal attempts; escalated tier APPROVEs.
    def _review_escalation_sensitive(
        c: TaskContract, a: List[Artifact], *, escalate: bool = False
    ) -> List[ReviewResult]:
        if escalate:
            return _review_approve(c, a, escalate=True)
        return _review_revise(c, a, escalate=False)

    # Act: max_revise=1 -> 2 normal attempts REVISE, then escalation approves.
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_escalation_sensitive,
        rubric_fns=_rubric_high(),
        max_revise=1,
    )

    # Assert: accepted specifically via the escalated review.
    assert outcome.status is OutcomeStatus.ACCEPTED
    joined = "\n".join(outcome.trail)
    assert "escalating review tier once" in joined
    assert "GATE 3 (review) (escalated) pass" in joined
    assert outcome.trail[-1] == "ACCEPTED after escalated review — all 5 gates passed."


# ===========================================================================
# 6. ESCALATION mechanics / boundedness
# ===========================================================================


def test_escalation_fires_exactly_once_and_review_sees_escalate_flag():
    # Arrange: record every escalate flag review_fn is called with.
    seen: List[bool] = []

    def _review_recording(
        c: TaskContract, a: List[Artifact], *, escalate: bool = False
    ) -> List[ReviewResult]:
        seen.append(escalate)
        return _review_revise(c, a, escalate=escalate)  # always REVISE -> forces full loop

    # Act: max_revise=2 -> attempts 0,1,2 (escalate False) + 1 escalated (True).
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_recording,
        rubric_fns=_rubric_high(),
        max_revise=2,
    )

    # Assert
    assert outcome.status is OutcomeStatus.PARKED
    # 3 normal attempts (False) + exactly 1 escalated (True)
    assert seen == [False, False, False, True]
    assert seen.count(True) == 1


def test_max_revise_zero_still_escalates_once_then_parks():
    # Arrange: max_revise=0 -> exactly 1 normal attempt + 1 escalation.
    attempts: List[bool] = []

    def _review_count(
        c: TaskContract, a: List[Artifact], *, escalate: bool = False
    ) -> List[ReviewResult]:
        attempts.append(escalate)
        return _review_revise(c, a, escalate=escalate)

    # Act
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_count,
        rubric_fns=_rubric_high(),
        max_revise=0,
    )

    # Assert
    assert outcome.status is OutcomeStatus.PARKED
    assert attempts == [False, True]  # 1 normal + 1 escalated
    assert outcome.revise_count == 0


def test_negative_max_revise_is_clamped_to_zero():
    # Arrange / Act: negative should behave like 0 (1 normal + escalation).
    attempts: List[bool] = []

    def _review_count(
        c: TaskContract, a: List[Artifact], *, escalate: bool = False
    ) -> List[ReviewResult]:
        attempts.append(escalate)
        return _review_revise(c, a, escalate=escalate)

    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_count,
        rubric_fns=_rubric_high(),
        max_revise=-5,
    )

    # Assert
    assert outcome.status is OutcomeStatus.PARKED
    assert attempts == [False, True]


# ===========================================================================
# 7. Robustness — injected callables raising
# ===========================================================================


def test_produce_raising_is_caught_not_propagated():
    # Arrange
    def _produce_boom() -> Tuple[str, List[Artifact]]:
        raise RuntimeError("worker crashed")

    # Act
    outcome = enforce(
        _contract(),
        _produce_boom,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,
        rubric_fns=_rubric_high(),
        max_revise=0,
    )

    # Assert: crash is absorbed into the trail, ends PARKED (not an exception).
    assert outcome.status is OutcomeStatus.PARKED
    assert "produce() raised" in "\n".join(outcome.trail)


def test_verify_fn_raising_is_caught():
    # Arrange
    def _verify_boom(c: TaskContract, a: List[Artifact]) -> List[VerifyResult]:
        raise OSError("subprocess exploded")

    # Act
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_boom,
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,
        rubric_fns=_rubric_high(),
        max_revise=0,
    )

    # Assert
    assert outcome.status is OutcomeStatus.PARKED
    assert "verify raised" in "\n".join(outcome.trail)


def test_review_fn_raising_is_caught():
    # Arrange
    def _review_boom(
        c: TaskContract, a: List[Artifact], *, escalate: bool = False
    ) -> List[ReviewResult]:
        raise RuntimeError("reviewer down")

    # Act
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_boom,
        rubric_fns=_rubric_high(),
        max_revise=0,
    )

    # Assert
    assert outcome.status is OutcomeStatus.PARKED
    assert "review raised" in "\n".join(outcome.trail)


def test_rubric_raising_is_caught():
    # Arrange
    def _score_boom(tt, arts, reasons) -> float:
        raise ValueError("bad score")

    # Act
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,
        rubric_fns=(_score_boom, (lambda tt: 0.7)),
        max_revise=0,
    )

    # Assert
    assert outcome.status is OutcomeStatus.PARKED
    assert "rubric raised" in "\n".join(outcome.trail)


# ===========================================================================
# 8. Model-agnostic: identical path for any worker id
# ===========================================================================


@pytest.mark.parametrize("task_type", ["code", "research", "content"])
def test_same_gate_path_for_every_task_type(task_type):
    # Arrange / Act
    outcome = enforce(
        _contract(task_type),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_approve,
        rubric_fns=_rubric_high(),
    )

    # Assert: the harness itself is type/model-agnostic — always accepts here.
    assert outcome.status is OutcomeStatus.ACCEPTED
    assert outcome.contract.task_type == task_type


# ===========================================================================
# 9. Evidence accumulation across attempts
# ===========================================================================


def test_evidence_accumulates_every_attempt_not_just_last():
    # Arrange: always REVISE at review so every attempt runs fully through g3.
    def _review_always_revise(
        c: TaskContract, a: List[Artifact], *, escalate: bool = False
    ) -> List[ReviewResult]:
        return _review_revise(c, a, escalate=escalate)

    # Act: 1 normal + 1 escalated attempt, each produces 1 artifact + 1 verify + 1 review.
    outcome = enforce(
        _contract(),
        _produce_ok,
        verify_fn=_verify_pass,
        antisim_fn=_antisim_clean,
        review_fn=_review_always_revise,
        rubric_fns=_rubric_high(),
        max_revise=0,
    )

    # Assert: evidence from BOTH attempts is retained.
    assert outcome.status is OutcomeStatus.PARKED
    assert len(outcome.artifacts) == 2       # 1 per attempt (normal + escalated)
    assert len(outcome.verifications) == 2
    assert len(outcome.reviews) == 2
