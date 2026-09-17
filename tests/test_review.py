"""Tests for grok_fleet.review — Gate 3 adversarial reviewer pipeline.

All model calls are FAKE Callers (deterministic). No network, no Hermes.
The fakes assert the real artifact text was placed into the review prompt,
proving reviewers see actual artifacts and not a summary.
"""
from __future__ import annotations

from typing import List, Optional

import pytest

from grok_fleet.review import review
from grok_fleet.types import (
    AcceptanceCriterion,
    Artifact,
    ReviewResult,
    TaskContract,
    Verdict,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_contract(task_type: str = "code") -> TaskContract:
    return TaskContract(
        task_id="T-1",
        deliverable="Ship a working widget with tests.",
        criteria=[
            AcceptanceCriterion(kind="pytest", target="tests/test_widget.py"),
            AcceptanceCriterion(kind="file_exists", target="widget.py"),
        ],
        task_type=task_type,  # type: ignore[arg-type]
    )


def make_artifacts(content: str = "def widget():\n    return 42  # UNIQUE_ARTIFACT_TOKEN") -> List[Artifact]:
    return [
        Artifact(kind="file", ref="widget.py", content=content, observed=True),
        Artifact(kind="report", ref="", content="All tests pass locally.", observed=None),
    ]


class RecordingCaller:
    """A fake Caller that records prompts and returns a canned reply.

    Carries ``__gqf_model_id__`` so the review module can stamp/guard on the
    reviewer's model id without breaking the frozen (model_id, prompt) shape.
    """

    def __init__(self, reply: str, *, model_id: str) -> None:
        self.reply = reply
        self.__gqf_model_id__ = model_id
        self.prompts: List[str] = []
        self.model_ids: List[str] = []
        self.call_count = 0

    def __call__(self, model_id: str, prompt: str) -> str:
        self.call_count += 1
        self.model_ids.append(model_id)
        self.prompts.append(prompt)
        return self.reply


class RaisingCaller:
    def __init__(self, *, model_id: str, exc: Exception) -> None:
        self.__gqf_model_id__ = model_id
        self._exc = exc
        self.call_count = 0

    def __call__(self, model_id: str, prompt: str) -> str:
        self.call_count += 1
        raise self._exc


APPROVE_REPLY = "VERDICT: APPROVE\nSCORE: 0.92\nREASONS:\n- Artifacts satisfy the contract.\n- Tests present."
REVISE_REPLY = "VERDICT: REVISE\nSCORE: 0.55\nREASONS:\n- Missing edge-case handling.\n- Docstring absent."
REJECT_REPLY = "VERDICT: REJECT\nSCORE: 0.1\nREASONS:\n- widget.py does not implement the contract."


# ---------------------------------------------------------------------------
# Fast tier always runs; reviewer sees REAL artifacts
# ---------------------------------------------------------------------------


def test_fast_tier_always_runs_and_returns_one_result() -> None:
    fast = RecordingCaller(APPROVE_REPLY, model_id="gemini-fast")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)

    assert len(results) == 1
    assert results[0].tier == "fast"
    assert results[0].model == "gemini-fast"
    assert results[0].verdict is Verdict.APPROVE
    assert fast.call_count == 1


def test_reviewer_receives_actual_artifact_text_in_prompt() -> None:
    fast = RecordingCaller(APPROVE_REPLY, model_id="gemini-fast")
    review(make_contract(), make_artifacts(), fast_caller=fast)

    prompt = fast.prompts[0]
    assert "UNIQUE_ARTIFACT_TOKEN" in prompt  # real code body, not a summary
    assert "All tests pass locally." in prompt  # second artifact body
    assert "widget.py" in prompt  # the ref locator
    assert "ADVERSARIAL" in prompt  # refute-first framing
    assert "REFUTE" in prompt


def test_prompt_includes_contract_deliverable_and_criteria() -> None:
    fast = RecordingCaller(APPROVE_REPLY, model_id="gemini-fast")
    review(make_contract(), make_artifacts(), fast_caller=fast)

    prompt = fast.prompts[0]
    assert "Ship a working widget with tests." in prompt
    assert "pytest" in prompt
    assert "tests/test_widget.py" in prompt


# ---------------------------------------------------------------------------
# Escalation to trusted tier
# ---------------------------------------------------------------------------


def test_no_trusted_call_when_fast_approves_and_not_escalated() -> None:
    fast = RecordingCaller(APPROVE_REPLY, model_id="gemini-fast")
    trusted = RecordingCaller(APPROVE_REPLY, model_id="local-trusted")

    results = review(
        make_contract(), make_artifacts(),
        fast_caller=fast, trusted_caller=trusted,
    )

    assert len(results) == 1
    assert trusted.call_count == 0  # not consulted on a clean approve


def test_trusted_runs_when_fast_flags_revise() -> None:
    fast = RecordingCaller(REVISE_REPLY, model_id="gemini-fast")
    trusted = RecordingCaller(APPROVE_REPLY, model_id="local-trusted")

    results = review(
        make_contract(), make_artifacts(),
        fast_caller=fast, trusted_caller=trusted,
    )

    assert len(results) == 2
    assert results[0].tier == "fast" and results[0].verdict is Verdict.REVISE
    assert results[1].tier == "trusted" and results[1].model == "local-trusted"
    assert trusted.call_count == 1


def test_trusted_runs_when_fast_flags_reject() -> None:
    fast = RecordingCaller(REJECT_REPLY, model_id="gemini-fast")
    trusted = RecordingCaller(APPROVE_REPLY, model_id="local-trusted")

    results = review(
        make_contract(), make_artifacts(),
        fast_caller=fast, trusted_caller=trusted,
    )

    assert len(results) == 2
    assert results[0].verdict is Verdict.REJECT
    assert trusted.call_count == 1


def test_trusted_runs_when_escalate_forced_even_on_approve() -> None:
    fast = RecordingCaller(APPROVE_REPLY, model_id="gemini-fast")
    trusted = RecordingCaller(APPROVE_REPLY, model_id="local-trusted")

    results = review(
        make_contract(), make_artifacts(),
        fast_caller=fast, trusted_caller=trusted, escalate=True,
    )

    assert len(results) == 2
    assert trusted.call_count == 1


def test_escalation_flag_but_no_trusted_caller_returns_only_fast() -> None:
    fast = RecordingCaller(REJECT_REPLY, model_id="gemini-fast")
    results = review(
        make_contract(), make_artifacts(),
        fast_caller=fast, trusted_caller=None, escalate=True,
    )
    # Warranted escalation but no trusted tier available -> just the fast result.
    assert len(results) == 1
    assert results[0].tier == "fast"


def test_trusted_also_sees_real_artifacts() -> None:
    fast = RecordingCaller(REVISE_REPLY, model_id="gemini-fast")
    trusted = RecordingCaller(APPROVE_REPLY, model_id="local-trusted")
    review(
        make_contract(), make_artifacts(),
        fast_caller=fast, trusted_caller=trusted,
    )
    assert "UNIQUE_ARTIFACT_TOKEN" in trusted.prompts[0]


# ---------------------------------------------------------------------------
# Self-review guard (never review your own output)
# ---------------------------------------------------------------------------


def test_self_review_guard_blocks_reviewer_that_produced_artifacts() -> None:
    # Artifacts carry a produced_by marker equal to the fast reviewer's id.
    arts = [
        Artifact(
            kind="file",
            ref="widget.py",
            content="produced_by: gemini-fast\ndef widget():\n    return 1",
            observed=True,
        )
    ]
    fast = RecordingCaller(APPROVE_REPLY, model_id="gemini-fast")

    results = review(make_contract(), arts, fast_caller=fast)

    assert len(results) == 1
    assert results[0].verdict is Verdict.REJECT  # fail-closed placeholder
    assert results[0].score == 0.0
    assert fast.call_count == 0  # the model was never actually called
    assert "self-review" in " ".join(results[0].reasons).lower()


def test_self_review_guard_lets_a_different_reviewer_pass() -> None:
    arts = [
        Artifact(
            kind="file",
            ref="widget.py",
            content="produced_by: some-worker\ndef widget():\n    return 1",
            observed=True,
        )
    ]
    fast = RecordingCaller(APPROVE_REPLY, model_id="gemini-fast")
    results = review(make_contract(), arts, fast_caller=fast)

    assert results[0].verdict is Verdict.APPROVE
    assert fast.call_count == 1


def test_self_review_guard_on_trusted_tier() -> None:
    arts = [
        Artifact(
            kind="file",
            ref="widget.py",
            content="produced_by: local-trusted\ndef widget():\n    return 1",
            observed=True,
        )
    ]
    fast = RecordingCaller(REJECT_REPLY, model_id="gemini-fast")
    trusted = RecordingCaller(APPROVE_REPLY, model_id="local-trusted")

    results = review(
        make_contract(), arts, fast_caller=fast, trusted_caller=trusted,
    )

    assert len(results) == 2
    assert results[1].tier == "trusted"
    assert results[1].verdict is Verdict.REJECT  # blocked, fail-closed
    assert trusted.call_count == 0  # trusted model never called on its own work


# ---------------------------------------------------------------------------
# Robust / fail-closed parsing
# ---------------------------------------------------------------------------


def test_unparseable_reply_fails_closed_to_reject() -> None:
    fast = RecordingCaller("I have no opinion whatsoever about this.", model_id="m")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)

    assert results[0].verdict is Verdict.REJECT
    assert results[0].score == 0.0


def test_empty_reply_fails_closed_to_reject() -> None:
    fast = RecordingCaller("", model_id="m")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)
    assert results[0].verdict is Verdict.REJECT
    assert results[0].score == 0.0


def test_caller_raising_fails_closed_to_reject() -> None:
    fast = RaisingCaller(model_id="m", exc=RuntimeError("boom"))
    results = review(make_contract(), make_artifacts(), fast_caller=fast)
    assert results[0].verdict is Verdict.REJECT
    assert results[0].score == 0.0
    assert "RuntimeError" in " ".join(results[0].reasons)


def test_verdict_word_without_label_is_detected() -> None:
    fast = RecordingCaller("After review I APPROVE this work. Solid.", model_id="m")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)
    assert results[0].verdict is Verdict.APPROVE


def test_verdict_precedence_reject_beats_approve() -> None:
    # No explicit VERDICT: line; both words present -> conservative REJECT wins.
    fast = RecordingCaller("I would normally APPROVE but I must REJECT this.", model_id="m")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)
    assert results[0].verdict is Verdict.REJECT


def test_explicit_verdict_line_beats_stray_words() -> None:
    reply = "There is a risk we might REJECT bad code.\nVERDICT: APPROVE\nSCORE: 0.8"
    fast = RecordingCaller(reply, model_id="m")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)
    assert results[0].verdict is Verdict.APPROVE


# ---------------------------------------------------------------------------
# Score parsing shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("VERDICT: APPROVE\nSCORE: 0.82", 0.82),
        ("VERDICT: APPROVE\nscore = 82%", 0.82),
        ("VERDICT: APPROVE\nquality: 0.5", 0.5),
        ("VERDICT: APPROVE\nrating 7/10", 0.7),
        ("VERDICT: APPROVE\nSCORE: 1.5", 1.0),   # clamps high
        ("VERDICT: APPROVE\nSCORE: -0.4", 0.0),  # clamps low
        ("VERDICT: APPROVE\nSCORE: 95", 0.95),   # bare > 1 read as percent
    ],
)
def test_score_parsing(reply: str, expected: float) -> None:
    fast = RecordingCaller(reply, model_id="m")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)
    assert results[0].score == pytest.approx(expected)


def test_score_synthesized_from_verdict_when_absent() -> None:
    fast = RecordingCaller("VERDICT: REVISE\nREASONS:\n- needs polish", model_id="m")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)
    assert results[0].verdict is Verdict.REVISE
    assert 0.0 < results[0].score < 1.0


# ---------------------------------------------------------------------------
# Reasons parsing
# ---------------------------------------------------------------------------


def test_reasons_bullets_are_extracted() -> None:
    fast = RecordingCaller(REVISE_REPLY, model_id="m")
    trusted = RecordingCaller(APPROVE_REPLY, model_id="t")
    results = review(
        make_contract(), make_artifacts(),
        fast_caller=fast, trusted_caller=trusted,
    )
    reasons = results[0].reasons
    assert any("edge-case" in r for r in reasons)
    assert any("Docstring" in r for r in reasons)
    # bullet markers stripped
    assert all(not r.startswith("-") for r in reasons)


def test_reasons_never_empty_for_a_parsed_verdict() -> None:
    fast = RecordingCaller("VERDICT: APPROVE\nSCORE: 0.9", model_id="m")
    results = review(make_contract(), make_artifacts(), fast_caller=fast)
    assert results[0].reasons  # falls back to a synthesized note


# ---------------------------------------------------------------------------
# Type / contract sanity
# ---------------------------------------------------------------------------


def test_results_are_reviewresult_instances() -> None:
    fast = RecordingCaller(REVISE_REPLY, model_id="m")
    trusted = RecordingCaller(APPROVE_REPLY, model_id="t")
    results = review(
        make_contract(), make_artifacts(),
        fast_caller=fast, trusted_caller=trusted,
    )
    assert all(isinstance(r, ReviewResult) for r in results)
    for r in results:
        assert 0.0 <= r.score <= 1.0
        assert r.tier in ("fast", "trusted")


def test_no_artifacts_still_reviews_fail_closed() -> None:
    fast = RecordingCaller("VERDICT: REJECT\nSCORE: 0\nREASONS:\n- nothing produced", model_id="m")
    results = review(make_contract(), [], fast_caller=fast)
    assert results[0].verdict is Verdict.REJECT
    assert "no artifacts" in fast.prompts[0].lower()


def test_reviewer_without_model_id_attr_gets_tier_fallback_id() -> None:
    # A plain function caller (no __gqf_model_id__) still works; id falls back.
    calls: List[str] = []

    def plain_caller(model_id: str, prompt: str) -> str:
        calls.append(model_id)
        return APPROVE_REPLY

    results = review(make_contract(), make_artifacts(), fast_caller=plain_caller)
    assert results[0].model == "reviewer_fast"
    assert calls == ["reviewer_fast"]
