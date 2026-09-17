"""Tests for grok_fleet.rubric (Gate 5: Quality-First gate).

Covers threshold() per task_type and score() across code (strict),
research (citation-weighted), and content (voice-weighted). Verifies the
score is always clamped to [0,1], is deterministic, penalizes blocking
reviewer reasons, and ignores approving notes. Pure stdlib + pytest.
AAA pattern (Arrange-Act-Assert).
"""
from __future__ import annotations

import pytest

from grok_fleet.rubric import score, threshold
from grok_fleet.types import TASK_TYPES, Artifact


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _art(kind="file", ref="out.txt", content="", observed=None) -> Artifact:
    return Artifact(kind=kind, ref=ref, content=content, observed=observed)


def _long(text_unit="This is a real sentence with substance. ", reps=15) -> str:
    return text_unit * reps


# ---------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_type", list(TASK_TYPES))
def test_threshold_in_range(task_type):
    # Act
    t = threshold(task_type)
    # Assert
    assert 0.0 <= t <= 1.0


def test_threshold_code_is_strictest():
    # Assert — code bar > research bar > content bar
    assert threshold("code") > threshold("research") > threshold("content")


def test_threshold_unknown_type_returns_default_not_zero():
    # Act
    t = threshold("nonsense")
    # Assert — never silently 0 (which would let anything pass)
    assert t > 0.0


# ---------------------------------------------------------------------------
# score — range + determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_type", list(TASK_TYPES))
def test_score_always_in_range(task_type):
    # Arrange
    arts = [_art(observed=True, content=_long())]
    # Act
    s = score(task_type, arts, [])
    # Assert
    assert 0.0 <= s <= 1.0


def test_score_empty_artifacts_is_zero():
    # Act
    s = score("code", [], [])
    # Assert — no work produced => no quality
    assert s == 0.0


def test_score_none_artifacts_is_zero():
    # Act
    s = score("code", None, None)
    # Assert
    assert s == 0.0


def test_score_is_deterministic():
    # Arrange
    arts = [_art(observed=True, content=_long()), _art(observed=False)]
    reasons = ["bug in loop", "missing edge case"]
    # Act
    a = score("code", arts, reasons)
    b = score("code", arts, reasons)
    # Assert
    assert a == b


def test_score_clamps_flood_of_complaints_to_zero():
    # Arrange — one weak artifact, many blocking reasons
    arts = [_art(observed=None)]
    reasons = [f"defect {i}" for i in range(20)]
    # Act
    s = score("code", arts, reasons)
    # Assert — floored, never negative
    assert s == 0.0


# ---------------------------------------------------------------------------
# score — code (strict, observed-driven)
# ---------------------------------------------------------------------------


def test_code_observed_scores_higher_than_unobserved():
    # Arrange
    observed = [_art(observed=True, content=_long())]
    unobserved = [_art(observed=False, content=_long())]
    # Act
    hi = score("code", observed, [])
    lo = score("code", unobserved, [])
    # Assert
    assert hi > lo


def test_code_all_observed_clears_threshold():
    # Arrange — clean, observed, substantial artifacts, no complaints
    arts = [_art(observed=True, content=_long()) for _ in range(2)]
    # Act
    s = score("code", arts, [])
    # Assert
    assert s >= threshold("code")


def test_code_observed_false_fails_threshold():
    # Arrange
    arts = [_art(observed=False, content=_long())]
    # Act
    s = score("code", arts, [])
    # Assert — a verifier that looked and saw failure must not pass strict bar
    assert s < threshold("code")


def test_code_blocking_reason_lowers_score():
    # Arrange
    arts = [_art(observed=True, content=_long())]
    # Act
    clean = score("code", arts, [])
    dinged = score("code", arts, ["off-by-one bug"])
    # Assert
    assert dinged < clean


# ---------------------------------------------------------------------------
# score — research (citation-weighted)
# ---------------------------------------------------------------------------


def test_research_citations_beat_no_citations():
    # Arrange
    cited = [
        _art(
            kind="report",
            ref="report.md",
            content=(
                "Finding one is supported. See https://example.com/a and "
                "https://example.org/b plus https://example.net/c for detail. "
                + _long()
            ),
            observed=True,
        )
    ]
    uncited = [
        _art(
            kind="report",
            ref="report.md",
            content="Just prose, no sources at all. " + _long(),
            observed=True,
        )
    ]
    # Act
    hi = score("research", cited, [])
    lo = score("research", uncited, [])
    # Assert
    assert hi > lo


def test_research_well_cited_clears_threshold():
    # Arrange — 3 resolvable URLs + substantial + observed
    arts = [
        _art(
            kind="report",
            ref="report.md",
            content=(
                "Body. https://example.com/1 https://example.com/2 "
                "https://example.com/3 " + _long()
            ),
            observed=True,
        )
    ]
    # Act
    s = score("research", arts, [])
    # Assert
    assert s >= threshold("research")


def test_research_doi_counts_as_citation():
    # Arrange
    arts = [
        _art(
            kind="report",
            content=(
                "Ref A 10.1000/abc123 Ref B 10.1000/def456 Ref C 10.1000/ghi789 "
                + _long()
            ),
            observed=True,
        )
    ]
    # Act
    s = score("research", arts, [])
    # Assert — DOIs push citation signal to full, clears the bar
    assert s >= threshold("research")


def test_research_bracket_refs_count_as_citations():
    # Arrange
    arts = [
        _art(
            kind="report",
            content="Claim [1] and claim [2] and claim [3]. " + _long(),
            observed=True,
        )
    ]
    # Act
    s = score("research", arts, [])
    # Assert
    assert s >= threshold("research")


# ---------------------------------------------------------------------------
# score — content (voice-weighted)
# ---------------------------------------------------------------------------


def test_content_substantial_voice_beats_thin():
    # Arrange
    rich = [_art(kind="text", ref="", content=_long(), observed=True)]
    thin = [_art(kind="text", ref="", content="ok.", observed=True)]
    # Act
    hi = score("content", rich, [])
    lo = score("content", thin, [])
    # Assert
    assert hi > lo


def test_content_rich_prose_clears_threshold():
    # Arrange — long, multi-sentence, observed content
    arts = [
        _art(
            kind="text",
            ref="",
            content=_long(reps=20),
            observed=True,
        )
    ]
    # Act
    s = score("content", arts, [])
    # Assert
    assert s >= threshold("content")


def test_content_does_not_require_citations():
    # Arrange — zero citations, but strong voice/length
    arts = [
        _art(
            kind="text",
            content=(
                "The morning light spilled across the room. She smiled softly. "
                "It was going to be a good day, and everyone knew it. "
                + _long(reps=10)
            ),
            observed=True,
        )
    ]
    # Act
    s = score("content", arts, [])
    # Assert — content bar reachable with no sources
    assert s >= threshold("content")


# ---------------------------------------------------------------------------
# score — reviewer reasons handling
# ---------------------------------------------------------------------------


def test_positive_review_notes_do_not_penalize():
    # Arrange
    arts = [_art(observed=True, content=_long())]
    # Act
    clean = score("code", arts, [])
    with_praise = score("code", arts, ["LGTM", "looks good", "no issues"])
    # Assert — approving notes cost nothing
    assert with_praise == clean


def test_blocking_reasons_penalize_more_for_code_than_content():
    # Arrange — same one blocking reason on comparable inputs
    reason = ["real defect here"]
    code_arts = [_art(observed=True, content=_long())]
    content_arts = [_art(kind="text", content=_long(), observed=True)]
    code_delta = score("code", code_arts, []) - score("code", code_arts, reason)
    content_delta = (
        score("content", content_arts, [])
        - score("content", content_arts, reason)
    )
    # Assert — code is stricter: a complaint hurts it more
    assert code_delta > content_delta


def test_blank_and_nonstring_reasons_ignored():
    # Arrange
    arts = [_art(observed=True, content=_long())]
    # Act
    clean = score("code", arts, [])
    noisy = score("code", arts, ["", "   ", None, 123])  # type: ignore[list-item]
    # Assert — junk reasons do not penalize
    assert noisy == clean
