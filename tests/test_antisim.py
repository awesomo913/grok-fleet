"""Tests for grok_fleet.antisim — Gate 4, the anti-simulation detector.

Includes the REQUIRED golden test: a fabricated "I ran the tests, all pass"
summary with NO observed passing artifact must yield a flag (=> NOT_DONE).
AAA pattern.
"""
from __future__ import annotations

from grok_fleet.types import Artifact
import grok_fleet.antisim as A


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _observed_artifact():
    """An artifact the verifier independently OBSERVED (observed is True)."""
    return Artifact(kind="file", ref="out.txt", content="", observed=True)


def _unobserved_artifact():
    """An artifact that was NOT observed (proof flag still None)."""
    return Artifact(kind="file", ref="out.txt", content="", observed=None)


def _looked_and_absent_artifact():
    """An artifact the verifier LOOKED for and did not find (observed False)."""
    return Artifact(kind="file", ref="out.txt", content="", observed=False)


# ---------------------------------------------------------------------------
# THE GOLDEN TEST (required)
# ---------------------------------------------------------------------------


def test_golden_fabricated_test_pass_with_no_real_artifact_is_flagged():
    # Arrange: the classic bluff. Confident assertive prose, zero proof.
    fabricated = "I ran the tests, all pass. Everything works, build succeeded."
    artifacts = []  # nothing produced, nothing observed

    # Act
    flags = A.scan(fabricated, artifacts)

    # Assert: the gate MUST refuse to bless this — non-empty => NOT_DONE upstream.
    assert flags, "golden bluff must be flagged"
    joined = " ".join(flags).lower()
    assert "no observed artifact" in joined or "no observed" in joined


def test_golden_bluff_still_flagged_when_artifact_looked_and_absent():
    # Even sharper: an artifact exists in the list but the verifier observed it
    # as ABSENT (False). That is NOT backing — the claim is still a bluff.
    fabricated = "I ran the tests, all pass."
    flags = A.scan(fabricated, [_looked_and_absent_artifact()])
    assert flags


# ---------------------------------------------------------------------------
# Backed claims are tolerated
# ---------------------------------------------------------------------------


def test_claim_backed_by_observed_artifact_passes():
    # Same words, but now Gate 2 actually observed a real artifact.
    text = "I ran the tests, all pass. This should be fine in production."
    flags = A.scan(text, [_observed_artifact()])
    assert flags == []  # proof exists elsewhere; prose is tolerated


def test_hedge_language_tolerated_when_backed():
    text = "I would normally simulate this, but assuming everything works it should pass."
    flags = A.scan(text, [_observed_artifact()])
    assert flags == []


# ---------------------------------------------------------------------------
# Individual hedge / simulation markers (no observed artifact)
# ---------------------------------------------------------------------------


def test_conditional_intent_flagged():
    flags = A.scan("I would run the suite and it would produce green output.", [])
    assert flags
    assert any("conditional intent" in f for f in flags)


def test_predictive_hedge_flagged():
    flags = A.scan("This should pass once wired up.", [_unobserved_artifact()])
    assert flags
    assert any("predictive hedge" in f for f in flags)


def test_simulation_word_flagged():
    flags = A.scan("Simulating the deployment now.", [])
    assert flags
    assert any("simulation language" in f for f in flags)


def test_in_a_real_scenario_tell_flagged():
    flags = A.scan("Here is the output. In a real scenario we would connect to the DB.", [])
    assert flags
    assert any("in a real scenario" in f.lower() for f in flags)


def test_assuming_flagged():
    flags = A.scan("Assuming the file exists, the parser returns the rows.", [])
    assert flags
    assert any("assumption" in f for f in flags)


def test_hypothetical_flagged():
    flags = A.scan("Hypothetically the endpoint returns 200.", [])
    assert flags


def test_placeholder_flagged():
    flags = A.scan("Returning a placeholder for now.", [])
    assert flags


# ---------------------------------------------------------------------------
# Empty / edge inputs
# ---------------------------------------------------------------------------


def test_empty_text_no_artifact_is_flagged():
    # An empty summary AND no observed artifact => nothing was proven at all.
    flags = A.scan("", [])
    assert flags
    assert any("no observed artifact" in f for f in flags)


def test_empty_text_with_observed_artifact_passes():
    # The artifact IS the deliverable; a blank summary is fine.
    flags = A.scan("   ", [_observed_artifact()])
    assert flags == []


def test_clean_summary_no_artifact_still_flagged_as_unbacked():
    # No hedge words, but also nothing observed and no verifiable claim: the
    # summary proves nothing, so the gate surfaces it rather than silently pass.
    flags = A.scan("The widget renders three tabs across the top.", [])
    assert flags
    assert any("unbacked" in f for f in flags)


def test_non_string_text_is_coerced_and_flagged():
    # Defensive: a non-str must not crash; with no artifact it flags unbacked.
    flags = A.scan(None, [])  # type: ignore[arg-type]
    assert flags


def test_done_claim_marker_variants_flagged():
    for phrase in [
        "We executed the pipeline successfully.",
        "I verified that the config loads.",
        "Confirmed that all rows import.",
        "It works end to end.",
    ]:
        flags = A.scan(phrase, [])
        assert flags, f"expected flag for: {phrase!r}"


def test_flags_are_deduped_by_label():
    # Repeating the same marker twice should not produce two identical flags.
    text = "I would run it. I would run it again."
    flags = A.scan(text, [])
    intent_flags = [f for f in flags if "conditional intent: 'I would'" in f]
    assert len(intent_flags) == 1
