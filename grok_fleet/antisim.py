"""grok_fleet.antisim — Gate 4: Anti-Simulation detector.

A worker that WROTE the deliverable talks about concrete, done things. A worker
that only PRETENDED to do the work leaks hedge/simulation language: "I would
run the tests", "this should pass", "simulating the deployment", "in a real
scenario we'd...", "assuming the file exists", "hypothetically". Gate 4 catches
those markers — but ONLY when there is no observed artifact backing the claim.

Grounding rule: a marker is a *violation* only when the produced work has no
artifact the verifier already OBSERVED (``Artifact.observed is True``). If real,
observed artifacts exist, hedge phrasing in a summary is tolerated (the proof is
elsewhere and Gate 2 already saw it). If NOTHING is observed and the text is
full of "I would / this should / simulating", that is a bluff => NOT_DONE.

GOLDEN CASE (see tests): a fabricated "I ran the tests, all pass." output with
NO observed passing artifact MUST produce at least one flag, so the harness
turns it into NOT_DONE. That is the whole point of this gate.

Pure stdlib. No network. No model calls.
"""
from __future__ import annotations

import logging
import re
from typing import List, Pattern, Tuple

from .types import Artifact

log = logging.getLogger(__name__)

__all__ = ["scan"]


# ---------------------------------------------------------------------------
# Marker catalogue
# ---------------------------------------------------------------------------
# Each entry: (compiled_regex, human_label). We compile once at import. Patterns
# are word-boundary anchored where sensible so "would" inside "wouldn't-matter"
# style tokens doesn't over-fire, while still catching the real hedges.

_SIM_MARKERS: Tuple[Tuple[str, str], ...] = (
    # Conditional / future intent instead of a done action.
    (r"\bi\s+would\b", "conditional intent: 'I would'"),
    (r"\bwe\s+would\b", "conditional intent: 'we would'"),
    (r"\bi'?d\s+(?:run|write|create|test|implement|add|check)\b", "conditional intent: \"I'd <do>\""),
    (r"\bi\s+could\b", "conditional intent: 'I could'"),
    (r"\bwould\s+(?:run|execute|produce|generate|create|write)\b", "conditional intent: 'would <do>'"),
    # Predictive hedge instead of an observed result.
    (r"\bthis\s+should\b", "predictive hedge: 'this should'"),
    (r"\bit\s+should\b", "predictive hedge: 'it should'"),
    (r"\bshould\s+(?:pass|work|succeed|return|produce|now\s+work)\b", "predictive hedge: 'should <succeed>'"),
    (r"\bshould\s+be\s+(?:fine|correct|passing|working)\b", "predictive hedge: 'should be fine'"),
    (r"\bwill\s+(?:probably|likely)\b", "predictive hedge: 'will probably'"),
    (r"\bought\s+to\b", "predictive hedge: 'ought to'"),
    # Explicit simulation / pretence.
    (r"\bsimulat(?:e|es|ed|ing|ion)\b", "simulation language: 'simulate/simulation'"),
    (r"\bpretend(?:ing|ed)?\b", "pretence language: 'pretend'"),
    (r"\bmock(?:ing|ed)?\s+(?:the\s+)?(?:result|output|response|run|test)s?\b", "mocked result language"),
    (r"\bhypothetical(?:ly)?\b", "hypothetical framing"),
    (r"\bfor\s+the\s+(?:sake\s+of\s+)?(?:example|demonstration|illustration)\b", "illustrative-only framing"),
    (r"\bimagine\s+(?:that|the|a|we)\b", "imaginative framing: 'imagine that'"),
    # "In a real ..." — the tell that what preceded was NOT real.
    (r"\bin\s+a\s+real\s+(?:scenario|system|environment|setup|deployment|world)\b", "'in a real scenario' tell"),
    (r"\bin\s+(?:an?\s+)?actual\s+(?:run|deployment|environment)\b", "'in an actual run' tell"),
    (r"\bin\s+production\s+(?:this\s+)?would\b", "'in production this would' tell"),
    # Assumption instead of verification.
    (r"\bassuming\s+(?:that\s+)?(?:the|this|it|everything|all)\b", "assumption instead of check: 'assuming'"),
    (r"\bassume\s+(?:that\s+)?(?:the|this|it|everything|all)\b", "assumption instead of check: 'assume'"),
    (r"\bif\s+everything\s+(?:is\s+)?(?:correct|works|passes)\b", "conditional-on-unknown: 'if everything works'"),
    # Placeholder / not-actually-done tells.
    (r"\bplaceholder\b", "placeholder content"),
    (r"\bnot\s+actually\s+(?:run|executed|tested|implemented)\b", "explicit 'not actually done'"),
    (r"\bwould\s+need\s+to\s+(?:be\s+)?(?:run|tested|verified|executed)\b", "'would need to be run' tell"),
    (r"\bto\s+the\s+best\s+of\s+my\s+knowledge\b", "unverified 'best of my knowledge'"),
)

_COMPILED_MARKERS: Tuple[Tuple[Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), label) for pat, label in _SIM_MARKERS
)

# Phrases that claim a concrete, checkable action was DONE. If the text asserts
# one of these but there is no observed artifact, that specific claim is a bluff
# regardless of whether a hedge word appears. This is what nails the golden
# "I ran the tests, all pass" case even though it uses assertive (not hedge)
# grammar.
_DONE_CLAIM_MARKERS: Tuple[Tuple[str, str], ...] = (
    (r"\b(?:i|we)\s+ran\s+(?:the\s+)?(?:tests?|suite|pytest|command|script)\b", "claims tests/command were run"),
    (r"\b(?:all\s+)?tests?\s+(?:pass(?:ed|ing)?|are\s+passing|green)\b", "claims tests passed"),
    (r"\b(?:i|we)\s+(?:have\s+)?executed\b", "claims execution happened"),
    (r"\b(?:i|we)\s+verified\s+(?:that\s+)?\b", "claims verification happened"),
    (r"\b(?:i|we)\s+(?:have\s+)?tested\s+(?:it|this|the)\b", "claims testing happened"),
    (r"\bconfirmed\s+(?:that\s+)?(?:it|this|the|all)\b", "claims confirmation happened"),
    (r"\b(?:i|we)\s+(?:have\s+)?created\s+(?:the\s+)?file\b", "claims a file was created"),
    (r"\bbuild\s+succeeded\b", "claims a build succeeded"),
    (r"\bit\s+works\b", "claims 'it works'"),
    (r"\beverything\s+(?:passes|works|is\s+working)\b", "claims 'everything works'"),
)

_COMPILED_DONE_CLAIMS: Tuple[Tuple[Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), label) for pat, label in _DONE_CLAIM_MARKERS
)


# ---------------------------------------------------------------------------
# Backing check
# ---------------------------------------------------------------------------


def _has_observed_artifact(artifacts: List[Artifact]) -> bool:
    """True iff at least one artifact was independently OBSERVED by the verifier.

    We consult ``Artifact.observed is True`` — the tri-state proof flag that only
    the verifier (Gate 2) is allowed to set. ``None`` (unchecked) and ``False``
    (looked, not there) both count as NOT backed. This is the anchor that lets
    hedge/claim language pass ONLY when real proof exists.
    """
    return any(art.observed is True for art in artifacts)


def _snippet(text: str, match: "re.Match[str]", width: int = 40) -> str:
    """Return a short context window around a regex match, for the flag string."""
    start = max(0, match.start() - width)
    end = min(len(text), match.end() + width)
    frag = text[start:end].replace("\n", " ").strip()
    return f"...{frag}..." if (start > 0 or end < len(text)) else frag


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scan(text: str, artifacts: List[Artifact]) -> List[str]:
    """Flag hedge/simulation language not backed by a concrete artifact (Gate 4).

    Scans ``text`` for two classes of markers:
      1. Hedge/simulation phrasing ("I would", "this should", "simulating",
         "in a real scenario", "assuming", "hypothetically", "pretend", ...).
      2. Assertive done-claims ("I ran the tests, all pass", "build succeeded",
         "it works") — assertive grammar, but still a bluff if unproven.

    A marker becomes a *violation* only when NO artifact has been independently
    observed by the verifier (``Artifact.observed is True``). When real observed
    artifacts exist, the proof lives in Gate 2 and hedge phrasing in a summary is
    tolerated => this returns ``[]``. When nothing is observed and such markers
    appear, each unique marker yields a human-readable flag string. A non-empty
    return => NOT_DONE upstream.

    Edge cases:
      * empty/whitespace text with no observed artifacts => one flag ("no
        substantive work, no observed artifact"), because an empty deliverable
        is itself unbacked.
      * empty/whitespace text WITH an observed artifact => ``[]`` (the artifact
        is the deliverable; the summary just happens to be blank).
    """
    if not isinstance(text, str):
        log.warning("antisim.scan: non-str text of type %s coerced to ''", type(text).__name__)
        text = ""

    backed = _has_observed_artifact(artifacts)

    # If the verifier already observed real work, we do not second-guess prose.
    if backed:
        return []

    flags: List[str] = []
    seen_labels: set = set()

    stripped = text.strip()
    if not stripped:
        # No prose AND no observed artifact => nothing was proven at all.
        flags.append(
            "unbacked: no observed artifact and an empty summary — no work to trust"
        )
        return flags

    # Nothing observed: every hedge/sim marker is now a violation.
    for pattern, label in _COMPILED_MARKERS:
        m = pattern.search(text)
        if m and label not in seen_labels:
            seen_labels.add(label)
            flags.append(f"{label} (no observed artifact backs it): {_snippet(text, m)!r}")

    # Assertive done-claims with nothing observed are the sharpest bluffs.
    for pattern, label in _COMPILED_DONE_CLAIMS:
        m = pattern.search(text)
        if m and label not in seen_labels:
            seen_labels.add(label)
            flags.append(f"{label} but NO observed artifact proves it: {_snippet(text, m)!r}")

    # If the text made no recognizable claim at all and nothing was observed, it
    # still failed to prove anything — surface that rather than silently pass.
    if not flags:
        flags.append(
            "unbacked: summary present but no observed artifact and no verifiable claim detected"
        )

    return flags
