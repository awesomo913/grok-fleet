"""grok_fleet.review — Gate 3: real adversarial review over the ACTUAL artifacts.

The reviewer pipeline is the fleet's independent second opinion. It is
deliberately *refute-first*: every reviewer is instructed to try to break the
work, hand back a Verdict (APPROVE / REVISE / REJECT), a 0.0..1.0 score, and
refutation bullets. Two tiers exist:

- **fast** free tier — runs on EVERY output. Cheap, always on.
- **trusted** local tier — runs ONLY when we escalate: the fast tier raised a
  flag (REVISE / REJECT), the caller forced ``escalate=True`` (e.g. a tie to
  break upstream), or the fast tier could not be parsed (fail-closed).

Hard invariants honored here:
- The reviewer sees the REAL artifact bytes, never the worker's own summary.
  We build the prompt from ``Artifact.content`` / ``Artifact.ref`` so a test's
  fake ``Caller`` can assert the artifact text was actually present.
- A model never reviews its own output. We derive the producing model id from
  an artifact marker (``produced_by:<id>`` embedded in ``ref``/``content``) and
  skip any reviewer whose id matches it, substituting a fail-closed placeholder
  result so the tier is never silently empty.
- We only talk to a model through the injected ``Caller``. Nothing here reaches
  a real API, opens a socket, or imports Hermes.
- Parsing is robust and FAIL-CLOSED: an unparseable reviewer reply becomes a
  REJECT with score 0.0, never an accidental APPROVE.

Public surface (frozen): :func:`review`.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from .types import (
    Artifact,
    Caller,
    ReviewResult,
    TaskContract,
    Verdict,
)

__all__ = ["review"]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants (named, not magic numbers)
# ---------------------------------------------------------------------------

#: Canonical tier labels stamped onto each ReviewResult.tier.
TIER_FAST = "fast"
TIER_TRUSTED = "trusted"

#: Model id recorded when no reviewer could run (self-review guard tripped and
#: no alternative was available). Fail-closed: pairs with a REJECT verdict.
_NO_REVIEWER_ID = "<no-reviewer>"

#: Score used for a fail-closed REJECT (unparseable reply / self-review block).
_FAILCLOSED_SCORE = 0.0

#: Marker convention a worker may embed so reviewers can honor the
#: "never review your own output" rule. Case-insensitive.
_PRODUCED_BY_RE = re.compile(r"produced[_\s-]*by\s*[:=]\s*([A-Za-z0-9._\-]+)", re.IGNORECASE)

#: How much of a single artifact body to inline into the prompt. Reviews need
#: the real bytes, but we cap absurdly large blobs so one artifact cannot
#: crowd out the rest. -1 style "no cap" is intentionally avoided.
_MAX_ARTIFACT_CHARS = 20_000

#: Verdict keyword patterns, matched case-insensitively as whole words.
_VERDICT_PATTERNS = (
    (Verdict.REJECT, re.compile(r"\bREJECT(?:ED|ION)?\b", re.IGNORECASE)),
    (Verdict.REVISE, re.compile(r"\bREVISE|\bREVISION\b|\bNEEDS?[_\s-]*WORK\b", re.IGNORECASE)),
    (Verdict.APPROVE, re.compile(r"\bAPPROVE[DS]?\b|\bAPPROVAL\b|\bLGTM\b", re.IGNORECASE)),
)

#: Above this, a bare (unlabelled-unit) number is read as a percentage rather
#: than an out-of-range float. So "95" -> 0.95, but "1.5" -> clamp to 1.0.
_PERCENT_SCALE_MIN = 10.0

#: Score extraction. A leading "-" is captured so negatives parse then clamp.
#: The value may be a fraction ("7/10"); we detect that via the trailing group
#: so the labelled matcher does not swallow just the numerator.
_SCORE_LABELLED_RE = re.compile(
    r"(?:score|quality|rating|confidence)\s*[:=]?\s*"
    r"(-?[0-9]+(?:\.[0-9]+)?)\s*(%|/\s*[0-9]+(?:\.[0-9]+)?)?",
    re.IGNORECASE,
)
_SCORE_FRACTION_RE = re.compile(r"(-?[0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")


# ---------------------------------------------------------------------------
# Prompt construction (refute-first, real artifacts inlined)
# ---------------------------------------------------------------------------


def _artifact_block(artifacts: List[Artifact]) -> str:
    """Render the real artifact bytes into a labelled block for the reviewer.

    Uses ``Artifact.content`` when present, otherwise falls back to the ``ref``
    locator so the reviewer at least knows what to inspect. This is the text a
    test's fake ``Caller`` asserts on — it MUST contain the genuine artifact
    body, never a summary.
    """
    lines: List[str] = []
    for idx, art in enumerate(artifacts, start=1):
        body = art.content or ""
        if len(body) > _MAX_ARTIFACT_CHARS:
            body = body[:_MAX_ARTIFACT_CHARS] + "\n...[truncated]..."
        observed = "unknown" if art.observed is None else str(bool(art.observed)).lower()
        header = f"--- ARTIFACT {idx}: kind={art.kind!r} ref={art.ref!r} observed={observed} ---"
        lines.append(header)
        if body:
            lines.append(body)
        elif art.ref:
            lines.append(f"[no inline content; inspect at ref: {art.ref}]")
        else:
            lines.append("[empty artifact]")
    if not lines:
        return "[no artifacts were produced]"
    return "\n".join(lines)


def _build_prompt(contract: TaskContract, artifacts: List[Artifact]) -> str:
    """Assemble the refute-first review prompt around the real artifacts."""
    criteria_lines = [
        f"  - kind={c.kind} target={c.target!r} extra={c.extra}"
        for c in contract.criteria
    ]
    criteria_text = "\n".join(criteria_lines) if criteria_lines else "  (none)"
    return (
        "You are an ADVERSARIAL reviewer. Your job is to REFUTE the work, not "
        "to praise it. Assume it is wrong until the artifacts below prove "
        "otherwise. Do not trust any claim you cannot see substantiated in the "
        "artifacts themselves.\n\n"
        f"TASK ID: {contract.task_id}\n"
        f"TASK TYPE: {contract.task_type}\n"
        f"DELIVERABLE (what 'done' must produce):\n{contract.deliverable}\n\n"
        "ACCEPTANCE CRITERIA (each must be genuinely satisfied):\n"
        f"{criteria_text}\n\n"
        "ACTUAL ARTIFACTS (the real produced work — review THESE, not a "
        "summary):\n"
        f"{_artifact_block(artifacts)}\n\n"
        "Respond in this shape:\n"
        "VERDICT: <APPROVE|REVISE|REJECT>\n"
        "SCORE: <0.0-1.0>\n"
        "REASONS:\n"
        "- <refutation or, if it truly holds, why the artifacts prove it>\n"
    )


# ---------------------------------------------------------------------------
# Robust, fail-closed parsing of a reviewer's raw reply
# ---------------------------------------------------------------------------


def _parse_verdict(text: str) -> Optional[Verdict]:
    """Extract a Verdict from raw reviewer text, or None if none is found.

    Precedence is REJECT > REVISE > APPROVE: if a reply mentions more than one
    (e.g. "not a REJECT, I APPROVE"), we take the most conservative signal that
    is present. Prefers an explicit ``VERDICT:`` line when one exists.
    """
    if not text:
        return None

    # Prefer an explicit "VERDICT: X" line — most reliable signal.
    verdict_line = re.search(r"verdict\s*[:=]\s*([A-Za-z]+)", text, re.IGNORECASE)
    if verdict_line:
        token = verdict_line.group(1).upper()
        if token.startswith("APPROV") or token == "LGTM":
            return Verdict.APPROVE
        if token.startswith("REVIS"):
            return Verdict.REVISE
        if token.startswith("REJECT"):
            return Verdict.REJECT

    # Fall back to scanning the whole reply, conservative-first.
    for verdict, pattern in _VERDICT_PATTERNS:
        if pattern.search(text):
            return verdict
    return None


def _parse_score(text: str, verdict: Verdict) -> float:
    """Extract a 0.0..1.0 score from raw reviewer text with a verdict fallback.

    Understands: bare floats ("0.82"), percentages ("82%"), and fractions
    ("7/10"). When no score is present we synthesize one from the verdict so a
    downstream tie-break / rubric always has a usable number, while keeping the
    ordering APPROVE > REVISE > REJECT.
    """
    labelled = _SCORE_LABELLED_RE.search(text or "")
    if labelled:
        raw = float(labelled.group(1))
        unit = labelled.group(2) or ""
        if unit.startswith("/"):
            # "rating 7/10" — labelled numerator + fraction denominator.
            den = float(unit[1:].strip())
            if den > 0:
                return _clamp01(raw / den)
        is_percent = unit == "%" or raw >= _PERCENT_SCALE_MIN
        value = raw / 100.0 if is_percent else raw
        return _clamp01(value)

    fraction = _SCORE_FRACTION_RE.search(text or "")
    if fraction:
        num = float(fraction.group(1))
        den = float(fraction.group(2))
        if den > 0:
            return _clamp01(num / den)

    # No explicit score — synthesize from the verdict so the value is sane.
    return {
        Verdict.APPROVE: 0.9,
        Verdict.REVISE: 0.5,
        Verdict.REJECT: 0.0,
    }[verdict]


def _clamp01(value: float) -> float:
    """Clamp a float into the closed interval [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _parse_reasons(text: str) -> List[str]:
    """Pull refutation bullets out of the reviewer reply.

    Collects lines that look like list items (``-``, ``*``, ``1.``) after a
    REASONS heading when present, otherwise any bullet-like lines in the reply.
    Falls back to a single trimmed line so a reason list is never empty for a
    reply that actually said something.
    """
    if not text:
        return []

    # If there is a REASONS: section, prefer everything after it.
    marker = re.search(r"reasons?\s*[:=]", text, re.IGNORECASE)
    scope = text[marker.end():] if marker else text

    bullets: List[str] = []
    for raw_line in scope.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        stripped = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line)
        if stripped != line:  # it was a bullet
            if stripped:
                bullets.append(stripped)

    if bullets:
        return bullets

    # No bullets — return the first non-empty, non-header content line.
    for raw_line in scope.splitlines():
        line = raw_line.strip()
        if line and not re.match(r"^(?:verdict|score)\s*[:=]", line, re.IGNORECASE):
            return [line]
    return []


def _parse_review(raw: str, *, tier: str, model_id: str) -> ReviewResult:
    """Turn one reviewer's raw text into a ReviewResult, FAIL-CLOSED.

    An unparseable reply (no recognizable verdict, or empty text) becomes a
    REJECT at score 0.0 — we never let ambiguity read as approval.
    """
    verdict = _parse_verdict(raw or "")
    if verdict is None:
        log.warning(
            "review: unparseable reply from model=%s tier=%s; failing closed to REJECT",
            model_id,
            tier,
        )
        reasons = ["Reviewer reply could not be parsed into a verdict; failing closed to REJECT."]
        return ReviewResult(
            tier=tier,
            model=model_id,
            verdict=Verdict.REJECT,
            score=_FAILCLOSED_SCORE,
            reasons=reasons,
        )

    score = _parse_score(raw, verdict)
    reasons = _parse_reasons(raw)
    if not reasons:
        reasons = [f"Reviewer returned verdict {verdict.value} with no explicit reasons."]
    return ReviewResult(tier=tier, model=model_id, verdict=verdict, score=score, reasons=reasons)


# ---------------------------------------------------------------------------
# Self-review guard
# ---------------------------------------------------------------------------


def _producing_model_id(artifacts: List[Artifact]) -> Optional[str]:
    """Best-effort detect which model produced these artifacts.

    Honors an optional ``produced_by:<id>`` marker embedded in an artifact's
    ``ref`` or ``content``. Returns the first id found, or None when the
    provenance is unknown (in which case the self-review guard cannot fire and
    review proceeds normally).
    """
    for art in artifacts:
        for field_value in (art.ref, art.content):
            if not field_value:
                continue
            match = _PRODUCED_BY_RE.search(field_value)
            if match:
                return match.group(1)
    return None


def _reviewer_id_from_call(caller: Caller, prompt: str, *, tier: str) -> str:
    """Resolve the reviewer's model id for stamping onto its ReviewResult.

    The frozen ``Caller`` signature is ``(model_id, prompt) -> raw_text`` and
    does not surface the model id back to us, so we cannot introspect it from
    the callable. Tests inject a caller bound to a known id; production wires a
    ModelSpec.id. We therefore accept the id out-of-band via the caller's
    ``__gqf_model_id__`` attribute when present, else fall back to a tier tag.
    This keeps the frozen signature intact while still letting the self-review
    guard and result stamping work.
    """
    model_id = getattr(caller, "__gqf_model_id__", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    return f"reviewer_{tier}"


def _run_tier(
    caller: Caller,
    prompt: str,
    *,
    tier: str,
    producer_id: Optional[str],
) -> ReviewResult:
    """Invoke one reviewer tier and parse its result, honoring the self-guard.

    If the resolved reviewer id equals the producing model id, we refuse to let
    it review its own work and return a fail-closed REJECT placeholder instead
    of calling the model. Any error raised by the injected caller is caught,
    logged, and converted to a fail-closed REJECT (never swallowed silently).
    """
    reviewer_id = _reviewer_id_from_call(caller, prompt, tier=tier)

    if producer_id is not None and reviewer_id == producer_id:
        log.warning(
            "review: reviewer %s is the producer of the artifacts; skipping self-review (tier=%s)",
            reviewer_id,
            tier,
        )
        return ReviewResult(
            tier=tier,
            model=reviewer_id,
            verdict=Verdict.REJECT,
            score=_FAILCLOSED_SCORE,
            reasons=[
                f"Reviewer {reviewer_id} produced these artifacts; self-review is not allowed. "
                "Escalate to a different reviewer."
            ],
        )

    try:
        raw = caller(reviewer_id, prompt)
    except Exception as exc:  # injected caller may raise; never let it crash the gate
        log.warning(
            "review: caller for model=%s tier=%s raised %r; failing closed to REJECT",
            reviewer_id,
            tier,
            exc,
        )
        return ReviewResult(
            tier=tier,
            model=reviewer_id,
            verdict=Verdict.REJECT,
            score=_FAILCLOSED_SCORE,
            reasons=[f"Reviewer call failed ({type(exc).__name__}); failing closed to REJECT."],
        )

    return _parse_review(raw, tier=tier, model_id=reviewer_id)


# ---------------------------------------------------------------------------
# Public entry point (FROZEN signature)
# ---------------------------------------------------------------------------


def review(
    contract: TaskContract,
    artifacts: List[Artifact],
    *,
    fast_caller: Caller,
    trusted_caller: Optional[Caller] = None,
    escalate: bool = False,
) -> List[ReviewResult]:
    """Run refute-first adversarial review over the ACTUAL artifacts (Gate 3).

    Always runs the fast free-tier reviewer via ``fast_caller``, handing it the
    real artifact contents (never the model's own summary). When ``escalate``
    is True (a tie needs breaking) OR the fast tier itself raised a flag
    (REVISE / REJECT), and ``trusted_caller`` is provided, also runs the
    trusted local reviewer and includes its ReviewResult.

    A model never reviews its own output: if the reviewer id matches the id
    that produced the artifacts (detected via a ``produced_by:<id>`` marker),
    that tier returns a fail-closed REJECT placeholder rather than calling the
    model. Callers are injected; this function never contacts a real API.

    Returns one ReviewResult per reviewer that ran (fast always; trusted when
    escalated/flagged and available).
    """
    prompt = _build_prompt(contract, artifacts)
    producer_id = _producing_model_id(artifacts)

    results: List[ReviewResult] = []

    fast_result = _run_tier(fast_caller, prompt, tier=TIER_FAST, producer_id=producer_id)
    results.append(fast_result)

    fast_flagged = fast_result.verdict in (Verdict.REVISE, Verdict.REJECT)
    should_escalate = escalate or fast_flagged

    if should_escalate and trusted_caller is not None:
        log.info(
            "review: escalating to trusted tier (escalate=%s, fast_verdict=%s)",
            escalate,
            fast_result.verdict.value,
        )
        trusted_result = _run_tier(
            trusted_caller, prompt, tier=TIER_TRUSTED, producer_id=producer_id
        )
        results.append(trusted_result)
    elif should_escalate and trusted_caller is None:
        log.warning(
            "review: escalation warranted (escalate=%s, fast_verdict=%s) but no trusted_caller provided",
            escalate,
            fast_result.verdict.value,
        )

    return results
