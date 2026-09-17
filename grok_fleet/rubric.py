"""grok_fleet.rubric — Gate 5: the Quality-First gate.

Gate 5 of the Grok Quality Fleet. After proof-of-work (Gate 2), anti-sim
(Gate 4), and adversarial review (Gate 3), the deliverable must still clear a
quality bar. This module scores a deliverable in ``[0.0, 1.0]`` and exposes the
per-task-type passing threshold.

Scoring philosophy (per task_type):
- ``code``     — STRICT. Reward artifacts the verifier actually OBSERVED
                 (``observed is True``) and penalize reviewer complaints hard.
                 The bar (threshold) is the highest of the three.
- ``research`` — CITATION-WEIGHTED. Reward resolvable citations (URLs, DOIs,
                 bracketed numeric refs) in the report text; thin, uncited prose
                 scores low.
- ``content``  — VOICE-WEIGHTED. Reward substance and voice signals (length,
                 structure, varied sentences) rather than citations.

Everything here is pure and deterministic: stdlib only, no network, no model
calls, no filesystem I/O. Identical inputs always produce an identical score so
the gate is reproducible.

Frozen public signatures honored EXACTLY (see grok_fleet/interfaces.py):
- ``score(task_type, artifacts, review_reasons) -> float``  (in [0.0, 1.0])
- ``threshold(task_type) -> float``
"""
from __future__ import annotations

import logging
import re
from typing import List

from .types import TASK_TYPES, Artifact, TaskType

__all__ = ["score", "threshold"]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-task-type passing thresholds (Gate 5 bar).
# code is strictest; content is most lenient; research sits in the middle.
# ---------------------------------------------------------------------------
_THRESHOLDS = {
    "code": 0.80,
    "research": 0.70,
    "content": 0.60,
}
_DEFAULT_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Scoring tunables (module constants, no magic numbers inline).
# ---------------------------------------------------------------------------

# How much a single reviewer complaint drags the score down, per task_type.
# code is punished hardest for complaints (strict); content the least.
_REVIEW_PENALTY_PER_REASON = {
    "code": 0.20,
    "research": 0.15,
    "content": 0.10,
}
_DEFAULT_REVIEW_PENALTY = 0.15

# Words in a reviewer reason that signal an *approving* / non-blocking note
# rather than a defect. Approving notes must not drag the score down.
_POSITIVE_REVIEW_MARKERS = (
    "looks good",
    "lgtm",
    "no issues",
    "no issue",
    "holds up",
    "approve",
    "approved",
    "passes",
    "solid",
    "well done",
    "correct",
    "no concerns",
)

# Substance thresholds (characters of inline content) used across task types.
_MIN_SUBSTANCE_CHARS = 40  # below this an artifact is "thin"
_FULL_SUBSTANCE_CHARS = 400  # at/above this an artifact is "substantial"

# research: citation targets.
_RESEARCH_TARGET_CITATIONS = 3  # citations needed for full citation credit.

# content: voice/length targets.
_CONTENT_TARGET_CHARS = 600  # chars of content for full length credit.

# Weight blend per task type: (base_artifact_weight, specialty_weight).
# base_artifact_weight rewards observed, present, substantial artifacts.
# specialty_weight rewards the task-type-specific signal (citations / voice).
# For code the specialty IS observed-ness, so it folds into the base weight.
_BLEND = {
    "code": (1.00, 0.00),
    "research": (0.50, 0.50),
    "content": (0.55, 0.45),
}
_DEFAULT_BLEND = (0.70, 0.30)

# Regexes for resolvable-citation detection (mirrors the spirit of
# verify.count_citations without importing another owner's stub module).
_URL_RE = re.compile(r"https?://[^\s)\]<>\"']+", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_BRACKET_REF_RE = re.compile(r"\[\d{1,3}\]")

# Sentence splitter for the content "voice" signal.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def threshold(task_type: TaskType) -> float:
    """Return the minimum passing quality score for ``task_type`` (Gate 5 bar).

    A :func:`score` below this triggers a bounded REVISE (<=2), then escalation,
    then PARK. Values are per-type constants (``code`` stricter than
    ``research`` stricter than ``content``). Unknown task types fall back to a
    conservative default rather than raising, so an out-of-vocab type never
    silently passes with a bar of 0.

    :param task_type: one of :data:`TASK_TYPES`.
    :returns: the passing threshold in ``[0.0, 1.0]``.
    """
    if task_type not in TASK_TYPES:
        log.warning(
            "rubric.threshold: unknown task_type %r; using default %.2f",
            task_type,
            _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD
    return _THRESHOLDS.get(task_type, _DEFAULT_THRESHOLD)


def score(
    task_type: TaskType,
    artifacts: List[Artifact],
    review_reasons: List[str],
) -> float:
    """Compute a deterministic ``0.0..1.0`` quality score (Gate 5).

    Blends an artifact-quality signal (presence, observed-ness, substance) with
    a task-type specialty signal (citations for ``research``, voice/length for
    ``content``; folded into observed-ness for ``code``), then applies a penalty
    for each *blocking* reviewer reason. Approving/non-blocking reviewer notes do
    not penalize.

    Deterministic: identical inputs always yield the identical float.

    :param task_type: one of :data:`TASK_TYPES` (drives the weight blend).
    :param artifacts: the produced artifacts (their ``observed`` flags and
        inline ``content`` drive the base signal).
    :param review_reasons: refute-first reviewer bullets; blocking ones lower the
        score.
    :returns: a float clamped to ``[0.0, 1.0]``.
    """
    arts = artifacts or []
    reasons = review_reasons or []

    base = _artifact_quality(arts)
    specialty = _specialty_signal(task_type, arts)

    base_w, spec_w = _BLEND.get(task_type, _DEFAULT_BLEND)
    total_w = base_w + spec_w
    if total_w <= 0.0:  # defensive; blend constants never sum to 0
        blended = base
    else:
        blended = (base * base_w + specialty * spec_w) / total_w

    penalty = _review_penalty(task_type, reasons)
    result = blended - penalty
    return _clamp01(result)


# ---------------------------------------------------------------------------
# Internal signals (pure, deterministic)
# ---------------------------------------------------------------------------


def _artifact_quality(artifacts: List[Artifact]) -> float:
    """Base signal in [0,1]: are there artifacts, observed, and substantial?

    An empty artifact set scores 0 — you cannot have quality work with nothing
    produced. Otherwise each artifact contributes a per-artifact score built
    from three parts:
      - presence  (has a ref or content at all): up to 0.30
      - observed  (verifier stamped observed is True): up to 0.40
        (an explicit observed is False is a hard zero on this part)
      - substance (length of inline content, if any): up to 0.30
    The base is the mean per-artifact score.
    """
    if not artifacts:
        return 0.0

    per_scores: List[float] = []
    for art in artifacts:
        present = bool(getattr(art, "ref", "") or getattr(art, "content", ""))
        presence_part = 0.30 if present else 0.0

        observed = getattr(art, "observed", None)
        if observed is True:
            observed_part = 0.40
        elif observed is False:
            observed_part = 0.0
        else:  # None -> not yet checked; give partial, not full, credit
            observed_part = 0.15

        substance_part = 0.30 * _substance_ratio(getattr(art, "content", "") or "")

        per_scores.append(presence_part + observed_part + substance_part)

    return _clamp01(sum(per_scores) / len(per_scores))


def _substance_ratio(text: str) -> float:
    """Map inline-content length to a [0,1] substance ratio.

    Below ``_MIN_SUBSTANCE_CHARS`` -> 0.0 (thin). At/above
    ``_FULL_SUBSTANCE_CHARS`` -> 1.0. Linear in between. Artifacts that live
    entirely at a ``ref`` (no inline content) get 0 here but still earn presence
    + observed credit elsewhere.
    """
    n = len(text.strip())
    if n <= _MIN_SUBSTANCE_CHARS:
        return 0.0
    if n >= _FULL_SUBSTANCE_CHARS:
        return 1.0
    span = _FULL_SUBSTANCE_CHARS - _MIN_SUBSTANCE_CHARS
    return (n - _MIN_SUBSTANCE_CHARS) / span


def _specialty_signal(task_type: TaskType, artifacts: List[Artifact]) -> float:
    """Task-type-specific quality signal in [0,1].

    - research -> citation density across all artifact text.
    - content  -> voice/length signal across all artifact text.
    - code / unknown -> reuse the observed-ness base (there is no separate
      specialty; strictness comes from the blend weighting + review penalty).
    """
    if task_type == "research":
        return _citation_signal(artifacts)
    if task_type == "content":
        return _voice_signal(artifacts)
    # code and any unknown type: specialty == observed-ness proxy.
    return _observed_ratio(artifacts)


def _observed_ratio(artifacts: List[Artifact]) -> float:
    """Fraction of artifacts the verifier stamped observed is True, in [0,1]."""
    if not artifacts:
        return 0.0
    observed = sum(1 for a in artifacts if getattr(a, "observed", None) is True)
    return observed / len(artifacts)


def _citation_signal(artifacts: List[Artifact]) -> float:
    """research specialty: resolvable-citation density -> [0,1].

    Counts URLs, DOIs, and bracketed numeric refs across all artifact content,
    then scales against ``_RESEARCH_TARGET_CITATIONS``. Zero citations -> 0.0.
    """
    text = _joined_text(artifacts)
    if not text:
        return 0.0
    count = _count_citations(text)
    if count <= 0:
        return 0.0
    if count >= _RESEARCH_TARGET_CITATIONS:
        return 1.0
    return count / _RESEARCH_TARGET_CITATIONS


def _voice_signal(artifacts: List[Artifact]) -> float:
    """content specialty: voice/length signal -> [0,1].

    Blends two deterministic proxies for readable, human-voiced prose:
      - length ratio: content length vs ``_CONTENT_TARGET_CHARS`` (60% weight)
      - variety ratio: sentence-count-based structure signal (40% weight)
    Empty content -> 0.0.
    """
    text = _joined_text(artifacts)
    stripped = text.strip()
    if not stripped:
        return 0.0

    length_ratio = min(1.0, len(stripped) / _CONTENT_TARGET_CHARS)

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
    # 3+ sentences reads as structured prose; scale up to that.
    variety_ratio = min(1.0, len(sentences) / 3.0)

    return _clamp01(0.60 * length_ratio + 0.40 * variety_ratio)


def _review_penalty(task_type: TaskType, reasons: List[str]) -> float:
    """Total penalty subtracted for *blocking* reviewer reasons.

    Each blocking reason costs ``_REVIEW_PENALTY_PER_REASON[task_type]``.
    Approving/non-blocking notes (matching ``_POSITIVE_REVIEW_MARKERS``) cost
    nothing. Blank reasons are ignored. The penalty is uncapped here; the final
    score is clamped to [0,1] by the caller, so a pile of complaints floors at 0.
    """
    per = _REVIEW_PENALTY_PER_REASON.get(task_type, _DEFAULT_REVIEW_PENALTY)
    blocking = 0
    for reason in reasons:
        if not isinstance(reason, str):
            continue
        text = reason.strip().lower()
        if not text:
            continue
        if any(marker in text for marker in _POSITIVE_REVIEW_MARKERS):
            continue
        blocking += 1
    return per * blocking


# ---------------------------------------------------------------------------
# Small pure utilities
# ---------------------------------------------------------------------------


def _joined_text(artifacts: List[Artifact]) -> str:
    """Concatenate all artifact inline content (newline-joined)."""
    return "\n".join(
        (getattr(a, "content", "") or "") for a in (artifacts or [])
    )


def _count_citations(text: str) -> int:
    """Count resolvable citations: URLs + DOIs + bracketed numeric refs.

    Deterministic and local (does not import verify.py, which another module
    owns). A URL that also embeds a DOI is counted once as a URL to avoid
    double-counting the same reference.
    """
    if not text:
        return 0
    urls = _URL_RE.findall(text)
    url_set = set(urls)
    # DOIs not already contained inside a matched URL.
    dois = [
        d for d in _DOI_RE.findall(text)
        if not any(d in u for u in url_set)
    ]
    brackets = _BRACKET_REF_RE.findall(text)
    return len(url_set) + len(set(dois)) + len(set(brackets))


def _clamp01(value: float) -> float:
    """Clamp a float into the closed interval [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
