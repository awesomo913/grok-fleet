"""grok_fleet.contract — Gate 1: the Task Contract.

Gate 1 of the Grok Quality Fleet. A contract is the explicit deliverable plus a
non-empty list of machine-checkable acceptance criteria. This module builds a
validated ``TaskContract`` and parses loose machine/JSON-ish specs into typed
``AcceptanceCriterion`` objects.

Everything here is pure and deterministic:
- stdlib only (``json`` for the parse boundary),
- no network, no model calls, no filesystem side effects,
- validation raises ``ValueError`` on any malformed input so a bad contract can
  never enter the harness.

Frozen public signatures honored EXACTLY (see grok_fleet/interfaces.py):
- ``build_contract(task_id, deliverable, criteria, task_type) -> TaskContract``
- ``parse_criteria(spec: object) -> List[AcceptanceCriterion]``
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from .types import (
    CRITERION_KINDS,
    TASK_TYPES,
    AcceptanceCriterion,
    TaskContract,
    TaskType,
)

__all__ = ["build_contract", "parse_criteria"]

log = logging.getLogger(__name__)

# Required per-kind keys inside ``extra`` for kinds that carry a knob. Kinds not
# listed here have no required ``extra`` keys. Kept as a plain module constant so
# both the parser and any future validator share one source of truth.
_REQUIRED_EXTRA_KEYS: Dict[str, tuple] = {
    "min_citations": ("min",),
    "regex_present": ("pattern",),
    "json_schema": ("schema",),
}


def build_contract(
    task_id: str,
    deliverable: str,
    criteria: List[AcceptanceCriterion],
    task_type: TaskType,
) -> TaskContract:
    """Assemble a validated :class:`TaskContract` (Gate 1).

    Validates that ``task_id`` and ``deliverable`` are non-empty strings,
    ``task_type`` is one of :data:`TASK_TYPES`, and ``criteria`` is a non-empty
    list of :class:`AcceptanceCriterion` whose kinds are all in
    :data:`CRITERION_KINDS`. Raises :class:`ValueError` on any violation so a
    malformed contract can never enter the harness.

    :param task_id: stable unique id (also the queue/job key). Non-empty.
    :param deliverable: plain-language statement of what "done" produces.
        Non-empty.
    :param criteria: the checkable conditions; must be a non-empty list of
        :class:`AcceptanceCriterion`.
    :param task_type: one of :data:`TASK_TYPES`.
    :returns: a validated, frozen :class:`TaskContract`.
    :raises ValueError: on any validation failure.
    """
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("build_contract: task_id must be a non-empty string")
    if not isinstance(deliverable, str) or not deliverable.strip():
        raise ValueError("build_contract: deliverable must be a non-empty string")
    if task_type not in TASK_TYPES:
        raise ValueError(
            f"build_contract: task_type must be one of {TASK_TYPES!r}, "
            f"got {task_type!r}"
        )
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(
            "build_contract: criteria must be a non-empty list of "
            "AcceptanceCriterion"
        )

    for idx, crit in enumerate(criteria):
        _validate_criterion(crit, idx)

    return TaskContract(
        task_id=task_id,
        deliverable=deliverable,
        criteria=list(criteria),  # defensive copy; contract owns its list
        task_type=task_type,
    )


def parse_criteria(spec: object) -> List[AcceptanceCriterion]:
    """Parse a machine/JSON-ish spec into typed :class:`AcceptanceCriterion`.

    Accepts one of:
      - a JSON string encoding a list of criterion dicts (or a single dict),
      - a list of dicts (each with ``kind``/``target`` and optional ``extra``),
      - a list of already-typed :class:`AcceptanceCriterion` (passed through
        after validation),
      - a single dict / single :class:`AcceptanceCriterion` (wrapped in a list).

    Each dict must carry a ``kind`` in :data:`CRITERION_KINDS` and a ``target``
    string. ``extra`` is optional and defaults to ``{}``; when present it must be
    a dict, and kinds that require a knob (``min_citations`` -> ``min``,
    ``regex_present`` -> ``pattern``, ``json_schema`` -> ``schema``) must supply
    it. Raises :class:`ValueError` on unknown kinds, missing required fields, or
    malformed input. This is the boundary that turns external task descriptions
    into checkable criteria.

    :param spec: the loose spec to parse (see accepted forms above).
    :returns: a list of validated :class:`AcceptanceCriterion`.
    :raises ValueError: on any malformed / unknown-kind input.
    """
    items = _normalize_spec_to_list(spec)
    if not items:
        raise ValueError("parse_criteria: spec yielded zero criteria")

    criteria: List[AcceptanceCriterion] = []
    for idx, item in enumerate(items):
        criteria.append(_coerce_item(item, idx))
    return criteria


# ---------------------------------------------------------------------------
# Internal helpers (pure)
# ---------------------------------------------------------------------------


def _normalize_spec_to_list(spec: object) -> List[object]:
    """Turn the accepted spec forms into a flat list of items to coerce.

    JSON strings are decoded first. A single dict / criterion is wrapped in a
    one-element list. Anything else that is not a list is a hard error.
    """
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            raise ValueError("parse_criteria: empty JSON string")
        try:
            spec = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            # Bind + surface — never a silent swallow.
            log.warning("parse_criteria: JSON decode failed: %s", exc)
            raise ValueError(f"parse_criteria: invalid JSON spec: {exc}") from exc

    if isinstance(spec, AcceptanceCriterion):
        return [spec]
    if isinstance(spec, dict):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return list(spec)

    raise ValueError(
        "parse_criteria: spec must be a JSON string, a dict, an "
        f"AcceptanceCriterion, or a list thereof; got {type(spec).__name__}"
    )


def _coerce_item(item: object, idx: int) -> AcceptanceCriterion:
    """Coerce one spec item (dict or already-typed criterion) into a criterion."""
    if isinstance(item, AcceptanceCriterion):
        _validate_criterion(item, idx)
        return item

    if not isinstance(item, dict):
        raise ValueError(
            f"parse_criteria: item {idx} must be a dict or AcceptanceCriterion, "
            f"got {type(item).__name__}"
        )

    kind = item.get("kind")
    target = item.get("target")
    extra = item.get("extra", {})

    if not isinstance(kind, str) or not kind:
        raise ValueError(f"parse_criteria: item {idx} missing non-empty 'kind'")
    if kind not in CRITERION_KINDS:
        raise ValueError(
            f"parse_criteria: item {idx} unknown kind {kind!r}; "
            f"must be one of {CRITERION_KINDS!r}"
        )
    if not isinstance(target, str) or not target:
        raise ValueError(
            f"parse_criteria: item {idx} ({kind}) missing non-empty 'target'"
        )
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError(
            f"parse_criteria: item {idx} ({kind}) 'extra' must be a dict, "
            f"got {type(extra).__name__}"
        )

    crit = AcceptanceCriterion(kind=kind, target=target, extra=dict(extra))
    _validate_criterion(crit, idx)
    return crit


def _validate_criterion(crit: object, idx: int) -> None:
    """Validate a fully-typed criterion: correct type, known kind, required extra.

    Shared by both :func:`build_contract` (which receives already-typed
    criteria) and the parser (post-coercion), so the exact same rules apply no
    matter which door the criterion came through.
    """
    if not isinstance(crit, AcceptanceCriterion):
        raise ValueError(
            f"criterion {idx} must be an AcceptanceCriterion, "
            f"got {type(crit).__name__}"
        )
    if crit.kind not in CRITERION_KINDS:
        raise ValueError(
            f"criterion {idx} unknown kind {crit.kind!r}; "
            f"must be one of {CRITERION_KINDS!r}"
        )
    if not isinstance(crit.target, str) or not crit.target:
        raise ValueError(f"criterion {idx} ({crit.kind}) must have a non-empty target")
    if not isinstance(crit.extra, dict):
        raise ValueError(
            f"criterion {idx} ({crit.kind}) extra must be a dict, "
            f"got {type(crit.extra).__name__}"
        )

    required = _REQUIRED_EXTRA_KEYS.get(crit.kind, ())
    for key in required:
        if key not in crit.extra:
            raise ValueError(
                f"criterion {idx} ({crit.kind}) requires extra[{key!r}]"
            )

    # Kind-specific value sanity checks (still deterministic, no I/O).
    if crit.kind == "min_citations":
        _validate_min(crit, idx)
    elif crit.kind == "regex_present":
        _validate_pattern(crit, idx)


def _validate_min(crit: AcceptanceCriterion, idx: int) -> None:
    """min_citations: extra['min'] must be a non-negative integer."""
    value: Any = crit.extra.get("min")
    # bool is a subclass of int; reject it so True/False can't sneak in as 1/0.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"criterion {idx} (min_citations) extra['min'] must be an int, "
            f"got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(
            f"criterion {idx} (min_citations) extra['min'] must be >= 0, got {value}"
        )


def _validate_pattern(crit: AcceptanceCriterion, idx: int) -> None:
    """regex_present: extra['pattern'] must be a compilable regex string."""
    import re  # local import: only needed on this validation path

    pattern: Any = crit.extra.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(
            f"criterion {idx} (regex_present) extra['pattern'] must be a "
            f"non-empty string, got {type(pattern).__name__}"
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        log.warning("parse_criteria: bad regex in criterion %d: %s", idx, exc)
        raise ValueError(
            f"criterion {idx} (regex_present) extra['pattern'] is not a valid "
            f"regex: {exc}"
        ) from exc
