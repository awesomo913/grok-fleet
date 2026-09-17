"""grok_fleet.verify — Gate 2: Proof-of-Work, independently verified.

THE ANTI-BLUFF HEART.

Given a TaskContract and the artifacts a worker claims prove its work, this
module RE-RUNS every acceptance criterion ITSELF and records only what it can
OBSERVE. A model's own claim ("I ran the tests, all pass") is worthless here —
we never read ``Artifact.observed`` as an input; we recompute it. If we cannot
independently observe a criterion as satisfied (test didn't pass, file isn't
there, command exited non-zero, schema didn't validate, too few citations,
regex absent), the result is ``observed=False``.

Every subprocess / filesystem touch is wrapped so a broken invocation can never
crash the gate: on ``OSError`` (or a subprocess timeout) we log the reason and
return an observed=False result. A failing test suite is a *normal* False, not
an exception.

Pure stdlib. No network. No live model calls. The only model boundary in the
whole package is the injected ``Caller`` (used by review.py, not here).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from .types import AcceptanceCriterion, Artifact, TaskContract, VerifyResult

# A Runner maps one criterion + the artifacts to an observed result. Kept here
# (as well as in interfaces.py) so implementers can import the alias from the
# concrete module. Signature MUST match interfaces.Runner exactly.
try:  # pragma: no cover - trivial import alias
    from .interfaces import Runner  # noqa: F401
except Exception as _exc:  # pragma: no cover - defensive: never let import fail
    from typing import Callable as _Callable

    Runner = _Callable[[AcceptanceCriterion, List[Artifact]], VerifyResult]  # type: ignore[misc,assignment]

log = logging.getLogger(__name__)

__all__ = [
    "verify",
    "run_pytest",
    "file_exists",
    "cmd_zero_exit",
    "count_citations",
]

# ---------------------------------------------------------------------------
# Tunable limits (named constants, no magic numbers)
# ---------------------------------------------------------------------------

#: How many chars of subprocess output we keep in a `detail` string.
_DETAIL_CAP = 2000

#: Default per-runner timeouts (seconds). Match the frozen signature defaults.
_PYTEST_TIMEOUT = 300
_CMD_TIMEOUT = 120

#: Citation-detection patterns. A "resolvable" citation is a concrete reference
#: we could in principle go check: a URL, a DOI, or a bracketed numeric ref.
_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_BRACKET_REF_RE = re.compile(r"\[(\d{1,3})\]")


# ---------------------------------------------------------------------------
# Artifact resolution helpers
# ---------------------------------------------------------------------------


def _resolve_text(target: str, artifacts: List[Artifact]) -> Tuple[Optional[str], str]:
    """Resolve the text body a content-oriented criterion should inspect.

    ``target`` for json_schema / min_citations / regex_present names an artifact
    by its ``ref`` (or, failing that, by ``kind``). We return that artifact's
    inline ``content``; if the artifact only lives on disk (empty content but a
    real ``ref`` path), we read the file. Returns ``(text_or_None, detail)``.

    We NEVER consult the artifact's ``observed`` flag — presence of usable text
    is what matters, and we recompute everything downstream.
    """
    match = _find_artifact(target, artifacts)
    if match is None:
        return None, f"no artifact matched target ref/kind {target!r}"

    if match.content:
        return match.content, f"used inline content of artifact ref={match.ref!r}"

    # No inline content — try reading from the ref path if it looks like a file.
    if match.ref and os.path.exists(match.ref):
        try:
            with open(match.ref, "r", encoding="utf-8", errors="replace") as fh:
                data = fh.read()
            return data, f"read {len(data)} chars from {match.ref!r}"
        except OSError as exc:
            log.warning("verify: could not read artifact ref %r: %s", match.ref, exc)
            return None, f"failed to read artifact ref {match.ref!r}: {exc}"

    return None, f"artifact ref={match.ref!r} has no inline content and no readable file"


def _find_artifact(target: str, artifacts: List[Artifact]) -> Optional[Artifact]:
    """Find the artifact a content criterion refers to.

    Preference order: exact ``ref`` match, then exact ``kind`` match, then — if
    the target is empty and there is exactly one artifact — that lone artifact.
    Returns None if nothing sensible matches.
    """
    for art in artifacts:
        if art.ref and art.ref == target:
            return art
    for art in artifacts:
        if art.kind == target:
            return art
    if not target and len(artifacts) == 1:
        return artifacts[0]
    return None


def _strip_wrapping_quotes(token: str) -> str:
    """Remove one layer of matched surrounding quotes from a token.

    Windows ``shlex.split(posix=False)`` preserves quotes inside tokens, so a
    quoted path token ``"C:\\x\\py.exe"`` arrives with its quotes. Strip a
    single matched ``"..."`` or ``'...'`` wrapper; leave everything else intact.
    """
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _trim(text: str) -> str:
    """Trim combined process output to a bounded, human-readable snippet."""
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= _DETAIL_CAP:
        return text
    head = text[: _DETAIL_CAP // 2]
    tail = text[-_DETAIL_CAP // 2 :]
    return f"{head}\n...[trimmed {len(text) - _DETAIL_CAP} chars]...\n{tail}"


# ---------------------------------------------------------------------------
# Individual observation runners (each returns (ok, detail); never trusts model)
# ---------------------------------------------------------------------------


def run_pytest(path: str, *, timeout: int = _PYTEST_TIMEOUT) -> Tuple[bool, str]:
    """Run pytest against ``path`` in a subprocess and observe the real result.

    Returns ``(passed, detail)`` where ``passed`` is True ONLY on exit code 0.
    A failing suite is a normal ``(False, ...)`` — never an exception. A broken
    invocation (missing path, OSError launching the interpreter, timeout) is
    logged and also returned as ``(False, ...)`` so the gate never crashes.
    """
    if not path or not os.path.exists(path):
        detail = f"pytest target path does not exist: {path!r}"
        log.warning("verify.run_pytest: %s", detail)
        return False, detail

    cmd = [sys.executable, "-m", "pytest", "-q", path]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        detail = f"pytest timed out after {timeout}s on {path!r}: {exc}"
        log.warning("verify.run_pytest: %s", detail)
        return False, detail
    except OSError as exc:
        detail = f"pytest could not be launched for {path!r}: {exc}"
        log.warning("verify.run_pytest: %s", detail)
        return False, detail

    combined = _trim((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))
    passed = proc.returncode == 0
    detail = f"exit={proc.returncode} :: {combined}"
    if not passed:
        log.info("verify.run_pytest: suite did not pass (exit=%s) for %r", proc.returncode, path)
    return passed, detail


def file_exists(path: str) -> Tuple[bool, str]:
    """Stat ``path`` and report whether it exists. Pure observation.

    Returns ``(exists, detail)``; detail notes type/size when present, or the
    missing path when absent. Any ``OSError`` while stat-ing is logged and
    treated as "not observable" => ``(False, ...)``.
    """
    if not path:
        return False, "empty path given to file_exists"
    try:
        if not os.path.exists(path):
            return False, f"path does not exist: {path!r}"
        if os.path.isdir(path):
            return True, f"directory exists: {path!r}"
        size = os.path.getsize(path)
        return True, f"file exists: {path!r} ({size} bytes)"
    except OSError as exc:
        detail = f"could not stat {path!r}: {exc}"
        log.warning("verify.file_exists: %s", detail)
        return False, detail


def cmd_zero_exit(cmd: str, *, cwd: Optional[str] = None, timeout: int = _CMD_TIMEOUT) -> Tuple[bool, str]:
    """Run a shell command and report whether it exited 0.

    Returns ``(ok, detail)`` where ``ok`` is True iff the process exited with
    code 0. Enforces ``timeout``; a timeout is ``ok=False`` (not an exception).
    A launch failure (``OSError``) is logged and returned as ``ok=False``. The
    command is tokenized with ``shlex`` (POSIX split) rather than run through a
    shell, so no shell-injection surface is opened.
    """
    if not cmd or not cmd.strip():
        return False, "empty command given to cmd_zero_exit"

    posix = os.name != "nt"
    try:
        argv = shlex.split(cmd, posix=posix)
    except ValueError as exc:
        detail = f"could not tokenize command {cmd!r}: {exc}"
        log.warning("verify.cmd_zero_exit: %s", detail)
        return False, detail
    if not argv:
        return False, f"command tokenized to nothing: {cmd!r}"

    # On Windows shlex(posix=False) keeps surrounding quotes inside each token,
    # so a quoted path like "C:\...\python.exe" would fail to launch. Strip a
    # single layer of matched surrounding quotes from each token there.
    if not posix:
        argv = [_strip_wrapping_quotes(tok) for tok in argv]

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        detail = f"command timed out after {timeout}s: {cmd!r}: {exc}"
        log.warning("verify.cmd_zero_exit: %s", detail)
        return False, detail
    except OSError as exc:
        detail = f"command could not be launched {cmd!r}: {exc}"
        log.warning("verify.cmd_zero_exit: %s", detail)
        return False, detail

    combined = _trim((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""))
    ok = proc.returncode == 0
    detail = f"exit={proc.returncode} :: {combined}"
    if not ok:
        log.info("verify.cmd_zero_exit: non-zero exit=%s for %r", proc.returncode, cmd)
    return ok, detail


def count_citations(text: str) -> int:
    """Count resolvable citations in ``text`` for the min_citations criterion.

    A "resolvable" citation is a concrete, checkable reference — NOT a bare
    claim. We count distinct:
      * URLs (http/https),
      * DOIs (10.xxxx/...),
      * bracketed numeric refs ``[n]`` on a reference-list line that carries NO
        URL/DOI but does point somewhere resolvable-looking (e.g.
        ``[3] arXiv:2401.00001``) — a real reference-list entry we couldn't
        otherwise catch. A lone ``[n]`` in prose does NOT count, and a
        ``[n] https://...`` line is counted once via its URL (never double).

    De-duplicated so repeating the same URL five times is one citation, and a
    reference-list line is never counted twice (once as URL and once as ref).
    Returns the integer count; never raises on odd input (returns 0).
    """
    if not text or not isinstance(text, str):
        return 0

    seen: set = set()

    for m in _URL_RE.finditer(text):
        seen.add(("url", m.group(0).rstrip(".,);]")))
    for m in _DOI_RE.finditer(text):
        seen.add(("doi", m.group(0).rstrip(".,);]")))

    # Bracketed reference-list refs count ONLY when the line has no URL/DOI (that
    # would already be counted above) but still names a resolvable source token
    # (arXiv id, ISBN, or another URL-ish/identifier token). This avoids double
    # counting a "[1] https://..." line while still crediting non-URL refs.
    _RESOLVABLE_TOKEN = re.compile(r"\b(?:arxiv|isbn|pmid|issn)\b", re.IGNORECASE)
    for line in text.splitlines():
        stripped = line.strip()
        head = _BRACKET_REF_RE.match(stripped)
        if not head:
            continue
        if _URL_RE.search(stripped) or _DOI_RE.search(stripped):
            continue  # already counted via URL/DOI; do not double count
        if _RESOLVABLE_TOKEN.search(stripped):
            seen.add(("ref", head.group(1)))

    return len(seen)


# ---------------------------------------------------------------------------
# json_schema / regex_present observers (content criteria)
# ---------------------------------------------------------------------------


def _check_json_schema(text: Optional[str], extra: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate ``text`` (parsed as JSON) against a minimal schema in ``extra``.

    We support a deliberately small, dependency-free subset of JSON-Schema so
    the gate needs no third-party validator:
      * ``type``       — one of object/array/string/number/integer/boolean/null
      * ``required``   — list of keys that must be present (object types)
      * ``properties`` — nested {key: subschema} recursively checked when present

    Anything we don't understand is ignored (not a failure) — the point is to
    OBSERVE structural truth, not to be a full validator. Returns (ok, detail).
    """
    if text is None:
        return False, "json_schema: no text to validate"
    schema = extra.get("schema")
    if not isinstance(schema, dict):
        return False, "json_schema: criterion.extra['schema'] missing or not a dict"
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        detail = f"json_schema: content is not valid JSON: {exc}"
        log.info("verify._check_json_schema: %s", detail)
        return False, detail

    ok, why = _validate_node(data, schema, path="$")
    return ok, ("json_schema: valid" if ok else f"json_schema: {why}")


_JSON_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _validate_node(value: Any, schema: Dict[str, Any], *, path: str) -> Tuple[bool, str]:
    """Recursively validate one JSON node against a minimal schema subset."""
    expected_type = schema.get("type")
    if expected_type is not None:
        checker = _JSON_TYPE_CHECKS.get(expected_type)
        if checker is not None and not checker(value):
            return False, f"{path}: expected type {expected_type!r}, got {type(value).__name__}"

    required = schema.get("required")
    if isinstance(required, list) and isinstance(value, dict):
        for key in required:
            if key not in value:
                return False, f"{path}: missing required key {key!r}"

    props = schema.get("properties")
    if isinstance(props, dict) and isinstance(value, dict):
        for key, subschema in props.items():
            if key in value and isinstance(subschema, dict):
                ok, why = _validate_node(value[key], subschema, path=f"{path}.{key}")
                if not ok:
                    return False, why

    return True, "ok"


def _check_regex_present(text: Optional[str], extra: Dict[str, Any]) -> Tuple[bool, str]:
    """Report whether ``extra['pattern']`` matches somewhere in ``text``."""
    if text is None:
        return False, "regex_present: no text to search"
    pattern = extra.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return False, "regex_present: criterion.extra['pattern'] missing or empty"
    flags = re.MULTILINE
    if extra.get("ignorecase"):
        flags |= re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        detail = f"regex_present: bad pattern {pattern!r}: {exc}"
        log.warning("verify._check_regex_present: %s", detail)
        return False, detail
    m = compiled.search(text)
    if m:
        return True, f"regex_present: matched {m.group(0)!r}"
    return False, f"regex_present: pattern {pattern!r} not found"


def _check_min_citations(text: Optional[str], extra: Dict[str, Any]) -> Tuple[bool, str]:
    """Observe whether ``text`` carries >= ``extra['min']`` resolvable citations."""
    if text is None:
        return False, "min_citations: no text to inspect"
    try:
        minimum = int(extra.get("min", 1))
    except (TypeError, ValueError) as exc:
        log.warning("verify._check_min_citations: bad 'min' value: %s", exc)
        minimum = 1
    found = count_citations(text)
    ok = found >= minimum
    return ok, f"min_citations: found {found}, need {minimum}"


# ---------------------------------------------------------------------------
# The default dispatcher runner
# ---------------------------------------------------------------------------


def _default_runner(criterion: AcceptanceCriterion, artifacts: List[Artifact]) -> VerifyResult:
    """Dispatch one criterion to the correct observer and box a VerifyResult.

    This is the built-in Runner. It NEVER inspects any artifact's ``observed``
    flag — it recomputes truth from scratch. Unknown kinds are a hard False
    (we refuse to bless a criterion we cannot check).
    """
    kind = criterion.kind
    target = criterion.target
    extra = criterion.extra or {}

    try:
        if kind == "pytest":
            ok, detail = run_pytest(target, timeout=int(extra.get("timeout", _PYTEST_TIMEOUT)))
        elif kind == "file_exists":
            ok, detail = file_exists(target)
        elif kind == "cmd_zero_exit":
            ok, detail = cmd_zero_exit(
                target,
                cwd=extra.get("cwd"),
                timeout=int(extra.get("timeout", _CMD_TIMEOUT)),
            )
        elif kind == "json_schema":
            text, src = _resolve_text(target, artifacts)
            ok, detail = _check_json_schema(text, extra)
            detail = f"{detail} ({src})"
        elif kind == "min_citations":
            text, src = _resolve_text(target, artifacts)
            ok, detail = _check_min_citations(text, extra)
            detail = f"{detail} ({src})"
        elif kind == "regex_present":
            text, src = _resolve_text(target, artifacts)
            ok, detail = _check_regex_present(text, extra)
            detail = f"{detail} ({src})"
        else:
            ok, detail = False, f"unknown criterion kind {kind!r} — cannot observe, refusing to bless"
            log.warning("verify._default_runner: %s", detail)
    except Exception as exc:  # last-resort guard: a runner must never crash the gate
        ok, detail = False, f"runner raised for kind {kind!r} target {target!r}: {exc}"
        log.warning("verify._default_runner: unexpected error: %s", exc)

    return VerifyResult(criterion=criterion, observed=ok, detail=detail)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _stamp_related_artifacts(criterion: AcceptanceCriterion, artifacts: List[Artifact], observed: bool) -> None:
    """Stamp ``Artifact.observed`` for artifacts a criterion speaks to.

    The verifier — and ONLY the verifier — sets ``observed``. For content
    criteria (json_schema/min_citations/regex_present) we stamp the matched
    artifact. For file_exists we stamp any artifact whose ``ref`` equals the
    checked path. We never downgrade a True to False on a second unrelated pass:
    once observed True stays True; a False only sets a still-unchecked (None)
    artifact, so an artifact proven by one criterion isn't unproven by another.
    """
    if criterion.kind in ("json_schema", "min_citations", "regex_present"):
        match = _find_artifact(criterion.target, artifacts)
        if match is not None:
            if observed:
                match.observed = True
            elif match.observed is None:
                match.observed = False
    elif criterion.kind == "file_exists":
        for art in artifacts:
            if art.ref and art.ref == criterion.target:
                if observed:
                    art.observed = True
                elif art.observed is None:
                    art.observed = False


def verify(
    contract: TaskContract,
    artifacts: List[Artifact],
    *,
    runner: Optional["Runner"] = None,
) -> List[VerifyResult]:
    """RE-RUN every acceptance criterion ourselves and record what we observe.

    For each criterion in ``contract.criteria``, dispatch to ``runner``
    (defaults to the built-in dispatcher). A criterion the harness cannot itself
    observe as passing yields ``VerifyResult(observed=False)``. Relevant
    ``Artifact.observed`` flags are stamped by the verifier (never by the
    model). The model's own success claim is irrelevant — only what this
    function observes counts.

    Returns one VerifyResult per criterion, in criterion order.
    """
    run = runner or _default_runner
    results: List[VerifyResult] = []

    for criterion in contract.criteria:
        try:
            result = run(criterion, artifacts)
        except Exception as exc:  # an injected runner may misbehave — never crash
            log.warning(
                "verify: runner raised for criterion kind=%s target=%r: %s",
                criterion.kind,
                criterion.target,
                exc,
            )
            result = VerifyResult(
                criterion=criterion,
                observed=False,
                detail=f"runner raised: {exc}",
            )

        results.append(result)
        _stamp_related_artifacts(criterion, artifacts, result.observed)

    return results
