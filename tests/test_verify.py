"""Tests for grok_fleet.verify — Gate 2, the anti-bluff heart.

These prove the verifier INDEPENDENTLY observes each criterion for real and
NEVER trusts an artifact's self-reported ``observed`` flag. AAA pattern.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

import pytest

# conftest.py puts the repo root on sys.path.
from grok_fleet.types import (
    AcceptanceCriterion,
    Artifact,
    TaskContract,
    VerifyResult,
)
import grok_fleet.verify as V


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _contract(criteria, task_type="code"):
    return TaskContract(
        task_id="t1",
        deliverable="do a thing",
        criteria=list(criteria),
        task_type=task_type,
    )


# ---------------------------------------------------------------------------
# file_exists
# ---------------------------------------------------------------------------


def test_file_exists_true_for_real_file(tmp_path):
    # Arrange
    p = tmp_path / "hello.txt"
    p.write_text("hi", encoding="utf-8")
    # Act
    ok, detail = V.file_exists(str(p))
    # Assert
    assert ok is True
    assert "bytes" in detail


def test_file_exists_true_for_directory(tmp_path):
    ok, detail = V.file_exists(str(tmp_path))
    assert ok is True
    assert "directory" in detail


def test_file_exists_false_for_missing(tmp_path):
    ok, detail = V.file_exists(str(tmp_path / "nope.txt"))
    assert ok is False
    assert "does not exist" in detail


def test_file_exists_false_for_empty_path():
    ok, detail = V.file_exists("")
    assert ok is False


# ---------------------------------------------------------------------------
# cmd_zero_exit
# ---------------------------------------------------------------------------


def test_cmd_zero_exit_true_on_success():
    # a trivially-zero command that works cross-platform via the py interpreter
    cmd = f'"{sys.executable}" -c "import sys; sys.exit(0)"'
    ok, detail = V.cmd_zero_exit(cmd, timeout=30)
    assert ok is True
    assert "exit=0" in detail


def test_cmd_zero_exit_false_on_nonzero():
    cmd = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    ok, detail = V.cmd_zero_exit(cmd, timeout=30)
    assert ok is False
    assert "exit=3" in detail


def test_cmd_zero_exit_false_on_missing_binary():
    ok, detail = V.cmd_zero_exit("this_binary_does_not_exist_zzz --flag", timeout=10)
    assert ok is False


def test_cmd_zero_exit_false_on_empty():
    ok, detail = V.cmd_zero_exit("   ", timeout=10)
    assert ok is False


def test_cmd_zero_exit_timeout_is_false_not_exception():
    # Sleep longer than the timeout; must return False, never raise.
    cmd = f'"{sys.executable}" -c "import time; time.sleep(5)"'
    ok, detail = V.cmd_zero_exit(cmd, timeout=1)
    assert ok is False
    assert "timed out" in detail.lower()


# ---------------------------------------------------------------------------
# run_pytest  (real subprocess shell-out)
# ---------------------------------------------------------------------------


def test_run_pytest_passing_suite(tmp_path):
    # Arrange: write a tiny passing test file
    test_file = tmp_path / "test_pass_sample.py"
    test_file.write_text(
        textwrap.dedent(
            """
            def test_ok():
                assert 1 + 1 == 2
            """
        ),
        encoding="utf-8",
    )
    # Act
    passed, detail = V.run_pytest(str(test_file), timeout=120)
    # Assert
    assert passed is True, detail
    assert "exit=0" in detail


def test_run_pytest_failing_suite_returns_false_not_exception(tmp_path):
    test_file = tmp_path / "test_fail_sample.py"
    test_file.write_text(
        textwrap.dedent(
            """
            def test_bad():
                assert 1 == 2
            """
        ),
        encoding="utf-8",
    )
    passed, detail = V.run_pytest(str(test_file), timeout=120)
    assert passed is False
    assert "exit=" in detail


def test_run_pytest_missing_path_is_false():
    passed, detail = V.run_pytest("C:/no/such/path/zzz_test.py", timeout=10)
    assert passed is False
    assert "does not exist" in detail


# ---------------------------------------------------------------------------
# count_citations
# ---------------------------------------------------------------------------


def test_count_citations_counts_urls_and_dois():
    text = (
        "See https://example.com/a and http://foo.org/b plus DOI 10.1000/xyz123."
    )
    assert V.count_citations(text) == 3


def test_count_citations_dedupes_repeated_url():
    text = "ref https://example.com/x again https://example.com/x and https://example.com/x"
    assert V.count_citations(text) == 1


def test_count_citations_bracket_ref_needs_backing_source_line():
    # A lone [1] in prose is NOT a citation.
    prose_only = "As shown in [1] and [2], the claim holds."
    assert V.count_citations(prose_only) == 0

    # But a reference-list line [1] http://... IS.
    with_refs = textwrap.dedent(
        """
        As shown in [1] and [2].

        [1] https://example.com/paper
        [2] 10.1234/abcd.efgh
        """
    )
    assert V.count_citations(with_refs) == 2


def test_count_citations_non_url_backed_ref_counts_once():
    # A reference-list line with an arXiv id (no URL/DOI) is one citation.
    text = "Body.\n\n[1] arXiv:2401.00001 Some Title\n[2] https://example.com/x"
    assert V.count_citations(text) == 2


def test_count_citations_empty_and_garbage():
    assert V.count_citations("") == 0
    assert V.count_citations(None) == 0  # type: ignore[arg-type]
    assert V.count_citations("no references here at all") == 0


# ---------------------------------------------------------------------------
# verify() end-to-end: dispatch + observed stamping
# ---------------------------------------------------------------------------


def test_verify_file_exists_criterion_stamps_artifact(tmp_path):
    # Arrange
    p = tmp_path / "out.txt"
    p.write_text("data", encoding="utf-8")
    art = Artifact(kind="file", ref=str(p), content="", observed=None)
    crit = AcceptanceCriterion(kind="file_exists", target=str(p))
    contract = _contract([crit])
    # Act
    results = V.verify(contract, [art])
    # Assert
    assert len(results) == 1
    assert results[0].observed is True
    assert art.observed is True  # verifier stamped it


def test_verify_never_trusts_self_reported_observed(tmp_path):
    # Artifact LIES: says observed=True, but the file does not exist.
    missing = str(tmp_path / "ghost.txt")
    liar = Artifact(kind="file", ref=missing, content="", observed=True)
    crit = AcceptanceCriterion(kind="file_exists", target=missing)
    contract = _contract([crit])

    results = V.verify(contract, [liar])

    # The verifier recomputes: file is absent => observed False, regardless of
    # the artifact's own claim.
    assert results[0].observed is False


def test_verify_json_schema_pass_and_fail():
    good = Artifact(
        kind="text",
        ref="report.json",
        content=json.dumps({"name": "x", "count": 3}),
    )
    schema = {"type": "object", "required": ["name", "count"], "properties": {"count": {"type": "integer"}}}
    crit = AcceptanceCriterion(kind="json_schema", target="report.json", extra={"schema": schema})
    results = V.verify(_contract([crit]), [good])
    assert results[0].observed is True

    bad = Artifact(kind="text", ref="report.json", content=json.dumps({"name": "x"}))
    results2 = V.verify(_contract([crit]), [bad])
    assert results2[0].observed is False
    assert "missing required key" in results2[0].detail


def test_verify_json_schema_invalid_json_is_false():
    art = Artifact(kind="text", ref="r.json", content="{not valid json,,,}")
    crit = AcceptanceCriterion(kind="json_schema", target="r.json", extra={"schema": {"type": "object"}})
    results = V.verify(_contract([crit]), [art])
    assert results[0].observed is False


def test_verify_regex_present():
    art = Artifact(kind="report", ref="r", content="the answer is 42 units")
    hit = AcceptanceCriterion(kind="regex_present", target="r", extra={"pattern": r"answer is \d+"})
    miss = AcceptanceCriterion(kind="regex_present", target="r", extra={"pattern": r"nope\d\d\d"})
    res = V.verify(_contract([hit, miss]), [art])
    assert res[0].observed is True
    assert res[1].observed is False


def test_verify_min_citations():
    art = Artifact(
        kind="research",
        ref="r",
        content="Body.\n\n[1] https://a.com/x\n[2] https://b.com/y\n[3] 10.1234/zzz.abc",
    )
    ok_crit = AcceptanceCriterion(kind="min_citations", target="r", extra={"min": 3})
    too_many = AcceptanceCriterion(kind="min_citations", target="r", extra={"min": 5})
    res = V.verify(_contract([ok_crit, too_many], task_type="research"), [art])
    assert res[0].observed is True
    assert res[1].observed is False


def test_verify_unknown_kind_is_false_via_injected_runner():
    # We can't build an unknown kind through the frozen Literal-typed dataclass
    # cleanly at type-check time, but at runtime the dataclass stores whatever we
    # pass. Force an unknown kind and confirm the default runner refuses it.
    crit = AcceptanceCriterion(kind="totally_made_up", target="x")  # type: ignore[arg-type]
    res = V.verify(_contract([crit]), [])
    assert res[0].observed is False
    assert "unknown criterion kind" in res[0].detail


def test_verify_uses_injected_runner():
    calls = []

    def fake_runner(criterion, artifacts):
        calls.append(criterion.kind)
        return VerifyResult(criterion=criterion, observed=True, detail="faked")

    crit = AcceptanceCriterion(kind="file_exists", target="whatever")
    res = V.verify(_contract([crit]), [], runner=fake_runner)
    assert res[0].detail == "faked"
    assert calls == ["file_exists"]


def test_verify_survives_misbehaving_runner():
    def boom(criterion, artifacts):
        raise RuntimeError("runner exploded")

    crit = AcceptanceCriterion(kind="file_exists", target="x")
    res = V.verify(_contract([crit]), [], runner=boom)
    # Must not propagate — verifier converts it to observed=False.
    assert res[0].observed is False
    assert "runner raised" in res[0].detail


def test_verify_resolves_text_from_disk_when_no_inline_content(tmp_path):
    p = tmp_path / "report.txt"
    p.write_text("the magic token QWERTY appears here", encoding="utf-8")
    art = Artifact(kind="report", ref=str(p), content="")  # empty inline; on disk
    crit = AcceptanceCriterion(kind="regex_present", target=str(p), extra={"pattern": r"QWERTY"})
    res = V.verify(_contract([crit]), [art])
    assert res[0].observed is True


def test_verify_multiple_criteria_order_preserved(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x", encoding="utf-8")
    c1 = AcceptanceCriterion(kind="file_exists", target=str(p))
    c2 = AcceptanceCriterion(kind="file_exists", target=str(tmp_path / "missing"))
    res = V.verify(_contract([c1, c2]), [])
    assert [r.observed for r in res] == [True, False]
    assert res[0].criterion is c1
    assert res[1].criterion is c2
