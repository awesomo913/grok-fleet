"""Tests for grok_fleet.contract (Gate 1: Task Contract).

Covers build_contract validation + parse_criteria across every accepted input
form and every rejection path. Pure stdlib + pytest; no network, no models.
AAA pattern (Arrange-Act-Assert).
"""
from __future__ import annotations

import json

import pytest

from grok_fleet.contract import build_contract, parse_criteria
from grok_fleet.types import (
    CRITERION_KINDS,
    TASK_TYPES,
    AcceptanceCriterion,
    TaskContract,
)


# ---------------------------------------------------------------------------
# build_contract — happy path
# ---------------------------------------------------------------------------


def _one_criterion() -> AcceptanceCriterion:
    return AcceptanceCriterion(kind="file_exists", target="out.txt")


def test_build_contract_returns_taskcontract():
    # Arrange
    crits = [_one_criterion()]
    # Act
    contract = build_contract("t1", "produce out.txt", crits, "code")
    # Assert
    assert isinstance(contract, TaskContract)
    assert contract.task_id == "t1"
    assert contract.deliverable == "produce out.txt"
    assert contract.task_type == "code"
    assert contract.criteria == crits


def test_build_contract_defensive_copies_criteria():
    # Arrange
    crits = [_one_criterion()]
    # Act
    contract = build_contract("t1", "d", crits, "code")
    crits.append(_one_criterion())  # mutate caller list after build
    # Assert — contract kept its own snapshot
    assert len(contract.criteria) == 1


@pytest.mark.parametrize("task_type", list(TASK_TYPES))
def test_build_contract_accepts_all_task_types(task_type):
    # Arrange / Act
    contract = build_contract("id", "deliver", [_one_criterion()], task_type)
    # Assert
    assert contract.task_type == task_type


@pytest.mark.parametrize("kind", list(CRITERION_KINDS))
def test_build_contract_accepts_all_kinds(kind):
    # Arrange — supply required extra where the kind needs it
    extra = {}
    if kind == "min_citations":
        extra = {"min": 2}
    elif kind == "regex_present":
        extra = {"pattern": r"foo\d+"}
    elif kind == "json_schema":
        extra = {"schema": {"type": "object"}}
    crit = AcceptanceCriterion(kind=kind, target="ref", extra=extra)
    # Act
    contract = build_contract("id", "deliver", [crit], "code")
    # Assert
    assert contract.criteria[0].kind == kind


# ---------------------------------------------------------------------------
# build_contract — rejection paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["", "   ", None, 123])
def test_build_contract_rejects_bad_task_id(bad_id):
    with pytest.raises(ValueError):
        build_contract(bad_id, "d", [_one_criterion()], "code")


@pytest.mark.parametrize("bad_deliverable", ["", "   ", None, []])
def test_build_contract_rejects_bad_deliverable(bad_deliverable):
    with pytest.raises(ValueError):
        build_contract("id", bad_deliverable, [_one_criterion()], "code")


@pytest.mark.parametrize("bad_type", ["", "codey", "CODE", None, "test"])
def test_build_contract_rejects_bad_task_type(bad_type):
    with pytest.raises(ValueError):
        build_contract("id", "d", [_one_criterion()], bad_type)


@pytest.mark.parametrize("bad_criteria", [[], None, "not a list", {}])
def test_build_contract_rejects_bad_criteria_container(bad_criteria):
    with pytest.raises(ValueError):
        build_contract("id", "d", bad_criteria, "code")


def test_build_contract_rejects_non_criterion_in_list():
    with pytest.raises(ValueError):
        build_contract("id", "d", [{"kind": "file_exists", "target": "x"}], "code")


def test_build_contract_rejects_unknown_kind_criterion():
    # Arrange — construct a criterion with an out-of-vocab kind
    bad = AcceptanceCriterion(kind="not_a_kind", target="x")
    # Act / Assert
    with pytest.raises(ValueError):
        build_contract("id", "d", [bad], "code")


def test_build_contract_rejects_empty_target():
    bad = AcceptanceCriterion(kind="file_exists", target="")
    with pytest.raises(ValueError):
        build_contract("id", "d", [bad], "code")


def test_build_contract_rejects_min_citations_without_min():
    bad = AcceptanceCriterion(kind="min_citations", target="report", extra={})
    with pytest.raises(ValueError):
        build_contract("id", "d", [bad], "research")


def test_build_contract_rejects_regex_without_pattern():
    bad = AcceptanceCriterion(kind="regex_present", target="report", extra={})
    with pytest.raises(ValueError):
        build_contract("id", "d", [bad], "content")


def test_build_contract_rejects_json_schema_without_schema():
    bad = AcceptanceCriterion(kind="json_schema", target="out.json", extra={})
    with pytest.raises(ValueError):
        build_contract("id", "d", [bad], "code")


def test_build_contract_rejects_negative_min_citations():
    bad = AcceptanceCriterion(kind="min_citations", target="r", extra={"min": -1})
    with pytest.raises(ValueError):
        build_contract("id", "d", [bad], "research")


def test_build_contract_rejects_bool_min_citations():
    # bool is a subclass of int; must be rejected explicitly
    bad = AcceptanceCriterion(kind="min_citations", target="r", extra={"min": True})
    with pytest.raises(ValueError):
        build_contract("id", "d", [bad], "research")


def test_build_contract_rejects_bad_regex_pattern():
    bad = AcceptanceCriterion(
        kind="regex_present", target="r", extra={"pattern": "([unclosed"}
    )
    with pytest.raises(ValueError):
        build_contract("id", "d", [bad], "content")


# ---------------------------------------------------------------------------
# parse_criteria — list of dicts
# ---------------------------------------------------------------------------


def test_parse_criteria_list_of_dicts():
    # Arrange
    spec = [
        {"kind": "file_exists", "target": "a.txt"},
        {"kind": "pytest", "target": "tests/"},
    ]
    # Act
    crits = parse_criteria(spec)
    # Assert
    assert len(crits) == 2
    assert all(isinstance(c, AcceptanceCriterion) for c in crits)
    assert crits[0].kind == "file_exists"
    assert crits[1].target == "tests/"


def test_parse_criteria_preserves_extra():
    # Arrange
    spec = [{"kind": "min_citations", "target": "report", "extra": {"min": 3}}]
    # Act
    crits = parse_criteria(spec)
    # Assert
    assert crits[0].extra == {"min": 3}


def test_parse_criteria_defaults_missing_extra_to_empty_dict():
    # Arrange
    spec = [{"kind": "file_exists", "target": "a.txt"}]
    # Act
    crits = parse_criteria(spec)
    # Assert
    assert crits[0].extra == {}


def test_parse_criteria_none_extra_becomes_empty_dict():
    # Arrange
    spec = [{"kind": "file_exists", "target": "a.txt", "extra": None}]
    # Act
    crits = parse_criteria(spec)
    # Assert
    assert crits[0].extra == {}


def test_parse_criteria_copies_extra_dict():
    # Arrange
    inner = {"min": 3}
    spec = [{"kind": "min_citations", "target": "r", "extra": inner}]
    # Act
    crits = parse_criteria(spec)
    inner["min"] = 999  # mutate source after parse
    # Assert — parsed criterion kept its own copy
    assert crits[0].extra == {"min": 3}


# ---------------------------------------------------------------------------
# parse_criteria — JSON string
# ---------------------------------------------------------------------------


def test_parse_criteria_json_string_list():
    # Arrange
    spec = json.dumps([{"kind": "file_exists", "target": "a.txt"}])
    # Act
    crits = parse_criteria(spec)
    # Assert
    assert len(crits) == 1
    assert crits[0].kind == "file_exists"


def test_parse_criteria_json_string_single_dict():
    # Arrange
    spec = json.dumps({"kind": "pytest", "target": "tests/"})
    # Act
    crits = parse_criteria(spec)
    # Assert
    assert len(crits) == 1
    assert crits[0].kind == "pytest"


def test_parse_criteria_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_criteria("{not valid json")


def test_parse_criteria_rejects_empty_json_string():
    with pytest.raises(ValueError):
        parse_criteria("   ")


# ---------------------------------------------------------------------------
# parse_criteria — single dict / single criterion / passthrough
# ---------------------------------------------------------------------------


def test_parse_criteria_single_dict_wrapped():
    # Arrange
    spec = {"kind": "cmd_zero_exit", "target": "echo hi"}
    # Act
    crits = parse_criteria(spec)
    # Assert
    assert len(crits) == 1
    assert crits[0].kind == "cmd_zero_exit"


def test_parse_criteria_passthrough_typed_criterion():
    # Arrange
    crit = AcceptanceCriterion(kind="file_exists", target="a.txt")
    # Act
    crits = parse_criteria([crit])
    # Assert
    assert crits[0] is crit


def test_parse_criteria_single_typed_criterion_wrapped():
    # Arrange
    crit = AcceptanceCriterion(kind="file_exists", target="a.txt")
    # Act
    crits = parse_criteria(crit)
    # Assert
    assert len(crits) == 1
    assert crits[0] is crit


def test_parse_criteria_mixed_dicts_and_criteria():
    # Arrange
    typed = AcceptanceCriterion(kind="pytest", target="tests/")
    spec = [typed, {"kind": "file_exists", "target": "a.txt"}]
    # Act
    crits = parse_criteria(spec)
    # Assert
    assert crits[0] is typed
    assert crits[1].kind == "file_exists"


# ---------------------------------------------------------------------------
# parse_criteria — rejection paths
# ---------------------------------------------------------------------------


def test_parse_criteria_rejects_empty_list():
    with pytest.raises(ValueError):
        parse_criteria([])


def test_parse_criteria_rejects_unknown_kind():
    with pytest.raises(ValueError):
        parse_criteria([{"kind": "bogus", "target": "x"}])


def test_parse_criteria_rejects_missing_kind():
    with pytest.raises(ValueError):
        parse_criteria([{"target": "x"}])


def test_parse_criteria_rejects_missing_target():
    with pytest.raises(ValueError):
        parse_criteria([{"kind": "file_exists"}])


def test_parse_criteria_rejects_empty_target():
    with pytest.raises(ValueError):
        parse_criteria([{"kind": "file_exists", "target": ""}])


def test_parse_criteria_rejects_non_dict_item():
    with pytest.raises(ValueError):
        parse_criteria([42])


def test_parse_criteria_rejects_non_dict_extra():
    with pytest.raises(ValueError):
        parse_criteria([{"kind": "file_exists", "target": "x", "extra": "nope"}])


def test_parse_criteria_rejects_bad_spec_type():
    with pytest.raises(ValueError):
        parse_criteria(42)


def test_parse_criteria_rejects_required_extra_missing():
    with pytest.raises(ValueError):
        parse_criteria([{"kind": "min_citations", "target": "r"}])


def test_parse_criteria_rejects_bad_regex_in_dict():
    with pytest.raises(ValueError):
        parse_criteria(
            [{"kind": "regex_present", "target": "r", "extra": {"pattern": "([bad"}}]
        )


# ---------------------------------------------------------------------------
# round-trip: parse then build
# ---------------------------------------------------------------------------


def test_parse_then_build_roundtrip():
    # Arrange
    spec = [
        {"kind": "pytest", "target": "tests/"},
        {"kind": "min_citations", "target": "report", "extra": {"min": 3}},
    ]
    # Act
    crits = parse_criteria(spec)
    contract = build_contract("job-1", "ship it", crits, "research")
    # Assert
    assert isinstance(contract, TaskContract)
    assert len(contract.criteria) == 2
    assert contract.task_type == "research"
