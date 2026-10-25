"""Unit tests for the trusted-guidance knowledge-base validator.

All data here is SYNTHETIC test data (example.org URLs, invented text). Nothing is fetched and no real source content
is used.
"""
import ast
import json
import sys
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from backend.services import knowledge_base as kb
from backend.services.knowledge_base import (
    MAX_EXCERPT_CHARS,
    MAX_REPORTED_ERRORS,
    REQUIRED_FIELDS,
    KnowledgeBaseError,
    KnowledgeBaseRecord,
    load_knowledge_base,
    parse_jsonl,
    validate_record,
)

TODAY = date(2026, 10, 25)


def _record(**overrides) -> dict:
    rec = {
        "source_id": "synthetic-001",
        "publisher": "Synthetic Publisher (test only)",
        "title": "Synthetic guidance on upfront fees",
        "url": "https://example.org/synthetic/001",
        "retrieved_date": "2026-10-25",
        "license_note": "Synthetic test fixture; not real content.",
        "excerpt": "Synthetic text describing a placeholder warning about upfront fees.",
        "warning_sign_tags": ["payment_request"],
    }
    rec.update(overrides)
    return rec


def _line(**overrides) -> str:
    return json.dumps(_record(**overrides))


def _second(**overrides) -> str:
    """A valid record that differs from _record() in id, url and excerpt."""
    base = dict(source_id="synthetic-002", url="https://example.org/synthetic/002", excerpt="Different synthetic text.")
    base.update(overrides)
    return _line(**base)


def _errors_of(obj, **kwargs) -> list[str]:
    with pytest.raises(KnowledgeBaseError) as exc:
        validate_record(obj, today=TODAY, **kwargs)
    return exc.value.errors


def _field_errors(obj, field, **kwargs) -> list[str]:
    return [e for e in _errors_of(obj, **kwargs) if f"field '{field}'" in e]


# --------------------------------------------------------------------------- happy paths
def test_valid_record_round_trips():
    rec = validate_record(_record(), today=TODAY)
    assert isinstance(rec, KnowledgeBaseRecord)
    assert rec.retrieved_date == date(2026, 10, 25)
    assert rec.warning_sign_tags == ("payment_request",)
    assert rec.to_dict() == _record()


def test_today_defaults_to_system_date_for_past_records():
    assert validate_record(_record(retrieved_date="2000-01-01")).retrieved_date == date(2000, 1, 1)


def test_parse_preserves_order_skips_blank_lines_and_bom():
    text = "﻿" + _line() + "\n\n   \n" + _second() + "\r\n"
    records = parse_jsonl(text, today=TODAY)
    assert isinstance(records, tuple)
    assert [r.source_id for r in records] == ["synthetic-001", "synthetic-002"]


def test_unicode_line_separator_inside_a_string_does_not_split_the_record():
    text = json.dumps(_record(excerpt="Synthetic text"), ensure_ascii=False)
    records = parse_jsonl(text, today=TODAY)
    assert len(records) == 1
    assert records[0].excerpt == "Synthetic text"


def test_load_knowledge_base_reads_local_file_with_bom(tmp_path):
    path = tmp_path / "kb.jsonl"
    path.write_bytes(("﻿" + _line() + "\n").encode("utf-8"))
    records = load_knowledge_base(path, today=TODAY)
    assert [r.source_id for r in records] == ["synthetic-001"]


def test_load_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_knowledge_base(tmp_path / "missing.jsonl")


def test_load_rejects_oversize_file(tmp_path):
    path = tmp_path / "kb.jsonl"
    path.write_text(_line() + "\n", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="larger than 10 bytes"):
        load_knowledge_base(path, max_file_bytes=10, today=TODAY)


def test_load_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "kb.jsonl"
    path.write_bytes(b"\xff\xfe{}\n")
    with pytest.raises(KnowledgeBaseError, match="UTF-8"):
        load_knowledge_base(path, today=TODAY)


# --------------------------------------------------------------------------- structure and types
@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_each_required_field_is_enforced(field):
    rec = _record()
    del rec[field]
    assert _errors_of(rec, line_no=3) == [f"line 3: field '{field}': missing"]


def test_unknown_field_is_rejected():
    errors = _errors_of(_record(extra="x"))
    assert len(errors) == 1 and "unknown field 'extra'" in errors[0]


@pytest.mark.parametrize("field,value", [
    ("source_id", 123), ("publisher", None), ("title", ["a"]), ("url", 5), ("retrieved_date", 20261025),
    ("license_note", True), ("excerpt", {"a": 1}), ("warning_sign_tags", "payment_request"),
    ("warning_sign_tags", ["payment_request", 5]),
])
def test_wrong_types_are_rejected_without_coercion(field, value):
    assert _field_errors(_record(**{field: value}), field)


@pytest.mark.parametrize("field", ["publisher", "title", "license_note", "excerpt"])
def test_whitespace_only_text_is_rejected(field):
    assert any("must not be empty" in e for e in _field_errors(_record(**{field: "   \n\t "}), field))


@pytest.mark.parametrize("field,limit", [("publisher", 200), ("title", 300), ("license_note", 500)])
def test_text_length_limits(field, limit):
    assert validate_record(_record(**{field: "x" * limit}), today=TODAY)
    assert _field_errors(_record(**{field: "x" * (limit + 1)}), field)


@pytest.mark.parametrize("field", ["publisher", "title", "license_note"])
def test_control_characters_rejected_in_single_line_text_fields(field):
    assert _field_errors(_record(**{field: "Bad\x00value"}), field)


def test_non_object_record_is_rejected():
    assert _errors_of([1, 2], line_no=4) == ["line 4: record must be a JSON object"]
    assert _errors_of("text") == ["record must be a JSON object"]


# --------------------------------------------------------------------------- source_id and duplicates
@pytest.mark.parametrize("bad", ["Synthetic-001", "has space", "ab", "-start", "a" * 65, "_abc", "abc.def"])
def test_source_id_pattern_violations(bad):
    assert _field_errors(_record(source_id=bad), "source_id")


@pytest.mark.parametrize("good", ["abc", "a" * 64, "0synthetic_1-x"])
def test_source_id_boundaries_accepted(good):
    assert validate_record(_record(source_id=good), today=TODAY).source_id == good


def test_duplicate_source_id_names_both_lines():
    text = "\n".join([_line(), _second(), _second(source_id="synthetic-001", url="https://example.org/synthetic/003",
                                                excerpt="Third synthetic text.")])
    with pytest.raises(KnowledgeBaseError) as exc:
        parse_jsonl(text, today=TODAY)
    assert exc.value.errors == ["line 3: field 'source_id': duplicate of line 1"]


def test_same_url_and_excerpt_under_different_ids_is_rejected():
    text = "\n".join([_line(), _line(source_id="synthetic-002")])
    with pytest.raises(KnowledgeBaseError) as exc:
        parse_jsonl(text, today=TODAY)
    assert len(exc.value.errors) == 1
    assert "line 2" in exc.value.errors[0] and "line 1" in exc.value.errors[0]


def test_same_url_with_different_excerpts_is_allowed():
    text = "\n".join([_line(), _line(source_id="synthetic-002", excerpt="A different synthetic excerpt.")])
    assert len(parse_jsonl(text, today=TODAY)) == 2


# --------------------------------------------------------------------------- url
@pytest.mark.parametrize("bad", [
    "http://example.org/x", "file:///etc/passwd", "javascript:alert(1)", "https://", "https://user:pw@example.org/x",
    "https://example.org/a b", " https://example.org/x", "https://example.org/\x07", "https://[::1/x", "",
    "https://example.org/" + "a" * 2048,
])
def test_url_violations(bad):
    assert _field_errors(_record(url=bad), "url")


# --------------------------------------------------------------------------- retrieved_date
@pytest.mark.parametrize("bad", [
    "25/10/2026", "2026-13-01", "2026-02-30", "2026-10-25T10:00:00", "20261025", "2026-W43-7",
    "２０２６-10-25", "2026-10-26",
])
def test_retrieved_date_violations(bad):
    assert _field_errors(_record(retrieved_date=bad), "retrieved_date")


def test_retrieved_date_today_and_past_are_accepted():
    assert validate_record(_record(retrieved_date="2026-10-25"), today=TODAY)
    assert validate_record(_record(retrieved_date="2025-01-01"), today=TODAY)


# --------------------------------------------------------------------------- license_note
@pytest.mark.parametrize("bad", ["TODO", "tbd", " Unknown ", "N/A"])
def test_license_note_placeholders_are_rejected(bad):
    assert _field_errors(_record(license_note=bad), "license_note")


# --------------------------------------------------------------------------- excerpt
def test_excerpt_limit_is_provisional_default_and_configurable():
    assert MAX_EXCERPT_CHARS == 1000
    assert "provisional" in (kb.__doc__ or "").lower()
    assert validate_record(_record(excerpt="x" * 1000), today=TODAY)
    assert _field_errors(_record(excerpt="x" * 1001), "excerpt")
    assert _field_errors(_record(excerpt="x" * 21), "excerpt", max_excerpt_chars=20)
    assert validate_record(_record(excerpt="x" * 1001), today=TODAY, max_excerpt_chars=2000)


def test_invalid_limit_arguments_raise_value_error():
    with pytest.raises(ValueError):
        validate_record(_record(), today=TODAY, max_excerpt_chars=0)
    with pytest.raises(ValueError):
        parse_jsonl(_line(), today=TODAY, max_records=0)


def test_excerpt_control_characters_rejected_but_newlines_and_tabs_collapse():
    assert _field_errors(_record(excerpt="bad\x00text"), "excerpt")
    assert _field_errors(_record(excerpt="bad\x07text"), "excerpt")
    rec = validate_record(_record(excerpt="Line one.\n\tLine   two.\r\n"), today=TODAY)
    assert rec.excerpt == "Line one. Line two."


def test_excerpt_is_nfc_normalised():
    rec = validate_record(_record(excerpt="café text"), today=TODAY)
    assert rec.excerpt == "café text"


def test_non_latin_text_with_joiners_is_accepted():
    text = "यह एक कृत्रिम पाठ है।‍"
    assert validate_record(_record(excerpt=text), today=TODAY).excerpt == text


# --------------------------------------------------------------------------- warning_sign_tags
@pytest.mark.parametrize("bad", [
    [], ["a" + str(i) for i in range(11)], ["payment_request", "payment_request"], ["Payment Request"],
    ["payment-request"], ["_x"], [""], ["x" * 65],
])
def test_tag_violations(bad):
    assert _field_errors(_record(warning_sign_tags=bad), "warning_sign_tags")


def test_allowed_tags_vocabulary_is_enforced_only_when_given():
    assert validate_record(_record(warning_sign_tags=["any_slug"]), today=TODAY)
    assert _field_errors(_record(warning_sign_tags=["pressure_tactic"]), "warning_sign_tags",
                         allowed_tags={"payment_request"})
    assert validate_record(_record(), today=TODAY, allowed_tags=["payment_request"])


# --------------------------------------------------------------------------- JSON handling and error reporting
def test_invalid_json_and_duplicate_keys_are_reported_with_line_numbers():
    dup_key_line = _line().replace('{"source_id"', '{"source_id": "synthetic-009", "source_id"', 1)
    text = "\n".join([_line(), "{not json", dup_key_line])
    with pytest.raises(KnowledgeBaseError) as exc:
        parse_jsonl(text, today=TODAY)
    assert exc.value.errors[0].startswith("line 2: invalid JSON")
    assert exc.value.errors[1] == "line 3: invalid JSON (duplicate key in object)"


@pytest.mark.parametrize("line", ["[1, 2]", '"text"', "42", "null"])
def test_non_object_json_lines_are_rejected(line):
    with pytest.raises(KnowledgeBaseError) as exc:
        parse_jsonl(line, today=TODAY)
    assert exc.value.errors == ["line 1: record must be a JSON object"]


def test_error_messages_never_echo_record_text():
    marker = "SECRET-SYNTHETIC-MARKER"
    bad_url = _line(url="http://example.org/x", excerpt=marker)
    broken = '{"excerpt": "' + marker
    for text in (bad_url, broken):
        with pytest.raises(KnowledgeBaseError) as exc:
            parse_jsonl(text, today=TODAY)
        assert marker not in str(exc.value)
        assert all(marker not in e for e in exc.value.errors)


def test_multiple_errors_are_aggregated_and_capped():
    few = "\n".join([_line(url="http://a.example.org/1"), _second(retrieved_date="bad"), "{oops"])
    with pytest.raises(KnowledgeBaseError) as exc:
        parse_jsonl(few, today=TODAY)
    assert [e.split(":")[0] for e in exc.value.errors] == ["line 1", "line 2", "line 3"]

    many = "\n".join("{bad" for _ in range(MAX_REPORTED_ERRORS + 10))
    with pytest.raises(KnowledgeBaseError) as exc:
        parse_jsonl(many, today=TODAY)
    assert len(exc.value.errors) == MAX_REPORTED_ERRORS
    assert exc.value.total_errors == MAX_REPORTED_ERRORS + 10
    assert "10 more" in str(exc.value)


def test_empty_input_rules_and_record_cap():
    for empty in ("", "\n  \n"):
        with pytest.raises(KnowledgeBaseError, match="no records"):
            parse_jsonl(empty, today=TODAY)
        assert parse_jsonl(empty, today=TODAY, allow_empty=True) == ()
    text = "\n".join([_line(), _second(), _second(source_id="synthetic-003", url="https://example.org/synthetic/003")])
    with pytest.raises(KnowledgeBaseError, match="too many records"):
        parse_jsonl(text, today=TODAY, max_records=2)


# --------------------------------------------------------------------------- immutability and isolation
def test_records_are_immutable():
    rec = parse_jsonl(_line(), today=TODAY)[0]
    with pytest.raises(FrozenInstanceError):
        rec.title = "changed"
    assert isinstance(rec.warning_sign_tags, tuple)


def test_module_is_standard_library_only_offline_and_decoupled():
    tree = ast.parse(Path(kb.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports would couple this module to other backend modules"
            imported.add(node.module)
    assert {name.split(".")[0] for name in imported} <= set(sys.stdlib_module_names)
    assert "backend" not in {name.split(".")[0] for name in imported}
    banned = ("socket", "ssl", "http", "ftplib", "smtplib", "xmlrpc", "subprocess", "asyncio", "urllib.request")
    assert not [n for n in imported if any(n == b or n.startswith(b + ".") for b in banned)]
