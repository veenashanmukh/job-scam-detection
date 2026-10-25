"""Trusted-guidance knowledge base: record schema, strict validation and JSONL parsing.

SCOPE (this module only does validation and parsing):
  * It does NOT fetch URLs, call external services, check that a URL is live, or process source documents.
  * It does NOT retrieve, rank, embed, or call any LLM, and it imports nothing from other ``backend`` modules.
  * It cannot judge whether a ``license_note`` is *correct*, only that one was recorded. Reuse terms for every real
    source must be reviewed by a person before real content is added.
  * It cannot reliably detect prompt-injection text inside excerpts. Excerpts are untrusted data; that is handled
    later, when prompts are built.
  * Guidance records are a separate evidence type from labelled job postings and are not part of the API response
    contract (``backend/schemas/analysis_response.md``).

PROVISIONAL LIMIT: ``MAX_EXCERPT_CHARS = 1000`` is an initial, provisional default. It is NOT a final limit for real
source content; set the real limit after the source/reuse-terms review. Override it per call with the
``max_excerpt_chars`` argument.

File format: JSONL, one JSON object per line, exactly these keys:
    source_id, publisher, title, url, retrieved_date (YYYY-MM-DD), license_note, excerpt, warning_sign_tags (list)
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

__all__ = [
    "REQUIRED_FIELDS", "MAX_EXCERPT_CHARS", "MAX_FILE_BYTES", "MAX_RECORDS", "MAX_REPORTED_ERRORS",
    "KnowledgeBaseError", "KnowledgeBaseRecord", "validate_record", "parse_jsonl", "load_knowledge_base",
]

REQUIRED_FIELDS = ("source_id", "publisher", "title", "url", "retrieved_date", "license_note", "excerpt",
                   "warning_sign_tags")

MAX_EXCERPT_CHARS = 1000      # PROVISIONAL initial limit; not final for real source content (see module docstring)
MAX_PUBLISHER_CHARS = 200
MAX_TITLE_CHARS = 300
MAX_LICENSE_NOTE_CHARS = 500
MAX_URL_CHARS = 2048
MAX_TAGS = 10
MAX_TAG_CHARS = 64
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_RECORDS = 5000
MAX_REPORTED_ERRORS = 50

_SOURCE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}")
_TAG_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_WHITESPACE_RE = re.compile(r"\s+")
_PLACEHOLDER_LICENSE_NOTES = frozenset({"todo", "tbd", "unknown", "n/a"})


class KnowledgeBaseError(ValueError):
    """Raised when knowledge-base data is invalid.

    ``errors`` holds at most ``MAX_REPORTED_ERRORS`` messages (line, field, reason; never the offending text);
    ``total_errors`` is the full count.
    """

    def __init__(self, errors: Collection[str]):
        all_errors = list(errors)
        self.total_errors = len(all_errors)
        self.errors = all_errors[:MAX_REPORTED_ERRORS]
        message = f"{self.total_errors} problem(s) in knowledge base data:\n" + "\n".join(f"  - {e}" for e in self.errors)
        hidden = self.total_errors - len(self.errors)
        if hidden:
            message += f"\n  ... and {hidden} more not shown"
        super().__init__(message)


@dataclass(frozen=True)
class KnowledgeBaseRecord:
    """One validated, immutable guidance record."""

    source_id: str
    publisher: str
    title: str
    url: str
    retrieved_date: date
    license_note: str
    excerpt: str
    warning_sign_tags: tuple[str, ...]

    def to_dict(self) -> dict:
        """JSON-compatible form (ISO date string, tags as a list)."""
        return {
            "source_id": self.source_id, "publisher": self.publisher, "title": self.title, "url": self.url,
            "retrieved_date": self.retrieved_date.isoformat(), "license_note": self.license_note,
            "excerpt": self.excerpt, "warning_sign_tags": list(self.warning_sign_tags),
        }


# --------------------------------------------------------------------------- field validators
class _FieldError(Exception):
    """Internal: one field failed validation; the message is the reason."""


def _has_control_char(text: str) -> bool:
    return any(unicodedata.category(ch) == "Cc" for ch in text)


def _check_source_id(value) -> str:
    if not isinstance(value, str):
        raise _FieldError("must be a string")
    if not _SOURCE_ID_RE.fullmatch(value):
        raise _FieldError("must be 3-64 characters: lowercase letters, digits, '_' or '-', starting with a letter or digit")
    return value


def _check_text(value, max_chars: int) -> str:
    if not isinstance(value, str):
        raise _FieldError("must be a string")
    text = value.strip()
    if not text:
        raise _FieldError("must not be empty")
    if _has_control_char(text):
        raise _FieldError("must not contain control characters")
    if len(text) > max_chars:
        raise _FieldError(f"must be at most {max_chars} characters")
    return text


def _check_url(value) -> str:
    if not isinstance(value, str):
        raise _FieldError("must be a string")
    if not value:
        raise _FieldError("must not be empty")
    if any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in value):
        raise _FieldError("must not contain whitespace or control characters")
    if len(value) > MAX_URL_CHARS:
        raise _FieldError(f"must be at most {MAX_URL_CHARS} characters")
    try:  # parsing only: the URL is never fetched
        parts = urlsplit(value)
        host = parts.hostname
    except ValueError:
        raise _FieldError("is not a valid URL") from None
    if parts.scheme != "https":
        raise _FieldError("must use https")
    if "@" in parts.netloc:
        raise _FieldError("must not contain credentials")
    if not host:
        raise _FieldError("must include a host")
    return value


def _check_date(value, today: date) -> date:
    if not isinstance(value, str):
        raise _FieldError("must be a string in YYYY-MM-DD format")
    if not _DATE_RE.fullmatch(value):
        raise _FieldError("must be a date in YYYY-MM-DD format")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise _FieldError("is not a real calendar date") from None
    if parsed > today:
        raise _FieldError("must not be in the future")
    return parsed


def _check_license_note(value) -> str:
    text = _check_text(value, MAX_LICENSE_NOTE_CHARS)
    if text.casefold() in _PLACEHOLDER_LICENSE_NOTES:
        raise _FieldError("must record real reuse terms, not a placeholder")
    return text


def _check_excerpt(value, max_chars: int) -> str:
    if not isinstance(value, str):
        raise _FieldError("must be a string")
    text = unicodedata.normalize("NFC", value)
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in text):
        raise _FieldError("must not contain control characters")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        raise _FieldError("must not be empty")
    if len(text) > max_chars:
        raise _FieldError(f"must be at most {max_chars} characters")
    return text


def _check_tags(value, allowed: frozenset[str] | None) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _FieldError("must be a list of strings")
    if not 1 <= len(value) <= MAX_TAGS:
        raise _FieldError(f"must contain between 1 and {MAX_TAGS} tags")
    seen: set[str] = set()
    for position, tag in enumerate(value, start=1):
        if not isinstance(tag, str) or len(tag) > MAX_TAG_CHARS or not _TAG_RE.fullmatch(tag):
            raise _FieldError(f"tag #{position} must be a lowercase slug such as 'payment_request'")
        if tag in seen:
            raise _FieldError(f"duplicate tag {tag!r}")
        seen.add(tag)
        if allowed is not None and tag not in allowed:
            raise _FieldError(f"tag {tag!r} is not in the allowed vocabulary")
    return tuple(value)


# --------------------------------------------------------------------------- record / file validation
def _check_record(obj, line_no: int | None, today: date, allowed: frozenset[str] | None,
                  max_excerpt_chars: int) -> tuple[KnowledgeBaseRecord | None, list[str]]:
    """Validate one decoded JSON value. Returns (record, []) or (None, errors)."""
    prefix = f"line {line_no}: " if line_no is not None else ""
    if not isinstance(obj, dict):
        return None, [f"{prefix}record must be a JSON object"]

    errors: list[str] = []
    for key in obj:
        if key not in REQUIRED_FIELDS:
            errors.append(f"{prefix}unknown field {str(key)[:64]!r}")
    for name in REQUIRED_FIELDS:
        if name not in obj:
            errors.append(f"{prefix}field '{name}': missing")

    checks = {
        "source_id": _check_source_id,
        "publisher": lambda v: _check_text(v, MAX_PUBLISHER_CHARS),
        "title": lambda v: _check_text(v, MAX_TITLE_CHARS),
        "url": _check_url,
        "retrieved_date": lambda v: _check_date(v, today),
        "license_note": _check_license_note,
        "excerpt": lambda v: _check_excerpt(v, max_excerpt_chars),
        "warning_sign_tags": lambda v: _check_tags(v, allowed),
    }
    clean: dict = {}
    for name, check in checks.items():
        if name not in obj:
            continue
        try:
            clean[name] = check(obj[name])
        except _FieldError as exc:
            errors.append(f"{prefix}field '{name}': {exc}")
    if errors:
        return None, errors
    return KnowledgeBaseRecord(**clean), []


def _check_limits(max_excerpt_chars: int, max_records: int) -> None:
    if max_excerpt_chars < 1:
        raise ValueError("max_excerpt_chars must be at least 1")
    if max_records < 1:
        raise ValueError("max_records must be at least 1")


def validate_record(obj, *, line_no: int | None = None, today: date | None = None,
                    allowed_tags: Collection[str] | None = None,
                    max_excerpt_chars: int = MAX_EXCERPT_CHARS) -> KnowledgeBaseRecord:
    """Validate one decoded JSON object and return an immutable record, or raise ``KnowledgeBaseError``."""
    _check_limits(max_excerpt_chars, MAX_RECORDS)
    allowed = None if allowed_tags is None else frozenset(allowed_tags)
    record, errors = _check_record(obj, line_no, today or date.today(), allowed, max_excerpt_chars)
    if record is None:
        raise KnowledgeBaseError(errors)
    return record


def _reject_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key in object")
    return dict(pairs)


def parse_jsonl(text: str, *, today: date | None = None, allowed_tags: Collection[str] | None = None,
                allow_empty: bool = False, max_excerpt_chars: int = MAX_EXCERPT_CHARS,
                max_records: int = MAX_RECORDS) -> tuple[KnowledgeBaseRecord, ...]:
    """Parse and validate JSONL text. All problems are collected and raised together as ``KnowledgeBaseError``.

    Beyond per-record rules this rejects duplicate ``source_id`` values and records that repeat the same ``url`` and
    ``excerpt`` under different ids. (The same URL with different excerpts is allowed.) Blank lines are skipped.
    """
    _check_limits(max_excerpt_chars, max_records)
    today = today or date.today()
    allowed = None if allowed_tags is None else frozenset(allowed_tags)
    if text.startswith("﻿"):
        text = text[1:]

    records: list[KnowledgeBaseRecord] = []
    errors: list[str] = []
    id_line: dict[str, int] = {}
    content_line: dict[tuple[str, str], int] = {}
    nonblank = 0
    # Split on "\n" only: str.splitlines() would also split on U+2028 and similar characters that are legal inside JSON strings.
    for line_no, raw in enumerate(text.split("\n"), start=1):
        raw = raw.rstrip("\r")
        if not raw.strip():
            continue
        nonblank += 1
        if nonblank > max_records:
            errors.append(f"line {line_no}: too many records (limit {max_records})")
            break
        try:
            obj = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except RecursionError:
            errors.append(f"line {line_no}: invalid JSON (nesting too deep)")
            continue
        except json.JSONDecodeError as exc:  # message holds position info only, never the line's text
            errors.append(f"line {line_no}: invalid JSON ({exc.msg} at column {exc.colno})")
            continue
        except ValueError as exc:  # duplicate key
            errors.append(f"line {line_no}: invalid JSON ({exc})")
            continue

        record, problems = _check_record(obj, line_no, today, allowed, max_excerpt_chars)
        errors.extend(problems)
        if record is None:
            continue
        if record.source_id in id_line:
            errors.append(f"line {line_no}: field 'source_id': duplicate of line {id_line[record.source_id]}")
            continue
        id_line[record.source_id] = line_no
        key = (record.url, record.excerpt)
        if key in content_line:
            errors.append(f"line {line_no}: same url and excerpt as line {content_line[key]} under a different source_id")
            continue
        content_line[key] = line_no
        records.append(record)

    if errors:
        raise KnowledgeBaseError(errors)
    if not records and not allow_empty:
        raise KnowledgeBaseError(["file contains no records"])
    return tuple(records)


def load_knowledge_base(path: str | Path, *, max_file_bytes: int = MAX_FILE_BYTES, today: date | None = None,
                        allowed_tags: Collection[str] | None = None, allow_empty: bool = False,
                        max_excerpt_chars: int = MAX_EXCERPT_CHARS,
                        max_records: int = MAX_RECORDS) -> tuple[KnowledgeBaseRecord, ...]:
    """Read a local UTF-8 JSONL file and validate it. Local file read only; no network access."""
    with Path(path).open("rb") as handle:
        data = handle.read(max_file_bytes + 1)
    if len(data) > max_file_bytes:
        raise KnowledgeBaseError([f"file is larger than {max_file_bytes} bytes"])
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise KnowledgeBaseError(["file is not valid UTF-8"]) from None
    return parse_jsonl(text, today=today, allowed_tags=allowed_tags, allow_empty=allow_empty,
                       max_excerpt_chars=max_excerpt_chars, max_records=max_records)
