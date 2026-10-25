"""Load the local trusted-guidance file for retrieval, with clear status reporting and a small cache.

SCOPE: this module only reads ONE LOCAL JSONL file through ``knowledge_base.load_knowledge_base``. It does not fetch
URLs, call an LLM, rank or retrieve evidence, or touch the API response contract. Its output
(``GuidanceLoadResult.records``) is meant to be passed to ``evidence_retrieval.retrieve_evidence`` by a later step.

BEHAVIOUR
  * Missing, empty, invalid or unreadable file -> no records, a status, and a fixed user-safe note. Never an
    exception, and never invented evidence.
  * An invalid file is rejected AS A WHOLE. Lines that look valid are not used, because a half-loaded file hides
    mistakes. Details (line number, field, reason; never the text itself) go to the developer log, not to the user.
  * The result is cached per (path, modified time, size). Creating, editing or deleting the file refreshes it
    automatically; ``clear_guidance_cache()`` forces a reload.
  * Only the file reading and validation are cached. ``retrieve_evidence`` still builds its TF-IDF index on every
    call; caching that would need a separate, approved change to ``evidence_retrieval.py``.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from backend.services.knowledge_base import KnowledgeBaseError, KnowledgeBaseRecord, load_knowledge_base

logger = logging.getLogger(__name__)

# backend/services/guidance_store.py -> backend/data/trusted_guidance.jsonl
DEFAULT_GUIDANCE_PATH = Path(__file__).resolve().parents[1] / "data" / "trusted_guidance.jsonl"

STATUS_LOADED = "loaded"
STATUS_MISSING = "missing"
STATUS_EMPTY = "empty"
STATUS_INVALID = "invalid"
STATUS_UNREADABLE = "unreadable"

# Fixed text, safe to show to a user. Suitable for the existing ``uncertainty_notes`` list.
NO_GUIDANCE_NOTE = "No trusted-guidance sources were available, so no guidance references are shown."


@dataclass(frozen=True)
class GuidanceLoadResult:
    """Outcome of loading the guidance file. ``records`` is empty unless ``status`` is ``"loaded"``."""

    status: str
    records: tuple[KnowledgeBaseRecord, ...] = ()
    uncertainty_note: str | None = None

    @property
    def available(self) -> bool:
        return self.status == STATUS_LOADED


_lock = threading.Lock()
_cache: tuple[tuple, GuidanceLoadResult] | None = None  # (signature, result): one entry is enough


def _unavailable(status: str) -> GuidanceLoadResult:
    return GuidanceLoadResult(status=status, records=(), uncertainty_note=NO_GUIDANCE_NOTE)


def _read(path: Path) -> GuidanceLoadResult:
    """Read and validate the file once. Never raises for file problems."""
    try:
        records = load_knowledge_base(path, allow_empty=True)
    except FileNotFoundError:
        logger.warning("Trusted-guidance file not found: %s", path)
        return _unavailable(STATUS_MISSING)
    except KnowledgeBaseError as exc:
        logger.error(
            "Trusted-guidance file %s is invalid (%d problem(s)); guidance is disabled until it is fixed:\n%s",
            path, exc.total_errors, "\n".join(exc.errors),
        )
        return _unavailable(STATUS_INVALID)
    except OSError as exc:
        logger.error("Trusted-guidance file %s could not be read (%s).", path, type(exc).__name__)
        return _unavailable(STATUS_UNREADABLE)
    if not records:
        logger.warning("Trusted-guidance file %s contains no records.", path)
        return _unavailable(STATUS_EMPTY)
    logger.info("Loaded %d trusted-guidance record(s) from %s.", len(records), path)
    return GuidanceLoadResult(status=STATUS_LOADED, records=records)


def load_guidance(path: str | Path | None = None) -> GuidanceLoadResult:
    """Return the validated guidance records (or a clear "not available" result), using a small cache.

    ``path`` defaults to ``DEFAULT_GUIDANCE_PATH``. A cheap ``stat`` call on every request detects file changes;
    the file is only parsed again when it is new, edited or deleted.
    """
    global _cache
    target = Path(path) if path is not None else DEFAULT_GUIDANCE_PATH
    key = str(target.absolute())
    try:
        stat = target.stat()
        signature: tuple = (key, stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        signature = (key, "missing")
    except OSError:
        logger.error("Trusted-guidance file %s could not be inspected.", target)
        return _unavailable(STATUS_UNREADABLE)

    with _lock:
        if _cache is not None and _cache[0] == signature:
            return _cache[1]
        result = _read(target)
        _cache = (signature, result)
        return result


def clear_guidance_cache() -> None:
    """Forget the cached result so the next ``load_guidance`` call reads the file again."""
    global _cache
    with _lock:
        _cache = None
