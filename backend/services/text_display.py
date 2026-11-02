"""Display-only text helpers. Nothing here affects a score or a rule."""
from __future__ import annotations

ELLIPSIS = "…"
_TRAILING = " ,;:-–—"


def shorten_quote(text, limit: int) -> str:
    """
    Text of at most about `limit` characters for a quote. A longer text is cut at the last word boundary before the
    limit and ends with an ellipsis, so it never stops in the middle of a word. A single word longer than the limit
    is kept whole (up to twice the limit; only a longer unbroken run, such as a pasted link, is cut hard).
    """
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > 0:
        return cut[:space].rstrip(_TRAILING) + ELLIPSIS
    first_word = text.split(" ", 1)[0]
    if len(first_word) <= 2 * limit:
        return first_word + (ELLIPSIS if len(first_word) < len(text) else "")
    return cut + ELLIPSIS
