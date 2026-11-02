"""Display-only helper shared by job and chat reports: the shown score is never exactly 0 or 100."""
from __future__ import annotations

DISPLAY_MIN = 1
DISPLAY_MAX = 99


def clamp_display_score(value: float) -> int:
    """Truncate to a whole number and clamp to [1, 99]. Monotone: a higher input never gives a lower score."""
    return int(min(float(DISPLAY_MAX), max(float(DISPLAY_MIN), float(value))))
