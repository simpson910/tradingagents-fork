"""Shared 5-tier closing-lean vocabulary and a deterministic heuristic parser.

The same five-tier scale (Strong Bearish / Bearish Lean / Neutral /
Bullish Lean / Strong Bullish) is used by:
- The Research Manager (interim view)
- The Editor (closing view / final position lean)
- The signal processor (lean extracted for downstream consumers)
- The memory log (lean tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.

Steelman fork note: tuple values are the user-facing labels (multi-word)
rather than the Python-internal enum class names (BUY/OVERWEIGHT/...).
The class names are kept stable in :mod:`tradingagents.agents.schemas`
to keep upstream rebases clean; the parser here matches against the
``.value`` strings the LLM actually emits.
"""

from __future__ import annotations

import re
from typing import Tuple


# Canonical, ordered 5-tier scale (most bearish to most bullish).
RATINGS_5_TIER: Tuple[str, ...] = (
    "Strong Bearish",
    "Bearish Lean",
    "Neutral",
    "Bullish Lean",
    "Strong Bullish",
)

# Lower-cased canonical forms for tolerant comparison.
_RATING_LOWER = tuple(r.lower() for r in RATINGS_5_TIER)

# Matches "Lean: X" / "lean - X" / "Lean: **X**" / "**Lean**: X".
# Captures everything after the separator up to end-of-line so multi-word
# values like "Strong Bullish" survive.
_LEAN_LABEL_RE = re.compile(r"\blean\b.*?[:\-]\s*([^\n]+)", re.IGNORECASE)


def parse_rating(text: str, default: str = "Neutral") -> str:
    """Heuristically extract a 5-tier closing lean from prose text.

    Two-pass strategy:
    1. Look for an explicit "Lean: X" label (tolerant of markdown bold and
       hyphen vs colon separator).  Returns the canonical capitalisation.
    2. Fall back to scanning for any 5-tier lean phrase anywhere in the text.

    Returns one of the canonical strings in :data:`RATINGS_5_TIER`, or
    ``default`` if no lean phrase is found.
    """
    for line in text.splitlines():
        m = _LEAN_LABEL_RE.search(line)
        if m:
            captured = m.group(1).strip().strip("*").strip().lower()
            for canonical, lower in zip(RATINGS_5_TIER, _RATING_LOWER):
                if captured.startswith(lower):
                    return canonical

    text_lower = text.lower()
    for canonical, lower in zip(RATINGS_5_TIER, _RATING_LOWER):
        if lower in text_lower:
            return canonical

    return default
