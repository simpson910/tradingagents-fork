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

# Matches a markdown-bold phrase: **anything**. Used to count the tiers
# the editor explicitly emphasised vs. passing prose references.
_BOLD_PHRASE_RE = re.compile(r"\*\*\s*([^*]+?)\s*\*\*")


def parse_rating(text: str, default: str = "Neutral") -> str:
    """Heuristically extract a 5-tier closing lean from prose text.

    Three-pass strategy:
    1. Look for an explicit "Lean: X" label line (tolerant of markdown
       bold and hyphen vs colon separator).
    2. Count tiers inside markdown bold (``**X**``).  The editor emphasises
       the actual stance; passing references to adjacent tiers ("supports
       a **Bearish Lean** over a Strong Bearish stance") stay plain. Most
       frequent bold tier wins; ties resolve toward Neutral (more
       conservative).  This prevents a single unbolded mention of an
       adjacent tier from hijacking the parser.
    3. Fall back to scanning for any 5-tier phrase anywhere in the text
       (legacy behaviour for un-formatted output).

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

    bold_counts = {c: 0 for c in RATINGS_5_TIER}
    for m in _BOLD_PHRASE_RE.finditer(text):
        captured = m.group(1).strip().lower()
        for canonical, lower in zip(RATINGS_5_TIER, _RATING_LOWER):
            if captured == lower:
                bold_counts[canonical] += 1
                break
    nonzero = [(c, n) for c, n in bold_counts.items() if n > 0]
    if nonzero:
        nonzero.sort(key=lambda x: (-x[1], abs(RATINGS_5_TIER.index(x[0]) - 2)))
        return nonzero[0][0]

    text_lower = text.lower()
    for canonical, lower in zip(RATINGS_5_TIER, _RATING_LOWER):
        if lower in text_lower:
            return canonical

    return default
