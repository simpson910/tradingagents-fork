"""Extract the 5-tier closing lean from the Editor's decision.

The Editor produces a typed ``PortfolioDecision`` via structured output and
renders it to markdown that always carries a ``**Lean**: X`` header (see
:func:`tradingagents.agents.schemas.render_pm_decision`).  The deterministic
heuristic in :mod:`tradingagents.agents.utils.rating` is more than
sufficient to extract that lean; no extra LLM call is needed.

This module exists for backwards compatibility with callers that expect a
``SignalProcessor.process_signal(text)`` interface.
"""

from __future__ import annotations

from typing import Any

from tradingagents.agents.utils.rating import parse_rating


class SignalProcessor:
    """Read the 5-tier closing lean out of an Editor decision."""

    def __init__(self, quick_thinking_llm: Any = None):
        # The LLM argument is accepted for backwards compatibility but no
        # longer used: the Editor's structured output guarantees the lean is
        # parseable from the rendered markdown without a second LLM call.
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """Return one of Strong Bearish / Bearish Lean / Neutral / Bullish Lean / Strong Bullish."""
        return parse_rating(full_signal)
