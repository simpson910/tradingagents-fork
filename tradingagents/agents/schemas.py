"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Editor) so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the lean-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged

Steelman fork note: class symbol names (BUY/OVERWEIGHT/...) are deliberately
kept stable to make upstream rebases clean; only the ``.value`` strings,
field descriptions, render output, and docstring role names carry the
compliance-softened labels that reach the LLM and surface in user output.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared lean types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier closing lean used by the Research Manager and the Editor."""

    BUY = "Strong Bullish"
    OVERWEIGHT = "Bullish Lean"
    HOLD = "Neutral"
    UNDERWEIGHT = "Bearish Lean"
    SELL = "Strong Bearish"


class TraderAction(str, Enum):
    """3-tier directional sub-lean used by the Trader.

    The Trader's job is to translate the Research Manager's interim view into
    a concrete sub-lean: should the analytical case lean Bullish, Bearish, or
    sit Neutral.  The full 5-tier nuance (Bullish Lean vs Strong Bullish,
    etc.) is left to the Editor.
    """

    BUY = "Bullish"
    HOLD = "Neutral"
    SELL = "Bearish"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional lean,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    analytical follow-ups.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The directional lean on the long thesis. Exactly one of "
            "Strong Bearish / Bearish Lean / Neutral / Bullish Lean / "
            "Strong Bullish. Reserve Neutral for situations where the "
            "evidence on both sides is genuinely balanced; otherwise "
            "commit to the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the lean. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete analytical follow-ups for the trader: what would "
            "tighten the thesis, what would change the lean, what to "
            "monitor. Frame as research questions, not trade instructions."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Lean**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured directional sub-lean produced by the Trader.

    The trader reads the Research Manager's interim view and the analyst
    reports, then turns them into a concrete sub-lean: which way the
    analytical case leans, the reasoning that justifies it, and the
    practical levels (implied entry, thesis-implied stop, conviction
    sizing) the analytical case implies.
    """

    action: TraderAction = Field(
        description="The directional sub-lean. Exactly one of Bullish / Neutral / Bearish.",
    )
    reasoning: str = Field(
        description=(
            "The analytical case for this sub-lean, anchored in the analysts' "
            "reports and the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description=(
            "Optional implied entry zone in the instrument's quote currency. "
            "Frame as 'the analytical case implies an entry zone of $X', "
            "never as 'buy at $X'."
        ),
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description=(
            "Optional thesis-implied stop reference in the instrument's quote "
            "currency. Frame as 'the thesis-implied stop reference is $Y', "
            "never as 'set a stop-loss at $Y'."
        ),
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description=(
            "Optional conviction sizing implied by thesis strength "
            "(e.g. 'high conviction', 'moderate conviction', 'cautious'). "
            "Never reference the user's actual portfolio or recommend a "
            "percentage of it."
        ),
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``CLOSING SUB-LEAN: **BULLISH/NEUTRAL/BEARISH**`` line is
    preserved as the coordination signal between agents (was
    ``FINAL TRANSACTION PROPOSAL`` upstream; renamed in this fork for
    compliance posture).  Any external code that greps for the old string
    needs updating; ``parse_rating`` and other in-tree consumers are kept
    aligned.
    """
    parts = [
        f"**Sub-lean**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Implied Entry**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Thesis-Implied Stop**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Conviction Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"CLOSING SUB-LEAN: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Editor  (upstream name: Portfolio Manager)
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Editor.

    Class name kept as ``PortfolioDecision`` to keep upstream rebases clean;
    the user-facing role is the Editor in prompts and output.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required.  Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the lean-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The closing lean on the long thesis. Exactly one of "
            "Strong Bearish / Bearish Lean / Neutral / Bullish Lean / "
            "Strong Bullish, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise closing view covering the implied entry zone, "
            "conviction sizing, key thesis-implied levels, and time horizon. "
            "Two to four sentences.  Frame as analytical observation, never "
            "as personalized advice."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate.  If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional thesis-implied price level in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional analytical time horizon for the lean, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Lean**``,
    ``**Closing View**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Lean**: {decision.rating.value}",
        "",
        f"**Closing View**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Thesis-Implied Level**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)
