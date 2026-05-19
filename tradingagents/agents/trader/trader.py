"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


# STEELMAN PATCH (2026-05-15, marked 2026-05-19): the entire prompt below
# is a Steelman rewrite of the upstream Trader role. Key Steelman edits:
# - Vocabulary: "directional sub-lean" replaces upstream Buy/Hold/Sell
#   action words; uses Bullish/Neutral/Bearish trichotomy
# - Framing: "analytical observation… never what a reader should do"
#   guardrail to keep editorial-commentary posture
# - Structured output: TraderProposal Pydantic schema (no upstream
#   "TraderAction" enum with Buy/Sell values)
# On upstream re-port, replace any returned upstream prompt entirely
# rather than merging diffs — this rewrite is total, not additive.


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research-team trader analyzing market data to assess "
                    "the directional sub-lean for the long thesis. Based on your "
                    "analysis, provide a specific Bullish, Neutral, or Bearish "
                    "sub-lean. Anchor your reasoning in the analysts' reports and "
                    "the research plan. Frame everything as analytical observation "
                    "— describe what the case implies, never what a reader should do."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an interim "
                    f"view tailored for {company_name}. {instrument_context} This view incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this as a foundation for evaluating the "
                    f"directional sub-lean.\n\nResearch Manager's interim view: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed analytical assessment."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
