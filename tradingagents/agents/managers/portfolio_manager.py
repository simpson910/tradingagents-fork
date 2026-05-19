"""Editor: synthesises the risk-analyst debate into the closing view.

The class symbol name ``PortfolioDecision`` and the file path
``managers/portfolio_manager.py`` are kept stable in this fork to keep
upstream rebases clean. The LangGraph node label, the structured-output
label, and the prompt body all say "Editor" — the SaaS frontend streams
the node label to users, so any "Portfolio Manager" leak would hit
Steelman's fiduciary-vocab ban.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_style_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


# === STEELMAN PATCH: anti-template framing (2026-05-19) ===
# Reason: cross-report eval of 6 V4-Flash debates found the same
# asymmetric framing template ("the bull case requires X simultaneous
# victories / the bear case only needs Y failures") reused unchanged
# across NFLX, META, and MSFT — rhetorical pattern, not evidence-driven
# analysis. Detection lives in scripts/eval-reports.mjs as the
# `bull-requires-simultaneous` / `bear-only-needs` / `long-requires-
# execution` regex set. On upstream re-port, preserve this instruction
# by appending its body inside the Editor prompt's f-string.
# === END STEELMAN PATCH ===
_ANTI_TEMPLATE_INSTRUCTION = (
    "\n\nAvoid templated asymmetric framings of the form \"the bull case "
    "requires X simultaneous victories\" or \"the bear case only needs one "
    "to fail\" or \"the long thesis requires near-perfect execution.\" "
    "That structure exaggerates the asymmetry between the two sides "
    "through rhetoric rather than evidence. If one side genuinely has a "
    "lower bar to being right, identify the specific datapoint or "
    "condition that makes it so, name it, and explain why — do not "
    "assert the asymmetry as a structural truism."
)


# === STEELMAN PATCH: weighting transparency (2026-05-19) ===
# Reason: cross-report eval found Editor's final lean averaged 1.05
# rungs more bearish than the sub-lean average across 6 V4-Flash
# debates — every report negative. Most extreme: NVDA where all four
# sub-leans were Bullish but Editor closed Neutral; MSFT where three
# of four were Bullish but Editor closed Bearish Lean. The downshift
# is structural, not data-driven — Editor synthesises silently using
# language like "prudent caution warranted" rather than naming which
# sub-lean got more weight and why. This patch forces the weighting
# to be explicit and anchored to a specific datapoint whenever the
# Editor's lean differs from the analyst majority. Detection: the
# `Editor delta vs sub-lean average` axis in scripts/eval-reports.mjs;
# expected effect over time is mean delta converging toward zero.
# === END STEELMAN PATCH ===
_WEIGHTING_TRANSPARENCY_INSTRUCTION = (
    "\n\nIf your closing lean differs from the simple majority of analyst "
    "sub-leans (Market / Sentiment / News / Fundamentals), state explicitly "
    "which sub-lean(s) you weighted more heavily and name the specific "
    "datapoint that justifies the weighting. Acceptable form: \"I weight "
    "the technical setup over the unanimous fundamental Bullish reading "
    "because [named indicator + named historical precedent + named time "
    "horizon].\" Not acceptable: \"prudent caution is warranted,\" \"the "
    "evidence tilts bearish,\" or any phrasing that asserts a weighting "
    "without naming the data. The synthesis is a named, evidence-anchored "
    "weighting decision — never a default toward caution."
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Editor")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Editor, synthesize the risk analysts' debate and deliver the closing view.

{instrument_context}

---

**Lean Scale** (use exactly one):
- **Strong Bullish**: High confidence the bull case is right
- **Bullish Lean**: The bull case is well-supported and outweighs the bear case
- **Neutral**: The evidence on both sides is genuinely balanced — reserve for true uncertainty, not a default
- **Bearish Lean**: The bear case is well-supported and the long thesis has meaningful downside risk
- **Strong Bearish**: High confidence the bear case is right

**Context:**
- Research Manager's interim view: **{research_plan}**
- Trader's directional sub-lean: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts. Frame everything as analytical observation — describe what the case implies for the long thesis, not what a reader should do. Do not recommend, advocate for, or describe specific actions, position sizes, entries, exits, or price zones at which a reader should add or trim exposure — even when synthesising or quoting the analysts' framings. The closing view should read like a research-note analytical paragraph, not like a portfolio-management instruction.{_ANTI_TEMPLATE_INSTRUCTION}{_WEIGHTING_TRANSPARENCY_INSTRUCTION}{get_style_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Editor",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
