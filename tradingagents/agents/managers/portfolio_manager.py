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

Be decisive and ground every conclusion in specific evidence from the analysts. Frame everything as analytical observation — describe what the case implies for the long thesis, not what a reader should do. Do not recommend, advocate for, or describe specific actions, position sizes, entries, exits, or price zones at which a reader should add or trim exposure — even when synthesising or quoting the analysts' framings. The closing view should read like a research-note analytical paragraph, not like a portfolio-management instruction.{get_style_instruction()}"""

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
