from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_style_instruction() -> str:
    """Return prompt instructions for user-facing output style.

    Two rules concatenated:
    1. Editorial style — no emojis or decorative symbols. DeepSeek (and
       most modern LLMs) decorate Markdown with 📊 🟢 🔴 etc. by default,
       which breaks Steelman's editorial tone (Stratechery / Seeking
       Alpha / FinChat don't use emojis in research notes). We tell the
       model not to.
    2. Output language — only appended if a non-English language is
       configured. Internal debate agents stay in English for reasoning
       quality regardless.

    Applied to user-facing agents only (analysts, portfolio manager).
    """
    style = " Use plain prose and Markdown tables only. Do not use emojis, decorative icons, or symbol bullets."

    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return style
    return style + f" Write your entire response in {lang}."


# === STEELMAN PATCH: risk-panel new-dimension requirement (2026-05-19) ===
# Reason: cross-report eval of 6 V4-Flash debates found that the 3-way
# risk panel (Aggressive / Conservative / Neutral) consistently reworded
# arguments already raised in the Round-1 Bull/Bear debate or in the
# four analyst reports. ~25% of LangGraph compute wasted on
# rehash. Risk panel's editorial value comes from introducing dimensions
# the structural debate missed — tail risk, optionality, correlation /
# hedge implications, regulatory scenarios, time-horizon arbitrage.
# Detection in scripts/eval-reports.mjs is soft (keyword presence in
# risk_debate_history); the prompt-side requirement here is the real
# fix.
# === END STEELMAN PATCH ===
def get_risk_panel_new_dimension_instruction() -> str:
    """Append to risk-panel debater prompts. Forces new-dimension surfacing."""
    return (
        " Your responsibility is to surface at least one analytical "
        "dimension that the four analyst reports (Market / Sentiment / "
        "News / Fundamentals) did not cover. Acceptable dimensions "
        "include: tail-risk stress scenarios (what if the multiple "
        "compresses 30% or revenue contracts 20%), unpriced optionality "
        "(scenarios neither the bull nor bear case has discounted), "
        "correlation / hedge implications (how this name moves with its "
        "sector or with a paired ticker), regulatory or policy scenarios "
        "(legal, antitrust, tariff, rate environment), and time-horizon "
        "arbitrage (short-term mispricing vs long-term thesis). Do NOT "
        "reword arguments already in the analyst reports — that is "
        "wasted compute. Name the new dimension explicitly."
    )


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
