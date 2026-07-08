"""Shared-router path-map completeness with Steelman's Editor terminal node."""

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import DEBATE_PATH_MAP, RISK_ANALYSIS_PATH_MAP


def _risk_state(latest_speaker, count=0):
    return {"risk_debate_state": {"latest_speaker": latest_speaker, "count": count}}


def _debate_state(current_response, count=0):
    return {"investment_debate_state": {"current_response": current_response, "count": count}}


@pytest.mark.unit
@pytest.mark.parametrize(
    "latest_speaker",
    [
        "Aggressive",
        "Aggressive Analyst",
        "Conservative",
        "Conservative Analyst",
        "Neutral",
        "Neutral Analyst",
        "",
        "Aggressive Risk Analyst",
        "Agresivo",
    ],
)
def test_risk_router_return_always_routable(latest_speaker):
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    target = logic.should_continue_risk_analysis(_risk_state(latest_speaker))
    assert target in RISK_ANALYSIS_PATH_MAP


@pytest.mark.unit
def test_risk_router_terminates_at_editor():
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    assert logic.should_continue_risk_analysis(_risk_state("Neutral", count=3)) == "Editor"
    assert RISK_ANALYSIS_PATH_MAP["Editor"] == "Editor"


@pytest.mark.unit
def test_risk_path_map_covers_full_router_range():
    logic = ConditionalLogic(max_risk_discuss_rounds=1)
    returns = {
        logic.should_continue_risk_analysis(_risk_state(s, c))
        for s in ("Aggressive", "Conservative", "Neutral", "drift")
        for c in (0, 99)
    }
    assert returns <= set(RISK_ANALYSIS_PATH_MAP)
    assert "Editor" in returns


@pytest.mark.unit
@pytest.mark.parametrize(
    "current_response",
    [
        "Bull",
        "Bull Researcher",
        "Bear",
        "Bear Researcher",
        "",
        "Optimista",
    ],
)
def test_debate_router_return_always_routable(current_response):
    logic = ConditionalLogic(max_debate_rounds=1)
    target = logic.should_continue_debate(_debate_state(current_response))
    assert target in DEBATE_PATH_MAP


@pytest.mark.unit
def test_debate_path_map_covers_full_router_range():
    logic = ConditionalLogic(max_debate_rounds=1)
    returns = {
        logic.should_continue_debate(_debate_state(s, c))
        for s in ("Bull", "Bear", "drift")
        for c in (0, 99)
    }
    assert returns <= set(DEBATE_PATH_MAP)
    assert "Research Manager" in returns
