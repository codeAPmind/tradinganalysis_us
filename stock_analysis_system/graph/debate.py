from langgraph.graph import StateGraph, END
from typing import TypedDict


class DebateState(TypedDict):
    ticker: str
    reports: list
    bull_argument: str
    bear_argument: str
    round: int
    consensus: bool
    final_conclusion: dict


def build_debate_graph(bull, bear, manager, max_rounds: int = 3):
    """多轮辩论子图,增强研究结论的鲁棒性。"""
    workflow = StateGraph(DebateState)

    def bull_argue(state: DebateState) -> dict:
        context = f"当前轮次: {state['round']}\n对方论点: {state.get('bear_argument', '')}"
        thesis = bull.synthesize(state["reports"])
        return {"bull_argument": str(thesis), "round": state["round"] + 1}

    def bear_argue(state: DebateState) -> dict:
        context = f"当前轮次: {state['round']}\n对方论点: {state.get('bull_argument', '')}"
        thesis = bear.synthesize(state["reports"])
        return {"bear_argument": str(thesis)}

    def manager_check(state: DebateState) -> dict:
        conclusion = manager.moderate(
            {"thesis": state["bull_argument"]},
            {"thesis": state["bear_argument"]},
            state["reports"],
        )
        # 如果置信度足够高,提前结束辩论
        conviction = conclusion.get("conviction", 0)
        consensus = conviction > 0.8 or state["round"] >= max_rounds
        return {"final_conclusion": conclusion, "consensus": consensus}

    workflow.add_node("bull_argue", bull_argue)
    workflow.add_node("bear_argue", bear_argue)
    workflow.add_node("manager_check", manager_check)

    workflow.set_entry_point("bull_argue")
    workflow.add_edge("bull_argue", "bear_argue")
    workflow.add_edge("bear_argue", "manager_check")
    workflow.add_conditional_edges(
        "manager_check",
        lambda s: END if s["consensus"] else "bull_argue",
    )

    return workflow.compile()
