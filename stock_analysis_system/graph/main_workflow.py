from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from agents.base import AnalysisReport


class AnalysisState(TypedDict):
    ticker: str
    account_size: float
    portfolio: dict
    comparables: list        # 可选的手动可比公司列表
    reports: list
    bull_thesis: dict
    bear_thesis: dict
    research_conclusion: dict
    trade_proposal: dict
    risk_review: dict
    final_output: dict


def build_workflow(agents: dict) -> Any:
    workflow = StateGraph(AnalysisState)

    # L1 + L2 分析节点
    def make_analysis_node(agent_key: str):
        def node(state: AnalysisState) -> dict:
            report = agents[agent_key].analyze(state["ticker"])
            return {"reports": state["reports"] + [report]}
        return node

    for key in ["macro", "sector", "fundamentals", "insider", "capital", "technical", "sentiment", "events", "options", "history"]:
        workflow.add_node(f"run_{key}", make_analysis_node(key))

    # comparable 节点单独处理（需要传 comparables 参数）
    def comparable_node(state: AnalysisState) -> dict:
        comps = state.get("comparables") or None
        report = agents["comparable"].analyze(state["ticker"], comparables=comps)
        return {"reports": state["reports"] + [report]}
    workflow.add_node("run_comparable", comparable_node)

    # L4 辩论节点
    def bull_node(state: AnalysisState) -> dict:
        return {"bull_thesis": agents["bull"].synthesize(state["reports"])}

    def bear_node(state: AnalysisState) -> dict:
        return {"bear_thesis": agents["bear"].synthesize(state["reports"])}

    def moderate_node(state: AnalysisState) -> dict:
        conclusion = agents["manager"].moderate(
            state["bull_thesis"], state["bear_thesis"], state["reports"]
        )
        return {"research_conclusion": conclusion}

    # L5 交易决策节点
    def trade_node(state: AnalysisState) -> dict:
        options_report = next(
            (r for r in state["reports"] if r.agent_name == "OptionsChainAgent"),
            state["reports"][-1],
        )
        tech_report = next(
            (r for r in state["reports"] if r.agent_name == "TechnicalAgent"),
            state["reports"][-1],
        )
        proposal = agents["trader"].decide(
            state["research_conclusion"],
            options_report,
            tech_report,
            state["account_size"],
        )
        return {"trade_proposal": proposal}

    def risk_node(state: AnalysisState) -> dict:
        review = agents["risk"].review(
            state["trade_proposal"], state["portfolio"], state["account_size"]
        )
        return {"risk_review": review}

    def finalize_node(state: AnalysisState) -> dict:
        return {"final_output": _package_output(state)}

    workflow.add_node("bull_research", bull_node)
    workflow.add_node("bear_research", bear_node)
    workflow.add_node("moderate", moderate_node)
    workflow.add_node("trade", trade_node)
    workflow.add_node("risk_check", risk_node)
    workflow.add_node("finalize", finalize_node)

    # 编排边：顺序执行（简化版，生产可改为并行 fan-out）
    workflow.set_entry_point("run_macro")
    edges = [
        ("run_macro", "run_sector"),
        ("run_sector", "run_fundamentals"),
        ("run_fundamentals", "run_insider"),
        ("run_insider", "run_capital"),
        ("run_capital", "run_technical"),
        ("run_technical", "run_sentiment"),
        ("run_sentiment", "run_events"),
        ("run_events", "run_options"),
        ("run_options", "run_history"),
        ("run_history", "run_comparable"),
        ("run_comparable", "bull_research"),
        ("bull_research", "bear_research"),
        ("bear_research", "moderate"),
        ("moderate", "trade"),
        ("trade", "risk_check"),
        ("risk_check", "finalize"),
        ("finalize", END),
    ]
    for a, b in edges:
        workflow.add_edge(a, b)

    return workflow.compile()


def _package_output(state: AnalysisState) -> dict:
    approved = state.get("risk_review", {}).get("approved", False)
    return {
        "ticker": state["ticker"],
        "analysis_summary": {
            r.agent_name: {
                "score": r.score,
                "confidence": r.confidence,
                "findings": r.key_findings,
            }
            for r in state["reports"]
        },
        "bull_case": state.get("bull_thesis", {}),
        "bear_case": state.get("bear_thesis", {}),
        "final_view": state.get("research_conclusion", {}),
        "trade_recommendation": state.get("trade_proposal") if approved else None,
        "risk_warnings": state.get("risk_review", {}).get("warnings", []),
        "approved": approved,
    }
