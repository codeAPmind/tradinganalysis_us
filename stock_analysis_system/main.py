#!/usr/bin/env python3
"""
个股全景分析 & 期权决策系统
用法: python main.py AAPL --account 50000
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings
from data.gateway import DataGateway, DataConfig
from data.historical_iv import HistoricalIVDatabase
from langchain_openai import ChatOpenAI
from agents import (
    MacroAgent, SectorAgent, FundamentalsAgent, InsiderAgent,
    CapitalFlowAgent, TechnicalAgent, SentimentAgent, EventCalendarAgent,
    OptionsChainAgent, BullResearcher, BearResearcher, ResearchManager,
    TraderAgent, RiskManagerAgent,
)
from graph.main_workflow import build_workflow


def analyze(ticker: str, account_size: float, portfolio: dict = None) -> dict:
    settings = Settings.load()

    gateway = DataGateway(DataConfig(
        fmp_api_key=settings.fmp_key,
        fred_api_key=settings.fred_key,
        finnhub_api_key=settings.finnhub_key,
    ))

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=settings.deepseek_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    iv_db = HistoricalIVDatabase()

    agents = {
        "macro":        MacroAgent("Macro", llm, gateway),
        "sector":       SectorAgent("Sector", llm, gateway),
        "fundamentals": FundamentalsAgent("Fundamentals", llm, gateway),
        "insider":      InsiderAgent("Insider", llm, gateway),
        "capital":      CapitalFlowAgent("Capital", llm, gateway),
        "technical":    TechnicalAgent("Technical", llm, gateway),
        "sentiment":    SentimentAgent("Sentiment", llm, gateway),
        "events":       EventCalendarAgent("Events", llm, gateway),
        "options":      OptionsChainAgent("Options", llm, gateway, iv_db),
        "bull":         BullResearcher("Bull", llm, gateway),
        "bear":         BearResearcher("Bear", llm, gateway),
        "manager":      ResearchManager("Manager", llm, gateway),
        "trader":       TraderAgent("Trader", llm, gateway),
        "risk":         RiskManagerAgent("Risk", llm, gateway),
    }

    workflow = build_workflow(agents)

    print(f"\n{'='*60}")
    print(f"  开始分析: {ticker.upper()}  账户规模: ${account_size:,.0f}")
    print(f"{'='*60}\n")

    result = workflow.invoke({
        "ticker": ticker.upper(),
        "account_size": account_size,
        "portfolio": portfolio or {},
        "reports": [],
        "bull_thesis": {},
        "bear_thesis": {},
        "research_conclusion": {},
        "trade_proposal": {},
        "risk_review": {},
        "final_output": {},
    })

    output = result["final_output"]

    # 保存报告
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    from datetime import datetime
    fname = reports_dir / f"{ticker.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n报告已保存至: {fname}")

    return output


def main():
    parser = argparse.ArgumentParser(description="个股全景分析 & 期权决策系统")
    parser.add_argument("ticker", help="股票代码, 例如 AAPL")
    parser.add_argument("--account", type=float, default=50000, help="账户规模(美元), 默认 50000")
    args = parser.parse_args()

    output = analyze(args.ticker, args.account)
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
