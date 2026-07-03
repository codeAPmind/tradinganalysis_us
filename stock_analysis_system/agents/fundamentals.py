from datetime import datetime
import pandas as pd
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


class FundamentalsAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        fund = self.gateway.get_fundamentals(ticker)
        metrics = self._compute_metrics(fund)

        user_prompt = f"""
股票: {ticker}
营收增速(YoY): {metrics.get('revenue_growth', 'N/A')}
毛利率: {metrics.get('gross_margin', 'N/A')}
经营现金流: {metrics.get('ocf', 'N/A')}
当前PE: {metrics.get('pe', 'N/A')}
PS Ratio: {metrics.get('ps', 'N/A')}
市值: {metrics.get('market_cap', 'N/A')}
Net Debt / EBITDA: {metrics.get('net_debt_to_ebitda', 'N/A')}
自由现金流: {metrics.get('fcf', 'N/A')}
"""
        llm_out = self._call_llm(Prompts.FUNDAMENTALS, user_prompt)
        parsed = self._parse_json(llm_out)

        findings = parsed.get("green_flags", []) + parsed.get("red_flags", [])
        if not findings:
            findings = [parsed.get("bull_thesis", ""), parsed.get("bear_thesis", "")]

        return AnalysisReport(
            agent_name="FundamentalsAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.8,
            key_findings=findings,
            raw_data=metrics,
            llm_reasoning=llm_out,
        )

    def _compute_metrics(self, fund: dict) -> dict:
        info = fund.get("info", {})
        metrics: dict = {
            "pe": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "market_cap": info.get("marketCap"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "ocf": info.get("operatingCashflow"),
            "fcf": info.get("freeCashflow"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
        }

        # 粗略估算 Net Debt / EBITDA
        ebitda = info.get("ebitda")
        total_debt = info.get("totalDebt", 0) or 0
        total_cash = info.get("totalCash", 0) or 0
        net_debt = total_debt - total_cash
        if ebitda and ebitda != 0:
            metrics["net_debt_to_ebitda"] = round(net_debt / ebitda, 2)
        else:
            metrics["net_debt_to_ebitda"] = None

        return metrics
