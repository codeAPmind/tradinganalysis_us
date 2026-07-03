from datetime import datetime
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


class CapitalFlowAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        institutional = self.gateway.get_institutional_holdings(ticker)
        short_data = self.gateway.get_short_interest(ticker)
        chain = self.gateway.get_option_chain(ticker)

        # Put/Call OI 比率
        put_call_ratio = None
        if not chain["calls"].empty and not chain["puts"].empty:
            call_oi = chain["calls"]["openInterest"].sum()
            put_oi = chain["puts"]["openInterest"].sum()
            if call_oi > 0:
                put_call_ratio = round(put_oi / call_oi, 2)

        raw = {
            "short_ratio": short_data.get("short_ratio"),
            "short_percent_of_float": short_data.get("short_percent_of_float"),
            "put_call_ratio": put_call_ratio,
            "institutional_rows": len(institutional) if not institutional.empty else 0,
        }

        user_prompt = f"""
股票: {ticker}
空头数据: 空头比例={raw['short_percent_of_float']}, 空头比率={raw['short_ratio']}
期权Put/Call OI比率: {put_call_ratio}
机构持仓数据行数: {raw['institutional_rows']}
"""
        llm_out = self._call_llm(Prompts.CAPITAL_FLOW, user_prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="CapitalFlowAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.6,
            key_findings=[
                parsed.get("reasoning", ""),
                f"机构趋势: {parsed.get('institutional_trend', 'unknown')}",
                f"Put/Call比率: {put_call_ratio}",
            ],
            raw_data=raw,
            llm_reasoning=llm_out,
        )
