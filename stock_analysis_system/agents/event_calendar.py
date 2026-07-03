from datetime import datetime
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


class EventCalendarAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        cal = self.gateway.get_earnings_calendar(ticker)

        next_earnings = cal.get("next_earnings", {})
        recommendations = cal.get("recommendations")
        price_targets = cal.get("analyst_price_targets")

        # 格式化财报日期
        earnings_info = "未知"
        if isinstance(next_earnings, dict):
            earnings_date = next_earnings.get("Earnings Date") or next_earnings.get("earnings_date")
            if earnings_date:
                earnings_info = str(earnings_date)

        # 分析师评级汇总
        analyst_summary = "无分析师数据"
        try:
            if recommendations is not None and not recommendations.empty:
                recent = recommendations.tail(5)
                analyst_summary = recent.to_string()
        except Exception:
            pass

        raw = {
            "next_earnings": earnings_info,
            "analyst_summary": analyst_summary,
            "price_targets": str(price_targets) if price_targets is not None else None,
        }

        user_prompt = f"""
股票: {ticker}
下次财报日期: {earnings_info}
分析师近期评级:
{analyst_summary}
分析师目标价: {raw['price_targets']}
当前日期: {datetime.now().strftime('%Y-%m-%d')}
"""
        llm_out = self._call_llm(Prompts.EVENT_CALENDAR, user_prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="EventCalendarAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.7,
            key_findings=[
                f"下次财报: {earnings_info}",
                f"最大风险事件: {parsed.get('biggest_risk_event', 'N/A')}",
                f"最大催化剂: {parsed.get('biggest_catalyst', 'N/A')}",
                parsed.get("reasoning", ""),
            ],
            raw_data=raw,
            llm_reasoning=llm_out,
        )
