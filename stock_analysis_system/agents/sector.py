from datetime import datetime
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


class SectorAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        sector_etf = self.gateway.get_sector_etf(ticker)

        stock_hist = self.gateway.get_price_history(ticker, period="3mo")
        etf_hist = self.gateway.get_price_history(sector_etf, period="3mo")
        spy_hist = self.gateway.get_price_history("SPY", period="3mo")

        def pct_return(df):
            if df.empty or len(df) < 2:
                return 0.0
            return float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100)

        stock_ret = pct_return(stock_hist)
        etf_ret = pct_return(etf_hist)
        spy_ret = pct_return(spy_hist)

        raw = {
            "ticker": ticker,
            "sector_etf": sector_etf,
            "stock_3mo_return": stock_ret,
            "sector_3mo_return": etf_ret,
            "spy_3mo_return": spy_ret,
            "vs_sector": stock_ret - etf_ret,
            "vs_spy": stock_ret - spy_ret,
        }

        user_prompt = (
            f"股票 {ticker} 近3个月回报: {stock_ret:.1f}%\n"
            f"行业ETF ({sector_etf}) 近3个月回报: {etf_ret:.1f}%\n"
            f"大盘SPY 近3个月回报: {spy_ret:.1f}%\n"
            f"相对行业超额: {raw['vs_sector']:+.1f}%\n"
            f"相对大盘超额: {raw['vs_spy']:+.1f}%"
        )
        llm_out = self._call_llm(Prompts.SECTOR, user_prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="SectorAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.65,
            key_findings=[
                parsed.get("reasoning", ""),
                f"行业趋势: {parsed.get('sector_trend', 'unknown')}",
                f"相对大盘超额: {raw['vs_spy']:+.1f}%",
            ],
            raw_data=raw,
            llm_reasoning=llm_out,
        )
