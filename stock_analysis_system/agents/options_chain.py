from datetime import datetime
import numpy as np
import pandas as pd
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts
from data.historical_iv import HistoricalIVDatabase


class OptionsChainAgent(BaseAgent):
    def __init__(self, name: str, llm, gateway, iv_db: HistoricalIVDatabase = None):
        super().__init__(name, llm, gateway)
        self.iv_db = iv_db or HistoricalIVDatabase()

    def analyze(self, ticker: str) -> AnalysisReport:
        chain = self.gateway.get_option_chain(ticker)
        price_hist = self.gateway.get_price_history(ticker, period="1y")

        if price_hist.empty:
            return AnalysisReport(
                agent_name="OptionsChainAgent", ticker=ticker,
                timestamp=datetime.now().isoformat(),
                score=0, confidence=0.3,
                key_findings=["数据不足"],
            )

        spot = float(price_hist["Close"].iloc[-1])
        hv_30 = self._calc_hv(price_hist["Close"], window=30)
        iv_atm = self._atm_iv(chain, spot)

        # 保存当日 IV 快照
        if iv_atm > 0:
            self.iv_db.save_iv(ticker, iv_atm)

        iv_rank = self.iv_db.calc_iv_rank(ticker, iv_atm) if iv_atm > 0 else 50.0
        iv_hv_spread = iv_atm - hv_30

        skew = self._calc_skew(chain, spot)
        term_structure = self._term_structure(ticker)
        max_pain = self._max_pain(chain)

        # 多个到期日的 IV 比较
        near_iv = iv_atm

        summary = {
            "spot": spot,
            "atm_iv": round(iv_atm * 100, 1),
            "hv_30": round(hv_30 * 100, 1),
            "iv_hv_spread": round(iv_hv_spread * 100, 1),
            "iv_rank": round(iv_rank, 1),
            "put_call_skew": round(skew * 100, 2),
            "term_structure": term_structure,
            "max_pain": max_pain,
            "target_expiry": chain.get("target_expiry"),
        }

        user_prompt = f"期权数据: {summary}"
        llm_out = self._call_llm(Prompts.OPTIONS_CHAIN, user_prompt)
        parsed = self._parse_json(llm_out)

        findings = parsed.get("recommended_strategy_family", []) + parsed.get("warnings", [])
        findings.append(f"IV Rank: {iv_rank:.0f}, 定价环境: {parsed.get('regime', 'unknown')}")

        return AnalysisReport(
            agent_name="OptionsChainAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=0,
            confidence=0.8,
            key_findings=findings,
            raw_data=summary,
            llm_reasoning=llm_out,
        )

    def _calc_hv(self, prices: pd.Series, window: int = 30) -> float:
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return float(log_ret.tail(window).std() * np.sqrt(252))

    def _atm_iv(self, chain: dict, spot: float) -> float:
        calls = chain.get("calls", pd.DataFrame())
        puts = chain.get("puts", pd.DataFrame())
        if calls.empty or puts.empty:
            return 0.0
        try:
            atm_call = calls.iloc[(calls["strike"] - spot).abs().argmin()]
            atm_put = puts.iloc[(puts["strike"] - spot).abs().argmin()]
            iv = (atm_call["impliedVolatility"] + atm_put["impliedVolatility"]) / 2
            return float(iv) if not np.isnan(iv) else 0.0
        except Exception:
            return 0.0

    def _calc_skew(self, chain: dict, spot: float) -> float:
        """近似 25-delta skew: 0.9*spot Put IV - 1.1*spot Call IV。"""
        calls = chain.get("calls", pd.DataFrame())
        puts = chain.get("puts", pd.DataFrame())
        if calls.empty or puts.empty:
            return 0.0
        try:
            otm_call_strike = spot * 1.1
            otm_put_strike = spot * 0.9
            c = calls.iloc[(calls["strike"] - otm_call_strike).abs().argmin()]
            p = puts.iloc[(puts["strike"] - otm_put_strike).abs().argmin()]
            skew = p["impliedVolatility"] - c["impliedVolatility"]
            return float(skew) if not np.isnan(skew) else 0.0
        except Exception:
            return 0.0

    def _term_structure(self, ticker: str) -> str:
        """比较近月和远月 ATM IV。"""
        try:
            from yfinance import Ticker
            t = Ticker(ticker)
            expiries = t.options
            if len(expiries) < 2:
                return "insufficient_data"
            near = t.option_chain(expiries[0])
            far = t.option_chain(expiries[-1])
            spot = float(t.history(period="1d")["Close"].iloc[-1])

            def get_atm_iv(chain, s):
                c = chain.calls.iloc[(chain.calls["strike"] - s).abs().argmin()]
                return float(c["impliedVolatility"])

            near_iv = get_atm_iv(near, spot)
            far_iv = get_atm_iv(far, spot)
            return "backwardation" if near_iv > far_iv else "contango"
        except Exception:
            return "contango"

    def _max_pain(self, chain: dict) -> float:
        calls = chain.get("calls", pd.DataFrame())
        puts = chain.get("puts", pd.DataFrame())
        if calls.empty or puts.empty:
            return 0.0
        try:
            strikes = sorted(set(calls["strike"]) | set(puts["strike"]))
            pain = {}
            for k in strikes:
                call_pain = ((calls["strike"] < k) * (k - calls["strike"]) * calls["openInterest"]).sum()
                put_pain = ((puts["strike"] > k) * (puts["strike"] - k) * puts["openInterest"]).sum()
                pain[k] = call_pain + put_pain
            return float(min(pain, key=pain.get))
        except Exception:
            return 0.0
