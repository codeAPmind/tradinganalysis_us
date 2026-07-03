from datetime import datetime
import numpy as np
import pandas as pd
import ta
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


class TechnicalAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        df = self.gateway.get_price_history(ticker, period="1y").copy()
        if df.empty or len(df) < 50:
            return AnalysisReport(
                agent_name="TechnicalAgent", ticker=ticker,
                timestamp=datetime.now().isoformat(),
                score=0, confidence=0.3,
                key_findings=["数据不足，无法完成技术分析"],
            )

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        df["MA20"] = ta.trend.sma_indicator(close, window=20)
        df["MA50"] = ta.trend.sma_indicator(close, window=50)
        df["MA200"] = ta.trend.sma_indicator(close, window=200)
        df["RSI"] = ta.momentum.rsi(close, window=14)
        df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BBL"] = bb.bollinger_lband()
        df["BBU"] = bb.bollinger_hband()

        last = df.iloc[-1]
        price = float(last["Close"])
        ma20 = float(last.get("MA20") or 0)
        ma50 = float(last.get("MA50") or 0)
        ma200 = float(last.get("MA200") or 0)
        rsi = float(last.get("RSI") or 50)
        atr = float(last.get("ATR") or 0)
        bbl = float(last.get("BBL") or 0)
        bbu = float(last.get("BBU") or 0)

        bb_pos = round((price - bbl) / (bbu - bbl), 2) if bbu > bbl else None

        ma_alignment = (
            "bull" if price > ma50 > ma200
            else "bear" if price < ma50 < ma200
            else "mixed"
        )

        key_levels = self._find_key_levels(df)

        summary = {
            "price": price,
            "ma20": round(ma20, 2), "ma50": round(ma50, 2), "ma200": round(ma200, 2),
            "ma_alignment": ma_alignment,
            "rsi": round(rsi, 1),
            "atr_pct": round(atr / price * 100, 2) if price else 0,
            "bb_position": bb_pos,
            "key_levels": key_levels,
        }

        user_prompt = f"技术面数据: {summary}"
        llm_out = self._call_llm(Prompts.TECHNICAL, user_prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="TechnicalAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.6,
            key_findings=[
                f"趋势: {parsed.get('trend', 'unknown')}",
                f"支撑: {parsed.get('key_support', key_levels.get('recent_low'))}",
                f"阻力: {parsed.get('key_resistance', key_levels.get('recent_high'))}",
                parsed.get("reasoning", ""),
            ],
            raw_data=summary,
            llm_reasoning=llm_out,
        )

    def _find_key_levels(self, df: pd.DataFrame) -> dict:
        recent = df.tail(120)
        return {
            "recent_high": round(float(recent["High"].max()), 2),
            "recent_low": round(float(recent["Low"].min()), 2),
            "volume_poc": self._volume_profile_poc(recent),
        }

    def _volume_profile_poc(self, df: pd.DataFrame) -> float:
        try:
            bins = pd.cut(df["Close"], bins=30)
            vol_by_price = df.groupby(bins, observed=True)["Volume"].sum()
            poc_bin = vol_by_price.idxmax()
            return round((poc_bin.left + poc_bin.right) / 2, 2)
        except Exception:
            return 0.0
