"""
InstitutionalFlowAgent：机构资金流向 Agent

追踪两类核心信号（参考"机构口是心非"逻辑）：

1. ETF 资金流（ETF Flow）
   - 追踪目标股所在板块的 1x / 3x 杠杆 ETF 日成交额 vs 20日均值
   - SOXL 单日成交额暴增 4 倍 = 机构趁跌捞底的铁证
   - 比股价反应领先 0-2 天

2. 期权权利金流（Options Premium Flow）
   - Put/Call OI 比率是存量，容易失真
   - Premium Flow = Σ(当日成交量 × 期权中间价 × 100)，是当天真实花了多少钱
   - 机构买 Call 时，Call Premium Flow >> Put Premium Flow
   - 比 OI 比率领先 1-3 天

说明：
  - ETF 日成交额用 price × volume 近似（非真实申购赎回 AUM 变化）
  - Options premium 用 (bid+ask)/2 × volume × 100 近似
  - 免费数据源精度有限，信号为辅助参考，非精确数字
"""
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from .base import BaseAgent, AnalysisReport

SYSTEM_PROMPT = """你是机构资金流向分析专家，专注于通过 ETF 资金流和期权权利金流
判断"机构真实意图"（区别于言论和散户情绪）。

核心逻辑：
- ETF 成交额/均值 > 3：机构大规模建仓信号
- ETF 杠杆多空比（SOXL成交额/SOXS成交额）> 3：多头主导
- 期权 Call Premium Flow / Put Premium Flow > 2：主力在用真金白银押涨
- 大盘下跌但板块 ETF 流入暴增：借利空掩护捞底

输出 JSON:
{
  "etf_signal": "bullish|bearish|neutral",
  "options_flow_signal": "bullish|bearish|neutral",
  "composite_signal": "bullish|bearish|neutral",
  "key_evidence": ["证据1", "证据2"],
  "institutional_behavior": "机构当前行为的一句话总结",
  "score": -1到+1,
  "confidence": 0到1,
  "reasoning": "..."
}
只输出 JSON，不要其他内容。"""

# 各板块的 ETF 配置：1x + 3x做多 + 3x做空
SECTOR_ETF_MAP = {
    # 半导体
    "semiconductor": {
        "etf_1x": "SOXX",
        "etf_3x_bull": "SOXL",
        "etf_3x_bear": "SOXS",
        "stocks": ["NVDA", "AMD", "MU", "INTC", "AVGO", "QCOM", "TSM", "AMAT", "LRCX", "KLAC"],
    },
    # 科技
    "tech": {
        "etf_1x": "XLK",
        "etf_3x_bull": "TQQQ",
        "etf_3x_bear": "SQQQ",
        "stocks": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
    },
    # 房地产
    "real_estate": {
        "etf_1x": "XLRE",
        "etf_3x_bull": "DRN",
        "etf_3x_bear": "DRV",
        "stocks": ["OPEN", "Z", "RDFN", "EXPI"],
    },
    # 能源
    "energy": {
        "etf_1x": "XLE",
        "etf_3x_bull": "ERX",
        "etf_3x_bear": "ERY",
        "stocks": ["XOM", "CVX", "COP", "SLB"],
    },
    # 金融
    "financials": {
        "etf_1x": "XLF",
        "etf_3x_bull": "FAS",
        "etf_3x_bear": "FAZ",
        "stocks": ["JPM", "BAC", "GS", "MS", "WFC"],
    },
}


def _get_sector_for_ticker(ticker: str) -> dict | None:
    ticker = ticker.upper()
    for sector_config in SECTOR_ETF_MAP.values():
        if ticker in sector_config["stocks"]:
            return sector_config
    return None


class InstitutionalFlowAgent(BaseAgent):

    def analyze(self, ticker: str) -> AnalysisReport:
        sector_cfg = _get_sector_for_ticker(ticker)

        # ── 1. ETF 资金流 ──────────────────────────────────
        etf_data = self._calc_etf_flows(sector_cfg)

        # ── 2. 期权权利金流 ────────────────────────────────
        options_flow = self._calc_options_premium_flow(ticker)

        # ── 3. 个股近5日期权权利金趋势 ────────────────────
        flow_trend = self._calc_premium_flow_trend(ticker)

        raw = {
            "etf_flows": etf_data,
            "options_premium_flow": options_flow,
            "premium_flow_trend_5d": flow_trend,
        }

        # ── 4. LLM 综合判断 ────────────────────────────────
        user_prompt = self._build_prompt(ticker, etf_data, options_flow, flow_trend)
        llm_out = self._call_llm(SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(llm_out)

        score = float(parsed.get("score", 0))
        confidence = float(parsed.get("confidence", 0.6))

        findings = [
            f"ETF信号: {parsed.get('etf_signal', 'N/A')} | "
            f"期权流信号: {parsed.get('options_flow_signal', 'N/A')} | "
            f"综合: {parsed.get('composite_signal', 'N/A')}",
            f"机构行为: {parsed.get('institutional_behavior', 'N/A')}",
        ] + parsed.get("key_evidence", [])

        return AnalysisReport(
            agent_name="InstitutionalFlowAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=score,
            confidence=confidence,
            key_findings=findings,
            raw_data=raw,
            llm_reasoning=llm_out,
        )

    # ──────────────────────────────────────────────────────
    #  ETF 资金流计算
    # ──────────────────────────────────────────────────────

    def _calc_etf_flows(self, sector_cfg: dict | None) -> dict:
        if not sector_cfg:
            return {"available": False, "reason": "未找到对应板块ETF配置"}

        result = {"available": True}
        for key in ["etf_1x", "etf_3x_bull", "etf_3x_bear"]:
            etf_sym = sector_cfg.get(key)
            if not etf_sym:
                continue
            try:
                hist = yf.Ticker(etf_sym).history(period="30d")
                if hist.empty or len(hist) < 5:
                    continue

                # 日成交额 = 收盘价 × 成交量
                hist["dollar_vol"] = hist["Close"] * hist["Volume"]
                today_vol = float(hist["dollar_vol"].iloc[-1])
                avg_20d = float(hist["dollar_vol"].iloc[:-1].mean())
                ratio = round(today_vol / avg_20d, 2) if avg_20d > 0 else None

                result[key] = {
                    "ticker": etf_sym,
                    "today_dollar_vol_M": round(today_vol / 1e6, 1),
                    "avg_20d_dollar_vol_M": round(avg_20d / 1e6, 1),
                    "ratio_vs_avg": ratio,
                    "signal": (
                        "极强流入" if ratio and ratio > 3 else
                        "强流入" if ratio and ratio > 2 else
                        "正常" if ratio and ratio > 0.7 else
                        "流出"
                    ),
                }
            except Exception as e:
                result[key] = {"ticker": etf_sym, "error": str(e)}

        # 多空 ETF 比率（杠杆多头成交额 / 杠杆空头成交额）
        bull = result.get("etf_3x_bull", {})
        bear = result.get("etf_3x_bear", {})
        bull_vol = bull.get("today_dollar_vol_M", 0)
        bear_vol = bear.get("today_dollar_vol_M", 0)
        if bull_vol and bear_vol and bear_vol > 0:
            result["leveraged_bull_bear_ratio"] = round(bull_vol / bear_vol, 2)

        return result

    # ──────────────────────────────────────────────────────
    #  期权权利金流计算（当日快照）
    # ──────────────────────────────────────────────────────

    def _calc_options_premium_flow(self, ticker: str) -> dict:
        try:
            t = yf.Ticker(ticker)
            expiries = t.options
            if not expiries:
                return {"available": False}

            # 只取最近2个到期日，避免数据量过大
            total_call_premium = 0.0
            total_put_premium = 0.0
            total_call_vol = 0
            total_put_vol = 0

            for exp in expiries[:2]:
                chain = t.option_chain(exp)
                calls = chain.calls
                puts = chain.puts

                if not calls.empty:
                    calls = calls.dropna(subset=["bid", "ask", "volume"])
                    calls = calls[calls["volume"] > 0]
                    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
                    calls["premium_flow"] = calls["mid"] * calls["volume"] * 100
                    total_call_premium += calls["premium_flow"].sum()
                    total_call_vol += int(calls["volume"].sum())

                if not puts.empty:
                    puts = puts.dropna(subset=["bid", "ask", "volume"])
                    puts = puts[puts["volume"] > 0]
                    puts["mid"] = (puts["bid"] + puts["ask"]) / 2
                    puts["premium_flow"] = puts["mid"] * puts["volume"] * 100
                    total_put_premium += puts["premium_flow"].sum()
                    total_put_vol += int(puts["volume"].sum())

            if total_put_premium == 0:
                call_put_ratio = None
            else:
                call_put_ratio = round(total_call_premium / total_put_premium, 2)

            return {
                "available": True,
                "call_premium_flow_K": round(total_call_premium / 1000, 1),
                "put_premium_flow_K": round(total_put_premium / 1000, 1),
                "call_volume": total_call_vol,
                "put_volume": total_put_vol,
                "call_put_premium_ratio": call_put_ratio,
                "oi_based_put_call_ratio": None,  # 由 CapitalFlowAgent 提供
                "signal": (
                    "强烈看涨" if call_put_ratio and call_put_ratio > 2 else
                    "看涨" if call_put_ratio and call_put_ratio > 1.3 else
                    "中性" if call_put_ratio and call_put_ratio > 0.7 else
                    "看跌" if call_put_ratio and call_put_ratio > 0.4 else
                    "强烈看跌" if call_put_ratio else "数据不足"
                ),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    # ──────────────────────────────────────────────────────
    #  近5日期权权利金趋势（判断方向是否持续）
    # ──────────────────────────────────────────────────────

    def _calc_premium_flow_trend(self, ticker: str) -> dict:
        """
        用过去5日的期权成交量变化近似权利金流趋势。
        注意：yfinance 不提供历史期权成交量，这里用价格+成交量变化推断。
        """
        try:
            hist = self.gateway.get_price_history(ticker, period="10d")
            if hist.empty:
                return {}

            # 近5日价格和成交量趋势
            recent = hist.tail(5)
            price_change_5d = float(
                (recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1) * 100
            )
            vol_change_5d = float(
                (recent["Volume"].iloc[-1] / recent["Volume"].iloc[0] - 1) * 100
            )
            avg_vol_ratio = float(
                recent["Volume"].mean() / hist["Volume"].mean()
            )

            return {
                "price_change_5d_pct": round(price_change_5d, 2),
                "volume_change_5d_pct": round(vol_change_5d, 2),
                "recent_vol_vs_30d_avg": round(avg_vol_ratio, 2),
                "note": "yfinance 不提供历史期权快照，用正股成交量近似趋势",
            }
        except Exception:
            return {}

    # ──────────────────────────────────────────────────────
    #  组装 Prompt
    # ──────────────────────────────────────────────────────

    def _build_prompt(self, ticker: str, etf_data: dict,
                      options_flow: dict, flow_trend: dict) -> str:
        # ETF 段
        etf_lines = []
        if etf_data.get("available"):
            for key in ["etf_1x", "etf_3x_bull", "etf_3x_bear"]:
                d = etf_data.get(key, {})
                if d and "today_dollar_vol_M" in d:
                    etf_lines.append(
                        f"  {d['ticker']}: 今日成交额 ${d['today_dollar_vol_M']}M "
                        f"（20日均值 ${d['avg_20d_dollar_vol_M']}M，"
                        f"比率 {d['ratio_vs_avg']}x，{d['signal']}）"
                    )
            lev_ratio = etf_data.get("leveraged_bull_bear_ratio")
            if lev_ratio:
                etf_lines.append(f"  3x做多/做空成交额比率: {lev_ratio}x")
        else:
            etf_lines.append("  未找到对应板块ETF（OTC股票或数据缺失）")

        # 期权权利金段
        of = options_flow
        if of.get("available"):
            opt_lines = [
                f"  Call 权利金流: ${of.get('call_premium_flow_K', 0)}K  "
                f"Put 权利金流: ${of.get('put_premium_flow_K', 0)}K",
                f"  Call/Put 权利金比率: {of.get('call_put_premium_ratio', 'N/A')}  "
                f"信号: {of.get('signal', 'N/A')}",
                f"  Call成交量: {of.get('call_volume', 0):,}  "
                f"Put成交量: {of.get('put_volume', 0):,}",
            ]
        else:
            opt_lines = [f"  期权数据不可用: {of.get('error', '未知原因')}"]

        # 近期趋势段
        ft = flow_trend
        trend_lines = [
            f"  近5日股价变化: {ft.get('price_change_5d_pct', 'N/A')}%",
            f"  近5日成交量变化: {ft.get('volume_change_5d_pct', 'N/A')}%",
            f"  近5日均量/30日均量: {ft.get('recent_vol_vs_30d_avg', 'N/A')}x",
        ]

        return f"""
股票: {ticker}

【板块 ETF 资金流】
{chr(10).join(etf_lines)}

【期权权利金流（今日，近2个到期日加总）】
{chr(10).join(opt_lines)}

【近5日价量趋势】
{chr(10).join(trend_lines)}

参考逻辑：
- ETF 3x做多成交额暴增（>3x均值）= 机构借利空捞底信号
- Call/Put 权利金比率 > 2 = 主力用真金白银押涨（比 OI 更领先）
- 大盘跌但板块 ETF 流入暴增 = 借利空掩护建仓（机构口是心非的铁证）
"""
