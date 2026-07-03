"""
HistoricalEventAgent：事件研究 Agent
- 收集历史财报日、内部人交易事件
- 计算各事件窗口股价反应（T+1/T+5/T+20）
- 识别当前情形与历史的最相似案例
- 输出"历史类比"结论辅助当前决策
"""
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd

from .base import BaseAgent, AnalysisReport
from data.event_study import EventStudyEngine, EventRecord

SYSTEM_PROMPT = """你是专业的事件研究分析师，擅长通过历史事件与当前情形的类比来判断股价走势。

分析时请注意：
1. "相似条件"不代表"结果一定相同"，只是提供概率参考
2. meme股/散户驱动型标的的历史规律比价值股更不稳定
3. 需区分"消息驱动"和"基本面驱动"的不同事件类型
4. "买预期卖事实"的市场规律在消息已充分定价时尤其重要

输出 JSON:
{
  "most_similar_past_event": {"date": "...", "description": "...", "outcome_t5": ...},
  "historical_pattern": "...",
  "base_rate": {"earnings_beat_next_day_positive_pct": ..., "insider_buy_t20_positive_pct": ...},
  "current_analog": "当前情形最像哪段历史，以及那段历史之后的走向",
  "key_difference": "当前情形与历史类比的最大不同点",
  "score": -1到+1,
  "reasoning": "..."
}
只输出 JSON，不要其他内容。"""


class HistoricalEventAgent(BaseAgent):

    def analyze(self, ticker: str) -> AnalysisReport:
        # 1. 获取2年历史价格
        hist = self.gateway.get_price_history(ticker, period="2y")
        if hist.empty or len(hist) < 60:
            return AnalysisReport(
                agent_name="HistoricalEventAgent", ticker=ticker,
                timestamp=datetime.now().isoformat(),
                score=0, confidence=0.3,
                key_findings=["历史数据不足，无法完成事件研究"],
            )

        engine = EventStudyEngine(hist["Close"], hist["Volume"])
        events: list[EventRecord] = []

        # 2. 历史财报日（yfinance earnings history）
        events += self._fetch_earnings_events(ticker, hist)

        # 3. 历史内部人交易（OpenInsider，拉365天）
        events += self._fetch_insider_events(ticker, hist)

        # 4. 批量计算窗口收益
        events = engine.enrich_all(events)

        # 5. 汇总统计
        earnings_stats = engine.summary_stats(events, "earnings")
        insider_buy_stats = engine.summary_stats(events, "insider_buy")
        insider_sell_stats = engine.summary_stats(events, "insider_sell")

        # 6. 格式化给 LLM
        event_table = self._format_event_table(events)
        current_price = float(hist["Close"].iloc[-1])
        ytd_return = float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100)

        user_prompt = f"""
股票: {ticker}
当前价格: ${current_price:.2f}
过去2年涨跌幅: {ytd_return:+.1f}%

历史事件列表（最近2年）:
{event_table}

历史基准统计:
- 财报事件({earnings_stats.get('count', 0)}次): T+1平均{earnings_stats.get('avg_t+1')}%, T+5平均{earnings_stats.get('avg_t+5')}%, 次日上涨概率{earnings_stats.get('positive_t+1_rate')}%
- 内部人买入({insider_buy_stats.get('count', 0)}次): T+20平均{insider_buy_stats.get('avg_t+20')}%
- 内部人卖出({insider_sell_stats.get('count', 0)}次): T+20平均{insider_sell_stats.get('avg_t+20')}%

当前情形:
- 距离上次财报约60天，下次财报日期待确认
- 近期内部人行为参考InsiderAgent结论
- 请基于以上历史数据，找出当前情形最相似的历史片段，并给出类比分析
"""
        llm_out = self._call_llm(SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(llm_out)

        similar = parsed.get("most_similar_past_event", {})
        findings = [
            f"历史规律: {parsed.get('historical_pattern', 'N/A')}",
            f"最相似历史节点: {similar.get('date', 'N/A')} — {similar.get('description', '')}",
            f"当前类比: {parsed.get('current_analog', 'N/A')}",
            f"关键差异: {parsed.get('key_difference', 'N/A')}",
            f"财报胜率(次日正收益): {earnings_stats.get('positive_t+1_rate', 'N/A')}%",
        ]

        return AnalysisReport(
            agent_name="HistoricalEventAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.65,
            key_findings=findings,
            raw_data={
                "events": [self._event_to_dict(e) for e in events],
                "earnings_stats": earnings_stats,
                "insider_buy_stats": insider_buy_stats,
                "insider_sell_stats": insider_sell_stats,
            },
            llm_reasoning=llm_out,
        )

    # ------------------------------------------------------------------ #
    #  私有方法
    # ------------------------------------------------------------------ #

    def _fetch_earnings_events(self, ticker: str, hist: pd.DataFrame) -> list[EventRecord]:
        """从 yfinance 获取历史财报日期。"""
        events = []
        try:
            t = yf.Ticker(ticker)
            # yfinance earnings_dates 包含过去和未来的财报日
            ed = t.earnings_dates
            if ed is None or ed.empty:
                return events

            cutoff = hist.index[0]
            for dt, row in ed.iterrows():
                dt_ts = pd.Timestamp(dt).tz_localize(None) if dt.tzinfo else pd.Timestamp(dt)
                if dt_ts < cutoff or dt_ts > pd.Timestamp.now():
                    continue
                date_str = dt_ts.strftime("%Y-%m-%d")

                # EPS 超预期幅度
                eps_est = row.get("EPS Estimate")
                eps_act = row.get("Reported EPS")
                surprise = None
                desc = f"财报发布"
                if pd.notna(eps_est) and pd.notna(eps_act) and eps_est != 0:
                    surprise = round((eps_act - eps_est) / abs(eps_est) * 100, 1)
                    desc += f"（EPS超预期 {surprise:+.1f}%）" if surprise > 0 else f"（EPS不及预期 {surprise:+.1f}%）"

                events.append(EventRecord(
                    date=date_str,
                    event_type="earnings",
                    description=desc,
                    extra={"eps_surprise_pct": surprise},
                ))
        except Exception:
            pass
        return events

    def _fetch_insider_events(self, ticker: str, hist: pd.DataFrame) -> list[EventRecord]:
        """从 OpenInsider 获取过去365天内部人交易事件。"""
        events = []
        try:
            df = self.gateway.get_insider_trades(ticker, days=365)
            if df.empty:
                return events

            df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]

            for _, row in df.iterrows():
                trade_date = str(row.get("Trade Date", "")).strip()
                if not trade_date or trade_date == "nan":
                    continue
                try:
                    pd.Timestamp(trade_date)
                except Exception:
                    continue

                trade_type = str(row.get("Trade Type", ""))
                name = str(row.get("Insider Name", ""))
                title = str(row.get("Title", ""))
                value = str(row.get("Value", ""))

                is_buy = "Purchase" in trade_type or "P -" in trade_type
                etype = "insider_buy" if is_buy else "insider_sell"

                events.append(EventRecord(
                    date=trade_date,
                    event_type=etype,
                    description=f"{name}({title}) {'买入' if is_buy else '卖出'} {value}",
                    extra={"name": name, "title": title, "value": value},
                ))
        except Exception:
            pass
        return events

    def _format_event_table(self, events: list[EventRecord]) -> str:
        if not events:
            return "（无历史事件数据）"
        sorted_events = sorted(events, key=lambda e: e.date, reverse=True)
        lines = ["日期       | 类型          | 描述                          | T+1%  | T+5%  | T+20%"]
        lines.append("-" * 90)
        for e in sorted_events[:30]:  # 最多展示30条
            t1 = f"{e.t_plus_1:+.1f}" if e.t_plus_1 is not None else "N/A"
            t5 = f"{e.t_plus_5:+.1f}" if e.t_plus_5 is not None else "N/A"
            t20 = f"{e.t_plus_20:+.1f}" if e.t_plus_20 is not None else "N/A"
            desc = e.description[:35]
            lines.append(f"{e.date} | {e.event_type:<13} | {desc:<35} | {t1:>6} | {t5:>6} | {t20:>6}")
        return "\n".join(lines)

    def _event_to_dict(self, e: EventRecord) -> dict:
        return {
            "date": e.date,
            "type": e.event_type,
            "description": e.description,
            "t+1": e.t_plus_1,
            "t+5": e.t_plus_5,
            "t+20": e.t_plus_20,
            "volume_ratio": e.volume_ratio,
        }
