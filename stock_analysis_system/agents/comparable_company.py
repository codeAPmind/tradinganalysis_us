"""
ComparableCompanyAgent：横向对比 Agent

核心逻辑：
1. 对目标股找 1-3 个可比公司 + 锚点时间（业务里程碑对齐，而非日历对齐）
2. 计算可比公司在锚点时的关键指标快照
3. 计算可比公司锚点后 T+30/60/90/180/365 的股价表现
4. 让 LLM 判断"目标股现在最像可比公司那个阶段"，以及"那之后发生了什么"

支持两种模式：
- 手动指定：comparables = [{"ticker": "CVNA", "anchor_date": "2023-11-01", "anchor_reason": "..."}]
- 自动发现：传入 comparables=None，LLM 自动建议可比公司和锚点
"""
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import yfinance as yf
import pandas as pd
import numpy as np

from .base import BaseAgent, AnalysisReport

SYSTEM_PROMPT_AUTO = """你是资深股票研究员，擅长找到"处于相似商业阶段"的可比公司。

给定目标公司的基本信息，请建议 1-3 个最有参考价值的历史可比案例。
选择标准：
- 商业模式相似（而非行业相同）
- 当时处于相似的"业务拐点"（如亏损转正、濒临退市后反转、上市初期）
- 有足够的历史数据（至少2年前的事件）

输出 JSON（数组）:
[
  {
    "ticker": "CVNA",
    "anchor_date": "2023-11-01",
    "anchor_reason": "债务重组完成后第一个季度，EBITDA首次转正，与OPEN当前从亏损收缩期开始出现盈利迹象的阶段对应",
    "similarity_dimension": "高杠杆资产撮合平台从濒临破产到业务重构",
    "what_happened_after": "CVNA股价从锚点后12个月涨超500%，主要驱动是毛利率持续改善和GMV增长重新加速"
  }
]
只输出 JSON 数组，不要其他内容。"""

SYSTEM_PROMPT_ANALYSIS = """你是专业的跨股票横向对比分析师。

分析目标：将目标公司当前状态与历史可比公司的特定阶段进行对比，
找出相似点、关键差异，以及可比案例后续的走势规律。

输出 JSON:
{
  "overall_similarity_score": 0-100,
  "comparisons": [
    {
      "comparable_ticker": "CVNA",
      "anchor_date": "2023-11-01",
      "similarity_score": 0-100,
      "matching_dimensions": ["维度1", "维度2"],
      "key_differences": ["差异1", "差异2"],
      "what_happened_after": {
        "t30_return": ...,
        "t90_return": ...,
        "t180_return": ...,
        "t365_return": ...,
        "key_drivers": "..."
      },
      "applicability_to_target": "这个可比案例对目标公司的参考价值和局限性"
    }
  ],
  "composite_signal": "综合所有可比案例后，对目标公司的概率判断",
  "score": -1到+1,
  "reasoning": "..."
}
只输出 JSON，不要其他内容。"""


@dataclass
class ComparableSpec:
    ticker: str
    anchor_date: str        # YYYY-MM-DD，锚点日期（里程碑对齐点）
    anchor_reason: str      # 为什么选这个日期
    similarity_dimension: str = ""
    what_happened_after: str = ""  # LLM 预填的历史知识


@dataclass
class ComparableSnapshot:
    """可比公司在锚点时的指标快照 + 锚点后收益"""
    spec: ComparableSpec
    # 锚点时的指标
    price_at_anchor: float = None
    market_cap_at_anchor: float = None
    price_52w_low_pct: float = None    # 当时价格距52周低点的涨幅
    price_52w_high_pct: float = None   # 当时价格距52周高点的跌幅
    short_interest_pct: float = None
    # 锚点后收益
    t30: float = None
    t60: float = None
    t90: float = None
    t180: float = None
    t365: float = None
    # 附加
    notes: str = ""


class ComparableCompanyAgent(BaseAgent):

    def analyze(self, ticker: str, comparables: list[dict] = None) -> AnalysisReport:
        """
        comparables: 可选，格式如:
        [{"ticker": "CVNA", "anchor_date": "2023-11-01",
          "anchor_reason": "债务重组后首季EBITDA转正",
          "similarity_dimension": "高杠杆平台反转",
          "what_happened_after": "12个月涨500%"}]
        不传则让 LLM 自动建议。
        """
        # 获取目标股当前状态
        target_info = self._get_target_snapshot(ticker)

        # 确定可比公司列表
        if comparables:
            comp_specs = [ComparableSpec(**c) for c in comparables]
        else:
            comp_specs = self._auto_discover_comparables(ticker, target_info)

        if not comp_specs:
            return AnalysisReport(
                agent_name="ComparableCompanyAgent", ticker=ticker,
                timestamp=datetime.now().isoformat(),
                score=0, confidence=0.3,
                key_findings=["未能找到有效的可比公司"],
            )

        # 计算每个可比公司的锚点快照和后续收益
        snapshots = [self._build_snapshot(spec) for spec in comp_specs]

        # LLM 综合对比分析
        analysis = self._run_comparison(ticker, target_info, snapshots)

        # 格式化 findings
        findings = [f"综合相似度评分: {analysis.get('overall_similarity_score', 'N/A')}/100"]
        for comp in analysis.get("comparisons", []):
            t = comp.get("comparable_ticker", "")
            sim = comp.get("similarity_score", "N/A")
            after = comp.get("what_happened_after", {})
            t365 = after.get("t365_return")
            findings.append(
                f"[{t}@{comp.get('anchor_date','')}] 相似度{sim}/100 "
                f"→ 锚点后12个月: {f'{t365:+.1f}%' if t365 is not None else '数据不足'}"
            )
        findings.append(f"综合信号: {analysis.get('composite_signal', '')[:80]}")

        return AnalysisReport(
            agent_name="ComparableCompanyAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(analysis.get("score", 0)),
            confidence=0.7,
            key_findings=findings,
            raw_data={
                "target_snapshot": target_info,
                "comparables": [self._snapshot_to_dict(s) for s in snapshots],
                "analysis": analysis,
            },
            llm_reasoning=str(analysis),
        )

    # ------------------------------------------------------------------ #
    #  目标股当前快照
    # ------------------------------------------------------------------ #

    def _get_target_snapshot(self, ticker: str) -> dict:
        hist = self.gateway.get_price_history(ticker, period="1y")
        fund = self.gateway.get_fundamentals(ticker)
        info = fund.get("info", {})

        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
        high_52w = float(hist["High"].max()) if not hist.empty else 0
        low_52w = float(hist["Low"].min()) if not hist.empty else 0

        return {
            "ticker": ticker,
            "current_price": current_price,
            "market_cap": info.get("marketCap"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "price_vs_52w_high_pct": round((current_price / high_52w - 1) * 100, 1) if high_52w else None,
            "price_vs_52w_low_pct": round((current_price / low_52w - 1) * 100, 1) if low_52w else None,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": (info.get("longBusinessSummary", "") or "")[:300],
        }

    # ------------------------------------------------------------------ #
    #  自动发现可比公司
    # ------------------------------------------------------------------ #

    def _auto_discover_comparables(self, ticker: str, target_info: dict) -> list[ComparableSpec]:
        prompt = f"""
目标公司: {ticker}
行业: {target_info.get('industry')} / {target_info.get('sector')}
当前PS: {target_info.get('ps_ratio')}
营收增速: {target_info.get('revenue_growth')}
毛利率: {target_info.get('gross_margin')}
公司简介: {target_info.get('description')}
价格距52周高点: {target_info.get('price_vs_52w_high_pct')}%
空头比例: {target_info.get('short_percent_of_float')}

请建议 1-3 个最有参考价值的历史可比案例（选择处于相似商业拐点的公司的特定时间段）。
"""
        llm_out = self._call_llm(SYSTEM_PROMPT_AUTO, prompt)
        raw = self._parse_json(llm_out)
        if not isinstance(raw, list):
            raw = [raw] if raw else []

        specs = []
        for item in raw:
            try:
                specs.append(ComparableSpec(
                    ticker=item["ticker"],
                    anchor_date=item["anchor_date"],
                    anchor_reason=item.get("anchor_reason", ""),
                    similarity_dimension=item.get("similarity_dimension", ""),
                    what_happened_after=item.get("what_happened_after", ""),
                ))
            except (KeyError, TypeError):
                continue
        return specs

    # ------------------------------------------------------------------ #
    #  构建可比公司快照
    # ------------------------------------------------------------------ #

    def _build_snapshot(self, spec: ComparableSpec) -> ComparableSnapshot:
        snap = ComparableSnapshot(spec=spec)
        try:
            anchor_dt = pd.Timestamp(spec.anchor_date)
            # 拉从锚点前1年到锚点后1.5年的数据
            start = (anchor_dt - timedelta(days=365)).strftime("%Y-%m-%d")
            end = (anchor_dt + timedelta(days=550)).strftime("%Y-%m-%d")

            hist = yf.Ticker(spec.ticker).history(start=start, end=end)
            if hist.empty:
                return snap

            hist.index = hist.index.tz_localize(None)

            # 找锚点最近交易日
            idx = hist.index.searchsorted(anchor_dt)
            if idx >= len(hist):
                idx = len(hist) - 1
            anchor_price = float(hist["Close"].iloc[idx])
            snap.price_at_anchor = round(anchor_price, 2)

            # 52周区间（锚点前1年）
            pre = hist.iloc[:idx]
            if not pre.empty:
                high_52w = float(pre["High"].max())
                low_52w = float(pre["Low"].min())
                snap.price_52w_high_pct = round((anchor_price / high_52w - 1) * 100, 1)
                snap.price_52w_low_pct = round((anchor_price / low_52w - 1) * 100, 1)

            # 锚点后各窗口收益
            for days, attr in [(30, "t30"), (60, "t60"), (90, "t90"), (180, "t180"), (365, "t365")]:
                future_idx = idx + self._business_days_offset(hist, idx, days)
                if future_idx < len(hist):
                    ret = (float(hist["Close"].iloc[future_idx]) / anchor_price - 1) * 100
                    setattr(snap, attr, round(ret, 1))

        except Exception as e:
            snap.notes = str(e)
        return snap

    def _business_days_offset(self, hist: pd.DataFrame, start_idx: int, calendar_days: int) -> int:
        """把日历天数近似转为交易日数（约 *0.71）。"""
        return max(1, round(calendar_days * 0.71))

    # ------------------------------------------------------------------ #
    #  LLM 综合对比
    # ------------------------------------------------------------------ #

    def _run_comparison(self, ticker: str, target_info: dict,
                        snapshots: list[ComparableSnapshot]) -> dict:
        comp_blocks = []
        for s in snapshots:
            comp_blocks.append(f"""
可比公司: {s.spec.ticker}
锚点日期: {s.spec.anchor_date}
锚点原因: {s.spec.anchor_reason}
相似维度: {s.spec.similarity_dimension}
锚点时价格: ${s.price_at_anchor}
锚点时距52周高点: {s.price_52w_high_pct}%  距52周低点: {s.price_52w_low_pct}%
锚点后收益 T+30:{s.t30}% / T+60:{s.t60}% / T+90:{s.t90}% / T+180:{s.t180}% / T+365:{s.t365}%
已知后续故事: {s.spec.what_happened_after}
""")

        prompt = f"""
目标公司当前状态:
{target_info}

可比公司历史快照:
{''.join(comp_blocks)}

请深度分析：
1. 目标公司与每个可比公司在锚点时的相似程度（财务、情绪、技术面多维对比）
2. 最关键的相似点和差异点
3. 可比案例的后续走势规律是否可以类推到目标公司
4. 给出综合判断（方向 + 概率）
"""
        llm_out = self._call_llm(SYSTEM_PROMPT_ANALYSIS, prompt)
        return self._parse_json(llm_out) or {}

    # ------------------------------------------------------------------ #
    #  序列化
    # ------------------------------------------------------------------ #

    def _snapshot_to_dict(self, s: ComparableSnapshot) -> dict:
        return {
            "ticker": s.spec.ticker,
            "anchor_date": s.spec.anchor_date,
            "anchor_reason": s.spec.anchor_reason,
            "price_at_anchor": s.price_at_anchor,
            "price_52w_high_pct": s.price_52w_high_pct,
            "price_52w_low_pct": s.price_52w_low_pct,
            "returns": {"t30": s.t30, "t60": s.t60, "t90": s.t90,
                        "t180": s.t180, "t365": s.t365},
            "notes": s.notes,
        }
