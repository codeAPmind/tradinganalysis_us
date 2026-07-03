"""
事件研究工具：计算事件窗口内的股价反应。
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class EventRecord:
    date: str               # 事件日期 YYYY-MM-DD
    event_type: str         # earnings / insider_buy / insider_sell / macro / sentiment
    description: str        # 事件描述
    t_minus_1: float = None # 事件前1日收益率
    t_plus_1: float = None  # 事件后1日收益率
    t_plus_5: float = None  # 事件后5日收益率
    t_plus_20: float = None # 事件后20日收益率
    volume_ratio: float = None  # 成交量 / 20日均量
    extra: dict = field(default_factory=dict)  # 额外数据（如EPS超预期幅度）


class EventStudyEngine:
    """给定历史价格序列和事件列表，批量计算事件窗口收益。"""

    def __init__(self, prices: pd.Series, volumes: pd.Series = None):
        """
        prices: 日频收盘价 Series，index 为 DatetimeIndex
        volumes: 日频成交量 Series（可选）
        """
        self.prices = prices.sort_index()
        self.volumes = volumes.sort_index() if volumes is not None else None
        self._returns = self.prices.pct_change()

    def calc_window_return(self, event_date: str, days_forward: int) -> Optional[float]:
        """事件日（含）到 event_date + days_forward 的累计收益率。"""
        try:
            dt = pd.Timestamp(event_date)
            idx = self.prices.index.searchsorted(dt)
            if idx >= len(self.prices):
                return None
            start_price = self.prices.iloc[idx]
            end_idx = idx + days_forward
            if end_idx >= len(self.prices):
                end_idx = len(self.prices) - 1
            end_price = self.prices.iloc[end_idx]
            return round((end_price / start_price - 1) * 100, 2)
        except Exception:
            return None

    def calc_pre_return(self, event_date: str, days_back: int = 1) -> Optional[float]:
        """事件日前 days_back 个交易日到事件日的收益率。"""
        try:
            dt = pd.Timestamp(event_date)
            idx = self.prices.index.searchsorted(dt)
            start_idx = max(0, idx - days_back)
            start_price = self.prices.iloc[start_idx]
            end_price = self.prices.iloc[idx]
            return round((end_price / start_price - 1) * 100, 2)
        except Exception:
            return None

    def calc_volume_ratio(self, event_date: str, window: int = 20) -> Optional[float]:
        """事件日成交量 / 前 window 日平均成交量。"""
        if self.volumes is None:
            return None
        try:
            dt = pd.Timestamp(event_date)
            idx = self.volumes.index.searchsorted(dt)
            if idx >= len(self.volumes):
                return None
            event_vol = self.volumes.iloc[idx]
            avg_vol = self.volumes.iloc[max(0, idx - window):idx].mean()
            if avg_vol == 0:
                return None
            return round(event_vol / avg_vol, 2)
        except Exception:
            return None

    def enrich_event(self, event: EventRecord) -> EventRecord:
        """填充一条事件的所有时间窗口数据。"""
        event.t_minus_1 = self.calc_pre_return(event.date, 1)
        event.t_plus_1 = self.calc_window_return(event.date, 1)
        event.t_plus_5 = self.calc_window_return(event.date, 5)
        event.t_plus_20 = self.calc_window_return(event.date, 20)
        event.volume_ratio = self.calc_volume_ratio(event.date)
        return event

    def enrich_all(self, events: list[EventRecord]) -> list[EventRecord]:
        return [self.enrich_event(e) for e in events]

    def summary_stats(self, events: list[EventRecord], event_type: str = None) -> dict:
        """按事件类型汇总平均反应。"""
        filtered = [e for e in events if event_type is None or e.event_type == event_type]
        if not filtered:
            return {}

        def avg(vals):
            v = [x for x in vals if x is not None]
            return round(np.mean(v), 2) if v else None

        return {
            "count": len(filtered),
            "avg_t+1": avg([e.t_plus_1 for e in filtered]),
            "avg_t+5": avg([e.t_plus_5 for e in filtered]),
            "avg_t+20": avg([e.t_plus_20 for e in filtered]),
            "positive_t+1_rate": round(
                sum(1 for e in filtered if e.t_plus_1 and e.t_plus_1 > 0) / len(filtered) * 100, 1
            ),
        }
