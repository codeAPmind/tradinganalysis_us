import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd


class HistoricalIVDatabase:
    """存储每日 ATM IV 快照,用于计算 IV Rank / IV Percentile。"""

    def __init__(self, db_path: str = "./cache/historical_iv.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS iv_snapshots (
                ticker TEXT,
                date TEXT,
                atm_iv REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        self.conn.commit()

    def save_iv(self, ticker: str, atm_iv: float, date: str = None):
        date = date or datetime.now().strftime("%Y-%m-%d")
        self.conn.execute(
            "INSERT OR REPLACE INTO iv_snapshots (ticker, date, atm_iv) VALUES (?, ?, ?)",
            (ticker, date, atm_iv),
        )
        self.conn.commit()

    def get_iv_history(self, ticker: str, days: int = 252) -> pd.Series:
        rows = self.conn.execute(
            "SELECT date, atm_iv FROM iv_snapshots WHERE ticker = ? ORDER BY date DESC LIMIT ?",
            (ticker, days),
        ).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows, columns=["date", "atm_iv"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["atm_iv"].sort_index()

    def calc_iv_rank(self, ticker: str, current_iv: float, days: int = 252) -> float:
        """IV Rank: 当前 IV 在过去一年数据中的位置。"""
        history = self.get_iv_history(ticker, days)
        if history.empty:
            return 50.0
        iv_min = history.min()
        iv_max = history.max()
        if iv_max == iv_min:
            return 50.0
        return (current_iv - iv_min) / (iv_max - iv_min) * 100

    def calc_iv_percentile(self, ticker: str, current_iv: float, days: int = 252) -> float:
        """IV Percentile: 有多少历史天数的 IV 低于当前 IV。"""
        history = self.get_iv_history(ticker, days)
        if history.empty:
            return 50.0
        return (history < current_iv).mean() * 100
