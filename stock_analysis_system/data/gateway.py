from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from functools import lru_cache

@dataclass
class DataConfig:
    fmp_api_key: str
    fred_api_key: str
    finnhub_api_key: str
    cache_dir: str = "./cache"


class DataGateway:
    """统一数据网关：所有 Agent 只通过这里取数据，不直接调用外部 API。"""

    def __init__(self, config: DataConfig):
        self.config = config
        self._price_cache: dict = {}

    def get_price_history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        key = (ticker, period)
        if key not in self._price_cache:
            self._price_cache[key] = yf.Ticker(ticker).history(period=period)
        return self._price_cache[key]

    def get_fundamentals(self, ticker: str) -> dict:
        t = yf.Ticker(ticker)
        return {
            "info": t.info,
            "financials": t.financials,
            "balance_sheet": t.balance_sheet,
            "cashflow": t.cashflow,
            "quarterly_financials": t.quarterly_financials,
        }

    def get_option_chain(self, ticker: str, expiry: Optional[str] = None) -> dict:
        t = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return {"expiries": [], "target_expiry": None, "calls": pd.DataFrame(), "puts": pd.DataFrame()}
        target = expiry or expiries[0]
        chain = t.option_chain(target)
        return {
            "expiries": expiries,
            "target_expiry": target,
            "calls": chain.calls,
            "puts": chain.puts,
        }

    def get_insider_trades(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """从 OpenInsider 抓取内部人交易，用 BeautifulSoup 解析 HTML 表格。"""
        url = f"http://openinsider.com/screener?s={ticker}&fd={days}&action=1"
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.text, "lxml")
            # OpenInsider 的交易表格 class 为 "tinytable"
            table = soup.find("table", {"class": "tinytable"})
            if table is None:
                return pd.DataFrame()

            headers_row = [th.get_text(strip=True) for th in table.find_all("th")]
            rows = []
            for tr in table.find("tbody").find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if cells:
                    rows.append(cells)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=headers_row[:len(rows[0])] if headers_row else None)
            return df
        except Exception as e:
            return pd.DataFrame()

    def get_institutional_holdings(self, ticker: str) -> pd.DataFrame:
        """13F 机构持仓变化，通过 FMP（需要付费套餐，降级返回空）。"""
        if not self.config.fmp_api_key:
            return pd.DataFrame()
        # FMP v4 institutional-ownership 需要付费，改用 yfinance 的 institutional_holders
        try:
            t = yf.Ticker(ticker)
            holders = t.institutional_holders
            return holders if holders is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_macro_indicators(self) -> dict:
        def safe_last(ticker_sym: str) -> float:
            try:
                hist = yf.Ticker(ticker_sym).history(period="1mo")
                return float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
            except Exception:
                return 0.0

        return {
            "vix": safe_last("^VIX"),
            "us10y_yield": safe_last("^TNX"),
            "dxy": safe_last("DX-Y.NYB"),
        }

    def get_news(self, ticker: str, days: int = 7) -> list:
        if not self.config.finnhub_api_key:
            return []
        end = datetime.now()
        start = end - timedelta(days=days)
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "token": self.config.finnhub_api_key,
        }
        try:
            return requests.get(url, params=params, timeout=30).json()
        except Exception:
            return []

    def get_earnings_calendar(self, ticker: str) -> dict:
        t = yf.Ticker(ticker)
        result: dict = {}
        try:
            result["next_earnings"] = t.calendar
        except Exception:
            result["next_earnings"] = {}
        try:
            result["recommendations"] = t.recommendations
        except Exception:
            result["recommendations"] = pd.DataFrame()
        try:
            result["analyst_price_targets"] = t.analyst_price_targets
        except Exception:
            result["analyst_price_targets"] = None
        return result

    def get_short_interest(self, ticker: str) -> dict:
        try:
            info = yf.Ticker(ticker).info
            return {
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "shares_short": info.get("sharesShort"),
            }
        except Exception:
            return {}

    def get_sector_etf(self, ticker: str) -> str:
        try:
            sector = yf.Ticker(ticker).info.get("sector", "")
        except Exception:
            sector = ""
        sector_etf_map = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Basic Materials": "XLB",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Communication Services": "XLC",
        }
        return sector_etf_map.get(sector, "SPY")
