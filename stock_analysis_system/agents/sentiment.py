from datetime import datetime
import requests
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts

# ApeWisdom 支持的 Reddit 板块
APEWISDOM_FEEDS = ["wallstreetbets", "stocks", "investing", "options"]


def _get_apewisdom(ticker: str) -> dict:
    """
    ApeWisdom 公开 API，无需 key。
    返回该 ticker 在主要 Reddit 板块的提及次数、排名、24h 变化。
    """
    url = f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json().get("results", [])
        for item in data:
            if item.get("ticker", "").upper() == ticker.upper():
                return item
    except Exception:
        pass
    return {}


def _get_stockanalysis_sentiment(ticker: str) -> dict:
    """
    备用：StockAnalysis 新闻情绪（公开接口）。
    """
    try:
        url = f"https://api.stockanalysis.com/stocks/{ticker.lower()}/forecast/"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        return resp.json() if resp.ok else {}
    except Exception:
        return {}


class SentimentAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        # 1. ApeWisdom — Reddit 提及数据（无需 API Key）
        ape = _get_apewisdom(ticker)

        # 2. Finnhub 新闻（已有 key，可选）
        news = self.gateway.get_news(ticker, days=7)
        news_headlines = [n.get("headline", "") for n in news[:10] if isinstance(n, dict)]

        # 3. 期权 Put/Call OI 比率
        chain = self.gateway.get_option_chain(ticker)
        put_call_ratio = None
        if not chain["calls"].empty and not chain["puts"].empty:
            call_oi = chain["calls"]["openInterest"].sum()
            put_oi = chain["puts"]["openInterest"].sum()
            if call_oi > 0:
                put_call_ratio = round(put_oi / call_oi, 2)

        raw = {
            # ApeWisdom 字段
            "reddit_mentions_24h": ape.get("mentions", 0),
            "reddit_mentions_24h_ago": ape.get("mentions_24h_ago", 0),
            "reddit_rank": ape.get("rank"),
            "reddit_rank_24h_ago": ape.get("rank_24h_ago"),
            "reddit_upvotes": ape.get("upvotes", 0),
            # 其他
            "news_count": len(news),
            "put_call_ratio": put_call_ratio,
            "headlines": news_headlines,
        }

        # 提及量变化率
        mentions_now = raw["reddit_mentions_24h"] or 0
        mentions_prev = raw["reddit_mentions_24h_ago"] or 0
        mentions_change_pct = (
            round((mentions_now - mentions_prev) / mentions_prev * 100, 1)
            if mentions_prev > 0 else None
        )
        raw["mentions_change_pct"] = mentions_change_pct

        user_prompt = f"""
股票: {ticker}

Reddit 社区讨论数据 (来源: ApeWisdom):
- 过去24小时提及次数: {mentions_now}
- 前24小时提及次数: {mentions_prev}
- 提及量变化: {f'{mentions_change_pct:+.1f}%' if mentions_change_pct is not None else 'N/A'}
- 当前排名: {raw['reddit_rank']} (前24h: {raw['reddit_rank_24h_ago']})
- 帖子点赞数: {raw['reddit_upvotes']}

新闻标题 (过去7天):
{chr(10).join(f'- {h}' for h in news_headlines) or '暂无新闻'}

期权 Put/Call OI 比率: {put_call_ratio}

注意: 提及量暴涨往往是反向信号(散户追高顶部)，需结合排名和基本面综合判断。
"""
        llm_out = self._call_llm(Prompts.SENTIMENT, user_prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="SentimentAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.5,
            key_findings=[
                f"Reddit提及: {mentions_now}次 ({f'{mentions_change_pct:+.1f}%' if mentions_change_pct is not None else 'N/A'})",
                f"Reddit排名: #{raw['reddit_rank']}",
                f"情绪极端: {parsed.get('sentiment_extreme', False)}",
                f"反向信号: {parsed.get('contrarian_signal', False)}",
                parsed.get("reasoning", ""),
            ],
            raw_data=raw,
            llm_reasoning=llm_out,
        )
