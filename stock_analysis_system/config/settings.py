import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    deepseek_key: str
    fmp_key: str
    fred_key: str
    finnhub_key: str
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    cache_dir: str = "./cache"

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            deepseek_key=os.environ["DEEPSEEK_API_KEY"],
            fmp_key=os.getenv("FMP_API_KEY", ""),
            fred_key=os.getenv("FRED_API_KEY", ""),
            finnhub_key=os.getenv("FINNHUB_API_KEY", ""),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "StockAnalysisBot/1.0"),
        )
