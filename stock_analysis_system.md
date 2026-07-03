# 个股全景分析 & 期权决策系统 —— Python 实现方案

> 参考 TradingAgents 多智能体范式，融合宏观、行业、基本面、内部人、资金流、技术面、情绪面、事件八层分析，输出可执行的股票买卖 / 期权交易建议。
>
> 本方案仅用于研究与教育目的，不构成任何投资建议。

---

## 一、设计哲学

### 1.1 为什么用多智能体架构

单一模型试图同时处理宏观、财报、技术图、情绪、期权链会陷入两个问题：一是上下文过载导致关键信号被稀释，二是没有"辩论"机制，容易被单一叙事主导。多智能体（Multi-Agent）架构把复杂决策拆成"专家分工 + 交叉辩论 + 风控复核"三阶段，模拟真实交易团队的运作方式。

### 1.2 系统三大原则

**原则一：分析与决策解耦**。分析层只负责客观描述"当前处于什么状态"，决策层才根据分析结果 + 风险预算判断"该不该做、怎么做"。这样在复盘时能明确判断错误的是分析还是决策。

**原则二：概率化而非预测化**。每个 Agent 输出的都是"当前赔率结构"，不是"下周涨还是跌"。最终决策基于胜率、赔率、Kelly 仓位的综合评估。

**原则三：可回溯、可复盘**。每一次分析的输入、中间过程、结论都要落库。没有记录就没有改进。

---

## 二、系统架构

### 2.1 分层图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户输入：Ticker + 日期                  │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │      L0: 数据接入层          │
              │  (yfinance / FMP / SEC ...) │
              └──────────────┬──────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
┌────▼─────┐   ┌────────────▼────────────┐   ┌──────▼──────┐
│ L1 宏观   │   │      L2 个股分析层       │   │ L3 期权层   │
│ Macro    │   │ Fundamentals / Insider  │   │ Options    │
│ Sector   │   │ Capital / Technical     │   │ Chain +    │
│          │   │ Sentiment / Events      │   │ Strategy   │
└────┬─────┘   └────────────┬────────────┘   └──────┬──────┘
     │                      │                       │
     └──────────────────────┼───────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │    L4 辩论与研究层         │
              │  Bull ⇄ Bear ⇄ Manager   │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │    L5 交易决策层           │
              │  Trader → Risk Manager   │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  最终报告 + 交易建议        │
              │  (股票方向 + 期权策略)      │
              └───────────────────────────┘
```

### 2.2 各智能体职责一览

| Agent | 层级 | 输入 | 输出 |
|-------|------|------|------|
| MacroAgent | L1 | 宏观数据 | 市场水温评分（Risk-On / Risk-Off） |
| SectorAgent | L1 | 行业 ETF、同行 | 板块强弱 + 相对位置 |
| FundamentalsAgent | L2 | 财报、估值 | 基本面评分 + Bull/Bear 论点 |
| InsiderAgent | L2 | Form 4、回购 | 内部人信心指标 |
| CapitalFlowAgent | L2 | 13F、期权异动、空头 | 大资金方向 |
| TechnicalAgent | L2 | K线、成交量 | 关键位 + 趋势状态 |
| SentimentAgent | L2 | Reddit、新闻、Put/Call | 情绪极端度 |
| EventCalendarAgent | L2 | 财报日历、宏观日历 | 未来 30 天催化剂 |
| OptionsChainAgent | L3 | 期权链、IV、Greeks | 期权定价环境 |
| BullResearcher | L4 | 上述所有 | 看多论文 |
| BearResearcher | L4 | 上述所有 | 看空论文 |
| ResearchManager | L4 | Bull + Bear | 综合胜率与赔率 |
| Trader | L5 | 研究结论 | 交易结构（股票 / 期权组合） |
| RiskManager | L5 | Trader 方案 | 仓位、止损、放行/否决 |

---

## 三、数据层实现

### 3.1 数据源与 Python 库映射

| 数据类别 | 数据源 | Python 库 / API |
|---------|--------|----------------|
| 股价 K 线 | Yahoo Finance | `yfinance` |
| 财务报表 | FMP / SEC | `requests` + FMP API / `sec-edgar-downloader` |
| 期权链 | Yahoo / CBOE | `yfinance.Ticker().option_chain()` |
| 内部人交易 | OpenInsider | `requests` + BeautifulSoup |
| 13F 持仓 | WhaleWisdom / SEC | FMP API / SEC EDGAR |
| 宏观数据 | FRED | `fredapi` |
| 新闻 | NewsAPI / Finnhub | `requests` |
| 社交情绪 | Reddit / Twitter | `praw` / `tweepy` |
| 技术指标 | 本地计算 | `pandas-ta` / `talib` |
| 期权异动 | Unusual Whales | 付费 API |

### 3.2 统一数据网关代码骨架

```python
# data/gateway.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd
import requests
from functools import lru_cache

@dataclass
class DataConfig:
    fmp_api_key: str
    fred_api_key: str
    finnhub_api_key: str
    cache_dir: str = "./cache"

class DataGateway:
    """统一数据网关：所有 Agent 只通过这里取数据，不直接调用外部 API。
    好处：缓存、限流、失败重试统一处理，方便替换数据源。"""

    def __init__(self, config: DataConfig):
        self.config = config

    @lru_cache(maxsize=128)
    def get_price_history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """获取历史 K 线,含 OHLCV。"""
        return yf.Ticker(ticker).history(period=period)

    def get_fundamentals(self, ticker: str) -> dict:
        """基本面:营收、利润、现金流、资产负债、估值。"""
        t = yf.Ticker(ticker)
        return {
            "info": t.info,
            "financials": t.financials,
            "balance_sheet": t.balance_sheet,
            "cashflow": t.cashflow,
            "quarterly_financials": t.quarterly_financials,
        }

    def get_option_chain(self, ticker: str, expiry: Optional[str] = None) -> dict:
        """期权链:如果未指定 expiry 则返回所有到期日 + 最近一个到期日的链。"""
        t = yf.Ticker(ticker)
        expiries = t.options
        target = expiry or expiries[0]
        chain = t.option_chain(target)
        return {
            "expiries": expiries,
            "target_expiry": target,
            "calls": chain.calls,
            "puts": chain.puts,
        }

    def get_insider_trades(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """内部人交易:从 OpenInsider 抓取最近 N 天的 Form 4。"""
        url = f"http://openinsider.com/screener?s={ticker}&fd={days}"
        # ... 抓取 + 解析表格
        return pd.DataFrame()  # 简化示意

    def get_institutional_holdings(self, ticker: str) -> pd.DataFrame:
        """13F 机构持仓变化,通过 FMP。"""
        url = f"https://financialmodelingprep.com/api/v4/institutional-ownership/symbol-ownership"
        params = {"symbol": ticker, "apikey": self.config.fmp_api_key}
        r = requests.get(url, params=params, timeout=30)
        return pd.DataFrame(r.json())

    def get_macro_indicators(self) -> dict:
        """宏观:VIX、10Y 收益率、DXY、Fed Funds Rate。"""
        vix = yf.Ticker("^VIX").history(period="1mo")["Close"].iloc[-1]
        tnx = yf.Ticker("^TNX").history(period="1mo")["Close"].iloc[-1]
        dxy = yf.Ticker("DX-Y.NYB").history(period="1mo")["Close"].iloc[-1]
        return {"vix": vix, "us10y_yield": tnx, "dxy": dxy}

    def get_news(self, ticker: str, days: int = 7) -> list:
        """新闻:Finnhub 免费额度足够。"""
        end = datetime.now()
        start = end - timedelta(days=days)
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "token": self.config.finnhub_api_key,
        }
        return requests.get(url, params=params, timeout=30).json()

    def get_earnings_calendar(self, ticker: str) -> dict:
        """财报日历 + 分析师预期。"""
        t = yf.Ticker(ticker)
        return {
            "next_earnings": t.calendar,
            "recommendations": t.recommendations,
            "analyst_price_targets": t.analyst_price_targets if hasattr(t, "analyst_price_targets") else None,
        }
```

---

## 四、分析层：八个专业 Agent

每个 Agent 都遵循相同结构：`(data) -> AnalysisReport`。用 LLM 做定性推理，用 Python 做定量计算，两者结合。

### 4.1 通用 Agent 基类

```python
# agents/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from langchain_anthropic import ChatAnthropic

@dataclass
class AnalysisReport:
    agent_name: str
    ticker: str
    timestamp: str
    score: float           # -1 到 +1,负数看空正数看多
    confidence: float      # 0 到 1
    key_findings: list[str] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)
    llm_reasoning: str = ""

class BaseAgent(ABC):
    def __init__(self, name: str, llm: ChatAnthropic, gateway):
        self.name = name
        self.llm = llm
        self.gateway = gateway

    @abstractmethod
    def analyze(self, ticker: str) -> AnalysisReport:
        pass

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.llm.invoke([
            ("system", system_prompt),
            ("user", user_prompt),
        ])
        return response.content
```

### 4.2 MacroAgent（宏观 Agent）

判断当前市场处于 Risk-On 还是 Risk-Off，给出个股所处的宏观水温。

```python
# agents/macro.py
class MacroAgent(BaseAgent):
    SYSTEM_PROMPT = """你是宏观分析师。基于给定的宏观指标,判断当前市场风险偏好:
    - VIX < 15: 极度平静(可能酝酿风险); 15-20: 平静; 20-30: 警戒; >30: 恐慌
    - 10Y 收益率快速上升: 对成长股不利
    - DXY 走强: 对美元资产多头有利,对新兴市场不利
    输出 JSON:{"regime": "risk_on|neutral|risk_off", "score": -1 to 1, "reasoning": "..."}"""

    def analyze(self, ticker: str) -> AnalysisReport:
        macro = self.gateway.get_macro_indicators()
        user_prompt = f"当前指标:VIX={macro['vix']:.2f},10Y={macro['us10y_yield']:.2f}%,DXY={macro['dxy']:.2f}"
        llm_out = self._call_llm(self.SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="MacroAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=parsed["score"],
            confidence=0.7,
            key_findings=[parsed["reasoning"]],
            raw_data=macro,
            llm_reasoning=llm_out,
        )
```

### 4.3 FundamentalsAgent（基本面）

```python
# agents/fundamentals.py
class FundamentalsAgent(BaseAgent):
    SYSTEM_PROMPT = """你是基本面分析师。请从以下维度评估:
    1. 增长质量:营收增速趋势、毛利率变化、经营现金流
    2. 估值水平:当前 PE / PS 相对于历史百分位、行业中位数
    3. 财务健康:现金余额、有息负债、Free Cash Flow
    4. Guidance vs Consensus:管理层指引与市场预期的差距

    输出 JSON:{"score": -1 to 1, "bull_thesis": "...", "bear_thesis": "...",
              "red_flags": [...], "green_flags": [...]}"""

    def analyze(self, ticker: str) -> AnalysisReport:
        fund = self.gateway.get_fundamentals(ticker)
        # 量化计算:营收增速、毛利率变化、估值百分位
        metrics = self._compute_metrics(fund)

        user_prompt = f"""
        股票:{ticker}
        营收增速(YoY):{metrics['revenue_growth']}
        毛利率趋势:{metrics['gross_margin_trend']}
        经营现金流:{metrics['ocf']}
        当前 PE:{metrics['pe']},历史百分位:{metrics['pe_percentile']}
        Net Debt / EBITDA:{metrics['net_debt_to_ebitda']}
        """
        llm_out = self._call_llm(self.SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="FundamentalsAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=parsed["score"],
            confidence=0.8,
            key_findings=parsed["green_flags"] + parsed["red_flags"],
            raw_data=metrics,
            llm_reasoning=llm_out,
        )

    def _compute_metrics(self, fund: dict) -> dict:
        info = fund["info"]
        fin = fund["quarterly_financials"]
        # ... 计算逻辑
        return {
            "revenue_growth": ...,
            "gross_margin_trend": ...,
            "ocf": ...,
            "pe": info.get("trailingPE"),
            "pe_percentile": ...,
            "net_debt_to_ebitda": ...,
        }
```

### 4.4 TechnicalAgent（技术面）

```python
# agents/technical.py
import pandas_ta as ta

class TechnicalAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        df = self.gateway.get_price_history(ticker, period="1y")

        # 均线
        df["MA20"] = ta.sma(df["Close"], length=20)
        df["MA50"] = ta.sma(df["Close"], length=50)
        df["MA200"] = ta.sma(df["Close"], length=200)
        # 动量
        df["RSI"] = ta.rsi(df["Close"], length=14)
        # 波动
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        # 布林
        bb = ta.bbands(df["Close"], length=20)
        df = df.join(bb)

        last = df.iloc[-1]
        # 关键位识别:近 6 个月 High / Low + Volume Profile POC
        key_levels = self._find_key_levels(df)

        summary = {
            "price": last["Close"],
            "ma_alignment": "bull" if last["Close"] > last["MA50"] > last["MA200"] else "bear",
            "rsi": last["RSI"],
            "atr_pct": last["ATR"] / last["Close"],
            "bb_position": (last["Close"] - last["BBL_20_2.0"]) / (last["BBU_20_2.0"] - last["BBL_20_2.0"]),
            "key_levels": key_levels,
        }

        # 用 LLM 做结构性判断
        prompt = f"""基于以下技术面数据判断:{summary}
        RSI > 70 超买, < 30 超卖。价格在均线组合中的位置反映趋势。
        输出 JSON:{{"score": -1 to 1, "trend": "up|down|range", "key_support": ..., "key_resistance": ..., "reasoning": "..."}}"""
        llm_out = self._call_llm("你是技术分析师", prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="TechnicalAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=parsed["score"],
            confidence=0.6,  # 技术面本身噪音大,置信度不宜过高
            key_findings=[f"支撑 {parsed['key_support']}", f"阻力 {parsed['key_resistance']}"],
            raw_data=summary,
            llm_reasoning=llm_out,
        )

    def _find_key_levels(self, df: pd.DataFrame) -> dict:
        # 用 pivot points 或 volume profile 识别关键价位
        recent = df.tail(120)
        return {
            "recent_high": recent["High"].max(),
            "recent_low": recent["Low"].min(),
            "volume_poc": self._volume_profile_poc(recent),
        }

    def _volume_profile_poc(self, df: pd.DataFrame) -> float:
        # 用 pandas 简易实现 Volume Profile 的 Point of Control
        bins = pd.cut(df["Close"], bins=30)
        vol_by_price = df.groupby(bins, observed=True)["Volume"].sum()
        poc_bin = vol_by_price.idxmax()
        return (poc_bin.left + poc_bin.right) / 2
```

### 4.5 InsiderAgent、CapitalFlowAgent、SentimentAgent、EventCalendarAgent

结构相同，只是数据源和 prompt 不同。核心思路：

**InsiderAgent**：统计过去 90 天 CEO/CFO/董事的净买入金额，正值 +50 万美元以上给正分。回购规模超过流通市值 2% 也是正分。

**CapitalFlowAgent**：13F 的机构净增持家数、期权 Call/Put 比率、Short Interest 变化率、Dark Pool 印记率。这一层信息延迟大，只做趋势判断不做短线信号。

**SentimentAgent**：Reddit 提及次数变化率、新闻标题的情绪评分（用 FinBERT 或 LLM）、Put/Call Ratio 百分位。**极端情绪往往是反向指标**，Prompt 里要明确这一点。

**EventCalendarAgent**：找出未来 30 天内的所有已知事件（下次财报、FOMC、投资者日、锁定期解禁），标注每个事件的历史平均波动幅度。

### 4.6 OptionsChainAgent（期权定价环境）

这是期权决策的关键 Agent，独立于股票方向判断。

```python
# agents/options_chain.py
import numpy as np
from scipy.stats import norm

class OptionsChainAgent(BaseAgent):
    """分析期权链本身的定价状态,回答:现在是买方友好还是卖方友好?"""

    def analyze(self, ticker: str) -> AnalysisReport:
        chain = self.gateway.get_option_chain(ticker)
        price_hist = self.gateway.get_price_history(ticker, period="1y")

        # 1. IV vs HV
        hv_30 = self._calc_hv(price_hist["Close"], window=30)
        iv_atm = self._atm_iv(chain, price_hist["Close"].iloc[-1])
        iv_hv_spread = iv_atm - hv_30

        # 2. IV Rank / Percentile
        iv_rank = self._iv_rank(ticker)  # 需要历史 IV 序列

        # 3. Skew
        skew = self._calc_skew(chain, price_hist["Close"].iloc[-1])

        # 4. Term Structure
        term = self._term_structure(ticker)

        # 5. Max Pain
        max_pain = self._max_pain(chain)

        summary = {
            "atm_iv": iv_atm,
            "hv_30": hv_30,
            "iv_hv_spread": iv_hv_spread,
            "iv_rank": iv_rank,
            "put_call_skew": skew,
            "term_structure": term,   # "contango" | "backwardation"
            "max_pain": max_pain,
            "current_price": price_hist["Close"].iloc[-1],
        }

        prompt = f"""基于期权数据:
        {summary}

        判断逻辑参考:
        - IV Rank > 70:期权贵,适合卖方(Cash-Secured Put / Covered Call / Iron Condor)
        - IV Rank < 30:期权便宜,适合买方(Long Call/Put / Debit Spread)
        - Backwardation:短期恐慌,近月期权特别贵,通常是短期底部信号
        - 深度 Put Skew:市场买保险积极,可能是防御情绪高涨
        - 股价远离 Max Pain:期权到期日前可能向 Max Pain 收敛

        输出 JSON:{{
            "regime": "buyer_friendly|seller_friendly|neutral",
            "recommended_strategy_family": ["long_call", "credit_spread", ...],
            "warnings": [...],
            "reasoning": "..."
        }}"""

        llm_out = self._call_llm("你是期权定价专家", prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="OptionsChainAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=0,  # 期权 Agent 不给方向分,只给环境判断
            confidence=0.8,
            key_findings=parsed["recommended_strategy_family"] + parsed["warnings"],
            raw_data=summary,
            llm_reasoning=llm_out,
        )

    def _calc_hv(self, prices: pd.Series, window: int = 30) -> float:
        """年化历史波动率。"""
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return log_ret.tail(window).std() * np.sqrt(252)

    def _atm_iv(self, chain: dict, spot: float) -> float:
        """ATM 隐含波动率,取 Call 和 Put 的平均。"""
        calls, puts = chain["calls"], chain["puts"]
        atm_call = calls.iloc[(calls["strike"] - spot).abs().argmin()]
        atm_put = puts.iloc[(puts["strike"] - spot).abs().argmin()]
        return (atm_call["impliedVolatility"] + atm_put["impliedVolatility"]) / 2

    def _iv_rank(self, ticker: str) -> float:
        """IV Rank:当前 IV 在过去一年中的百分位。需要历史 IV 序列。
        简化:用 HV 序列近似,或者对接付费 API。"""
        # 实际实现要维护历史 IV 数据库
        return 50.0  # 占位

    def _calc_skew(self, chain: dict, spot: float) -> float:
        """25 Delta Put IV - 25 Delta Call IV,反映恐慌溢价。"""
        # 简化实现
        return 0.03  # 占位

    def _term_structure(self, ticker: str) -> str:
        """近月 IV vs 远月 IV。"""
        return "contango"  # 占位

    def _max_pain(self, chain: dict) -> float:
        """Max Pain:让期权买方总损失最大的行权价。"""
        calls, puts = chain["calls"], chain["puts"]
        strikes = sorted(set(calls["strike"]) | set(puts["strike"]))
        pain = {}
        for k in strikes:
            call_pain = ((calls["strike"] < k) * (k - calls["strike"]) * calls["openInterest"]).sum()
            put_pain = ((puts["strike"] > k) * (puts["strike"] - k) * puts["openInterest"]).sum()
            pain[k] = call_pain + put_pain
        return min(pain, key=pain.get)
```

---

## 五、辩论与研究层

### 5.1 Bull / Bear Researcher

参考 TradingAgents 的做法，让两个 Agent 在收到全部分析报告后各自写"论文"，然后进入辩论。

```python
# agents/researchers.py
class BullResearcher(BaseAgent):
    def synthesize(self, reports: list[AnalysisReport]) -> dict:
        context = self._format_reports(reports)
        prompt = f"""你是坚定的看多研究员。基于以下八份分析报告,写一份看多论文:

        {context}

        必须包含:
        1. 三个最有力的看多论点(引用具体数据)
        2. 未来 3 个月的价格路径推演
        3. 如果看错了,什么信号会让你放弃这个观点(Kill Switch)

        输出 JSON:{{"thesis": "...", "key_points": [...], "target_price": ..., "kill_switch": "..."}}"""
        return self._parse_json(self._call_llm("看多研究员", prompt))

class BearResearcher(BaseAgent):
    def synthesize(self, reports: list[AnalysisReport]) -> dict:
        # 同上,但从看空角度
        ...

class ResearchManager(BaseAgent):
    """研究主管:主持 Bull 和 Bear 的辩论,给出加权结论。"""

    def moderate(self, bull_thesis: dict, bear_thesis: dict, reports: list[AnalysisReport]) -> dict:
        prompt = f"""你是研究主管,主持辩论。

        看多论点:{bull_thesis}
        看空论点:{bear_thesis}

        请:
        1. 指出双方论点中最强和最弱的部分
        2. 判断当前的证据更支持哪一边(给出 0-100 的看多概率)
        3. 指出必须持续跟踪的三个变量(Watch List)
        4. 给出综合评级:强烈看多 / 看多 / 中性 / 看空 / 强烈看空

        输出 JSON:{{
            "bull_probability": 0-100,
            "rating": "strong_bull|bull|neutral|bear|strong_bear",
            "conviction": 0-1,
            "watch_list": [...],
            "reasoning": "..."
        }}"""
        return self._parse_json(self._call_llm("研究主管", prompt))
```

### 5.2 辩论循环（可选加强版）

TradingAgents 的原论文中，Bull 和 Bear 会进行多轮辩论（比如 3 轮），每轮针对对方的论点做反驳。用 LangGraph 的 conditional edge 实现：

```python
# graph/debate.py
from langgraph.graph import StateGraph, END

def build_debate_graph(bull, bear, manager, max_rounds=3):
    workflow = StateGraph(DebateState)

    workflow.add_node("bull_argue", bull.argue)
    workflow.add_node("bear_argue", bear.argue)
    workflow.add_node("manager_check", manager.check_consensus)

    workflow.set_entry_point("bull_argue")
    workflow.add_edge("bull_argue", "bear_argue")
    workflow.add_edge("bear_argue", "manager_check")
    workflow.add_conditional_edges(
        "manager_check",
        lambda s: "bull_argue" if s["round"] < max_rounds and not s["consensus"] else END
    )
    return workflow.compile()
```

---

## 六、交易决策层

### 6.1 Trader Agent：把研究结论翻译成具体交易结构

这一步是关键：**同样是"看多"，可以有五种不同的交易方式**，Trader 要根据 OptionsChainAgent 的定价环境和自己的风险偏好选择。

```python
# agents/trader.py
class TraderAgent(BaseAgent):
    STRATEGY_MENU = {
        "long_stock": "直接买入股票,无杠杆,无时间衰减",
        "long_call": "买入看涨期权,高杠杆,但 Theta 衰减",
        "bull_call_spread": "买低卖高 Call,降低成本但收益封顶",
        "cash_secured_put": "卖出看跌期权,收权利金,愿意在低位接货",
        "covered_call": "持股 + 卖 Call,增强收益",
        "long_put": "买入看跌期权,做空但风险有限",
        "bear_put_spread": "买高卖低 Put,做空且成本可控",
        "iron_condor": "同时卖跨式,赚取时间价值,适合震荡",
        "straddle": "买跨式,赌大波动,不关心方向",
    }

    def decide(self, research_conclusion: dict, options_env: AnalysisReport,
               technical: AnalysisReport, account_size: float) -> dict:
        prompt = f"""你是交易员,把研究结论翻译成可执行交易。

        研究结论:{research_conclusion}
        期权定价环境:{options_env.raw_data}
        技术面关键位:{technical.key_findings}
        账户规模:{account_size}

        策略菜单:{self.STRATEGY_MENU}

        决策原则:
        - 看多 + IV 高 → 优先 Bull Call Spread 或 Cash-Secured Put(卖方)
        - 看多 + IV 低 → 优先 Long Call 或直接买股(买方)
        - 看空 + IV 高 → 优先 Bear Put Spread
        - 看空 + IV 低 → 优先 Long Put
        - 观点不强 + 高 IV → Iron Condor / 空仓
        - 观点不强 + 事件驱动 → Straddle(赌波动)

        输出 JSON:{{
            "strategy": "策略名",
            "structure": {{
                "action": "buy|sell",
                "instrument": "stock|call|put|spread",
                "strikes": [...],
                "expiry": "YYYY-MM-DD",
                "quantity": ...,
                "estimated_cost": ...,
                "max_profit": ...,
                "max_loss": ...,
                "breakeven": ...
            }},
            "entry_trigger": "什么价格入场",
            "profit_target": "什么条件止盈",
            "stop_loss": "什么条件止损",
            "time_stop": "持有多久无进展就退出",
            "reasoning": "..."
        }}"""
        return self._parse_json(self._call_llm("交易员", prompt))
```

### 6.2 RiskManager Agent：最后一道防线

```python
# agents/risk.py
class RiskManagerAgent(BaseAgent):
    def review(self, trade: dict, portfolio: dict, account_size: float) -> dict:
        max_loss = trade["structure"]["max_loss"]
        loss_pct = max_loss / account_size

        # 硬性规则先过一遍
        hard_rules = self._check_hard_rules(trade, portfolio, account_size)
        if not hard_rules["passed"]:
            return {"approved": False, "reason": hard_rules["reason"]}

        # 软性 LLM 判断
        prompt = f"""你是风控经理,审核这笔交易。

        交易方案:{trade}
        当前组合:{portfolio}
        账户规模:{account_size}
        本笔最大亏损占账户:{loss_pct:.2%}

        审核要点:
        1. 单笔亏损上限:2%
        2. 期权总敞口上限:账户 10%
        3. 相关性:是否与已有持仓构成相同方向的重复押注
        4. 事件风险:是否有未考虑的财报 / FOMC / 解禁
        5. 流动性:期权 Open Interest 和 Bid-Ask Spread 是否可接受
        6. 心理准备:如果亏到最大损失,是否会影响后续决策

        输出 JSON:{{
            "approved": true|false,
            "adjustments": [...],  # 建议调整
            "warnings": [...],
            "final_position_size": ...,
            "reasoning": "..."
        }}"""
        return self._parse_json(self._call_llm("风控经理", prompt))

    def _check_hard_rules(self, trade: dict, portfolio: dict, account_size: float) -> dict:
        max_loss = trade["structure"]["max_loss"]
        if max_loss / account_size > 0.02:
            return {"passed": False, "reason": "单笔亏损超过账户 2%"}

        current_options_exposure = sum(p.get("options_value", 0) for p in portfolio.values())
        new_exposure = trade["structure"]["estimated_cost"]
        if (current_options_exposure + new_exposure) / account_size > 0.10:
            return {"passed": False, "reason": "期权总敞口将超过账户 10%"}

        return {"passed": True, "reason": ""}
```

---

## 七、编排：用 LangGraph 串起来

```python
# graph/main_workflow.py
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AnalysisState(TypedDict):
    ticker: str
    account_size: float
    portfolio: dict
    reports: list  # 各 Agent 的分析报告
    bull_thesis: dict
    bear_thesis: dict
    research_conclusion: dict
    trade_proposal: dict
    risk_review: dict
    final_output: dict

def build_workflow(agents: dict, llm) -> StateGraph:
    workflow = StateGraph(AnalysisState)

    # L1 + L2 并行:所有分析 Agent
    workflow.add_node("run_macro", lambda s: {"reports": s["reports"] + [agents["macro"].analyze(s["ticker"])]})
    workflow.add_node("run_sector", lambda s: {"reports": s["reports"] + [agents["sector"].analyze(s["ticker"])]})
    workflow.add_node("run_fundamentals", lambda s: {"reports": s["reports"] + [agents["fundamentals"].analyze(s["ticker"])]})
    workflow.add_node("run_insider", lambda s: {"reports": s["reports"] + [agents["insider"].analyze(s["ticker"])]})
    workflow.add_node("run_capital", lambda s: {"reports": s["reports"] + [agents["capital"].analyze(s["ticker"])]})
    workflow.add_node("run_technical", lambda s: {"reports": s["reports"] + [agents["technical"].analyze(s["ticker"])]})
    workflow.add_node("run_sentiment", lambda s: {"reports": s["reports"] + [agents["sentiment"].analyze(s["ticker"])]})
    workflow.add_node("run_events", lambda s: {"reports": s["reports"] + [agents["events"].analyze(s["ticker"])]})
    workflow.add_node("run_options", lambda s: {"reports": s["reports"] + [agents["options"].analyze(s["ticker"])]})

    # L3 辩论
    workflow.add_node("bull_research", lambda s: {"bull_thesis": agents["bull"].synthesize(s["reports"])})
    workflow.add_node("bear_research", lambda s: {"bear_thesis": agents["bear"].synthesize(s["reports"])})
    workflow.add_node("moderate", lambda s: {
        "research_conclusion": agents["manager"].moderate(s["bull_thesis"], s["bear_thesis"], s["reports"])
    })

    # L4 交易与风控
    workflow.add_node("trade", lambda s: {
        "trade_proposal": agents["trader"].decide(
            s["research_conclusion"],
            [r for r in s["reports"] if r.agent_name == "OptionsChainAgent"][0],
            [r for r in s["reports"] if r.agent_name == "TechnicalAgent"][0],
            s["account_size"],
        )
    })
    workflow.add_node("risk_check", lambda s: {
        "risk_review": agents["risk"].review(s["trade_proposal"], s["portfolio"], s["account_size"])
    })

    workflow.add_node("finalize", lambda s: {"final_output": _package_output(s)})

    # 编排边
    workflow.set_entry_point("run_macro")
    # 顺序 fan-out(简化,实际可用 parallel branch)
    for a, b in [
        ("run_macro", "run_sector"), ("run_sector", "run_fundamentals"),
        ("run_fundamentals", "run_insider"), ("run_insider", "run_capital"),
        ("run_capital", "run_technical"), ("run_technical", "run_sentiment"),
        ("run_sentiment", "run_events"), ("run_events", "run_options"),
        ("run_options", "bull_research"), ("bull_research", "bear_research"),
        ("bear_research", "moderate"), ("moderate", "trade"),
        ("trade", "risk_check"), ("risk_check", "finalize"), ("finalize", END),
    ]:
        workflow.add_edge(a, b)

    return workflow.compile()

def _package_output(state: AnalysisState) -> dict:
    return {
        "ticker": state["ticker"],
        "analysis_summary": {r.agent_name: {"score": r.score, "confidence": r.confidence, "findings": r.key_findings} for r in state["reports"]},
        "bull_case": state["bull_thesis"],
        "bear_case": state["bear_thesis"],
        "final_view": state["research_conclusion"],
        "trade_recommendation": state["trade_proposal"] if state["risk_review"]["approved"] else None,
        "risk_warnings": state["risk_review"]["warnings"],
        "approved": state["risk_review"]["approved"],
    }
```

---

## 八、项目结构

```
stock_analysis_system/
├── config/
│   ├── __init__.py
│   ├── settings.py           # API keys, LLM 配置
│   └── prompts.py            # 所有 Agent 的 system prompt
├── data/
│   ├── __init__.py
│   ├── gateway.py            # DataGateway
│   ├── cache.py              # SQLite/Redis 缓存层
│   └── historical_iv.py      # 历史 IV 数据库
├── agents/
│   ├── __init__.py
│   ├── base.py               # BaseAgent + AnalysisReport
│   ├── macro.py
│   ├── sector.py
│   ├── fundamentals.py
│   ├── insider.py
│   ├── capital_flow.py
│   ├── technical.py
│   ├── sentiment.py
│   ├── event_calendar.py
│   ├── options_chain.py
│   ├── researchers.py        # Bull, Bear, Manager
│   ├── trader.py
│   └── risk.py
├── graph/
│   ├── __init__.py
│   ├── main_workflow.py      # LangGraph 主流程
│   └── debate.py             # 辩论子图
├── reports/                  # 每次分析的输出存档(JSON + Markdown)
├── logs/
├── tests/
├── notebooks/                # Jupyter 探索 / 回测
├── requirements.txt
├── .env.example
└── main.py                   # 入口
```

### requirements.txt

```txt
langchain>=0.3.0
langchain-anthropic>=0.3.0
langgraph>=0.2.0
yfinance>=0.2.40
pandas>=2.2.0
numpy>=1.26.0
pandas-ta>=0.3.14
scipy>=1.13.0
fredapi>=0.5.2
requests>=2.32.0
beautifulsoup4>=4.12.0
praw>=7.7.0                    # Reddit
python-dotenv>=1.0.0
pydantic>=2.5.0
sqlalchemy>=2.0.0
matplotlib>=3.8.0
```

---

## 九、快速开始

```python
# main.py
from config.settings import Settings
from data.gateway import DataGateway, DataConfig
from langchain_anthropic import ChatAnthropic
from agents import (
    MacroAgent, SectorAgent, FundamentalsAgent, InsiderAgent,
    CapitalFlowAgent, TechnicalAgent, SentimentAgent, EventCalendarAgent,
    OptionsChainAgent, BullResearcher, BearResearcher, ResearchManager,
    TraderAgent, RiskManagerAgent,
)
from graph.main_workflow import build_workflow

def analyze(ticker: str, account_size: float, portfolio: dict = None):
    settings = Settings.load()
    gateway = DataGateway(DataConfig(
        fmp_api_key=settings.fmp_key,
        fred_api_key=settings.fred_key,
        finnhub_api_key=settings.finnhub_key,
    ))
    llm = ChatAnthropic(model="claude-opus-4-7", api_key=settings.anthropic_key)

    agents = {
        "macro": MacroAgent("Macro", llm, gateway),
        "sector": SectorAgent("Sector", llm, gateway),
        "fundamentals": FundamentalsAgent("Fundamentals", llm, gateway),
        "insider": InsiderAgent("Insider", llm, gateway),
        "capital": CapitalFlowAgent("Capital", llm, gateway),
        "technical": TechnicalAgent("Technical", llm, gateway),
        "sentiment": SentimentAgent("Sentiment", llm, gateway),
        "events": EventCalendarAgent("Events", llm, gateway),
        "options": OptionsChainAgent("Options", llm, gateway),
        "bull": BullResearcher("Bull", llm, gateway),
        "bear": BearResearcher("Bear", llm, gateway),
        "manager": ResearchManager("Manager", llm, gateway),
        "trader": TraderAgent("Trader", llm, gateway),
        "risk": RiskManagerAgent("Risk", llm, gateway),
    }

    workflow = build_workflow(agents, llm)
    result = workflow.invoke({
        "ticker": ticker,
        "account_size": account_size,
        "portfolio": portfolio or {},
        "reports": [],
        "bull_thesis": {},
        "bear_thesis": {},
        "research_conclusion": {},
        "trade_proposal": {},
        "risk_review": {},
        "final_output": {},
    })
    return result["final_output"]

if __name__ == "__main__":
    output = analyze("OPEN", account_size=50000)
    import json
    print(json.dumps(output, indent=2, ensure_ascii=False))
```

### 期望输出示例

```json
{
  "ticker": "OPEN",
  "analysis_summary": {
    "MacroAgent": {"score": -0.2, "confidence": 0.7, "findings": ["VIX 22 中性偏警戒"]},
    "FundamentalsAgent": {"score": -0.5, "confidence": 0.8, "findings": ["现金流为负", "跑道 4 季度"]},
    "TechnicalAgent": {"score": 0.3, "confidence": 0.6, "findings": ["支撑 8.5", "阻力 11.2"]},
    "SentimentAgent": {"score": 0.6, "confidence": 0.5, "findings": ["Reddit 提及暴涨"]},
    "OptionsChainAgent": {"score": 0, "confidence": 0.8, "findings": ["IV Rank 85 极高", "Backwardation"]}
  },
  "final_view": {
    "bull_probability": 45,
    "rating": "neutral",
    "conviction": 0.55,
    "watch_list": ["下次财报", "现金流转正拐点", "Reddit 热度是否延续"]
  },
  "trade_recommendation": {
    "strategy": "iron_condor",
    "structure": {
      "action": "sell",
      "instrument": "iron_condor",
      "strikes": [7, 8, 11, 12],
      "expiry": "2026-07-18",
      "quantity": 2,
      "estimated_cost": -180,
      "max_profit": 180,
      "max_loss": 820,
      "breakeven": [7.9, 11.1]
    },
    "reasoning": "观点不强 + IV 极高 → 卖跨赚时间价值,用 Iron Condor 限定尾部风险"
  },
  "approved": true,
  "risk_warnings": ["财报日在到期前,建议财报前平仓"]
}
```

---

## 十、进阶方向与常见陷阱

### 10.1 值得投入时间的进阶方向

**回测框架**：把整个 workflow 接入历史数据，用 `vectorbt` 或 `backtrader` 跑回测。注意要用 point-in-time 数据避免前视偏差。

**IV 历史数据库**：IV Rank 是期权决策的核心指标，但免费数据源不提供历史 IV。可以每天定时拉取期权链快照存到 SQLite，积累半年就有可用的历史 IV 序列。

**Gamma Exposure 分析**：市场做市商的 Gamma 敞口会影响股价波动。SpotGamma 付费提供，但可以自己用期权链 + Open Interest 估算，公式在 CBOE 官方文档里。

**信号权重学习**：初期八个 Agent 的意见简单平均，但不同股票在不同市场环境下最有效的 Agent 是不同的。可以用历史交易结果做监督学习，让权重自适应。

**多轮辩论增强**：Bull 和 Bear 单轮就下结论容易被叙事绑架。加入 3 轮辩论 + 引入第三方 "魔鬼代言人 Agent" 专门挑刺，能显著减少错误论证。

### 10.2 常见陷阱

**LLM 幻觉数据**：LLM 有时会在推理中编造数字。所有关键数据必须通过 Python 计算传给 LLM，Prompt 中明确说 "只能引用我提供的数据，不要自己生成"。

**API 限流**：yfinance 、Finnhub 免费版都有严格限流。DataGateway 必须做本地缓存（同一 ticker 同一天的数据只拉一次），否则跑不了几次就被封。

**期权流动性陷阱**：小盘股期权 Bid-Ask Spread 可能高达 20%，纸面上的策略实际根本执行不动。Trader Agent 必须检查 Volume 和 Open Interest，太低的直接放弃。

**回测过度拟合**：一套系统在历史上表现好，未必是逻辑好，可能是 curve fitting。Walk-Forward Analysis + 样本外验证是标配。

**过度自信**：系统跑通、有几笔漂亮的 paper trading 之后，很容易加大仓位。**永远记住：市场存在结构性变化，昨天有效的模式明天可能失效**。用小仓位、长周期验证系统健壮性，再逐步放大。

**心理层面的自动化陷阱**：即使系统给出信号，真的下单还是人在做。系统越准，人越容易在系统不确定的时候擅自决策；系统偶尔亏损，人越容易失去信任而弃用。最终决定盈亏的还是执行者的纪律，不是代码。

---

## 十一、法律与伦理声明

本系统输出的所有内容仅用于研究、学习和策略探索，**不构成任何投资建议**。使用者需自行承担交易决策的全部后果。作者不是持牌投资顾问，本文档中的所有观点和方法可能存在错误或偏差。

在使用真实资金前，请：
1. 用小仓位 paper trading 至少 3 个月
2. 完整记录每一笔交易的入场、离场、盈亏、复盘
3. 咨询持牌财务顾问了解你的实际风险承受能力
4. 确保投入的资金是你完全可以承受损失的部分

期权是杠杆工具，可能导致快速且完全的本金损失。市场存在你无法预知的风险。系统再完善也无法消除市场的根本不确定性。
