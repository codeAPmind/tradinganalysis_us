from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts

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


class TraderAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        raise NotImplementedError("TraderAgent uses decide()")

    def decide(
        self,
        research_conclusion: dict,
        options_env: AnalysisReport,
        technical: AnalysisReport,
        account_size: float,
    ) -> dict:
        prompt = f"""把研究结论翻译成可执行交易。

研究结论: {research_conclusion}
期权定价环境: {options_env.raw_data}
技术面关键位: {technical.key_findings}
账户规模: ${account_size:,.0f}

可用策略菜单: {STRATEGY_MENU}

决策原则:
- 看多 + IV高 → Bull Call Spread 或 Cash-Secured Put
- 看多 + IV低 → Long Call 或直接买股
- 看空 + IV高 → Bear Put Spread
- 看空 + IV低 → Long Put
- 观点不强 + 高IV → Iron Condor
- 事件驱动不知方向 → Straddle"""
        llm_out = self._call_llm(Prompts.TRADER, prompt)
        return self._parse_json(llm_out)
