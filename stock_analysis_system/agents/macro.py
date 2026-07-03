from datetime import datetime
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


class MacroAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        macro = self.gateway.get_macro_indicators()
        user_prompt = (
            f"当前指标:VIX={macro['vix']:.2f}, "
            f"10Y收益率={macro['us10y_yield']:.2f}%, "
            f"DXY={macro['dxy']:.2f}"
        )
        llm_out = self._call_llm(Prompts.MACRO, user_prompt)
        parsed = self._parse_json(llm_out)

        return AnalysisReport(
            agent_name="MacroAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.7,
            key_findings=[parsed.get("reasoning", ""), f"市场状态: {parsed.get('regime', 'unknown')}"],
            raw_data=macro,
            llm_reasoning=llm_out,
        )
