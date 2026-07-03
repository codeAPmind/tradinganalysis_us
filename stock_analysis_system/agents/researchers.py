from datetime import datetime
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


class BullResearcher(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        raise NotImplementedError("BullResearcher uses synthesize()")

    def synthesize(self, reports: list[AnalysisReport]) -> dict:
        context = self._format_reports(reports)
        ticker = reports[0].ticker if reports else "UNKNOWN"
        prompt = f"""基于以下分析报告,写看多论文:

{context}

必须包含:
1. 三个最有力的看多论点(引用具体数据)
2. 未来3个月的价格路径推演
3. Kill Switch:什么信号会让你放弃这个观点"""
        llm_out = self._call_llm(Prompts.BULL_RESEARCHER, prompt)
        return self._parse_json(llm_out)


class BearResearcher(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        raise NotImplementedError("BearResearcher uses synthesize()")

    def synthesize(self, reports: list[AnalysisReport]) -> dict:
        context = self._format_reports(reports)
        prompt = f"""基于以下分析报告,写看空论文:

{context}

必须包含:
1. 三个最有力的看空论点(引用具体数据)
2. 未来3个月的下跌路径推演
3. Kill Switch:什么信号会让你放弃这个观点"""
        llm_out = self._call_llm(Prompts.BEAR_RESEARCHER, prompt)
        return self._parse_json(llm_out)


class ResearchManager(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        raise NotImplementedError("ResearchManager uses moderate()")

    def moderate(
        self,
        bull_thesis: dict,
        bear_thesis: dict,
        reports: list[AnalysisReport],
    ) -> dict:
        prompt = f"""主持辩论并给出综合结论。

看多论点: {bull_thesis}
看空论点: {bear_thesis}

请:
1. 指出双方最强和最弱的论点
2. 给出看多概率(0-100)
3. 列出必须持续跟踪的三个变量
4. 给出综合评级"""
        llm_out = self._call_llm(Prompts.RESEARCH_MANAGER, prompt)
        return self._parse_json(llm_out)
