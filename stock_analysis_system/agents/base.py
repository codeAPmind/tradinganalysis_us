import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AnalysisReport:
    agent_name: str
    ticker: str
    timestamp: str
    score: float          # -1 到 +1,负数看空正数看多
    confidence: float     # 0 到 1
    key_findings: list[str] = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)
    llm_reasoning: str = ""


class BaseAgent(ABC):
    def __init__(self, name: str, llm, gateway):
        self.name = name
        self.llm = llm
        self.gateway = gateway

    @abstractmethod
    def analyze(self, ticker: str) -> AnalysisReport:
        pass

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return response.content

    def _parse_json(self, text: str) -> dict:
        # 尝试从文本中提取 JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 找 ```json ... ``` 块
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # 找第一个 { ... }
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _format_reports(self, reports: list[AnalysisReport]) -> str:
        lines = []
        for r in reports:
            lines.append(f"[{r.agent_name}] 评分:{r.score:.2f} 置信度:{r.confidence:.2f}")
            lines.append(f"  关键发现: {'; '.join(r.key_findings[:3])}")
            lines.append(f"  原始推理: {r.llm_reasoning[:300]}")
        return "\n".join(lines)
