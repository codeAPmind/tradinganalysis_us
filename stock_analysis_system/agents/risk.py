from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


class RiskManagerAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        raise NotImplementedError("RiskManagerAgent uses review()")

    def review(self, trade: dict, portfolio: dict, account_size: float) -> dict:
        structure = trade.get("structure", {})
        max_loss = abs(structure.get("max_loss", 0) or 0)
        loss_pct = max_loss / account_size if account_size > 0 else 0

        hard_rules = self._check_hard_rules(trade, portfolio, account_size)
        if not hard_rules["passed"]:
            return {
                "approved": False,
                "adjustments": [],
                "warnings": [hard_rules["reason"]],
                "final_position_size": 0,
                "reasoning": hard_rules["reason"],
            }

        prompt = f"""审核这笔交易方案。

交易方案: {trade}
当前组合持仓数: {len(portfolio)}
账户规模: ${account_size:,.0f}
本笔最大亏损占账户: {loss_pct:.2%}

审核要点: 单笔亏损≤2%, 期权总敞口≤10%, 相关性集中度, 流动性"""
        llm_out = self._call_llm(Prompts.RISK_MANAGER, prompt)
        result = self._parse_json(llm_out)
        if not result:
            result = {"approved": True, "adjustments": [], "warnings": [], "reasoning": "LLM解析失败,默认放行"}
        return result

    def _check_hard_rules(self, trade: dict, portfolio: dict, account_size: float) -> dict:
        structure = trade.get("structure", {})
        max_loss = abs(structure.get("max_loss", 0) or 0)

        if max_loss / account_size > 0.02:
            return {"passed": False, "reason": f"单笔亏损 ${max_loss:,.0f} 超过账户2% (${account_size * 0.02:,.0f})"}

        current_options_exposure = sum(
            abs(p.get("options_value", 0) or 0) for p in portfolio.values()
        )
        new_exposure = abs(structure.get("estimated_cost", 0) or 0)
        if (current_options_exposure + new_exposure) / account_size > 0.10:
            return {"passed": False, "reason": "期权总敞口将超过账户10%"}

        return {"passed": True, "reason": ""}
