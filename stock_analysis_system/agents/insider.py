from datetime import datetime
import re
from .base import BaseAgent, AnalysisReport
from config.prompts import Prompts


def _parse_value(val_str: str) -> float:
    """把 '+$487,800' / '-$322,090' 这类字符串转为数字。"""
    if not val_str or val_str == "":
        return 0.0
    cleaned = re.sub(r"[,$]", "", str(val_str)).replace("+", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


class InsiderAgent(BaseAgent):
    def analyze(self, ticker: str) -> AnalysisReport:
        df = self.gateway.get_insider_trades(ticker, days=90)

        if df.empty:
            raw = {"ticker": ticker, "trade_count": 0, "net_buy_value": 0, "trades": []}
            llm_out = self._call_llm(
                Prompts.INSIDER,
                f"股票: {ticker}\n过去90天内无内部人交易数据。"
            )
            parsed = self._parse_json(llm_out)
            return AnalysisReport(
                agent_name="InsiderAgent",
                ticker=ticker,
                timestamp=datetime.now().isoformat(),
                score=float(parsed.get("score", 0)),
                confidence=0.4,
                key_findings=["过去90天内无内部人交易数据"],
                raw_data=raw,
                llm_reasoning=llm_out,
            )

        # 解析净买入金额
        trades = []
        net_value = 0.0
        purchase_value = 0.0
        sale_value = 0.0

        # 列名含 \xa0，统一清洗
        df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]

        for _, row in df.iterrows():
            trade_type = str(row.get("Trade Type", ""))
            value = _parse_value(row.get("Value", ""))
            name = str(row.get("Insider Name", ""))
            title = str(row.get("Title", ""))
            trade_date = str(row.get("Trade Date", ""))
            price = str(row.get("Price", ""))
            qty = str(row.get("Qty", ""))

            is_purchase = "Purchase" in trade_type or "P -" in trade_type
            is_sale = "Sale" in trade_type and "S -" in trade_type

            trades.append({
                "name": name,
                "title": title,
                "date": trade_date,
                "type": trade_type,
                "price": price,
                "qty": qty,
                "value": value,
            })

            if is_purchase:
                purchase_value += abs(value)
                net_value += abs(value)
            elif is_sale:
                sale_value += abs(value)
                net_value -= abs(value)

        raw = {
            "ticker": ticker,
            "trade_count": len(df),
            "purchase_value": purchase_value,
            "sale_value": sale_value,
            "net_buy_value": net_value,
            "trades": trades,
        }

        # 格式化给 LLM
        trade_lines = []
        for t in trades:
            trade_lines.append(
                f"  {t['date']} | {t['name']} ({t['title']}) | {t['type']} | {t['price']} x {t['qty']} = {t['value']:+,.0f}$"
            )

        user_prompt = f"""
股票: {ticker}
过去90天内部人交易明细:
{chr(10).join(trade_lines)}

汇总:
- 净买入金额: ${net_value:+,.0f}
- 总买入: ${purchase_value:,.0f}
- 总卖出: ${sale_value:,.0f}
- 交易笔数: {len(df)}
"""
        llm_out = self._call_llm(Prompts.INSIDER, user_prompt)
        parsed = self._parse_json(llm_out)

        # 关键发现
        ceo_buys = [t for t in trades if "CEO" in t["title"] and "Purchase" in t["type"]]
        findings = []
        if ceo_buys:
            b = ceo_buys[0]
            findings.append(f"CEO {b['name']} 于 {b['date']} 买入 {b['qty']} 股，金额 {b['value']:+,.0f}$")
        findings += [
            f"净买入金额: ${net_value:+,.0f}（买入 ${purchase_value:,.0f} / 卖出 ${sale_value:,.0f}）",
            parsed.get("reasoning", ""),
        ]

        return AnalysisReport(
            agent_name="InsiderAgent",
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            score=float(parsed.get("score", 0)),
            confidence=0.65,
            key_findings=findings,
            raw_data=raw,
            llm_reasoning=llm_out,
        )
