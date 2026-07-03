"""
可比公司配置库
用法：
    from config.comparables import COMPARABLES
    comps = COMPARABLES["OPEN"]   # 取预设配置
    comps = COMPARABLES["_auto"]  # 让 LLM 自动发现

每条记录字段：
  ticker            : 可比公司股票代码
  anchor_date       : 锚点日期（业务里程碑对齐点，不是日历对齐）
  anchor_reason     : 为什么选这个日期（里程碑描述）
  similarity_dimension : 相似的核心维度
  what_happened_after  : LLM 预填的历史知识（可选，帮助 LLM 更准确）
"""

COMPARABLES: dict[str, list[dict]] = {

    # ───────────────────────────────────────────────
    # OPEN — Opendoor Technologies（iBuyer 房产撮合）
    # ───────────────────────────────────────────────
    "OPEN": [
        {
            "ticker": "CVNA",
            "anchor_date": "2023-11-01",
            "anchor_reason": (
                "Carvana 债务重组完成后第一个季度，调整后 EBITDA 首次转正，"
                "从濒临破产（2022年负债超$7B）到运营重构起点"
            ),
            "similarity_dimension": (
                "高杠杆资产撮合平台（房/车）从濒临退市到业务重构："
                "①收缩交易量降低亏损 ②毛利率改善先于营收增长 ③CEO 亲自增持信心"
            ),
            "what_happened_after": (
                "锚点后6个月+180%，12个月+849%。"
                "主要驱动：毛利率从3%→18%，GPU（单车毛利）连续创新高，"
                "市场重新定价从'破产风险'到'盈利转型'"
            ),
        },
        {
            "ticker": "ZILLOW",
            "anchor_date": "2021-11-02",
            "anchor_reason": (
                "Zillow 宣布关闭 iBuying 业务（Zillow Offers），"
                "确认 iBuyer 模式在高利率环境下的系统性风险"
            ),
            "similarity_dimension": (
                "iBuyer 同类竞争对手的失败案例对照："
                "Zillow 退出时的估值压缩路径 vs OPEN 当前是否会重蹈"
            ),
            "what_happened_after": (
                "宣布后股价当日暴跌-24%，此后6个月累计跌超50%。"
                "关键差异：Zillow 是主动关闭，OPEN 是收缩但坚持，路径不同"
            ),
        },
    ],

    # ───────────────────────────────────────────────
    # TSLA — Tesla（高估值成长股）
    # ───────────────────────────────────────────────
    "TSLA": [
        {
            "ticker": "AMZN",
            "anchor_date": "2001-10-01",
            "anchor_reason": (
                "亚马逊互联网泡沫后股价从$113跌至$5.5，"
                "市场质疑商业模式可行性，PE 为负，营收增速骤降"
            ),
            "similarity_dimension": (
                "高估值科技公司在增速放缓期的估值重定价："
                "①PE 极高但基本面韧性 ②市场质疑长期故事 ③管理层大幅减持"
            ),
            "what_happened_after": (
                "锚点后5年+3000%。但需注意：AMZN 当时 PS 仅1-2倍，"
                "TSLA 当前 PS 16倍，估值起点完全不同"
            ),
        },
        {
            "ticker": "NVDA",
            "anchor_date": "2022-10-14",
            "anchor_reason": (
                "英伟达股价从峰值跌超60%，市场担忧游戏需求下滑，"
                "内部人大量减持，PE 仍偏高"
            ),
            "similarity_dimension": (
                "高PE半导体/科技龙头在周期下行期的估值压缩后反弹："
                "核心驱动力转换（游戏→AI）成为反转催化剂"
            ),
            "what_happened_after": (
                "锚点后12个月+200%，AI 叙事重构估值体系。"
                "对 TSLA 的启示：需要找到类似'新叙事'（FSD/Robotaxi）"
            ),
        },
    ],

    # ───────────────────────────────────────────────
    # MU — Micron（存储芯片周期股）
    # ───────────────────────────────────────────────
    "MU": [
        {
            "ticker": "MU",
            "anchor_date": "2022-10-03",
            "anchor_reason": (
                "美光上一个周期低点：内存价格暴跌，PE 为负，"
                "内部人大规模减持，市场极度悲观"
            ),
            "similarity_dimension": (
                "美光自身历史周期：当前高管减持$9252万 "
                "与 2022年周期顶部的减持行为高度相似"
            ),
            "what_happened_after": (
                "2022年低点后12个月+80%，但随后因 HBM 供需紧张再涨一倍。"
                "历史规律：高管减持往往领先股价顶部3-6个月"
            ),
        },
        {
            "ticker": "SMSN",  # 三星，场外近似参考
            "anchor_date": "2023-06-01",
            "anchor_reason": (
                "三星宣布主动减产以支撑内存价格，行业供给侧开始收缩，"
                "对应 MU 受益于竞争格局改善的节点"
            ),
            "similarity_dimension": (
                "存储芯片行业供给侧改善驱动 ASP 上涨的历史规律"
            ),
            "what_happened_after": (
                "减产公告后 MU 股价在6个月内涨超40%，ASP 企稳是核心驱动"
            ),
        },
    ],

    # ───────────────────────────────────────────────
    # 通用模板：新增目标股时复制此块
    # ───────────────────────────────────────────────
    # "TICKER": [
    #     {
    #         "ticker": "COMP_TICKER",
    #         "anchor_date": "YYYY-MM-DD",
    #         "anchor_reason": "为什么选这个日期（业务里程碑）",
    #         "similarity_dimension": "相似的核心维度（商业模式/财务状态/市场情绪）",
    #         "what_happened_after": "锚点之后发生了什么（预填历史知识帮助LLM）",
    #     },
    # ],

    # 特殊值：传 "_auto" 让 LLM 自动发现可比公司
    "_auto": [],
}


def get_comparables(ticker: str) -> list[dict] | None:
    """
    返回目标股的可比公司配置。
    - 有预设 → 返回预设列表
    - 无预设 → 返回 None（触发 LLM 自动发现）
    """
    key = ticker.upper()
    if key in COMPARABLES:
        return COMPARABLES[key]
    return None  # None = 自动发现
