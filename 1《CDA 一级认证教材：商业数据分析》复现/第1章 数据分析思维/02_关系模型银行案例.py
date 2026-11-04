"""复现原 PBIX 两页：信用卡类型与性别、按年及性别统计贷款额。
核心：card → disp → clients；loans → 所有者disp → clients。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *


def main():
    c = Case(
        __file__,
        "关系模型银行案例",
        "依据本章 Power BI 报表配置和 PPT 复现两张图。所有数值来自本章 CSV，不使用随机数。",
    )
    base = SOURCE / "第1章 数据分析思维/关系模型演示-不建议用poweBI操作关系型数据模型"
    frames = {
        n: csv(c.source(base / (n + ".csv")))
        for n in ["card", "disp", "clients", "loans", "accounts"]
    }
    for name, d in frames.items():
        profile(c, d, name)
        key = {
            "card": "card_id",
            "disp": "disp_id",
            "clients": "client_id",
            "loans": "loan_id",
            "accounts": "account_id",
        }[name]
        assert d[key].notna().all() and d[key].is_unique, (name, "主键异常")
    card, disp, clients, loans = [
        frames[n] for n in ["card", "disp", "clients", "loans"]
    ]
    # 多对一连接校验防止错误连接造成行数膨胀。只统计实际存在的卡。
    cards = card.merge(
        disp[["disp_id", "client_id", "account_id"]],
        on="disp_id",
        how="left",
        validate="many_to_one",
    ).merge(
        clients[["client_id", "sex"]],
        on="client_id",
        how="left",
        validate="many_to_one",
    )
    assert len(cards) == len(card) and cards.sex.notna().all()
    counts = pd.crosstab(cards["type"], cards.sex)
    counts.columns.name = "性别"
    counts.index.name = "卡类型"
    c.table(counts, "卡类型与性别人数")
    c.table(counts.div(counts.sum(axis=1), axis=0), "各卡类型内性别比例")
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot.bar(stacked=True, ax=ax, rot=0)
    for container in ax.containers:
        ax.bar_label(container, label_type="center", color="white")
    ax.set(title="信用卡类型与持卡人性别", xlabel="卡类型", ylabel="卡记录数（张）")
    c.figure(fig, "信用卡类型与性别")
    owners = disp[disp.type.eq("所有者")]
    assert owners.account_id.is_unique
    joined = loans.merge(
        owners[["account_id", "client_id"]],
        on="account_id",
        how="left",
        validate="many_to_one",
    ).merge(
        clients[["client_id", "sex"]],
        on="client_id",
        how="left",
        validate="many_to_one",
    )
    assert len(joined) == len(loans) and joined.sex.notna().all()
    joined["年份"] = pd.to_datetime(joined.date).dt.year
    annual = joined.pivot_table(
        index="年份", columns="sex", values="amount", aggfunc="sum", fill_value=0
    )
    c.table(annual, "历年新增贷款金额按性别")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(
        annual.index,
        *(annual[col] / 10000 for col in annual),
        labels=annual.columns,
        alpha=0.85,
    )
    ax.legend(title="账户所有者性别")
    ax.set(
        title="历年新增贷款额：按账户所有者性别",
        xlabel="贷款发放年份",
        ylabel="贷款金额（万原始货币单位）",
    )
    ax.set_xticks(annual.index)
    c.figure(fig, "历年贷款额性别堆积图")
    assert counts.to_numpy().sum() == len(card)
    assert annual.to_numpy().sum() == loans.amount.sum()
    checks = pd.DataFrame(
        {
            "项目": [
                "信用卡关联前后行数",
                "贷款关联前后行数",
                "贷款金额汇总一致",
                "未匹配持卡人",
                "未匹配贷款所有者",
            ],
            "原始/期望": [len(card), len(loans), loans.amount.sum(), 0, 0],
            "复现": [
                len(cards),
                len(joined),
                annual.to_numpy().sum(),
                cards.sex.isna().sum(),
                joined.sex.isna().sum(),
            ],
        }
    )
    c.table(checks, "结果核验")
    c.notes += [
        f"信用卡 {len(card)} 张、持卡客户 {cards.client_id.nunique()} 人；贷款 {len(loans)} 笔、贷款总额 {loans.amount.sum():,.0f}。原始金额币种未在 CSV 中注明，不擅自改标人民币。",
        "卡图按卡记录计数，与 PBIX 的非空性别计数对应；这份数据中一个持卡客户是否有多张卡已通过客户去重数披露。未持卡客户不进入图中，与原报告排除“空白”卡类型一致。",
        "贷款图先筛选 disp.type=所有者，再关联客户。一个账户可能关联所有者和其他授权人，若不筛选，会重复统计贷款。使用 many_to_one 验证连接基数，连接后记录数及金额均与原始贷款表一致。",
        "accounts 用于检验账户主键与关系。trans、order、district 是关系模型的配套表，但原报告两张图不依赖它们，本练习不为了用完所有文件添加额外分析。",
        "Power BI 关系传播在 Python 中显式写为 merge，图形分别迁移为堆叠柱形图和堆积面积图。所有断言已在运行时执行。",
    ]
    assert cards.account_id.isin(frames["accounts"].account_id).all()
    assert loans.account_id.isin(frames["accounts"].account_id).all()
    c.done()


if __name__ == "__main__":
    main()
