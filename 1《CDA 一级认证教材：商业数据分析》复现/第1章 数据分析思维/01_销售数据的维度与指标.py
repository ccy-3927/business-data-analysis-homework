"""sale.csv：以年份、市场为维度，销售额、利润为度量；利润率必须先汇总再相除。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *


def main():
    c = Case(
        __file__,
        "销售数据的维度与指标",
        "配套 sale.csv 的 Python 分析练习。原目录没有对应成品图，因此这些图是基于教材数据的维度分析，不宣称为教材原图。",
    )
    d = csv(c.source(SOURCE / "第1章 数据分析思维/sale.csv"))
    assert not d[["year", "market", "sale", "profit"]].isna().any().any()
    assert not d.duplicated(["year", "market"]).any()
    profile(c, d, "销售数据")
    yearly = d.groupby("year")[["sale", "profit"]].sum()
    yearly["利润率"] = yearly.profit / yearly.sale
    yearly["销售额同比"] = yearly.sale.pct_change()
    c.table(yearly, "年度指标")
    bymarket = d.groupby("market")[["sale", "profit"]].sum()
    bymarket["利润率"] = bymarket.profit / bymarket.sale
    c.table(bymarket, "市场指标")
    pivot = d.pivot(index="year", columns="market", values="sale")
    pivot.index.name = "年份"
    pivot.columns.name = "市场"
    c.table(pivot, "年度市场销售额透视表")
    c.line(pivot, "各市场销售额趋势", "销售额（原数据单位）", interactive=True)
    c.bar(
        bymarket.sale.sort_values(ascending=False),
        "市场销售额对比",
        "销售额（原数据单位）",
        interactive=True,
    )
    c.bar(
        bymarket["利润率"].mul(100).sort_values(ascending=False),
        "市场整体利润率",
        "利润率（%）",
        interactive=True,
    )
    assert yearly.sale.sum() == d.sale.sum() == bymarket.sale.sum()
    c.notes += [
        f"原始记录 {len(d)} 条；销售额合计 {d.sale.sum():,.0f}，利润合计 {d.profit.sum():,.0f}，整体利润率 {d.profit.sum()/d.sale.sum():.2%}。数据未标注币种，图中保留原数据单位。",
        "groupby 对应按维度汇总；pivot 对应 Excel 透视表；pct_change 对应同比计算。首年的同比留空，不填造基期。利润率=sum(profit)/sum(sale)，不是各行利润率的简单平均。",
        f"累计销售额最高的市场是“{bymarket.sale.idxmax()}”；仅描述这份教材样本，不推断现实市场。校验通过：年份、市场交叉键唯一，各维度销售额总计一致。",
    ]
    c.done()


if __name__ == "__main__":
    main()
