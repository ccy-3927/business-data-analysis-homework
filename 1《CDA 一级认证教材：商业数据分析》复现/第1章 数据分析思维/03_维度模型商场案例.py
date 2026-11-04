"""复现日期维度筛选、产品销售额卡片、日销售总额及日内记录平均销售额。
用法：python 03_维度模型商场案例.py --year 2019 [--month 1]
"""

from pathlib import Path
import sys, argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--month", type=int, choices=range(1, 13))
    args = ap.parse_args()
    c = Case(
        __file__,
        "维度模型商场案例",
        "复现商场 Power BI 的日期筛选和销售指标。默认选择 2019 年，与原文件保存的筛选一致；可用 --year 和 --month 切换。",
    )
    p = c.source(SOURCE / "第1章 数据分析思维/维度模型演示/商场销售数据表.xlsx")
    sales = excel(p, "产品销售数据表")
    products = excel(p, "产品信息表")
    dates = excel(p, "日期表")
    profile(c, sales, "销售事实表")
    # Excel 编码大小写不同。先统一大小写，再验证维度唯一性。
    sales["产品编号"] = sales["产品编号"].str.strip().str.upper()
    products["产品编号"] = products["产品编号"].str.strip().str.upper()
    assert products["产品编号"].is_unique and dates["日期"].is_unique
    assert sales["产品编号"].isin(products["产品编号"]).all()
    assert sales["订单日期"].isin(dates["日期"]).all()
    mismatch = sales.loc[
        sales["销售数量"] * sales["销售单价"] != sales["销售额"]
    ].copy()
    mismatch["Excel行号"] = mismatch.index + 2
    mismatch["数量乘单价"] = mismatch["销售数量"] * mismatch["销售单价"]
    mismatch["原金额减重算金额"] = mismatch["销售额"] - mismatch["数量乘单价"]
    c.table(mismatch, "原始金额差异清单_保留原值")
    linked = sales.merge(
        products[["产品编号", "产品名称"]],
        on="产品编号",
        validate="many_to_one",
        suffixes=("", "_维表"),
    )
    assert linked["产品名称"].eq(linked["产品名称_维表"]).all()
    linked = linked.merge(
        dates[["日期", "年度", "月"]],
        left_on="订单日期",
        right_on="日期",
        validate="many_to_one",
    )
    assert len(linked) == len(sales) and linked["销售额"].sum() == sales["销售额"].sum()
    chosen = linked.loc[linked["年度"].eq(args.year)]
    if args.month:
        chosen = chosen.loc[chosen["月"].eq(args.month)]
    if chosen.empty:
        raise ValueError("筛选范围没有原始记录，请选择教材中存在的年月。")
    period = str(args.year) + (f"年{args.month}月" if args.month else "年")
    daily = chosen.groupby("订单日期")["销售额"].agg(
        销售总额="sum", 记录平均销售额="mean", 记录数="size"
    )
    product = chosen.groupby("产品名称")["销售额"].sum().sort_values(ascending=False)
    c.table(daily, period + "_每日指标")
    c.table(product, period + "_产品销售额")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(daily.index, daily["销售总额"] / 10000, alpha=0.3)
    ax.plot(daily.index, daily["销售总额"] / 10000, lw=1)
    ax.set(
        title=period + "每日销售额总和（原报表第1页）",
        ylabel="销售额（万元）",
        xlabel="日期",
    )
    c.figure(fig, period + "_日销售总额")
    c.line(
        daily["记录平均销售额"] / 10000,
        period + "_日内销售记录平均值",
        "每条销售记录平均销售额（万元）",
    )
    c.bar(product / 10000, period + "_产品销售额卡片迁移", "销售额（万元）")
    # 全期月度对比是对日期下钻的补充，不混入默认筛选指标。
    monthly = linked.groupby(["年度", "月"])["销售额"].sum().unstack(0) / 10000
    c.table(monthly, "全期月度销售额_万元")
    c.line(monthly, "全期月度趋势对比", "销售额（万元）", interactive=True)
    assert daily["销售总额"].sum() == product.sum() == chosen["销售额"].sum()
    c.table(
        pd.DataFrame(
            {
                "校验": [
                    "销售额=数量×单价",
                    "产品编号全部匹配",
                    "日期全部匹配",
                    "关联前后行数相同",
                    "产品与日期汇总一致",
                ],
                "通过": [mismatch.empty, True, True, True, True],
                "说明": [
                    f"{len(mismatch)}条源数据差异，已导出并保留原销售额",
                    "",
                    "",
                    "",
                    "",
                ],
            }
        ),
        "结果核验",
    )
    c.notes += [
        f"原始销售事实表 {len(sales):,} 行，{sales.订单日期.min():%Y-%m-%d} 至 {sales.订单日期.max():%Y-%m-%d}。完全相同记录 {sales.duplicated().sum()} 行；由于没有交易主键，不能断定为错误重复，保留原记录以匹配教材汇总。",
        f"当前筛选：{period}；{len(chosen):,} 条记录；销售额 {chosen.销售额.sum():,.0f} 元；销量 {chosen.销售数量.sum():,.0f}；销售额最高的产品为 {product.idxmax()}。",
        "Power BI 日期切片器 → 命令行参数和日期维度过滤；关系 → merge；SUM → groupby.sum；AVERAGE → groupby.mean；多行卡片 → 数值结果表及横向条形图。",
        "原 PBIX 第二页的字段名称仍保留 Sum 字样，但实际 Aggregation Function=1，显示名为“销售额 的平均值”。这里按日计算每条记录的平均销售额，不能解释成客单价，也不是全月销售额除以天数。",
        "原报表还放置了“关键影响因素”AI 控件，以日期解释销售单价汇总。配套文件未提供该控件的模型、输出或参数，无法声称严格复现其专有算法，本次明确不生成替代因果结论。空白切片器也不作硬复现。",
        f'金额公式检查发现 {len(mismatch)} 条不一致，原销售额合计比数量乘单价多 {mismatch["原金额减重算金额"].sum():,.0f} 元。差异清单含 Excel 原始行号；不假定是折扣或错误，不重写原数据。',
        "字段检查、编码规范化、维度主键唯一性、连接基数及不同维度汇总通过。金额公式检查的源数据异常单独披露。原始文件没有被修改。",
    ]
    c.done()


if __name__ == "__main__":
    main()
