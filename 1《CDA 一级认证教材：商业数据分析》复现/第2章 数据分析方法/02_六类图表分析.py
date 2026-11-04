"""趋势、对比、构成、分布、关系、桑基：只读工作簿保存值，不重算随机函数。"""

from chapter2 import *


def main():
    c = newcase(
        __file__,
        "六类图表分析",
        "依据无答案练习中的具体图表任务实现。所有含 RAND/RANDBETWEEN 的输入只读取 Excel 缓存值，不调用随机函数，不与另一版本的不同缓存混用。",
    )
    p = source(c, "1趋势分析.xlsx")
    yearly = block(p, "趋势图", "B2:C13").set_index("year")
    c.table(yearly, "年度订单量")
    c.line(yearly, "年度订单量走势", "订单量（个）", interactive=True)
    weekly = block(p, "趋势图", "B18:E29").set_index("周")
    weekly.iloc[:, 1] = weekly.iloc[:, 0].mean()
    c.table(weekly, "周处理时长")
    c.line(weekly, "周处理时长及预警", "小时")
    monthly = block(p, "趋势图", "B35:D47").set_index("月")
    monthly["环比增幅"] = monthly.iloc[:, 0].pct_change()
    c.table(monthly, "月度订单与环比")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(monthly.index, monthly.iloc[:, 0], marker="o")
    axes[0].set(title="月度订单量与环比变化", ylabel="订单量（个）")
    axes[1].bar(monthly.index, monthly["环比增幅"] * 100, color=COLORS[0])
    axes[1].axhline(0, color="#899AA8", lw=1)
    axes[1].set(xlabel="月", ylabel="环比（%）", xticks=monthly.index)
    for ax in axes:
        ax.grid(axis="y", alpha=0.5)
    c.figure(fig, "月度订单与环比_分轴展示")
    p = source(c, "2对比分析.xlsx")
    product = block(p, "对比分析", "B2:C13").set_index("产品类型").iloc[:, 0]
    c.table(product, "产品订单")
    c.bar(product.sort_values(ascending=False), "不同产品订单对比", "订单量（个）")
    long = block(p, "对比分析", "B18:C29").set_index("产品类型").iloc[:, 0]
    c.bar(long.sort_values(ascending=False), "长名称产品排序", "订单量（个）")
    radar = block(p, "对比分析", "B38:C44").set_index("某顾客画像").iloc[:, 0]
    c.table(radar, "六维画像得分")
    angles = np.linspace(0, 2 * np.pi, len(radar), endpoint=False)
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.plot(np.r_[angles, angles[0]], np.r_[radar.values, radar.iloc[0]], marker="o")
    ax.fill(np.r_[angles, angles[0]], np.r_[radar.values, radar.iloc[0]], alpha=0.14)
    ax.set_xticks(angles, radar.index)
    ax.set_ylim(0, 100)
    ax.set_title("教材六维得分画像", pad=28)
    c.figure(fig, "六维画像雷达图")
    p = source(c, "3构成分析.xlsx")
    s = block(p, "构成分析", "B4:C8").set_index("产品类型").iloc[:, 0]
    c.table(s / s.sum(), "产品销售占比")
    c.bar((s / s.sum() * 100).sort_values(ascending=False), "产品销售构成", "占比（%）")
    mix = block(p, "构成分析", "B17:E22").set_index("年/产品类型")
    c.table(mix, "五年产品销量")
    stacked(c, mix, "五年产品销量构成")
    mix2 = block(p, "构成分析", "B29:E34").set_index("年/产品类型")
    c.table(mix2.div(mix2.sum(axis=1), axis=0), "五年百分比构成")
    stacked(c, mix2, "五年产品百分比构成", percent=True)
    p = source(c, "4分布分析.xlsx")
    flow = block(p, "分布分析", "B2:C50").set_index("时点")
    c.table(flow, "半小时客流")
    c.line(flow, "门店日内客流走势", "客流量（原表单位）", interactive=True)
    age = block(p, "分布分析", "K2:L340").dropna()
    price = block(p, "分布分析", "T1:U339").dropna()
    for d, col, name, unit in [
        (age, "年龄", "用户年龄分布", "岁"),
        (price, "客单价", "客单价分布", "原表金额单位"),
    ]:
        profile(c, d, name)
        values = pd.to_numeric(d[col])
        c.table(values.describe(), "统计_" + name)
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        axes[0].hist(values, bins="fd", color=COLORS[0], edgecolor="white")
        axes[0].set(title=f"{name} · n={len(values)}", ylabel="频数")
        axes[1].boxplot(
            values,
            vert=False,
            patch_artist=True,
            boxprops={"facecolor": "#82B5CC"},
            medianprops={"color": "#19344A"},
            showfliers=True,
        )
        axes[1].set(xlabel=f"{col}（{unit}）", yticks=[])
        c.figure(fig, name)
    p = source(c, "5关系分析.xlsx")
    activity = block(p, "关系图", "B3:D8").set_index("地区/人数")
    c.table(activity, "营销活动下单人数")
    stacked(c, activity, "各营销活动下单比例", percent=True)
    satisfaction = block(p, "关系图", "B15:D18").set_index("忠诚度")
    c.table(satisfaction, "忠诚度与满意度")
    stacked(c, satisfaction, "不同忠诚度的满意构成", percent=True)
    spend = block(p, "关系图", "L1:O99").dropna()
    c.table(spend, "营销活动消费明细")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [spend[k] for k in spend.columns[1:]],
        labels=spend.columns[1:],
        patch_artist=True,
        boxprops={"facecolor": "#82B5CC"},
        medianprops={"color": "#19344A"},
    )
    ax.set(
        title=f"各营销活动消费分布 · 每组 n={len(spend)}", ylabel="销售金额（原表单位）"
    )
    ax.grid(axis="y", alpha=0.5)
    c.figure(fig, "活动消费分布箱线图")
    relation = block(p, "关系图", "Y1:Z59").dropna()
    c.table(relation, "投入产出数据")
    scatter(c, relation, "投入成本", "产出收益", "投入与产出散点关系")
    p = source(c, "6桑吉图.xlsx")
    links = excel(p, 0).dropna(subset=["来源", "目标", "流量"])
    labels = list(dict.fromkeys(links.来源.tolist() + links.目标.tolist()))
    ids = {s: i for i, s in enumerate(labels)}
    f = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=22,
                thickness=20,
                color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
            ),
            link=dict(
                source=links.来源.map(ids), target=links.目标.map(ids), value=links.流量
            ),
        )
    )
    c.interactive(f, "网站访问路径桑基图")
    c.table(links, "桑基图流量边")
    balances = pd.DataFrame(
        {
            "流入": links.groupby("目标").流量.sum(),
            "流出": links.groupby("来源").流量.sum(),
        }
    ).fillna(0)
    balances["流入减流出"] = balances.流入 - balances.流出
    c.table(balances, "节点流量核对")
    c.notes += [
        "趋势折线、预警线、产品比较、雷达、构成、直方图、箱线、分类关系与投入产出散点均对应原表的明确练习；构成的饼图表达改为按占比排序条形图，月度订单与环比拆成上下两个图，避免双轴尺度误导。",
        "箱线：箱体为Q1至Q3，中线为中位数，须为1.5×IQR内最远观察值，须外点为离群观察，不删除。样本量见各图标题。年龄和客单价是原表独立列，保留其原标签，不推定为同一业务变量。",
        "桑基图是独立交互图，可拖动节点并悬停查看边流量。原表边按原值保留；中间节点进出不平衡时不补造去向或流量。HTML 文件仅包含单张图，不是审阅页面。",
        f"年度订单峰值出现在 {yearly.iloc[:,0].idxmax()}；周处理时长峰值 {weekly.iloc[:,0].max():.2f} 小时，比样本平均高 {weekly.iloc[:,0].max()-weekly.iloc[:,0].mean():.2f} 小时。以上只描述教材保存样本。",
    ]
    c.done()


if __name__ == "__main__":
    main()
