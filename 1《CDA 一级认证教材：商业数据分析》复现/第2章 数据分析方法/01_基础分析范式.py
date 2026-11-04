"""从交易/登录明细重算波士顿矩阵、RFM、忠诚度、同期群；另复现漏斗和相关分析。"""

from chapter2 import *
from scipy.stats import pearsonr, linregress


def case(name, desc):
    return finish_sub(newcase(__file__, name, desc), name)


def bcg():
    c = case(
        "波士顿矩阵",
        "复现教材以销售额占比、环比增长率划分品牌的矩阵。阈值来自本数据；不是经典 BCG 的相对最大竞争者市场份额。",
    )
    p = source(c, "1-1波士顿矩阵-参考.xlsx")
    d = excel(p, "销售表", usecols="A:G")
    d = d.dropna(subset=["订单ID"])
    profile(c, d, "销售明细")
    d["月份"] = pd.to_datetime(d["销售日期"]).dt.to_period("M").astype(str)
    monthly = d.pivot_table(
        index="品牌", columns="月份", values="订单金额", aggfunc="sum", fill_value=0
    )
    assert len(monthly.columns) == 2
    old, new = monthly.columns
    result = monthly.copy()
    result["销售额占比"] = monthly.sum(axis=1) / monthly.to_numpy().sum()
    result["环比增长率"] = safe_div(monthly[new] - monthly[old], monthly[old])
    growth = monthly[new].sum() / monthly[old].sum() - 1
    share = result["销售额占比"].mean()
    result["分类"] = np.select(
        [
            (result.销售额占比 >= share) & (result.环比增长率 >= growth),
            (result.销售额占比 >= share) & (result.环比增长率 < growth),
            (result.销售额占比 < share) & (result.环比增长率 >= growth),
        ],
        ["明星", "金牛", "问题"],
        default="瘦狗",
    )
    c.table(result, "品牌指标及分类")
    ref = block(
        p, "参考答案", "J6:L19", ["品牌", "销售额占比", "环比增长率"]
    ).set_index("品牌")
    compare(c, result[ref.columns], ref, "与教材参考指标核对")
    assert np.allclose(result[ref.columns].sort_index(), ref.sort_index())
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, (name, g) in enumerate(result.groupby("分类")):
        ax.scatter(
            g.销售额占比 * 100,
            g.环比增长率 * 100,
            s=100 + g.销售额占比 * 1800,
            label=name,
            alpha=0.8,
            color=COLORS[i],
            edgecolor="white",
        )
        for brand, r in g.iterrows():
            ax.annotate(
                brand,
                (r.销售额占比 * 100, r.环比增长率 * 100),
                xytext=(7, 5),
                textcoords="offset points",
            )
    ax.axvline(share * 100, color="#899AA8", ls="--")
    ax.axhline(growth * 100, color="#899AA8", ls="--")
    ax.set(
        title="品牌组合：市场份额与增长",
        xlabel="两个月合计销售额占比（%）",
        ylabel=f"{new} 相对 {old} 增长率（%）",
    )
    ax.legend(bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.4)
    c.figure(fig, "波士顿矩阵")
    f = go.Figure()
    for name, g in result.groupby("分类"):
        f.add_trace(
            go.Scatter(
                x=g.销售额占比 * 100,
                y=g.环比增长率 * 100,
                text=g.index,
                mode="markers+text",
                textposition="top center",
                name=name,
                marker=dict(size=12 + g.销售额占比 * 100),
                hovertemplate="品牌%{text}<br>份额%{x:.2f}%<br>环比%{y:.2f}%<extra></extra>",
            )
        )
    f.add_vline(x=share * 100, line_dash="dash")
    f.add_hline(y=growth * 100, line_dash="dash")
    f.update_xaxes(title="销售额占比（%）")
    f.update_yaxes(title="环比增长率（%）")
    c.interactive(f, "波士顿矩阵")
    c.notes += [
        f"两期为 {old}、{new}；份额按两期合计计算，增长按后一月/前一月−1计算。份额阈值 {share:.2%}，市场整体环比阈值 {growth:.2%}。全部品牌与教材参考数值核对通过。",
        "气泡面积随销售份额变化。分区用于提出资源配置线索，不能仅凭两个月的样本判断长期经营表现。",
    ]
    c.done()


def rfm():
    c = case(
        "RFM用户分层",
        "从原始流水计算去重消费频次、消费总额及最近一次消费间隔，并按教材均值阈值分成八类。",
    )
    p = source(c, "1-2RFM模型-参考.xlsx")
    d = (
        excel(p, "原始数据", usecols="A:H")
        .dropna(subset=["用户编号"])
        .rename(columns={"销售日期": "消费日期"})
    )
    profile(c, d, "交易明细")
    now = pd.Timestamp(block(p, "参考答案", "B1:B1", ["时点"]).iloc[0, 0])
    d["消费日期"] = pd.to_datetime(d["消费日期"])
    assert (d.消费日期 <= now).all()
    r = d.groupby("用户编号").agg(
        最近消费=("消费日期", "max"), F=("流水号", "nunique"), M=("销售金额", "sum")
    )
    r["R"] = (now - r.最近消费).dt.days
    means = r[["R", "F", "M"]].mean()
    r["R程度"] = np.where(r.R < means.R, "高", "低")
    r["F程度"] = np.where(r.F > means.F, "高", "低")
    r["M程度"] = np.where(r.M > means.M, "高", "低")
    mapping = block(p, "业务背景", "O4:R11", ["R程度", "F程度", "M程度", "用户类别"])
    lookup = dict(zip(mapping.R程度 + mapping.F程度 + mapping.M程度, mapping.用户类别))
    r["用户类别"] = (r.R程度 + r.F程度 + r.M程度).map(lookup)
    ref = excel(p, "参考答案", header=5, usecols="A:I")
    ref = ref[pd.to_numeric(ref["用户编号"], errors="coerce").notna()].set_index(
        "用户编号"
    )
    check = r[["R", "F", "M"]].rename(
        columns={"R": "消费时间间隔", "F": "消费频次", "M": "消费金额"}
    )
    compare(c, check, ref[check.columns], "RFM与参考答案核对")
    assert np.allclose(check.sort_index(), ref[check.columns].sort_index())
    assert r["用户类别"].sort_index().equals(ref["用户类别"].sort_index())
    summary = (
        r.groupby("用户类别")
        .agg(
            人数=("R", "size"),
            消费额=("M", "sum"),
            平均间隔=("R", "mean"),
            平均频次=("F", "mean"),
        )
        .sort_values("消费额", ascending=False)
    )
    summary["人数占比"] = summary.人数 / len(r)
    summary["消费额占比"] = summary.消费额 / summary.消费额.sum()
    c.table(r, "用户RFM明细")
    c.table(summary, "八类用户汇总")
    c.table(means, "阈值")
    c.bar(summary.人数, "RFM分层人数", "用户数", interactive=True)
    c.bar(summary.消费额 / 10000, "RFM分层消费贡献", "消费额（万元）", interactive=True)
    assert np.isclose(r.M.sum(), d.销售金额.sum()) and summary.人数.sum() == len(r)
    c.notes += [
        f"分析时点由教材 B1 读取：{now.date()}。R 越小越好；R 严格小于均值算高，F/M 严格大于均值算高，与 Excel IF 保持一致。用户 {len(r):,} 人，阈值 R={means.R:.4f}天、F={means.F:.4f}次、M={means.M:.4f}。",
        f"“{summary.index[0]}”的消费贡献最高，占 {summary.消费额占比.iloc[0]:.2%}。该结果是历史分层，不是未来购买预测。明细数值和八类标签均与教材参考逐用户核对。",
    ]
    c.done()


def loyalty():
    c = case(
        "用户忠诚度",
        "先识别近13周未消费的易流失用户及仅首次消费的潜在用户，再对其余用户做频次与金额四分位评分。",
    )
    p = source(c, "1-3用户忠诚度模型-参考.xlsx")
    d = excel(p, "销售流水表", usecols="A:E").dropna(subset=["会员编号"])
    d["消费日期"] = pd.to_datetime(d["消费日期"])
    now = pd.Timestamp(
        block(p, "潜在&易流失用户-数据准备", "B1:B1", ["时点"]).iloc[0, 0]
    )
    cutoff = now - pd.Timedelta(weeks=13)
    assert (d.消费日期 < now).all()
    r = d.groupby("会员编号").agg(
        F=("订单编号", "nunique"), M=("消费金额", "sum"), 最近消费=("消费日期", "max")
    )
    r["之前次数"] = (
        d.loc[d.消费日期 < cutoff]
        .groupby("会员编号")
        .订单编号.nunique()
        .reindex(r.index, fill_value=0)
    )
    r["近13周次数"] = (
        d.loc[d.消费日期 >= cutoff]
        .groupby("会员编号")
        .订单编号.nunique()
        .reindex(r.index, fill_value=0)
    )
    r["用户类别"] = np.select(
        [r["近13周次数"].eq(0), r["之前次数"].eq(0) & r["近13周次数"].eq(1)],
        ["易流失用户", "潜在用户"],
        default="待评分",
    )
    active = r["用户类别"].eq("待评分")
    quant = r.loc[active, ["F", "M"]].quantile([0.25, 0.75])
    for col in ["F", "M"]:
        r.loc[active, col + "评分"] = np.select(
            [
                r.loc[active, col] < quant.loc[0.25, col],
                r.loc[active, col] > quant.loc[0.75, col],
            ],
            [1, 3],
            default=2,
        )
    score = r.loc[active, ["F评分", "M评分"]].sum(axis=1)
    r.loc[active, "用户类别"] = np.select(
        [score.eq(6), score.eq(5), score.eq(4)],
        ["忠诚用户", "进阶用户", "普通用户"],
        default="低价值用户",
    )
    ref = (
        excel(p, "参考答案", header=8, usecols="A:G")
        .dropna(subset=["会员编号"])
        .set_index("会员编号")
    )
    ref = ref.loc[ref.index.astype(str).str.startswith("M")]
    comparison = r.loc[ref.index, ["F", "M"]].rename(
        columns={"F": "消费次数", "M": "消费额"}
    )
    compare(c, comparison, ref[["消费次数", "消费额"]], "忠诚度基础指标核对")
    assert np.allclose(comparison, ref[["消费次数", "消费额"]])
    assert (r.loc[ref.index, "用户类别"] == ref["用户类别"]).all()
    c.table(r, "忠诚度用户明细")
    c.table(quant, "活跃用户四分位阈值")
    summary = r.groupby("用户类别").agg(人数=("F", "size"), 消费额=("M", "sum"))
    c.table(summary, "用户分类汇总")
    c.bar(
        summary.人数.sort_values(ascending=False),
        "忠诚度分层",
        "用户数",
        interactive=True,
    )
    c.notes += [
        f"教材分析时点 {now.date()}、13周界限 {cutoff.date()}；本实现将界限当日纳入近13周。对 {active.sum()} 名待评分用户，F/M 分别按 Q1、Q3 评分，小于Q1为1分、大于Q3为3分、其他2分。合计6/5/4/其他分对应忠诚/进阶/普通/低价值。",
        "与参考答案中的活跃用户消费次数、金额、标签逐一核对。潜在与易流失用户单列，不能参与活跃群体分位数计算。",
    ]
    c.done()


def cohort():
    c = case(
        "同期群留存",
        "以注册日期分群，在每个注册后第N日按用户去重计数；分母为该注册日的用户数。",
    )
    p = source(c, "1-4同期群分析-参考.xlsx")
    d = excel(p, "用户登录明细", usecols="A:C").dropna(subset=["用户编号"])
    d["注册日期"] = pd.to_datetime(d.注册日期).dt.normalize()
    d["登录日期"] = pd.to_datetime(d.登录时点).dt.normalize()
    assert d.groupby("用户编号").注册日期.nunique().max() == 1
    d["第N日"] = (d.登录日期 - d.注册日期).dt.days
    assert d["第N日"].ge(0).all()
    users = d[["用户编号", "注册日期"]].drop_duplicates()
    size = users.groupby("注册日期").size()
    counts = (
        d.pivot_table(
            index="注册日期", columns="第N日", values="用户编号", aggfunc="nunique"
        )
        .reindex(index=size.index, columns=range(d["第N日"].max() + 1))
        .fillna(0)
    )
    for dt in counts.index:
        # 未到观察期的格子置空，不能把尚未发生的时间当成流失。
        counts.loc[dt, counts.columns > (d.登录日期.max() - dt).days] = np.nan
    rate = counts.div(size, axis=0)
    assert rate.max().max() <= 1
    c.table(rate, "每日同期群留存率")
    c.table(counts, "每日留存人数")
    c.table(size, "注册人数")
    ref = block(p, "参考答案", "A20:AF28", ["注册日期"] + list(range(31))).set_index(
        "注册日期"
    )
    compare(c, rate, ref, "同期群参考值核对")
    mask = ref.notna()
    assert np.allclose(rate.to_numpy()[mask], ref.to_numpy(dtype=float)[mask])
    show = rate.copy()
    show.index = show.index.strftime("%m-%d")
    show.columns = ["D" + str(i) for i in show.columns]
    c.heat(show * 100, "注册同期群留存率（%）")
    f = go.Figure(
        go.Heatmap(
            z=show * 100,
            x=show.columns,
            y=show.index,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar=dict(title="%"),
            hovertemplate="注册%{y}<br>%{x}留存%{z:.1f}%<extra></extra>",
        )
    )
    f.update_yaxes(autorange="reversed")
    c.interactive(f, "注册同期群留存", static_names=("注册同期群留存率（%）",))
    daily = d.groupby("登录日期").用户编号.nunique()
    c.table(daily, "每日活跃用户")
    c.line(daily, "每日活跃用户趋势", "去重用户数", interactive=True)
    c.notes += [
        f"样本含 {len(users)} 名用户，观察截止 {d.登录日期.max().date()}。登录明细的重复登录不重复计人；留存率逐格核对参考表。",
        "注册分母只能由已提供登录明细中的用户推导；没有独立注册全集，不能声称包括从未登录的注册用户。空白格表示未观察到该时长，不是0%。",
    ]
    c.done()


def funnel_corr():
    c = case("漏斗分析", "从教材保存的各阶段人数重算逐步转化率与整体转化率。")
    p = source(c, "2-1漏斗图-参考.xlsx")
    d = block(p, "Sheet1", "A4:C9", ["阶段", "占位", "用户数"]).set_index("阶段")
    d["逐步转化率"] = d.用户数 / d.用户数.shift(1)
    d.iloc[0, d.columns.get_loc("逐步转化率")] = 1
    d["整体转化率"] = d.用户数 / d.用户数.iloc[0]
    c.table(d, "漏斗指标")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(
        d.index,
        d.用户数,
        left=(d.用户数.max() - d.用户数) / 2,
        color=COLORS[0],
        height=0.72,
    )
    for i, (n, r) in enumerate(d.iterrows()):
        ax.text(
            d.用户数.max() / 2,
            i,
            f"{r.用户数:.0f}人  |  总体 {r.整体转化率:.0%}",
            ha="center",
            va="center",
            color="white",
        )
    ax.invert_yaxis()
    ax.set(title="销售漏斗：从潜在到成交", xlabel="人数（居中布局）")
    c.figure(fig, "销售漏斗")
    c.interactive(
        go.Figure(go.Funnel(y=d.index, x=d.用户数, textinfo="value+percent initial")),
        "销售转化漏斗",
        static_names=("销售漏斗",),
    )
    c.notes.append(
        f"最终整体转化率 {d.整体转化率.iloc[-1]:.0%}。人数是教材的教学示例，不能据此推断真实业务转化。"
    )
    c.done()
    c = case(
        "相关分析",
        "使用教材提供的 x、y，计算 Pearson 相关及最小二乘直线；相关不代表因果。",
    )
    p = source(c, "3-1相关分析-参考.xlsx")
    d = excel(p, "原始数据")
    profile(c, d, "相关样本")
    reg = linregress(d.x, d.y)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(d.x, d.y, s=40, alpha=0.75, edgecolor="white")
    x = np.sort(d.x.to_numpy())
    ax.plot(
        x,
        reg.intercept + reg.slope * x,
        color=COLORS[1],
        label=f"线性拟合，r = {reg.rvalue:.3f}",
    )
    ax.set(
        title="x 与 y 的线性关系", xlabel="x（原数据单位）", ylabel="y（原数据单位）"
    )
    ax.legend()
    ax.grid(alpha=0.5)
    c.figure(fig, "相关散点与拟合")
    c.table(
        pd.Series(
            {
                "样本量": len(d),
                "Pearson_r": reg.rvalue,
                "斜率": reg.slope,
                "截距": reg.intercept,
                "R平方": reg.rvalue**2,
            }
        ),
        "相关与回归结果",
    )
    c.notes.append(
        f"配对样本 {len(d)} 条，Pearson r={reg.rvalue:.4f}。直线仅为样本内描述性拟合，不产生新样本或因果结论。"
    )
    c.done()


if __name__ == "__main__":
    bcg()
    rfm()
    loyalty()
    cohort()
    funnel_corr()
