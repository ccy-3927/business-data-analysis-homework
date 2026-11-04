"""将教材两套SQL迁移为pandas：严格以发卡时点截断交易窗口。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

c = Case(
    __file__,
    "信用卡客户画像",
    "按card_id一行构建画像；保留教材180天和365天两套窗口，不将其混称同一口径。",
)
base = SOURCE / "第9章 用户标签体系与用户画像专题"
c.source(base / "信用卡客户画像.sql")
c.source(base / "信用卡客户画像.pptx")


def read(name):
    return csv(c.source(base / "data" / f"{name}.csv"), low_memory=False)


card, disp, clients, trans = [read(n) for n in ["card", "disp", "clients", "trans"]]
card["issued"] = pd.to_datetime(card["issued"])
clients["birth_date"] = pd.to_datetime(clients["birth_date"])
trans["date"] = pd.to_datetime(trans["date"])
for col in ["amount", "balance"]:
    trans[col] = pd.to_numeric(
        trans[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="raise"
    )
assert card["card_id"].is_unique and trans["trans_id"].is_unique
master = card.merge(
    disp[["disp_id", "client_id", "account_id"]], on="disp_id", validate="many_to_one"
).merge(clients, on="client_id", validate="many_to_one")
assert len(master) == len(card)
master["发卡年龄_教材日数除365"] = (
    master["issued"] - master["birth_date"]
).dt.days / 365
annual = card.groupby(card["issued"].dt.year).size().rename("发卡数")
c.table(annual, "年度发卡量")
c.line(annual, "年度发卡趋势", "卡数", True)
types = card["type"].value_counts()
c.table(pd.DataFrame({"卡数": types, "占比": types / len(card)}), "持卡类型构成")
c.bar(types, "持卡类型构成", "卡数", True)
joined = master[["card_id", "account_id", "issued"]].merge(
    trans, on="account_id", how="left"
)
joined["距发卡天数"] = (joined["issued"] - joined["date"]).dt.days
before = joined[joined["距发卡天数"].gt(0)].copy()
last = before[before["date"].eq(before.groupby("card_id")["date"].transform("max"))]
c.table(last, "发卡前最后交易日全部记录")
latest = last.groupby("card_id").agg(
    最后交易日=("date", "max"),
    距最后交易天数=("距发卡天数", "first"),
    最后交易日余额均值=("balance", "mean"),
    最后交易日笔数=("trans_id", "size"),
)
c.table(latest, "最后交易日画像")
for window in [180, 365]:
    a = before[before["距发卡天数"].lt(window)].copy()
    # 完全保留SQL中“借/贷”的筛选，不因标签名称推断或改写账务方向。
    a["借类金额"] = a["amount"].where(a["type"].eq("借"), 0)
    a["贷类金额"] = a["amount"].where(a["type"].eq("贷"), 0)
    g = a.groupby("card_id").agg(
        交易笔数=("trans_id", "size"),
        借类金额=("借类金额", "sum"),
        贷类金额=("贷类金额", "sum"),
        逐笔余额均值=("balance", "mean"),
    )
    result = master.merge(
        latest, on="card_id", how="left", validate="one_to_one"
    ).merge(g, on="card_id", how="left", validate="one_to_one")
    result["交易笔数"] = result["交易笔数"].fillna(0).astype(int)
    result["贷类金额除逐笔余额均值"] = result["贷类金额"] / result[
        "逐笔余额均值"
    ].replace(0, np.nan)
    result["贷类金额除借类金额"] = result["贷类金额"] / result["借类金额"].replace(
        0, np.nan
    )
    assert result["card_id"].is_unique and len(result) == len(card)
    assert (
        a["date"].lt(a["issued"]).all() and a["距发卡天数"].between(1, window - 1).all()
    )
    c.table(result, f"发卡前{window}天客户画像")
    if window == 365:
        profile(c, result, "画像数据")
        s = pd.crosstab(result["type"], result["sex"])
        c.table(s, "卡类型与性别")
        c.heat(s, "卡类型与性别分布")
        c.table(s.div(s.sum(axis=1), axis=0), "卡类型内性别占比")
        comparison = result.groupby("type").agg(
            卡数=("card_id", "size"),
            平均发卡年龄=("发卡年龄_教材日数除365", "mean"),
            平均借类金额=("借类金额", "mean"),
            平均逐笔余额=("逐笔余额均值", "mean"),
        )
        c.table(comparison, "持卡类型定量画像比较")
        for col, title in [
            ("发卡年龄_教材日数除365", "持卡类型的年龄比较"),
            ("借类金额", "办卡前一年借类总金额分布"),
            ("逐笔余额均值", "办卡前一年逐笔平均余额分布"),
        ]:
            groups = [(k, g[col].dropna()) for k, g in result.groupby("type")]
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.boxplot(
                [v.to_numpy() for k, v in groups],
                tick_labels=[f"{k}\nn={len(v)}" for k, v in groups],
                patch_artist=True,
                boxprops={"facecolor": "#82B5CC"},
                medianprops={"color": "#19344A", "linewidth": 2},
            )
            ax.set(title=title, ylabel=col)
            ax.grid(axis="y", alpha=0.7)
            fig.text(
                0.5,
                0.01,
                "箱体：四分位距；中线：中位数；须：1.5倍四分位距内；点：离群观测",
                ha="center",
                fontsize=9,
            )
            fig.subplots_adjust(bottom=0.18)
            c.figure(fig, title)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].hist(
            result["发卡年龄_教材日数除365"],
            bins=15,
            color=COLORS[0],
            edgecolor="white",
        )
        axes[0].set(title="发卡时年龄分布", xlabel="年龄（日数/365）", ylabel="卡数")
        axes[1].hist(result["交易笔数"], bins=20, color=COLORS[2], edgecolor="white")
        axes[1].set(title="发卡前365天交易笔数", xlabel="交易笔数", ylabel="卡数")
        c.figure(fig, "年龄与交易频次分布")
        f = go.Figure(
            go.Scatter(
                x=result["发卡年龄_教材日数除365"],
                y=result["交易笔数"],
                mode="markers",
                text=result["type"],
                marker={"size": 6, "opacity": 0.5},
                hovertemplate="年龄%{x:.1f}<br>笔数%{y}<br>%{text}<extra></extra>",
            )
        )
        f.update_xaxes(title="发卡年龄")
        f.update_yaxes(title="交易笔数")
        c.interactive(f, "发卡年龄与历史交易频次")
c.notes += [
    f"共{len(card)}张卡；最后交易日保留{len(last)}条交易记录，聚合后为{len(latest)}张卡。存在同日多笔，按第二套SQL使用当日逐笔余额平均值，不能称为确定的日终余额。",
    "SQL第一套窗口为0<间隔<180，第二套为0<间隔<365；注释中的“开卡半年内”与实际发卡前条件不同，本实现遵循时间条件。所有发卡当日及之后交易排除，防止未来信息泄漏。",
    "教材把借类金额命名为入账、贷类为出账，但数据的“信贷资金”等操作名称需要业务字典核实。本实现使用借类/贷类中性列名，保持SQL计算方向。逐笔余额均值不是时间加权平均资产。",
    "未提供信用违约标签与标签评分规则，因此不训练风险模型，不自创客户评分。参考标签表和画像报告仅作方法材料，不以其图示数字生成客户记录。",
]
c.done()
