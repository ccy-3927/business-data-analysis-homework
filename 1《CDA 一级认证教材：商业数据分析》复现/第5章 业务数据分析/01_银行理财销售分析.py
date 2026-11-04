"""迁移PBIX的销售目标、客群分解、品类、期限和购买动机分析。"""

from pathlib import Path
import sys, json, zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import plotly.express as px

c = Case(
    __file__,
    "银行理财销售分析",
    "教材业务背景：月目标2500万元，当前为31天月份的第29日；数据是当月时点快照。",
)
p = c.source(find("银行案例数据 - 学员用.xlsx"))
d = excel(p, "销售数据")
for sheet, key in [
    ("风险偏好说明表", "风险编号"),
    ("品类说明表", "品类编号"),
    ("学历说明表", "学历编号"),
]:
    dim = excel(p, sheet)
    d = d.merge(dim, on=key, how="left", validate="many_to_one")
assert len(d) == 440 and d[["风险说明", "品类名称", "学历名称"]].notna().all().all()
profile(c, d, "销售记录")
c.table(d, "关联后销售明细")
amount = d["购买金额"].sum()
goal = 25000000
c.table(
    pd.Series(
        {
            "当月累计销售额": amount,
            "月度目标": goal,
            "月度目标达成率": amount / goal,
            "时间进度": 29 / 31,
            "按时间进度目标": goal * 29 / 31,
            "距月目标差额": goal - amount,
            "剩余2日所需日均销售额": max(goal - amount, 0) / 2,
        }
    ),
    "目标进度",
)
c.bar(pd.Series({"月度目标": goal, "累计销售额": amount}), "月度目标与累计销售额", "元")
g = d.groupby("品类名称").agg(
    销售额=("购买金额", "sum"),
    记录数=("购买金额", "size"),
    单笔金额均值=("购买金额", "mean"),
    期限均值=("期限", "mean"),
)
g["销售占比"] = g["销售额"] / amount
c.table(g, "品类指标")
c.bar(g["销售额"].sort_values(ascending=False), "各品类销售额", "元", True)
# 原图把两个不同单位放双轴；迁移后分开呈现。
c.bar(g["单笔金额均值"].sort_values(ascending=False), "各品类单笔金额均值", "元")
c.bar(g["期限均值"].sort_values(ascending=False), "各品类期限均值", "期限（源表单位）")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
groups = list(d.groupby("品类名称"))
for ax, col in zip(axes, ["购买金额", "期限"]):
    ax.boxplot(
        [a[col].to_numpy() for _, a in groups],
        vert=False,
        tick_labels=[f"{k} (n={len(a)})" for k, a in groups],
        patch_artist=True,
        boxprops={"facecolor": "#82B5CC"},
        medianprops={"color": "#19344A"},
    )
    ax.set(title=col + "分布", xlabel=col)
    ax.grid(axis="x", alpha=0.6)
fig.text(
    0.5,
    0.01,
    "箱体为四分位距，中线为中位数，须为1.5倍四分位距内，点为离群观测",
    ha="center",
    fontsize=9,
)
c.figure(fig, "各品类金额与期限分布")
for col in ["家庭年收入", "年龄", "学历名称", "性别", "风险说明"]:
    s = d.groupby(col)["购买金额"].sum().sort_values(ascending=False)
    c.table(s, col + "销售额")
    c.bar(s, col + "销售分布", "元")
for col in ["购买目的", "选择本行的原因"]:
    s = d[col].value_counts()
    c.table(s, col + "频数")
    c.bar(s, col, "记录数", True)
group = (
    d.groupby(["家庭年收入", "年龄", "学历名称", "性别"], dropna=False)["购买金额"]
    .sum()
    .reset_index()
)
c.table(group, "客群层级分解")
c.interactive(
    px.treemap(
        group, path=["家庭年收入", "年龄", "学历名称", "性别"], values="购买金额"
    ),
    "销售额客群逐层下钻",
)
matrix = d.pivot_table(
    index="品类名称", columns="风险说明", values="购买金额", aggfunc="sum", fill_value=0
)
c.table(matrix, "品类与风险偏好")
c.heat(matrix, "品类与风险偏好销售额")
pbix = next(p.parent.glob("*.pbix"))
c.source(pbix)
layout = json.loads(zipfile.ZipFile(pbix).read("Report/Layout").decode("utf-16-le"))
mapping = []
for page in layout["sections"]:
    for v in page["visualContainers"]:
        sv = json.loads(v["config"]).get("singleVisual", {})
        mapping.append(
            {
                "原图类型": sv.get("visualType"),
                "字段": sv.get("projections"),
                "查询": sv.get("prototypeQuery"),
            }
        )
(c.out / "PowerBI字段与聚合口径.json").write_text(
    json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
)
c.notes += [
    f"累计销售额{amount:,.0f}元，目标完成率{amount/goal:.2%}。记录没有客户ID，单笔均值不能冒称独立客户客单价。",
    "分解树迁移为可逐层点击的矩形树图；词云迁移为频数条形图；双轴组合图分成两张。客户购买偏好仅为本批记录的描述。",
    "源数据没有逐日销售，不能构造月度日趋势。剩余日均需求是目标差额的计算，不是未来销售预测。收益字段含负值，原值保留。",
]
c.done()
