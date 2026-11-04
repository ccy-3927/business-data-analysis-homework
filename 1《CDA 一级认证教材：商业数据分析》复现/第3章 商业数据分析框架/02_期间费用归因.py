"""两份教材版本分别分析，不把不同金额混在同一条归因链中。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

c = Case(
    __file__,
    "期间费用归因",
    "原始版与完整版分别计算费用差额、环比和贡献度，逐层对账。金额沿用源表单位。",
)
reconcile = []
for version, pattern, sheets in [
    (
        "原始版",
        "案例2：财务费用分析-原始数据.xlsx",
        ["总费用数据", "财务三大期间费用数据", "管理费用数据"],
    ),
    (
        "完整版",
        "案例2：财务费用分析-完整案例.xlsx",
        ["总费用数据-原始数据", "财务三大期间费用-原始数据", "管理费用-原始数据"],
    ),
]:
    p = c.source(find(pattern))
    frames = []
    for sheet, width in zip(sheets, [3, 4, 4]):
        a = pd.read_excel(p, sheet_name=sheet, usecols=list(range(width)))
        a = a[pd.to_numeric(a.iloc[:, 0], errors="coerce").eq(2020)].copy()
        a[a.columns[1]] = pd.to_numeric(
            a.iloc[:, 1].astype(str).str.replace("月", "", regex=False)
        ).astype(int)
        frames.append(a)
    total, period, mgmt = frames
    t = total.set_index(total.columns[1])["金额"]
    c.table(t, version + "_总费用")
    c.bar(t.rename(index={6: "6月", 7: "7月"}), version + "_总费用对比", "金额")
    pivots = []
    for a, label in [(period, "期间费用"), (mgmt, "管理费用")]:
        a.iloc[:, 2] = a.iloc[:, 2].str.strip()
        q = a.pivot_table(
            index=a.columns[2], columns=a.columns[1], values="金额", aggfunc="sum"
        )
        q["差额"] = q[7] - q[6]
        q["环比"] = q["差额"] / q[6].replace(0, np.nan)
        q["层内增量贡献"] = (
            q["差额"] / q["差额"].sum() if q["差额"].sum() != 0 else np.nan
        )
        c.table(q, version + "_" + label + "归因")
        c.bar(
            q["差额"].sort_values(ascending=False),
            version + "_" + label + "增量",
            "7月减6月",
            True,
        )
        pivots.append(q)
    for month in [6, 7]:
        reconcile.append(
            {
                "版本": version,
                "月份": month,
                "总费用": t.loc[month],
                "期间费用合计": pivots[0][month].sum(),
                "期间费用与总额差": pivots[0][month].sum() - t.loc[month],
                "管理明细合计": pivots[1][month].sum(),
                "管理分类金额": pivots[0].loc["管理费用", month],
            }
        )
    c.notes.append(
        f'{version}：总额增长{t.loc[7]-t.loc[6]:,.0f}；期间费用明细增长{pivots[0]["差额"].sum():,.0f}；管理明细增长{pivots[1]["差额"].sum():,.0f}。'
    )
c.table(pd.DataFrame(reconcile), "两版本逐层对账")
c.notes += [
    "原始版期间费用合计与总额一致；完整版期间费用合计与总额不一致。完整版保留教材的管理费用下钻，但不得用其明细解释原始版或总额的全部增长。差异不靠填造调整项消除。",
    "贡献度是差额的算术分解，不能作为因果证据。",
]
c.done()
