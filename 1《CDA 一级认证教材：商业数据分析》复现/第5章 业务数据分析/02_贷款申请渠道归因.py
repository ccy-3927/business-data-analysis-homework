"""重算贷款申请趋势、渠道构成和2021年5月环比贡献。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import openpyxl

c = Case(
    __file__,
    "贷款申请渠道归因",
    "从申请书主键计数，解释总量变化的渠道贡献；不把数值归因写成已证实的业务原因。",
)
d = excel(c.source(find("归因分析数据.xlsx")))
ref = c.source(find("归因分析 - 结果.xlsx"))
assert d.iloc[:, 0].is_unique
d["申请日期"] = pd.to_datetime(d["申请日期"])
d["月份"] = d["申请日期"].dt.to_period("M").astype(str)
m = pd.crosstab(d["月份"], d["申请渠道"]).sort_index()
totals = m.sum(axis=1)
c.table(m, "月度渠道申请量")
c.table(pd.DataFrame({"申请量": totals, "环比": totals.pct_change()}), "月度趋势")
c.line(totals, "贷款申请量月度趋势", "申请数", True)
c.line(m, "各渠道贷款申请量", "申请数", True)
share = m.div(totals, axis=0)
c.table(share, "渠道构成")
fig, ax = plt.subplots(figsize=(11, 5))
(share * 100).plot.bar(stacked=True, ax=ax, rot=0)
ax.set(title="贷款申请渠道构成", ylabel="占比（%）", xlabel="月份")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
c.figure(fig, "贷款申请渠道构成")
a = pd.DataFrame({"4月": m.loc["2021-04"], "5月": m.loc["2021-05"]})
a["差额"] = a["5月"] - a["4月"]
delta = a["差额"].sum()
a["对总变化贡献"] = a["差额"] / delta
a["对整体环比贡献百分点"] = a["差额"] / totals.loc["2021-04"] * 100
assert delta == -39 and np.isclose(a["对总变化贡献"].sum(), 1)
w = openpyxl.load_workbook(ref, read_only=True, data_only=True)
assert totals.tolist() == [w["业务描述"].cell(r, 3).value for r in range(16, 23)]
c.table(a, "5月渠道增减贡献")
c.bar(a["差额"].sort_values(), "5月各渠道申请增减", "相对4月变化（笔）", True)
water = go.Figure(
    go.Waterfall(
        x=["4月"] + a.index.tolist() + ["5月"],
        measure=["absolute"] + ["relative"] * len(a) + ["total"],
        y=[totals.loc["2021-04"]] + a["差额"].tolist() + [0],
        connector={"line": {"color": "#AAB8C2"}},
    )
)
c.interactive(water, "4月至5月贷款申请变化分解")
c.notes += [
    f"5月230笔，较4月269笔减少39笔（{delta/269:.2%}）；柜台减少23笔、手机银行减少19笔、自助机增加3笔。",
    "贡献分母采用总变化−39，所以两下降渠道贡献为正，自助机为负（抵消下降）。教材部分单元格用绝对差额作分母，符号不统一；此处明确采用有符号总变化。",
    "渠道占比变化不能单独证明柜台工作压力降低。宣传不足、APP路径过深是教材提出的待验证假设；没有曝光、点击或实验数据，不能复现转化漏斗或A/B实验结论。",
]
c.done()
