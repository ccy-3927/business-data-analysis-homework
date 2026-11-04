"""性别→年龄→品类→支付偏好；输出可复核的目标群体与营销假设。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import openpyxl

c = Case(
    __file__,
    "电商精准营销",
    "沿用教材2023年1月客户数据和筛选路径。无金额、曝光和转化字段，不计算虚构ROI。",
)
d = excel(c.source(find("案例3：电商平台的精准营销-原始数据.xlsx"))).iloc[:, :8]
ref = c.source(find("案例3：电商平台的精准营销-完整案例.xlsx"))
assert d["客户ID"].is_unique, "客户重复，需要改为去重客户口径"
profile(c, d, "客户数据")
labels = ["18-25岁", "26-35岁", "36-49岁", "50岁以上"]
d["年龄段"] = pd.cut(d["年龄"], [17, 25, 35, 49, np.inf], labels=labels)
female = d[d["性别"].eq("Female")]
senior = female[female["年龄"].ge(50)]
target = senior[senior["品类"].isin(["Clothing", "Cosmetics"])]
for a, col, name in [
    (d, "性别", "购买客户性别分布"),
    (female, "年龄段", "女性客户年龄分布"),
    (senior, "品类", "50岁及以上女性品类偏好"),
    (target, "支付方式", "目标客群支付方式"),
]:
    s = a.groupby(col, observed=False)["客户ID"].nunique()
    c.table(pd.DataFrame({"客户数": s, "占比": s / s.sum()}), name)
    c.bar(s.sort_values(ascending=False), name, "客户数", True)
w = openpyxl.load_workbook(ref, read_only=True, data_only=True)
sh = w["准备区"]
assert (
    len(d) == sh["B9"].value
    and len(senior) == sh["C93"].value
    and len(target) == sh["C113"].value
)
age = (
    female.groupby("年龄段", observed=False)
    .size()
    .rename("Python女性客户数")
    .to_frame()
)
age["教材准备区H27至H30"] = [sh.cell(r, 8).value for r in range(27, 31)]
age["差额"] = age.iloc[:, 0] - age.iloc[:, 1]
c.table(age, "年龄汇总差异")
c.table(target, "目标客群明细")
cross = pd.crosstab(d["年龄段"], d["品类"])
c.table(cross, "年龄与品类客户数")
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(cross.to_numpy(), aspect="auto", cmap="Blues")
fig.colorbar(im, ax=ax, label="客户数")
ax.set_xticks(
    range(len(cross.columns)),
    [x.replace(" & ", " &\n") for x in cross.columns],
    rotation=0,
)
ax.set_yticks(range(len(cross)), cross.index)
ax.set_title("年龄与品类客户分布")
c.figure(fig, "年龄与品类客户分布")
c.notes += [
    f"共{len(d)}位独立客户，女性{len(female)}位；50岁及以上女性{len(senior)}位，其中服饰和化妆品客群{len(target)}位。",
    "教材“女性年龄段分布”H27:H30合计1555，与男性总数一致，不能当作女性分布。本实现从原始客户重新分组。教材标签“50岁以上”实际上包含50岁，代码明确使用年龄≥50。",
    "可测试面向该客群的服饰与化妆品组合活动，支付入口参考支付偏好。但人数多不等于购买倾向显著更高，更不能证明活动提升销量。需后续真实实验数据验证；本作业不模拟实验结果。",
]
c.done()
