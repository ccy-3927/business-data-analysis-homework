"""只复现模板中确实存在的统计表和标准定义，不生成银行经营数据。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import openpyxl

c = Case(
    __file__,
    "数据标准模板迁移",
    "使用教材已保存的标准分布汇总及标准定义。2011是模板汇总中的标准条数，并非2011条业务交易。",
)
p = c.source(find("1、基础数据标准编制模板.xlsx"))
w = openpyxl.load_workbook(p, read_only=True, data_only=True)
rows = list(w["标准分布统计"].values)
matrix = pd.DataFrame(
    [r[1:15] for r in rows[4:14]],
    index=[r[0] for r in rows[4:14]],
    columns=rows[3][1:15],
).astype(float)
c.table(matrix, "模板原汇总_保留空白")
# 透视计数空格按0汇总；与教材行列总计逐一核对。
values = matrix.fillna(0)
np.testing.assert_allclose(values.sum(axis=1), [r[15] for r in rows[4:14]])
np.testing.assert_allclose(values.sum(axis=0), rows[14][1:15])
assert values.to_numpy().sum() == rows[14][15] == 2011
c.table(values, "标准分布统计")
c.bar(
    values.sum(axis=1).sort_values(ascending=False), "各主题标准数量", "标准条数", True
)
c.bar(values.sum(axis=0).sort_values(ascending=False), "各部门标准数量", "标准条数")
fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(values.to_numpy(), aspect="auto", cmap="Blues")
fig.colorbar(im, ax=ax, label="标准条数")
ax.set_xticks(range(len(values.columns)), values.columns, rotation=90)
ax.set_yticks(range(len(values)), values.index)
ax.set_title("主题与部门标准分布")
c.figure(fig, "主题与部门标准分布")
c.table(excel(p, "客户主题为例", header=1), "客户数据标准示例")
p2 = c.source(find("2、指标数据标准编制模板.xlsx"))
for name, header in [("标准定义", 1), ("维度定义", 0), ("维度值代码定义", 0)]:
    d = excel(p2, name, header=header)
    c.table(d, name)
c.notes += [
    "源表只提供汇总计数和一个客户标准示例，不能据此反推2011条标准明细。图表是对已有汇总的Python重绘，行列总计均已核对。",
    "指标标准中有占位SQL（如dw_XXX.pXXX、T1.XXX），未提供可执行的仓库表；只迁移定义，不伪造贷款余额或不良率结果。维度定义和代码保留原文本。",
]
c.done()
