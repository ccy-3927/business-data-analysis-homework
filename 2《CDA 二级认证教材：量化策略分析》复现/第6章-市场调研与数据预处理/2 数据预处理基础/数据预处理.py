"""教材Notebook的可运行迁移版；原单元格及变化见同目录对照文档。"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *

c = start(
    __file__,
    "营销数据预处理",
    "复现AvgIncome分布检查和教材指定的0值缺失处理，保留原始列和修改标记。",
)
d = data(c, "teleco_camp_orig.csv")
profile(c, d, "营销原始数据")
clean = d.copy()
clean["AvgIncome原值"] = d.AvgIncome
clean["AvgIncome原值为零"] = d.AvgIncome.eq(0)
clean["AvgIncome"] = d.AvgIncome.replace(0, np.nan)
c.table(clean, "清洗后明细")
c.table(
    pd.DataFrame(
        {"原值": d.AvgIncome.describe(), "零转缺失后": clean.AvgIncome.describe()}
    ),
    "收入处理前后统计",
)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, s, title in zip(
    axes, [d.AvgIncome, clean.AvgIncome], ["原始AvgIncome", "0转缺失后AvgIncome"]
):
    ax.hist(s.dropna(), bins=20, color="#82B5CC", edgecolor="white")
    ax.set(title=title, xlabel="AvgIncome", ylabel="非空记录数")
c.figure(fig, "收入零值处理前后")
c.table(d[["AvgHomeValue", "AvgIncome", "ARPU", "Age"]].describe(), "重点字段描述统计")
c.bar(clean.isna().sum().loc[lambda x: x > 0], "缺失字段记录数", "记录数")
c.notes += [
    f"按教材将AvgIncome中的{int(d.AvgIncome.eq(0).sum())}个0转为缺失，没有均值填充。ARPU和Age已有缺失，仍保留。",
    "教材对AvgHomeValue只提出核查，未给出确认规则，因此不自行删除或填补。随附woe.py的main演示用随机数据，不执行；另以本数据真实Suc_flag演示其二元WOE逻辑，见配套WOE实操。",
]
c.done()
