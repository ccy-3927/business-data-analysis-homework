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
    "Python数据可视化",
    "原股票折线和固定降水示例复现；随机图示改用教材现有房产数据演示绘图方法，并明确区分。",
)
d = data(c, "2017股价.xlsx")
d["日期"] = pd.to_datetime(d["日期"])
d = d.sort_values("日期")
profile(c, d, "股价")
c.line(
    d.set_index("日期")[["收盘价", "开盘价", "最高价", "最低价"]],
    "2017股价走势",
    "股价",
    True,
)
rain = literal(c, 15, "df")
c.table(rain, "Notebook固定降水产量示例")
scatter(c, rain["降水量"], rain["产量"], "降水与产量_教材固定示例")
h = data(c, "sndHsPr.csv")
profile(c, h, "房产数据")
c.bar(h["dist"].value_counts(), "房源区域构成", "记录数")
hist(c, h.price, "房价直方图与核密度", "price（源数据单位）", True)
box(c, h, "floor", "price", "各楼层房价分布")
fig, ax = plt.subplots(figsize=(10, 6))
hb = ax.hexbin(h.AREA, h.price, gridsize=40, mincnt=1, cmap="Blues")
fig.colorbar(hb, ax=ax, label="观测数")
ax.set(title="房屋面积与价格二维分布", xlabel="AREA", ylabel="price")
c.figure(fig, "房屋面积与价格二维分布")
heat(c, h[["AREA", "roomnum", "halls", "price"]].corr(), "房屋连续变量相关图", True)
c.notes += [
    "cell6、12、18、21、24、30依赖随机生成数据，未重新生成投票或成绩。构成图、直方图、箱线图、二维分布和热图改用同章sndHsPr.csv；原随机示例及其绘图代码保留在单元格对照文档。",
    "构成使用条形图以便比较；16,210条房产数据用二维计数图减少散点遮盖。核密度曲线是对观测分布的平滑估计，不是新增样本。",
]
c.done()
