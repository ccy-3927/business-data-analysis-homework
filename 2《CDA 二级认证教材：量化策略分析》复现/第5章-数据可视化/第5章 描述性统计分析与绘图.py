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
    "描述性统计分析与绘图",
    "复现房价集中趋势、离散程度、分位数及三种相关系数。",
)
d = data(c, "sndHsPr.csv")
profile(c, d, "房产数据")
s = d.price
c.table(
    pd.Series(
        {
            "均值": s.mean(),
            "中位数": s.median(),
            "标准差_ddof1": s.std(),
            "方差_ddof1": s.var(),
            "极差": s.max() - s.min(),
            "偏度": s.skew(),
            "超额峰度": s.kurt(),
            "Q25": s.quantile(0.25),
            "Q75": s.quantile(0.75),
        }
    ),
    "房价描述统计",
)
hist(c, s, "房价分布", "price（源数据单位）")
result = pd.Series(
    {
        m: d[["AREA", "price"]].corr(method=m).iloc[0, 1]
        for m in ["pearson", "spearman", "kendall"]
    },
    name="面积与价格相关系数",
)
c.table(result, "三种相关系数")
c.bar(result, "面积与价格相关系数", "相关系数")
c.notes += [
    "教材cell12随机正态数据不重新生成；偏度、峰度改在真实提供的房价列计算。相关关系不代表面积导致价格变化，未补造地理位置或房产价格单位。"
]
c.done()
