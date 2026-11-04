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
    "地区经济主成分与因子旋转",
    "复现9项经济指标的标准化、两维主成分与正交最大方差旋转。",
)
d = data(c, "cities_10.csv")
x = d.loc[:, "X1":]
profile(c, d, "地区经济")
score, w, pca, z = rotated_components(x, 2)
pca_tables(c, x, pca)
weights = pd.DataFrame(w, index=x.columns, columns=["旋转分量1", "旋转分量2"])
c.table(weights, "旋转权重")
heat(c, weights, "旋转后变量权重", True)
out = pd.DataFrame(score, index=d.AREA, columns=["旋转分量1", "旋转分量2"])
c.table(out, "旋转得分")
f = go.Figure(
    go.Scatter(
        x=out.iloc[:, 0],
        y=out.iloc[:, 1],
        mode="markers+text",
        text=out.index,
        textposition="top center",
        marker={"size": 10},
    )
)
f.update_xaxes(title=out.columns[0])
f.update_yaxes(title=out.columns[1])
c.interactive(f, "地区经济因子空间")
c.table(
    pd.DataFrame(
        {
            "指标": [
                "GDP",
                "人均GDP",
                "工业增加值",
                "第三产业增加值",
                "固定资产投资",
                "基本建设投资",
                "社会消费品零售总额",
                "海关出口总额",
                "地方财政收入",
            ]
        },
        index=x.columns,
    ),
    "指标解释",
)
c.notes += [
    f"样本10个地区，9项指标。前两分量累计解释{pca.explained_variance_ratio_[:2].sum():.2%}。样本很少，解释仅针对本教材数据。",
    "原fa_kit提取路线是主成分权重加正交varimax旋转，此处以NumPy SVD实现；不是最大似然共同因子分析。权重矩阵和得分完整导出，不强行把任意列命名为总量或均量；因子符号与顺序不唯一。",
]
c.done()
