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
    "破产标识与坏账列联分析",
    "按教材bankruptcy_ind和bad_ind构造频数、行比例与卡方独立性检验。",
)
d = data(c, "accepts.csv")
clean = d[["bankruptcy_ind", "bad_ind"]].dropna()
tab = pd.crosstab(clean.bankruptcy_ind, clean.bad_ind)
c.table(tab, "列联频数")
c.table(tab.div(tab.sum(axis=1), axis=0), "行比例")
heat(c, tab, "破产标识与坏账频数")
results = []
for correction in [True, False]:
    chi, p, df, expected = stats.chi2_contingency(tab, correction=correction)
    results.append(
        {
            "连续性校正": correction,
            "卡方": chi,
            "p值": p,
            "自由度": df,
            "最小期望频数": expected.min(),
        }
    )
c.table(pd.DataFrame(results), "卡方检验")
c.table(pd.DataFrame(expected, index=tab.index, columns=tab.columns), "期望频数")
c.table(
    pd.Series(
        {
            "原始行数": len(d),
            "完整标识行数": len(clean),
            "缺失排除行数": len(d) - len(clean),
            "CramerV": np.sqrt(
                chi / (tab.to_numpy().sum() * min(tab.shape[0] - 1, tab.shape[1] - 1))
            ),
        }
    ),
    "检验样本与效应量",
)
c.bar((tab[1] / tab.sum(axis=1)) * 100, "不同破产标识的坏账比例", "坏账比例（%）")
assert tab.to_numpy().sum() + len(d) - len(clean) == len(d)
c.notes += [
    "卡方使用纯频数矩阵，不把All合计行列纳入检验；缺失破产标识单独统计而非归为N。提供与原代码默认一致的连续性校正值及未校正Pearson值。",
    "这是已获批样本的关联分析，不表示全部申请人群，不证明破产标识造成坏账。",
]
c.done()
