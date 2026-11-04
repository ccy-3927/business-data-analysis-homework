"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.decomposition import SparsePCA

c = start(
    __file__,
    "变量聚类与代表变量选择",
    "沿教材SparsePCA路线，修正原varSelec相关系数列表跨变量累积及best_alpha未定义的问题。",
)
d = data(c, "cities_10.csv")
x = d.loc[:, "X1":]
z = StandardScaler().fit_transform(x)
candidates = []
for a in np.arange(0.1, 10, 0.2):
    m = SparsePCA(n_components=2, alpha=a, random_state=42).fit(z)
    weights = m.components_.T
    counts = (weights != 0).sum(axis=1)
    error = np.abs(counts - 1).sum()
    candidates.append((error, a, m))
    if error == 0:
        break
err, a, m = min(candidates, key=lambda v: (v[0], v[1]))
score = m.transform(z)
cor = np.corrcoef(z.T, score.T)[: x.shape[1], x.shape[1] :]
group = np.argmax(np.abs(cor), axis=1)
ratios = (1 - cor**2) / (1 - cor[:, ::-1] ** 2 + 1e-12)
selected = []
records = []
for g in range(2):
    ids = np.where(group == g)[0]
    pick = ids[np.argmin(ratios[ids, g])]
    selected.append(x.columns[pick])
for i, col in enumerate(x):
    records.append(
        {
            "变量": col,
            "组": group[i],
            "与F1相关": cor[i, 0],
            "与F2相关": cor[i, 1],
            "1-R平方比": ratios[i, group[i]],
            "代表变量": col in selected,
        }
    )
c.table(pd.DataFrame(records), "变量聚类与选择")
c.table(d[["AREA"] + selected], "代表变量数据")
heat(
    c,
    pd.DataFrame(weights, index=x.columns, columns=["F1", "F2"]),
    "SparsePCA权重",
    True,
)
c.notes.append(
    f"alpha={a:.2f}，每变量单一非零归属误差={err}；代表变量{selected}。这是稀疏分量启发式变量选择，不是唯一分组。"
)
c.done()
