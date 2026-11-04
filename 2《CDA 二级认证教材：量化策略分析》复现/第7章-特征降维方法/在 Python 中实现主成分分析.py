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
    "电信业务主成分分析",
    "复现标准化、两主成分提取、白化及得分，说明不同通信行为的共同变化。",
)
d = data(c, "profile_telecom.csv")
x = d.loc[:, "cnt_call":]
profile(c, d, "电信数据")
z = StandardScaler().fit_transform(x)
full = PCA(svd_solver="full").fit(z)
pca = PCA(n_components=2, whiten=True, svd_solver="full").fit(z)
scores = pca.transform(z)
pca_tables(c, x, full)
c.table(
    pd.DataFrame(pca.components_.T, index=x.columns, columns=["PC1权重", "PC2权重"]),
    "主成分权重",
)
score = pd.DataFrame(scores, columns=["PC1白化", "PC2白化"], index=d.index)
c.table(pd.concat([d, score], axis=1), "白化主成分得分")
np.testing.assert_allclose(score.cov(), np.eye(2), atol=1e-10)
f = go.Figure(
    go.Scatter(
        x=score.iloc[:, 0],
        y=score.iloc[:, 1],
        mode="markers",
        text=d.ID,
        marker={"size": 6, "opacity": 0.6},
        hovertemplate="ID%{text}<br>PC1%{x:.2f}<br>PC2%{y:.2f}<extra></extra>",
    )
)
f.update_xaxes(title=score.columns[0])
f.update_yaxes(title=score.columns[1])
c.interactive(f, "电信客户主成分空间")
c.notes += [
    f"前两主成分累计解释{pca.explained_variance_ratio_.sum():.2%}变异，白化后样本协方差核对为单位矩阵。白化只改变得分尺度，不改变解释比例。",
    "此Notebook只做降维，不自行补聚类标签。ID不参与标准化或主成分提取。主成分符号可整体翻转，不改变含义。",
]
c.done()
