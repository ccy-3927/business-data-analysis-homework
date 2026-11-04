"""教材Notebook的可运行迁移版；原单元格及变化见同目录对照文档。"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform

c = start(
    __file__,
    "地区层次聚类",
    "复现主成分、旋转因子、欧氏距离和Ward树状图，核对教材已保存的因子得分。",
)
d = data(c, "cities_10.csv")
x = d.loc[:, "X1":]
score, w, pca, z = rotated_components(x, 2)
pca_tables(c, x, pca)
ref = data(c, "fa_cities_10.csv")
assert ref.AREA.tolist() == d.AREA.tolist()
np.testing.assert_allclose(ref[x.columns], x)
r, _ = orthogonal_procrustes(score, ref[["Gross", "Avg"]].to_numpy())
aligned = score @ r
c.table(
    pd.DataFrame(
        {
            "地区": d.AREA,
            "重算对齐1": aligned[:, 0],
            "教材Gross": ref.Gross,
            "重算对齐2": aligned[:, 1],
            "教材Avg": ref.Avg,
        }
    ),
    "因子得分核对",
)
delta = float(np.abs(aligned - ref[["Gross", "Avg"]].to_numpy()).max())
c.table(pd.DataFrame(w, index=x.columns, columns=["F1", "F2"]), "重算旋转权重")
# 树状图使用源文件已保存的得分，避免悄悄混用不同算法的中间结果。
dist = pdist(ref[["Gross", "Avg"]], metric="euclidean")
Z = linkage(dist, method="ward")
c.table(pd.DataFrame(squareform(dist), index=d.AREA, columns=d.AREA), "地区欧氏距离")
c.table(
    pd.DataFrame(Z, columns=["左节点", "右节点", "合并距离", "成员数"]), "Ward合并过程"
)
fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(
    Z,
    labels=d.AREA.tolist(),
    ax=ax,
    leaf_font_size=11,
    color_threshold=0,
    above_threshold_color=COLORS[0],
)
ax.set(title="地区经济Ward层次聚类", ylabel="Ward合并距离")
c.figure(fig, "地区Ward树状图")
out = d[["AREA"]].copy()
for k in [2, 3, 4]:
    out[f"切为{k}群"] = fcluster(Z, k, criterion="maxclust")
c.table(out, "不同切分数对照")
f = go.Figure(
    go.Scatter(
        x=ref.Gross,
        y=ref.Avg,
        mode="markers+text",
        text=ref.AREA,
        textposition="top center",
        marker={"size": 11},
    )
)
f.update_xaxes(title="教材Gross得分")
f.update_yaxes(title="教材Avg得分")
c.interactive(f, "地区因子与聚类观察")
c.notes += [
    f"重算得分与教材检查点经正交对齐后的最大差异为{delta:.8g}。正交对齐用于消除旋转/符号不唯一，距离比较不受这种变换影响；没有把差异强行修成0。",
    "最终树状图按教材fa_cities_10.csv重绘；另导出从原始指标重算的权重及得分对照，完整保留计算链。2/3/4群仅是对同一树的切分示例，不宣称哪一个是真实经济分类。10个地区数据不能推广到全国。",
]
c.done()
