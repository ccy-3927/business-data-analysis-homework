"""地区PCA、正交旋转与层次聚类共用计算。"""

from model_common import *
from sklearn.decomposition import SparsePCA
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree


def cities(c, full=True):
    d = data(c, "cities_10.csv")
    x = d.loc[:, "X1":]
    scores, w, pca, z = rotated_components(x, 2)
    pca_tables(c, x, pca)
    heat(c, pd.DataFrame(w, index=x.columns, columns=["F1", "F2"]), "旋转权重", True)
    out = d.assign(F1=scores[:, 0], F2=scores[:, 1])
    c.table(out, "地区因子得分")
    if full:
        for name, arr in [
            ("PCA", pca.transform(z)[:, :2]),
            ("旋转因子", scores),
            (
                "SparsePCA",
                SparsePCA(n_components=2, alpha=0.65, random_state=42).fit_transform(z),
            ),
        ]:
            f = go.Figure(
                go.Scatter(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    mode="markers+text",
                    text=d.AREA,
                    textposition="top center",
                    marker={"size": 12},
                )
            )
            f.update_xaxes(title=name + "1")
            f.update_yaxes(title=name + "2")
            c.interactive(f, name + "地区分布")
        rows = []
        for a in np.arange(0.1, 1, 0.05):
            m = SparsePCA(n_components=2, alpha=a, random_state=42).fit(z)
            rows.append({"alpha": a, "非零系数数": np.count_nonzero(m.components_)})
        c.table(pd.DataFrame(rows), "稀疏参数比较")
    Z = linkage(scores, method="ward")
    labels = cut_tree(Z, n_clusters=2).ravel()
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        Z,
        labels=d.AREA.tolist(),
        ax=ax,
        color_threshold=0,
        above_threshold_color=COLORS[0],
    )
    ax.set(title="地区经济Ward聚类", ylabel="Ward合并距离")
    c.figure(fig, "地区Ward聚类")
    c.table(out.assign(群=labels), "地区聚类结果")
    c.table(x.groupby(labels).mean(), "两群经济指标均值")
    c.notes += [
        "fa_kit以PCA单位权重的正交varimax迁移，因子方向/符号不唯一，按权重解释。10个地区的结果只适用于这份教材数据。"
    ]
