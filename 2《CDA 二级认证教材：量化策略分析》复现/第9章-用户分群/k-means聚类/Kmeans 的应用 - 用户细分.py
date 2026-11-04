"""教材Notebook的可运行迁移版；原单元格及变化见同目录对照文档。"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *

from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from threadpoolctl import threadpool_limits

c = start(
    __file__,
    "银行用户细分",
    "复现标准化、三分量旋转、异常分布观察、分位数正态变换及K-means三群，并用原始行为解释。",
)
d = data(c, "profile_bank.csv")
x = d.loc[:, "CNT_TBM":"CNT_CSC"]
profile(c, x, "渠道次数")
score, w, pca, z = rotated_components(x, 3)
pca_tables(c, x, pca)
c.table(pd.DataFrame(w, index=x.columns, columns=["F1", "F2", "F3"]), "旋转分量权重")
q = QuantileTransformer(
    n_quantiles=1000, output_distribution="normal", random_state=0, subsample=None
)
trans = q.fit_transform(score)
c.table(
    pd.DataFrame(
        {
            "旋转后偏度": pd.DataFrame(score).skew(),
            "分位变换后偏度": pd.DataFrame(trans).skew(),
        }
    ),
    "变换前后偏度",
)
with threadpool_limits(limits=1):
    raw = KMeans(n_clusters=3, random_state=2025, n_init=10).fit(score)
    final = KMeans(n_clusters=3, random_state=2025, n_init=10).fit(trans)
    sil = silhouette_score(trans, final.labels_, sample_size=3000, random_state=2025)
out = d.assign(变换前群=raw.labels_, 用户群=final.labels_)
assert len(out) == len(d)
c.table(out, "客户分群明细")
c.table(
    pd.DataFrame(
        {
            "变换前": pd.Series(raw.labels_).value_counts(),
            "变换后": pd.Series(final.labels_).value_counts(),
        }
    ).fillna(0),
    "群体规模对照",
)
c.bar(out["用户群"].value_counts().sort_index(), "用户群人数", "人数")
centers = out.groupby("用户群")[x.columns].mean()
medians = out.groupby("用户群")[x.columns].median()
c.table(centers, "群体原始行为均值")
c.table(medians, "群体原始行为中位数")
heat(c, (centers - x.mean()) / x.std(), "群体行为标准化均值")
for col in x.columns:
    box(c, out, "用户群", col, col + "分群分布")
c.table(
    pd.Series(
        {
            "样本数": len(d),
            "K": 3,
            "抽取3000真实行的轮廓系数": sil,
            "变换前后ARI": adjusted_rand_score(raw.labels_, final.labels_),
            "最终inertia": final.inertia_,
        }
    ),
    "聚类诊断",
)
# 解释树只输入四个原始行为，避免ID和总次数进入模型。
tree = DecisionTreeClassifier(
    max_depth=3, min_samples_split=100, min_samples_leaf=100, random_state=12345
).fit(x, final.labels_)
(c.out / "用户群解释规则.txt").write_text(
    export_text(tree, feature_names=list(x.columns)), encoding="utf-8"
)
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    tree,
    feature_names=x.columns,
    class_names=["群0", "群1", "群2"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    proportion=True,
)
ax.set_title("用户群行为解释树")
c.figure(fig, "用户群解释树")
c.table(
    pd.Series(tree.feature_importances_, index=x.columns, name="重要性"),
    "解释树变量重要性",
)
c.notes += [
    "固定随机种子使K-means初始化可重现，未生成客户记录。分位数变换是对已有特征的派生处理，并非生成正态样本。群编号没有大小顺序；重跑或不同实现的编号可置换，解释应看行为指标。",
    "原Notebook解释树把ID和总次数也输入，本实现只用四个原始行为以免ID产生伪规则。树仅解释已得到的聚类，样本内规则不能声称预测未来业务效果。",
    "fa_kit替换为PCA单位权重的正交varimax旋转，得分列采用F1-F3，避免因旋转列顺序不同而误用ATM_POS/TBM/CSC固定名字。输出权重、均值、中位数与分布支持命名。",
]
c.done()
