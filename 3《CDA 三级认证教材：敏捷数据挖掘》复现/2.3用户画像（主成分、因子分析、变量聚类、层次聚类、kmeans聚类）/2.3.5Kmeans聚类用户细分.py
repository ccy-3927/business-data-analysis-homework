"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier

c = start(
    __file__,
    "家庭与爱好双维度用户细分",
    "按教材对16万余真实客户清洗、分位变换、因子旋转，分别聚类家庭和爱好画像。",
)
d = data(c, "data_clust.csv", skipinitialspace=True)
raw = d.copy()
flags = d.isna().add_suffix("_原缺失")
for col in ["interested_travel", "computer_owner", "HH_adults_num"]:
    d[col] = d[col].fillna(d[col].mode()[0])
d["HH_has_children"] = d.HH_has_children.map({"N": 0, "Y": 1}).fillna(0)
d["interested_reading"] = (
    pd.to_numeric(d.interested_reading.replace(".", "0")).gt(1.5).astype(int)
)
interest = [
    "interested_travel",
    "computer_owner",
    "interested_golf",
    "interested_gambling",
    "interested_reading",
]
d["interest"] = d[interest].mean(axis=1)
continuous = [
    "age",
    "home_value",
    "risk_score",
    "interested_sport",
    "HH_dieting",
    "auto_member",
    "HH_grandparent",
    "HH_head_age",
    "loan_ratio",
]
ordered = ["marital", "interest", "HH_adults_num"]
z = d.copy()
z[continuous] = QuantileTransformer(
    n_quantiles=100, output_distribution="normal", subsample=None, random_state=42
).fit_transform(d[continuous])
z[ordered] = StandardScaler().fit_transform(d[ordered])
household = [
    "age",
    "marital",
    "HH_adults_num",
    "home_value",
    "risk_score",
    "HH_grandparent",
    "HH_head_age",
    "loan_ratio",
]
hobby = ["HH_dieting", "auto_member", "interest", "interested_sport"]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for j, col in enumerate(["age", "home_value", "loan_ratio"]):
    axes[0, j].hist(d[col], bins=30, color=COLORS[0])
    axes[0, j].set(title=col + "原值", ylabel="人数")
    axes[1, j].hist(z[col], bins=30, color=COLORS[2])
    axes[1, j].set(title=col + "分位变换", ylabel="人数")
c.figure(fig, "连续特征变换前后")
for name, cols, n, k in [("家庭", household, 4, 3), ("爱好", hobby, 3, 2)]:
    scores, w, pca, _ = rotated_components(z[cols], n)
    scaled = StandardScaler().fit_transform(scores)
    heat(
        c,
        pd.DataFrame(w, index=cols, columns=["F" + str(i + 1) for i in range(n)]),
        name + "旋转权重",
        True,
    )
    rows = []
    models = {}
    with threadpool_limits(1):
        for kk in range(2, 12):
            model = KMeans(n_clusters=kk, n_init=5, random_state=42).fit(scaled)
            models[kk] = model
            rows.append(
                {
                    "K": kk,
                    "SSD": model.inertia_,
                    "轮廓系数_2000真实行": silhouette_score(
                        scaled, model.labels_, sample_size=2000, random_state=42
                    ),
                }
            )
    table = pd.DataFrame(rows).set_index("K")
    c.table(table, name + "K选择")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, table):
        ax.plot(table.index, table[col], marker="o")
        ax.set(title=name + col, xlabel="K")
    c.figure(fig, name + "K选择")
    d[name + "群"] = models[k].labels_
    means = d.groupby(name + "群")[cols].mean()
    c.table(means, name + "群原指标均值")
    heat(c, (means - d[cols].mean()) / d[cols].std(), name + "群标准化画像")
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=200, random_state=42
    ).fit(d[cols], d[name + "群"])
    treeplot(c, tree, cols, name + "解释树")
c.table(pd.crosstab(d["家庭群"], d["爱好群"]), "家庭爱好交叉人数")
heat(c, pd.crosstab(d["家庭群"], d["爱好群"]), "家庭爱好交叉人数")
c.table(d.join(flags), "用户分群明细")
c.notes += [
    f"处理{len(d)}条，未生成客户。众数填补及将缺失孩子标记记0、阅读点号记0均沿教材假设，输出保留原缺失标记。",
    "保留教材家庭3群、爱好2群，并输出K=2至11诊断。群编号随实现可置换，画像应查本次原指标均值；分位变换不保证群体等规模。解释树用于解释既有分群，不能当作营销因果规则。",
]
c.done()
