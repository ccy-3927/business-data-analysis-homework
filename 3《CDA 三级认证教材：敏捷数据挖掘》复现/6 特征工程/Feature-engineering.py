"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    RFE,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

c = start(
    __file__,
    "电信特征工程",
    "复现异常处理、缺失编码、分箱、标准化、特征构造、PCA/LDA与过滤/包裹/嵌入选择。",
)
raw = data(c, "telecom_bill.csv")
d = raw.loc[raw.traffic < 5000].copy()
c.table(raw.loc[raw.traffic >= 5000], "教材规则排除记录")
d["state原值"] = d.state
d["IMEI原缺失"] = d.IMEI.isna()
d["state原缺失"] = d.state.isna()
d["state"] = d.state.replace({2: np.nan, 4: np.nan})
d["state"] = d.state.fillna(d.state.mode()[0])
d["IMEI"] = d.IMEI.fillna(0)
mu, sigma = d.expenditure.mean(), d.expenditure.std()
d["expenditure盖帽"] = d.expenditure.clip(mu - 3 * sigma, mu + 3 * sigma)
d["log_expd"] = np.log1p(d["expenditure盖帽"])
d["log_call"] = np.log1p(d.call)
d["log_traffic"] = np.log1p(d.traffic)
d["has_IMEI"] = d.IMEI.gt(0).astype(int)
d["duration"] = (pd.Timestamp("2015-01-01") - pd.to_datetime(d.join_time)).dt.days / 30
d["unit_price"] = d.expenditure / (d.traffic + 1)
d["TAC"] = d.IMEI.map(lambda x: str(int(x))[:6] if x else "缺失")
c.bar(d.state.value_counts().sort_index(), "处理后状态人数", "人数")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(d.expenditure, bins=40)
axes[0].set(title="原支出", xlabel="expenditure", ylabel="人数")
axes[1].hist(d.log_expd, bins=30)
axes[1].set(title="3σ盖帽后log1p支出", xlabel="log1p", ylabel="人数")
c.figure(fig, "支出变换")
for name, bins in [
    ("等宽5箱", pd.cut(d.duration, 5)),
    ("等频14箱", pd.qcut(d.duration, 14, duplicates="drop")),
    ("人工边界", pd.cut(d.duration, [-np.inf, 5, 12, 36, np.inf])),
]:
    t = d.groupby(bins, observed=True).churn.agg(["count", "mean", "sum"])
    c.table(t, name + "_流失统计")
    rates = t["mean"] * 100
    rates.index = [f"({v.left:.2f}, {v.right:.2f}]" for v in t.index]
    c.bar(rates, name + "_流失率", "流失率%")
cols = ["log_expd", "log_call", "log_traffic"]
z = StandardScaler().fit_transform(d[cols])
zcols = ["s_expd", "s_call", "s_traffic"]
d[zcols] = z
c.table(
    pd.DataFrame(MinMaxScaler().fit_transform(d[cols]), index=d.index, columns=cols),
    "MinMax变换",
)
c.table(
    pd.DataFrame(Normalizer().fit_transform(z), index=d.index, columns=zcols),
    "行向量归一化",
)
pcs = PCA(n_components=2, whiten=True).fit(z)
score = pcs.transform(z)
c.table(pd.DataFrame(pcs.components_, columns=zcols), "PCA权重")
d["PC1"] = score[:, 0]
d["PC2"] = score[:, 1]
lda = LinearDiscriminantAnalysis().fit(d[zcols], d.churn)
d["LDA"] = lda.transform(d[zcols]).ravel()
c.table(pd.DataFrame(lda.coef_, columns=zcols), "LDA投影系数")
sample = d.sample(min(3000, len(d)), random_state=42)
f = go.Figure()
for label, g in sample.groupby("churn"):
    f.add_trace(
        go.Scatter(
            x=g.PC1,
            y=g.PC2,
            mode="markers",
            name="churn=" + str(label),
            marker={"size": 6, "opacity": 0.6},
        )
    )
c.interactive(f, "真实客户PCA投影")
x = d[["s_expd", "s_call", "s_traffic", "duration"]]
y = d.churn
selector = RFE(LogisticRegression(max_iter=4000), n_features_to_select=2).fit(x, y)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1).fit(x, y)
embedded = SelectFromModel(GradientBoostingClassifier(random_state=42)).fit(x, y)
f, p = f_classif(x, y)
f2, p2 = f_regression(x.drop(columns="s_expd"), d.s_expd)
c.table(
    pd.DataFrame(
        {
            "ANOVA_F": f,
            "ANOVA_p": p,
            "RFE保留": selector.support_,
            "随机森林重要性": rf.feature_importances_,
            "GBDT保留": embedded.get_support(),
        },
        index=x.columns,
    ),
    "连续特征筛选",
)
c.table(
    pd.DataFrame({"F": f2, "p": p2}, index=x.columns.drop("s_expd")),
    "支出关联筛选_排除目标自身",
)
cat = pd.get_dummies(d[["vip", "state", "has_IMEI"]].astype("category"), dtype=int)
stat, p = chi2(cat, y)
c.table(pd.DataFrame({"卡方": stat, "p": p}, index=cat.columns), "类别OneHot卡方")
v = VarianceThreshold(0.09).fit(d[["vip", "has_IMEI"]])
c.table(
    pd.DataFrame(
        {"方差": v.variances_, "保留": v.get_support()}, index=["vip", "has_IMEI"]
    ),
    "方差筛选",
)
c.table(d, "特征工程结果")
c.notes += [
    f"原{len(raw)}条，按教材traffic<5000保留{len(d)}条；固定基准日2015-01-01，未用当前日期。修正原例new_state未回写的问题，保留原值及缺失标记。",
    "LDA和特征筛选使用全体教材观测用于讲解，不输出样本外预测效果。人工伪测试向量不运行；TAC从原设备值取前6位，缺失设备不伪装为真实编码。回归筛选排除目标本身，类别卡方先独热编码避免把状态码当作频数。",
]
c.done()
