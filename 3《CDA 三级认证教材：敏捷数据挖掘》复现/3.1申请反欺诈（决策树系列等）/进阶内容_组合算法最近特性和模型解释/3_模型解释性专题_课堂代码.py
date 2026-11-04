"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

c = start(
    __file__,
    "模型解释与规则提取",
    "使用教材保险数据复现逻辑系数、树重要性、TreeSHAP、偏依赖和决策路径。",
)
a, b, ya, yb = insurance(c)
prep = preprocess(a)
xa = pd.DataFrame(
    prep.fit_transform(a), columns=prep.get_feature_names_out(), index=a.index
)
xb = pd.DataFrame(prep.transform(b), columns=xa.columns, index=b.index)
lr = Pipeline(
    [("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=4000))]
).fit(xa, ya)
c.table(
    pd.Series(lr.named_steps["lr"].coef_[0], index=xa.columns, name="标准化Logit系数"),
    "逻辑回归系数",
)
tree = DecisionTreeClassifier(random_state=42).fit(xa, ya)
rf = RandomForestClassifier(
    n_estimators=100, oob_score=True, n_jobs=1, random_state=42
).fit(xa, ya)
dm = xgb.DMatrix(xa, label=ya)
model = xgb.train(
    {"objective": "binary:logistic", "max_depth": 4, "seed": 42, "nthread": 1},
    dm,
    num_boost_round=50,
)
lm = lgb.train(
    {"objective": "binary", "verbosity": -1, "num_threads": 1, "seed": 42},
    lgb.Dataset(xa, label=ya),
    num_boost_round=100,
)
importance = pd.DataFrame(
    {
        "决策树不纯度": tree.feature_importances_,
        "随机森林不纯度": rf.feature_importances_,
        "XGB分裂次数": pd.Series(model.get_score(importance_type="weight"))
        .reindex(xa.columns)
        .fillna(0)
        .values,
        "LGB分裂次数": lm.feature_importance(),
    },
    index=xa.columns,
)
c.table(importance, "不同定义特征重要性")
# XGBoost原生pred_contribs就是TreeSHAP，避免依赖绘图库；检查可加性到原始margin。
sample = xb.iloc[:1000]
contrib = model.predict(xgb.DMatrix(sample), pred_contribs=True)
margin = model.predict(xgb.DMatrix(sample), output_margin=True)
np.testing.assert_allclose(contrib.sum(axis=1), margin, atol=1e-5)
shap = pd.DataFrame(contrib[:, :-1], index=sample.index, columns=xa.columns)
imp = shap.abs().mean().sort_values(ascending=False)
c.bar(imp.head(12), "TreeSHAP全局重要性", "平均绝对SHAP（对数几率）")
c.bar(
    shap.iloc[0].reindex(
        shap.iloc[0].abs().sort_values(ascending=False).head(12).index
    ),
    "单个真实客户SHAP",
    "SHAP（对数几率）",
)
c.table(shap.assign(基线=contrib[:, -1]), "真实测试客户SHAP")
top = imp.head(10).index
fig, ax = plt.subplots(figsize=(10, 7))
for j, col in enumerate(top):
    # 确定性排列错开点，不新增数据。
    vals = shap[col].to_numpy()
    offset = ((np.arange(len(vals)) * 37) % 101 / 100 - 0.5) * 0.5
    colors = sample[col].rank(pct=True)
    pts = ax.scatter(
        vals, j + offset, c=colors, cmap="viridis", s=6, alpha=0.5, vmin=0, vmax=1
    )
ax.set_yticks(range(len(top)), top)
ax.invert_yaxis()
ax.set(title="真实客户SHAP分布", xlabel="SHAP（对数几率）")
fig.colorbar(pts, ax=ax, label="变量值分位数")
c.figure(fig, "SHAP分布")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# 显式平均预测实现PDP，避免本机sklearn/SciPy组合中的网格reshape兼容错误。
for ax, col in zip(axes.flat, ["GEND", "BANK", "age", "meda"]):
    grid = np.unique(np.quantile(xa[col], np.linspace(0.05, 0.95, 20)))
    values = []
    for value in grid:
        changed = xa.iloc[:1000].copy()
        changed[col] = value
        values.append(rf.predict_proba(changed)[:, 1].mean())
    ax.plot(
        grid, values, marker="o", linestyle="none" if col in ["GEND", "BANK"] else "-"
    )
    ax.set(title=col, xlabel="特征值", ylabel="平均预测响应概率")
c.figure(fig, "四变量偏依赖")
features = importance["决策树不纯度"].nlargest(5).index
simple = DecisionTreeClassifier(max_depth=3, random_state=42).fit(xa[features], ya)
treeplot(c, simple, features, "简化解释树")
sample1 = xb[features].iloc[[0]]
nodes = simple.decision_path(sample1).indices
rows = []
for node in nodes:
    feature = simple.tree_.feature[node]
    if feature >= 0:
        col = features[feature]
        value = sample1.iloc[0, feature]
        threshold = simple.tree_.threshold[node]
        rows.append(
            {
                "节点": node,
                "变量": col,
                "真实客户值": value,
                "阈值": threshold,
                "分支": "左 ≤" if value <= threshold else "右 >",
            }
        )
c.table(pd.DataFrame(rows), "真实客户决策路径")
evaluate(
    c,
    yb,
    {
        "逻辑回归": lr.predict_proba(xb)[:, 1],
        "随机森林": rf.predict_proba(xb)[:, 1],
        "XGBoost": model.predict(xgb.DMatrix(xb)),
    },
    "解释模型测试评估",
)
c.notes += [
    "SHAP为前1000条真实测试记录，贡献和已核对等于logit margin；不能直接把SHAP相加解释为概率变化。随机森林feature_importances_仍是不纯度重要性，oob_score=True不会自动变成置换重要性。",
    "偏依赖改变模型输入以计算响应曲线，是模型函数分析，不增加客户记录，相关特征下不作因果解释。原鸢尾花示例改用同章保险数据。按不造数据要求，不生成DiCE反事实客户。",
]
c.done()
