"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold

c = start(
    __file__,
    "客户价值线性回归",
    "复现线性/对数模型、前向AIC、残差与影响点、分类交互、岭/Lasso/弹性网络。",
)
d = data(c, "creditcard_exp.csv", skipinitialspace=True)
oos = data(c, "creditcard_exp_out_of_sample.csv", skipinitialspace=True)
train = d.loc[d.avg_exp.notna()].copy()
for frame in [train, oos]:
    for col in ["Income", "dist_home_val", "dist_avg_income"]:
        frame[col + "_ln"] = np.log(frame[col])
    frame["high_avg_ratio"] = frame.high_avg / frame.dist_avg_income
specs = {
    "简单": "avg_exp~Income",
    "多元": "avg_exp~Income+dist_home_val+dist_avg_income",
    "对数目标": "avg_exp_ln~Income",
    "双对数": "avg_exp_ln~Income_ln",
}
models = {name: ols(f, train).fit() for name, f in specs.items()}
# 按教材识别影响点，但主分析保留观测，删点拟合仅作敏感性对照。
base = models["双对数"]
influence = OLSInfluence(base).summary_frame()
c.table(influence, "影响点诊断")
keep = ((base.resid - base.resid.mean()) / base.resid.std()).abs() <= 2
models["删点双对数_仅敏感性"] = ols(specs["双对数"], train.loc[keep]).fit()
for title, cols, target in [
    ("前向金额", ["Income", "Age", "dist_home_val", "dist_avg_income"], "avg_exp"),
    (
        "前向对数",
        ["Age", "high_avg_ratio", "dist_home_val_ln", "dist_avg_income_ln"],
        "avg_exp_ln",
    ),
]:
    chosen = []
    best = ols(target + "~1", train).fit().aic
    path = []
    while cols:
        aic, col, m = min(
            (
                ols(target + "~" + "+".join(chosen + [x]), train).fit().aic,
                x,
                ols(target + "~" + "+".join(chosen + [x]), train).fit(),
            )
            for x in cols
        )
        if aic >= best:
            break
        chosen.append(col)
        cols.remove(col)
        best = aic
        path.append({"变量": col, "AIC": aic})
        models[title] = m
    c.table(pd.DataFrame(path), title + "_路径")
models["分类交互"] = ols(
    "avg_exp_ln~dist_avg_income_ln+dist_home_val_ln+C(Selfempl)+C(gender)*C(edu_class)",
    train,
).fit()
rows = []
for name, m in models.items():
    model_output(c, m, name)
    rows.append(
        {
            "模型": name,
            "n": m.nobs,
            "R2_同目标尺度才可比": m.rsquared,
            "AIC_同样本同尺度才可比": m.aic,
        }
    )
c.table(pd.DataFrame(rows), "模型比较")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, name in zip(axes.flat, list(specs)):
    m = models[name]
    ax.scatter(m.fittedvalues, m.resid, alpha=0.7, s=22)
    ax.axhline(0, color="gray", ls="--")
    ax.set(title=name, xlabel="样本内拟合值", ylabel="残差")
c.figure(fig, "函数形式与残差")
scatter(c, train.Income, train.avg_exp, "收入与月均消费")
heat(
    c,
    train[["Income", "Age", "dist_home_val", "dist_avg_income"]].corr(),
    "解释变量相关性",
    True,
)
fig, ax = plt.subplots(figsize=(7, 5))
stats.probplot(base.resid, plot=ax)
ax.set_title("双对数残差Q-Q")
c.figure(fig, "残差Q-Q")
pred = oos[["id"]].copy()
pred["教育原值"] = oos.edu_class
unseen = ~oos.edu_class.isin(train.edu_class.unique())
oos.loc[unseen, "edu_class"] = 1
pred["教育按教材映射"] = oos.edu_class
pred["交互模型_几何均值预测"] = np.exp(models["分类交互"].predict(oos))
x = train[["Age", "Income", "dist_home_val", "dist_avg_income"]]
y = train.avg_exp_ln
alphas = np.logspace(-3, 3, 50)
cv = KFold(5, shuffle=True, random_state=42)
for name, est, grid in [
    ("Ridge", Ridge(), {"model__alpha": alphas}),
    ("Lasso", Lasso(max_iter=20000), {"model__alpha": alphas}),
    (
        "ElasticNet",
        ElasticNet(max_iter=20000),
        {
            "model__alpha": np.logspace(-3, 1, 20),
            "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
        },
    ),
]:
    pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
    g = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=1).fit(
        x, y
    )
    c.table(
        pd.DataFrame(g.cv_results_)[["params", "mean_test_score", "std_test_score"]],
        name + "_交叉验证",
    )
    pred[name + "_几何均值预测"] = np.exp(g.predict(oos[x.columns]))
    c.notes.append(
        f"{name}最优参数{g.best_params_}；交叉验证对数MSE={-g.best_score_:.5f}。"
    )
    if name != "ElasticNet":
        z = StandardScaler().fit_transform(x)
        coefs = [est.set_params(alpha=a).fit(z, y).coef_.copy() for a in alphas]
        fig, ax = plt.subplots(figsize=(10, 5))
        for j, col in enumerate(x):
            ax.plot(
                alphas, np.array(coefs)[:, j], label=col, ls=["-", "--", ":", "-."][j]
            )
        ax.set(
            xscale="log",
            xlabel="alpha",
            ylabel="标准化特征系数",
            title=name + "系数路径",
        )
        ax.legend()
        c.figure(fig, name + "系数路径")
c.table(pred, "本期待预测客户结果")
c.notes += [
    f"有金额观测{len(train)}条；影响点规则标记{int((~keep).sum())}条，主模型不删除。样本外文件{len(oos)}条没有真实消费目标，不能计算样本外误差。",
    f"按教材将待预测集中训练未见教育类别映射为1，共{int(unseen.sum())}条，输出保留原值。exp(log预测)是几何均值/对数正态中位数口径，不冒称算术均值。",
    "正则化的标准化放入每折Pipeline；所有统计拟合、残差诊断为教学分析，不作为因果证据。",
]
c.done()
