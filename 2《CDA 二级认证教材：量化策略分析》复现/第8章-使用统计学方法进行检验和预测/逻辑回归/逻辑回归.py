"""教材Notebook的可运行迁移版；原单元格及变化见同目录对照文档。"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *

import statsmodels.formula.api as smf
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor

c = start(
    __file__,
    "电信流失逻辑回归",
    "复现交叉表、单变量与多变量logit、前向AIC、分类变量检验、ROC及阈值评估。churn=1为正类。",
)
d = data(c, "telecom_churn.csv")
profile(c, d, "流失数据")
assert d.subscriberID.is_unique and d.notna().all().all()
train = d.sample(frac=0.7, random_state=666).copy()
test = d.loc[~d.index.isin(train.index)].copy()
assert set(train.index).isdisjoint(test.index)
c.table(
    pd.DataFrame(
        {
            "subscriberID": d.subscriberID,
            "分组": np.where(d.index.isin(train.index), "训练", "测试"),
        }
    ),
    "训练测试划分",
)
tab = pd.crosstab(d.posTrend, d.churn)
c.table(tab, "posTrend列联表")
c.table(tab.div(tab.sum(axis=1), axis=0), "posTrend行比例")
chi = stats.chi2_contingency(tab)
c.table(pd.Series({"卡方": chi.statistic, "p值": chi.pvalue}), "posTrend卡方")
continuous = [
    "duration",
    "AGE",
    "incomeCode",
    "nrProm",
    "peakMinAv",
    "peakMinDiff",
    "call_10086",
]
selected = []
remaining = continuous.copy()
current = smf.logit("churn~1", train).fit(disp=False).aic
history = []
while remaining:
    candidates = []
    for name in remaining:
        model = smf.logit("churn~" + "+".join(selected + [name]), train).fit(
            disp=False, maxiter=200
        )
        if model.mle_retvals["converged"]:
            candidates.append((model.aic, name))
    if not candidates:
        break
    best, name = min(candidates)
    if best >= current - 1e-8:
        break
    selected.append(name)
    remaining.remove(name)
    current = best
    history.append({"新增变量": name, "AIC": best})
c.table(pd.DataFrame(history), "训练集前向AIC路径")
cats = [
    "posTrend",
    "prom",
    "edu_class",
    "feton",
    "curPlan",
    "avgplan",
    "gender",
    "negTrend",
    "planChange",
    "posPlanChange",
    "negPlanChange",
]
rows = []
for name in cats:
    t = pd.crosstab(train[name], train.churn)
    chi = stats.chi2_contingency(t)
    rows.append(
        {"变量": name, "p值": chi.pvalue, "最小期望频数": chi.expected_freq.min()}
    )
c.table(pd.DataFrame(rows), "训练集分类变量筛查")
original_formula = "churn~duration+AGE+incomeCode+peakMinDiff+call_10086+C(edu_class)+C(feton)+C(posTrend)+C(negTrend)+C(curPlan)+C(avgplan)+C(gender)"
# 保留原公式失败的诊断；不把优化器给出的极大系数解释为可靠效应。
import warnings

with warnings.catch_warnings(record=True) as fit_warnings:
    warnings.simplefilter("always")
    original_fit = smf.logit(original_formula, train).fit(disp=False, maxiter=300)
c.table(
    train.groupby(["curPlan", "avgplan"]).churn.agg(["count", "sum"]),
    "训练集套餐组合计数",
)
c.table(
    pd.Series(
        {
            "收敛": original_fit.mle_retvals["converged"],
            "最大系数绝对值": original_fit.params.abs().max(),
            "设计矩阵秩": np.linalg.matrix_rank(original_fit.model.exog),
            "列数": original_fit.model.exog.shape[1],
        }
    ),
    "教材原公式估计诊断",
)
c.table(original_fit.params, "教材原公式系数_不稳定仅诊断")
(c.out / "教材原公式警告.txt").write_text(
    "\n".join(str(w.message) for w in fit_warnings), encoding="utf-8"
)
specs = {
    "单变量": "churn~duration",
    "多变量": original_formula.replace("+C(avgplan)", ""),
}
evaluations = []
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], ls=":", color="#888888")
for label, formula in specs.items():
    m = smf.logit(formula, train).fit(disp=False, maxiter=300)
    assert m.mle_retvals["converged"]
    model_output(c, m, label)
    tr = m.predict(train)
    te = m.predict(test)
    fpr, tpr, thresholds = metrics.roc_curve(train.churn, tr)
    chosen = thresholds[np.argmax(tpr - fpr)]
    out = test[["subscriberID", "churn"]].copy()
    out["流失概率"] = te
    out["训练集约登阈值"] = chosen
    out["预测标签"] = te.ge(chosen).astype(int)
    c.table(out, label + "_测试预测")
    for threshold in [0.3, 0.5, chosen]:
        pred = te.ge(threshold).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(
            test.churn, pred, labels=[0, 1]
        ).ravel()
        evaluations.append(
            {
                "模型": label,
                "阈值": threshold,
                "阈值来源": "训练集约登" if threshold == chosen else "固定教学阈值",
                "AUC": metrics.roc_auc_score(test.churn, te),
                "准确率": metrics.accuracy_score(test.churn, pred),
                "precision_正类1": metrics.precision_score(
                    test.churn, pred, zero_division=0
                ),
                "recall_正类1": metrics.recall_score(test.churn, pred),
                "specificity": tn / (tn + fp),
                "F1_正类1": metrics.f1_score(test.churn, pred),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )
    fpr, tpr, _ = metrics.roc_curve(test.churn, te)
    ax.plot(
        fpr,
        tpr,
        label=f"{label} AUC={metrics.roc_auc_score(test.churn,te):.3f}",
        ls="-" if label == "单变量" else "--",
    )
    heat(
        c,
        pd.DataFrame(
            metrics.confusion_matrix(test.churn, out["预测标签"], labels=[0, 1]),
            index=["实际0", "实际1"],
            columns=["预测0", "预测1"],
        ),
        label + "测试混淆矩阵",
    )
    if label == "多变量":
        c.table(
            pd.Series(
                {
                    name: variance_inflation_factor(m.model.exog, j)
                    for j, name in enumerate(m.model.exog_names)
                    if name != "Intercept"
                },
                name="VIF",
            ),
            "多变量设计矩阵VIF",
        )
ax.set(title="测试集ROC曲线", xlabel="假阳性率", ylabel="真阳性率")
ax.legend(loc="lower right")
ax.grid(alpha=0.4)
c.figure(fig, "测试集ROC对比")
c.table(pd.DataFrame(evaluations), "测试集指标")
c.notes += [
    f"沿用教材random_state=666的70%抽样，训练{len(train)}条、测试{len(test)}条。抽样只选已有记录。",
    "修正原代码precision/recall/specificity的分母和类别混用，统一以churn=1为正类。分类筛查只用训练集；约登阈值也只从训练集选择，测试集仅评价。原代码在测试集上取最优阈值和手工0.456不作为本次最优阈值。",
    "前向AIC只在训练集运行，与按教材变量组合拟合分开输出；显著性筛查不是因果识别。模型概率不等于实际流失事件。",
    "教材完整公式在该训练集上Newton迭代300次仍不收敛。curPlan与avgplan几乎重合，若干变更套餐组合只有1至4条且结局单一，造成估计不稳定。已保留原公式、警告和训练组合计数；用于后续评价的多变量模型删除C(avgplan)，保留当前套餐curPlan，并验证收敛。这一修正仅依据训练数据，未用测试指标挑选模型。原公式不稳定系数不作显著性解释。",
]
c.done()
