"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.feature_selection import RFECV
from statsmodels.formula.api import logit

c = start(
    __file__,
    "开卡响应逻辑回归",
    "复现单变量Logit、递归特征消除、正则调参、ROC与本期评分。",
)
d = data(c, "creditcard_exp.csv", skipinitialspace=True)
new = data(c, "creditcard_exp_out_of_sample.csv", skipinitialspace=True)
cols = [
    "Age",
    "Income",
    "dist_home_val",
    "dist_avg_income",
    "high_avg",
    "gender",
    "Ownrent",
    "Selfempl",
    "edu_class",
]
tr = d.sample(frac=0.7, random_state=1234)
te = d.drop(tr.index)
x = tr[cols]
y = tr.Acc
m = logit("Acc~Income", tr).fit(disp=False)
model_output(c, m, "单变量Logit")
# RFECV的估计器本身包含标准化，避免先全训练集标准化再交叉验证。
est = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=3000))])
rfe = RFECV(
    est, cv=3, scoring="roc_auc", importance_getter="named_steps.lr.coef_", n_jobs=1
).fit(x, y)
selected = x.columns[rfe.support_].tolist()
c.table(
    pd.DataFrame({"变量": cols, "排名": rfe.ranking_, "保留": rfe.support_}),
    "递归特征筛选",
)
best = search(
    c, est, {"lr__C": np.logspace(-3, 3, 10)}, x[selected], y, "正则调参", cv=4
)
p = best.predict_proba(te[selected])[:, 1]
evaluate(c, te.Acc, {"单变量": m.predict(te), "筛选后多变量": p}, "测试集评估")
thresholds = []
for t in np.arange(0.1, 1, 0.2):
    thresholds.append(
        {
            "阈值": t,
            "precision_1": precision_score(te.Acc, p >= t, zero_division=0),
            "recall_1": recall_score(te.Acc, p >= t),
            "F1_1": f1_score(te.Acc, p >= t),
        }
    )
c.table(pd.DataFrame(thresholds), "固定阈值对照")
c.table(
    new[["id"]].assign(开卡概率=best.predict_proba(new[selected])[:, 1]), "本期客户评分"
)
joblib.dump(best, c.out / "开卡模型.joblib")
np.testing.assert_allclose(
    joblib.load(c.out / "开卡模型.joblib").predict_proba(te[selected])[:, 1], p
)
c.notes += [
    "仅使用发生响应前的属性，排除avg_exp及其对数等事后字段。沿用教材分类字段数值编码以便RFE逐原始变量筛选；不能将该编码视为类别之间等距的业务事实。",
    "原Notebook的WOE部分已改成Markdown且注明不建议运行，本案例复现其实际执行的RFE路线。precision/recall统一以开卡1为正类，修正原代码分母和类别混用。测试集小，评分只作教学。",
]
c.done()
