"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

import xgboost as xgb

c = start(
    __file__,
    "XGBoost基础与约束",
    "复现sklearn/原生接口、早停、单调约束、交互约束及并行树。",
)
a, b, ya, yb = insurance(c)
prep = preprocess(a)
xa = pd.DataFrame(
    prep.fit_transform(a), columns=prep.get_feature_names_out(), index=a.index
)
xb = pd.DataFrame(prep.transform(b), columns=xa.columns, index=b.index)
aa, v, yaa, yv = train_test_split(xa, ya, test_size=0.3, stratify=ya, random_state=42)
da = xgb.DMatrix(xa, label=ya)
db = xgb.DMatrix(xb)
dt = xgb.DMatrix(aa, label=yaa)
dv = xgb.DMatrix(v, label=yv)
base = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "eta": 0.4,
    "seed": 42,
    "nthread": 1,
}
probs = {}
sk = xgb.XGBClassifier(n_estimators=100, n_jobs=1, random_state=42).fit(xa, ya)
probs["sklearn接口"] = sk.predict_proba(xb)[:, 1]
mono = [0] * xa.shape[1]
mono[1] = 1
mono[2] = -1
specs = {
    "原生15轮": {},
    "单调约束": {"monotone_constraints": tuple(mono)},
    "交互约束": {"interaction_constraints": "[[0,2],[1,3,4],[5,6]]"},
    "随机森林": {
        "num_parallel_tree": 100,
        "eta": 1.0,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
    },
}
for name, extra in specs.items():
    m = xgb.train(base | extra, da, num_boost_round=1 if name == "随机森林" else 15)
    probs[name] = m.predict(db)
history = {}
early = xgb.train(
    base,
    dt,
    num_boost_round=200,
    evals=[(dt, "train"), (dv, "validation")],
    early_stopping_rounds=3,
    evals_result=history,
    verbose_eval=False,
)
probs["训练内早停"] = early.predict(db, iteration_range=(0, early.best_iteration + 1))
c.line(
    pd.DataFrame({key: val["auc"] for key, val in history.items()}),
    "早停验证AUC",
    "AUC",
)
evaluate(c, yb, probs, "XGBoost测试比较")
c.table(pd.DataFrame({"变量": xa.columns, "单调约束": mono}), "教学约束映射")
c.notes += [
    f"早停最佳轮次{early.best_iteration+1}；预测显式限定最佳迭代区间。",
    "原生模型明确设binary:logistic，避免默认回归目标被误当二分类概率。前3列的单调方向和交互分组沿教材仅作接口示例，不宣称这些保险变量有真实单调因果关系。",
]
c.done()
