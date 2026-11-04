"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

c = start(
    __file__,
    "LightGBM基础与约束",
    "复现直方图梯度提升、LightGBM原生接口、类别特征、早停与约束。",
)
a, b, ya, yb = insurance(c)
prep = preprocess(a)
xa = pd.DataFrame(
    prep.fit_transform(a), columns=prep.get_feature_names_out(), index=a.index
)
xb = pd.DataFrame(prep.transform(b), columns=xa.columns, index=b.index)
aa, v, yaa, yv = train_test_split(xa, ya, test_size=0.3, stratify=ya, random_state=42)
cat = a.select_dtypes("object").columns.tolist()
probs = {}
base = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "seed": 42,
    "num_threads": 1,
}
with threadpool_limits(1):
    h = HistGradientBoostingClassifier(random_state=42).fit(xa, ya)
probs["HistGradientBoosting"] = h.predict_proba(xb)[:, 1]
sk = lgb.LGBMClassifier(n_jobs=1, verbosity=-1, random_state=42).fit(xa, ya)
probs["sklearn接口"] = sk.predict_proba(xb)[:, 1]
train = lgb.Dataset(xa, label=ya)
m = lgb.train(base, train, num_boost_round=100)
probs["原生接口"] = m.predict(xb)
history = {}
dt = lgb.Dataset(aa, label=yaa, categorical_feature=cat)
dv = lgb.Dataset(v, label=yv, reference=dt)
early = lgb.train(
    base,
    dt,
    valid_sets=[dv],
    num_boost_round=200,
    callbacks=[lgb.early_stopping(5, verbose=False), lgb.record_evaluation(history)],
)
probs["类别特征早停"] = early.predict(xb)
(c.out / "LightGBM模型.txt").write_text(early.model_to_string(), encoding="utf-8")
# 对类别特征的任意编码施加单调约束并不合理；沿教材接口教学，对数值age设置单调+1，其余0。
mono = [int(col == "age") for col in xa.columns]
for name, extra in [
    (
        "单调接口示例",
        {"monotone_constraints": mono, "monotone_constraints_method": "advanced"},
    ),
    ("交互接口示例", {"interaction_constraints": [[1, 3, 4], [5, 6]]}),
]:
    model = lgb.train(base | extra, lgb.Dataset(xa, label=ya), num_boost_round=100)
    probs[name] = model.predict(xb)
evaluate(c, yb, probs, "LightGBM测试比较")
c.line(pd.Series(history["valid_0"]["auc"]), "内部验证AUC", "AUC")
c.notes += [
    f"早停最佳轮次{early.best_iteration}。HistGradientBoosting是独立算法，不称为LightGBM的完全实现。",
    "原教材对任意类别编码加单调约束，本例改在age数值列展示约束接口；此约束仅教学，不作业务断言。省去无分析作用的二进制数据副本，保留可用模型。",
]
c.done()
