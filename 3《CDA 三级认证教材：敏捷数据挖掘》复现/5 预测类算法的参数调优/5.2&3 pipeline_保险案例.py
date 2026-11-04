"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.tree import DecisionTreeClassifier

c = start(
    __file__,
    "保险响应Pipeline实操",
    "复现缺失处理、类别编码、标准化、模型调参、持久化与三名新用户预测，并完成教材决策树练习。",
)
a, b, ya, yb = insurance(c)
probs = {}
predictions = data(c, "保险案例_新进用户.csv").drop(columns="resp_flag")
out = predictions.copy()
for name, est, grid in [
    (
        "逻辑回归",
        LogisticRegression(max_iter=4000),
        {"model__C": [0.1, 1, 1.5, 1.9], "model__class_weight": [None, "balanced"]},
    ),
    (
        "决策树",
        DecisionTreeClassifier(random_state=42),
        {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [20, 100],
            "model__class_weight": [None, "balanced"],
        },
    ),
]:
    pipe = Pipeline([("prep", preprocess(a, onehot=True, scale=True)), ("model", est)])
    model = search(c, pipe, grid, a, ya, name)
    probs[name] = model.predict_proba(b)[:, 1]
    out[name + "_响应概率"] = model.predict_proba(predictions)[:, 1]
    out[name + "_预测标签"] = model.predict(predictions)
    joblib.dump(model, c.out / (name + "_完整管道.joblib"))
    loaded = joblib.load(c.out / (name + "_完整管道.joblib"))
    np.testing.assert_allclose(
        loaded.predict_proba(predictions), model.predict_proba(predictions)
    )
evaluate(c, yb, probs, "保险Pipeline测试比较")
c.table(out, "三名新用户预测")
c.notes += [
    "新增用户没有真实resp_flag，因此只输出概率和预测，不计算伪准确率。类别字段按训练列名单识别，保留新用户整列缺失字段；未知类别由OneHotEncoder处理。",
    "保存的是完整预处理+模型管道，已用加载后的模型核对三名用户预测一致。教材原模型文件未直接加载，避免旧版本不兼容与遗漏预处理。",
]
c.done()
