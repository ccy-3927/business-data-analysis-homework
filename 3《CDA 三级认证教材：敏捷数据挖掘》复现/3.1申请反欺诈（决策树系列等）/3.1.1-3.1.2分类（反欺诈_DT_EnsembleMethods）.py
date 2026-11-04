"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

c = start(
    __file__,
    "申请反欺诈树模型与集成",
    "复现决策树、随机森林、AdaBoost、XGBoost、LightGBM，比较测试ROC、PR与KS。",
)
d = data(c, "loan_net.csv")
x = d.iloc[:, :4]
y = d.flag
a, b, ya, yb = train_test_split(x, y, test_size=0.4, random_state=1234, stratify=y)
specs = [
    (
        "决策树",
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [2, 3, 4, 5],
            "min_samples_leaf": [2, 6, 10],
            "class_weight": [None, "balanced"],
        },
    ),
    (
        "随机森林",
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        {"max_depth": [4, 8], "max_features": [0.5, 0.7], "min_samples_leaf": [2, 6]},
    ),
    (
        "AdaBoost",
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [15, 20, 25], "learning_rate": [0.5, 1, 1.5]},
    ),
    (
        "XGBoost",
        XGBClassifier(n_jobs=1, random_state=42, eval_metric="logloss"),
        {
            "max_depth": [2, 3, 5],
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 0.5],
        },
    ),
    (
        "LightGBM",
        LGBMClassifier(n_jobs=1, random_state=42, verbosity=-1),
        {
            "num_leaves": [7, 15, 31],
            "n_estimators": [50, 100],
            "learning_rate": [0.04, 0.08],
        },
    ),
]
probs = {}
for name, est, grid in specs:
    m = search(c, est, grid, a, ya, name, cv=4)
    probs[name] = m.predict_proba(b)[:, 1]
    if name == "决策树":
        treeplot(c, m, x.columns, "反欺诈决策树")
        joblib.dump(m, c.out / "反欺诈决策树.joblib")
    c.table(
        pd.Series(m.feature_importances_, index=x.columns, name="重要性"),
        name + "_重要性",
    )
evaluate(c, yb, probs, "反欺诈测试集比较")
c.notes += [
    f"共{len(d)}条，正类{int(y.sum())}条；分层60/40拆分以保留少数类。CV只用训练集，统一ROC-AUC作搜索目标，阈值固定0.5，结合AP和召回率理解极度不平衡。",
    "为可重复运行，将教材大规模网格收敛到代码列出的候选组合，所有算法均实际拟合；不冒称教材全网格最优结果。",
]
c.done()
