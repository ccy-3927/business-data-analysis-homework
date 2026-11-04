"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from scipy.stats import uniform, randint
import optuna

c = start(
    __file__,
    "预测算法四种参数搜索",
    "复现网格、逐次减半、随机搜索和Optuna TPE；最终测试集不参与任何参数选择。",
)
a, b, ya, yb = insurance(c)
prep = preprocess(a)
base = Pipeline(
    [
        ("prep", prep),
        ("model", XGBClassifier(n_jobs=1, random_state=42, eval_metric="logloss")),
    ]
)
cv = StratifiedKFold(3, shuffle=True, random_state=42)
grid = {
    "model__n_estimators": [100, 140],
    "model__max_depth": [3, 5],
    "model__subsample": [0.8, 0.9],
    "model__colsample_bytree": [0.8, 0.9],
}
jobs = {
    "网格": GridSearchCV(
        base, grid, cv=cv, scoring="roc_auc", n_jobs=1, error_score="raise"
    ),
    "逐次减半": HalvingGridSearchCV(
        base,
        {
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.6, 0.8],
            "model__colsample_bytree": [0.6, 0.8],
        },
        resource="model__n_estimators",
        min_resources=20,
        max_resources=180,
        factor=3,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        error_score="raise",
    ),
    "随机": RandomizedSearchCV(
        base,
        {
            "model__n_estimators": randint(100, 150),
            "model__max_depth": [3, 4, 5],
            "model__subsample": uniform(0.5, 0.5),
            "model__colsample_bytree": uniform(0.5, 0.5),
        },
        n_iter=20,
        random_state=42,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        error_score="raise",
    ),
}
probs = {}
results = []
for name, job in jobs.items():
    with threadpool_limits(1):
        job.fit(a, ya)
    c.table(
        pd.DataFrame(job.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ],
        name + "_候选结果",
    )
    probs[name] = job.predict_proba(b)[:, 1]
    results.append(
        {"方法": name, "训练CV_AUC": job.best_score_, "最优参数": str(job.best_params_)}
    )
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    params = {
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10),
        "num_leaves": trial.suggest_int("num_leaves", 2, 64),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1),
        "subsample": trial.suggest_float("subsample", 0.4, 1),
    }
    pipe = Pipeline(
        [
            ("prep", preprocess(a)),
            (
                "model",
                LGBMClassifier(
                    **params,
                    n_estimators=100,
                    subsample_freq=1,
                    n_jobs=1,
                    verbosity=-1,
                    random_state=42
                ),
            ),
        ]
    )
    with threadpool_limits(1):
        return cross_val_score(pipe, a, ya, cv=cv, scoring="roc_auc", n_jobs=1).mean()


study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=20)
final = Pipeline(
    [
        ("prep", preprocess(a)),
        (
            "model",
            LGBMClassifier(
                **study.best_params,
                n_estimators=100,
                subsample_freq=1,
                n_jobs=1,
                verbosity=-1,
                random_state=42
            ),
        ),
    ]
).fit(a, ya)
probs["TPE_LightGBM"] = final.predict_proba(b)[:, 1]
c.table(study.trials_dataframe(), "TPE搜索记录")
results.append(
    {
        "方法": "TPE_LightGBM",
        "训练CV_AUC": study.best_value,
        "最优参数": str(study.best_params),
    }
)
c.table(pd.DataFrame(results), "搜索最优参数")
evaluate(c, yb, probs, "搜索结果独立测试")
c.line(
    pd.Series(
        [max(t.value for t in study.trials[: i + 1]) for i in range(len(study.trials))]
    ),
    "TPE最佳验证AUC进程",
    "AUC",
)
c.notes += [
    "采用可在普通电脑重复运行的搜索范围：网格16组、减半12组初始候选、随机20次、TPE20次，实际候选均有表格，不声称复刻教材数千组合的最优值。",
    "减半以树数作为资源，保留所有训练行；预处理在每折内拟合。TPE修正原例直接在测试集优化AUC的问题，并使subsample_freq=1让行采样参数实际生效。TPE使用不同算法，不能将差异只归因于搜索方法。",
]
c.done()
