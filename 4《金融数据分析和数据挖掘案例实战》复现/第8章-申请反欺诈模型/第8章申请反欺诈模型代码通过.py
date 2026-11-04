"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from fraud_features import AnomalyFeatures, historical_network
from network_common import network_plot, communities, nx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from bayes_opt import BayesianOptimization

c = start(
    __file__,
    "申请反欺诈与关系网络",
    "重建全量按日网络特征，对比基础异常特征、网络特征和贝叶斯优化LightGBM。",
)
d = data(c, "data/loan1_org_basic.csv").set_index("userid")
edges = data(c, "data/loan1_org_net.csv")
profile(c, d, "申请数据")
conflicts = d.index[d.index.duplicated(False)].unique()
c.table(d.loc[conflicts], "重复客户冲突记录")
d = d.loc[~d.index.isin(conflicts)].copy()
assert d.index.is_unique
c.notes.append(
    f"排除{len(conflicts)}个重复且信息冲突的客户号，原始冲突记录保留供核对，不任意选择某一条或改标签。清洗后{len(d)}位唯一客户再分区。"
)
net = historical_network(edges)
c.table(net, "按申请日重建网络特征")
g = nx.from_pandas_edgelist(edges, "userid", "connid")
groups, partition = communities(g)
c.table(pd.Series(partition, name="社群"), "全网社群归属")
c.bar(
    pd.Series([len(x) for x in groups]).value_counts().sort_index().head(20),
    "社群规模频数",
    "社群数",
)
focus = g.subgraph(nx.node_connected_component(g, "id103572")).copy()
flags = edges.drop_duplicates("userid").set_index("userid").flag.to_dict()
network_plot(c, focus, "教材目标申请人的关联网络", flags)
for date in [42767, 42768]:
    sub = edges[
        edges.in_date.le(date)
        & edges.userid.isin(focus.nodes)
        & edges.connid.isin(focus.nodes)
    ]
    gg = nx.from_pandas_edgelist(sub, "userid", "connid")
    network_plot(
        c,
        gg,
        f"关联网络截至{date}",
        {
            n: (
                flags.get(n, 9)
                if n in edges.loc[edges.flag_date.le(date), "userid"].values
                else 9
            )
            for n in gg
        },
    )
x = d.drop(columns="flag").join(net.drop(columns="date")).fillna(0)
y = d.flag
a, b, ya, yb = train_test_split(x, y, test_size=0.3, random_state=12345, stratify=y)
models = {}
base = ["age", "gender", "city_level", "edu", "income"]
for label, cols, grid, scoring in [
    (
        "异常特征",
        base,
        {
            "gb__loss": ["log_loss", "exponential"],
            "gb__learning_rate": [0.2],
            "gb__n_estimators": [80],
            "gb__max_depth": [2],
        },
        "roc_auc",
    ),
    (
        "加入网络",
        list(x.columns),
        {
            "gb__loss": ["log_loss", "exponential"],
            "gb__learning_rate": [0.1, 0.3],
            "gb__n_estimators": [35],
            "gb__max_depth": [1, 2],
            "gb__min_samples_split": [2, 4],
        },
        "f1",
    ),
]:
    model = search(
        c,
        Pipeline(
            [
                ("features", AnomalyFeatures()),
                ("gb", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        grid,
        a[cols],
        ya,
        label,
        score=scoring,
        cv=4,
    )
    models[label] = model.predict_proba(b[cols])[:, 1]
    joblib.dump(model, c.out / (label + "模型.joblib"))
prepared = AnomalyFeatures().fit(a)
aa = prepared.transform(a)
bb = prepared.transform(b)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ["income_resid", "income_resid_st_abs"]):
    ax.boxplot(
        [aa.loc[ya.eq(k), col] for k in [0, 1]],
        tick_labels=["正常", "欺诈"],
        showfliers=False,
    )
    ax.set(title=col, ylabel="训练集值")
c.figure(fig, "收入异常特征分组")
space = {
    "n_estimators": (10, 120),
    "learning_rate": (0.05, 0.2),
    "max_depth": (1, 3),
    "min_split_gain": (0, 10),
    "min_child_weight": (0, 10),
    "subsample": (0.5, 1),
    "colsample_bytree": (0.5, 1),
    "reg_lambda": (0, 1),
}


def estimator(params):
    params = params.copy()
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = max(0.001, params["min_child_weight"])
    return Pipeline(
        [
            ("features", AnomalyFeatures()),
            (
                "lgb",
                LGBMClassifier(
                    **params,
                    num_leaves=2 ** params["max_depth"],
                    subsample_freq=1,
                    random_state=123,
                    n_jobs=1,
                    verbosity=-1,
                ),
            ),
        ]
    )


failures = []


def objective(**params):
    try:
        with threadpool_limits(1):
            return float(
                cross_val_score(
                    estimator(params),
                    a,
                    ya,
                    cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring="roc_auc",
                    n_jobs=1,
                    error_score="raise",
                ).mean()
            )
    except LightGBMError as exc:
        failures.append({"参数": str(params), "错误": str(exc)})
        return 0.0  # 搜索惩罚值，不是有效AUC；在结果中明确标为失败。


opt = BayesianOptimization(f=objective, pbounds=space, random_state=42, verbose=0)
opt.maximize(init_points=5, n_iter=15)
best = estimator(opt.max["params"]).fit(a, ya)
models["贝叶斯LightGBM"] = best.predict_proba(b)[:, 1]
joblib.dump(best, c.out / "贝叶斯模型.joblib")
c.table(
    pd.DataFrame(
        [
            dict(
                **v["params"],
                CV_AUC=v["target"] if v["target"] > 0 else np.nan,
                有效=v["target"] > 0,
            )
            for v in opt.res
        ]
    ),
    "贝叶斯搜索记录",
)
if failures:
    c.table(pd.DataFrame(failures), "无效候选原因")
c.notes.append(
    f"LightGBM叶子数限制为2的树深次方，最小Hessian下限0.001以避免零值分裂错误。无效候选{len(failures)}个，失败候选不作为有效AUC或最终模型。"
)
results = evaluate(c, yb, models, "独立测试比较")
c.table(results, "模型比较")
# 复制采样仅复制真实行，不合成客户；在固定模型下独立比较。
from imblearn.over_sampling import RandomOverSampler

resampler = RandomOverSampler(sampling_strategy=0.2, random_state=0)
ar, yr = resampler.fit_resample(aa, ya)
np.testing.assert_allclose(ar.to_numpy(), aa.iloc[resampler.sample_indices_].to_numpy())
over = GradientBoostingClassifier(
    n_estimators=80, learning_rate=0.2, max_depth=2, random_state=42
).fit(ar, yr)
plain = GradientBoostingClassifier(
    n_estimators=80, learning_rate=0.2, max_depth=2, random_state=42
).fit(aa, ya)
evaluate(
    c,
    yb,
    {
        "原始训练": plain.predict_proba(bb)[:, 1],
        "真实行重复抽样": over.predict_proba(bb)[:, 1],
    },
    "重复抽样对照",
)
c.notes += [
    f"重建{len(net)}个申请人网络特征；不使用教材仅16行的不完整中间网络CSV。目标联系人图有{len(focus)}节点；社群编号由本次计算确定，不硬编码旧编号。",
    "网络每日只加入截至该日关系；风险占比只使用此前已发生事件，排除自身及同日未知事件。图示网络按日批处理，不代表日内实时顺序。随机分区用于教材对照，不替代未来时间外验证。",
    "异常特征的编码、收入回归和标准化在训练折内拟合。贝叶斯搜索保持教材参数范围，执行5次初始+15次迭代（教材50次），5折AUC。RandomOverSampler仅重复真实训练行；跳过所有SMOTE合成样本。",
]
c.done()
