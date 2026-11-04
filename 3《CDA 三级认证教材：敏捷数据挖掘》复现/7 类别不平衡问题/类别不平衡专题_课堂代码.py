"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from xgboost import XGBClassifier
from imblearn.under_sampling import (
    RandomUnderSampler,
    TomekLinks,
    EditedNearestNeighbours,
)

c = start(
    __file__,
    "类别不平衡真实案例",
    "不生成make_blobs样本，也不使用SMOTE/SMOTENC等合成记录；复现真实银行营销数据上的欠采样与损失加权。",
)
d = data(c, "train_set.csv").dropna().copy()
x = d.drop(columns=["ID", "y"])
y = d.y
a, b, ya, yb = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
prep = preprocess(a, onehot=True, scale=True)
xa = prep.fit_transform(a)
xb = prep.transform(b)
probs = {}
counts = []
samplers = {
    "原始": None,
    "随机欠采样": RandomUnderSampler(sampling_strategy=0.3, random_state=42),
    "TomekLinks": TomekLinks(n_jobs=1),
    "ENN": EditedNearestNeighbours(n_neighbors=20, n_jobs=1),
}
for name, sampler in samplers.items():
    with threadpool_limits(1):
        xx, yy = (xa, ya) if sampler is None else sampler.fit_resample(xa, ya)
    if sampler is not None:
        selected = sampler.sample_indices_
        np.testing.assert_allclose(xx, xa[selected])
        c.table(
            pd.DataFrame({"原教材ID": d.loc[a.index[selected], "ID"].to_numpy()}),
            name + "_保留真实记录ID",
        )
    model = XGBClassifier(
        n_estimators=100, max_depth=4, n_jobs=1, random_state=42, eval_metric="logloss"
    ).fit(xx, yy)
    probs[name] = model.predict_proba(xb)[:, 1]
    counts.append(
        {
            "处理": name,
            "0类": int((np.asarray(yy) == 0).sum()),
            "1类": int((np.asarray(yy) == 1).sum()),
        }
    )
weight = float((ya == 0).sum() / (ya == 1).sum())
for name, params in [
    ("损失加权", {"scale_pos_weight": weight}),
    ("限制更新步长", {"max_delta_step": 0.6}),
]:
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        n_jobs=1,
        random_state=42,
        eval_metric="logloss",
        **params,
    ).fit(xa, ya)
    probs[name] = model.predict_proba(xb)[:, 1]
evaluate(c, yb, probs, "不平衡处理独立测试")
c.table(pd.DataFrame(counts), "训练类人数对照")
fig, ax = plt.subplots(figsize=(10, 5))
table = pd.DataFrame(counts).set_index("处理")
table.plot.bar(ax=ax, color=COLORS[:2])
ax.set(title="欠采样前后真实记录数", ylabel="人数", xlabel="处理方式")
ax.tick_params(axis="x", rotation=0)
c.figure(fig, "训练类人数对照")
c.notes += [
    f"完整记录{len(d)}条；测试集保持原分布{len(b)}条。类别权重={weight:.4f}，只改变训练损失，不复制或创造记录。欠采样只删除训练记录，已核对保留行与原数据一致。",
    "教材SMOTE、SMOTENC及联合合成采样因“不造数据”要求未执行；Tomek/ENN依赖编码后距离，采用OneHot和标准化减少任意类别编码的距离问题。duration是通话后才知道的字段，因此本教材模型不能用于通话前名单决策。",
]
c.done()
