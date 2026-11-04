"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import adjusted_rand_score

c = start(
    __file__,
    "交易异常识别",
    "复现孤立森林与LOF。使用既有真实记录，标签仅供独立评估，异常不直接等同欺诈。",
)
d = data(c, "creditcard.csv")
x = d.loc[:, "V1":"V28"]
y = d.Class
a, b, ya, yb = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
# 精确LOF在28万行上需巨量距离计算，训练上限20000真实行；测试保留完整30%。
fit = a.sample(min(20000, len(a)), random_state=42)
iso = IsolationForest(
    contamination=0.02,
    n_estimators=100,
    max_samples=0.6,
    max_features=0.6,
    random_state=42,
    n_jobs=1,
).fit(fit)
with threadpool_limits(1):
    lof = LocalOutlierFactor(
        n_neighbors=500, contamination=0.02, novelty=True, n_jobs=1
    ).fit(fit)
    scores = {"孤立森林": -iso.score_samples(b), "LOF": -lof.score_samples(b)}
    labels = {
        "孤立森林": (iso.predict(b) == -1).astype(int),
        "LOF": (lof.predict(b) == -1).astype(int),
    }
evaluate(c, yb, scores, "独立测试异常识别", labels=labels)
c.table(
    pd.Series(
        {name: adjusted_rand_score(yb, pred) for name, pred in labels.items()},
        name="ARI",
    ),
    "异常标签ARI",
)
c.table(pd.DataFrame({"教材原行号": fit.index}), "真实训练行索引")
c.bar(y.value_counts().sort_index(), "全量交易类别人数", "人数")
c.bar(x.kurtosis().sort_values(ascending=False), "原特征峰度", "峰度")
# 教材以标签选参应明确叫半监督选择；内部验证，不碰最终测试集。
aa, v, yaa, yv = train_test_split(a, ya, test_size=0.2, stratify=ya, random_state=17)
aa = aa.sample(min(10000, len(aa)), random_state=42)
rows = []
for contamination in [0.015, 0.02, 0.025]:
    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        max_samples=0.6,
        max_features=0.6,
        random_state=42,
        n_jobs=1,
    ).fit(aa)
    rows.append(
        {
            "contamination": contamination,
            "内部验证ARI": adjusted_rand_score(yv, model.predict(v)),
        }
    )
c.table(pd.DataFrame(rows), "训练内部标签辅助选参")
c.notes += [
    f"全量{len(d)}条，独立测试{len(b)}条。LOF和孤立森林共同在{len(fit)}条真实训练记录上拟合，上限为计算资源调整，不冒称原全量LOF结果。",
    "novelty=True只对未参与拟合的测试记录预测，修正原代码在训练样本调用novelty预测。孤立森林并不会自动按峰度抽取特征；contamination=0.02是教学报警比例，不是实际欺诈率。",
    "标签辅助选参表仅使用内部验证数据，因此不称纯无监督调优；主测试比较保持教材固定0.02报警口径。",
]
c.done()
