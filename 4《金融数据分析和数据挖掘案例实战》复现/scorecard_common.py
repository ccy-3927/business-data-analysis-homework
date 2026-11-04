"""A/B卡共享的训练分箱、WOE筛选和可核对评分。"""

from model_common import *
import toad
from sklearn.metrics import log_loss


def stepwise_logit(frame, y):
    """以伯努利对数似然计算AIC，逐次增加或删除一列，直至无改善。"""
    cache = {}
    trail = []

    def aic(cols):
        key = tuple(sorted(cols))
        if key not in cache:
            if not cols:
                pred = np.full(len(y), y.mean())
            else:
                model = LogisticRegression(penalty=None, max_iter=3000).fit(
                    frame[list(key)], y
                )
                pred = model.predict_proba(frame[list(key)])[:, 1]
            cache[key] = 2 * log_loss(y, pred, normalize=False) + 2 * (len(cols) + 1)
        return cache[key]

    selected = []
    current = aic(selected)
    while True:
        candidates = [selected + [k] for k in frame if k not in selected] + [
            [j for j in selected if j != k] for k in selected
        ]
        if not candidates:
            break
        scores = [aic(cols) for cols in candidates]
        idx = int(np.argmin(scores))
        if scores[idx] >= current - 1e-6:
            break
        selected = candidates[idx]
        current = scores[idx]
        trail.append(
            {"步骤": len(trail) + 1, "AIC": current, "特征": ",".join(selected)}
        )
    return frame[selected], pd.DataFrame(trail)


def scorecard(c, d, target, seed, corr=0.7, initial=True):
    y = d[target].astype(int)
    x = d.drop(columns=target).replace([np.inf, -np.inf], np.nan)
    a, b, ya, yb = train_test_split(x, y, test_size=0.33, random_state=seed, stratify=y)
    profile(c, d, "建模数据")
    a = toad.selection.drop_empty(a, threshold=0.9)
    a = toad.selection.drop_var(a, threshold=0)
    if initial:
        a = toad.selection.drop_iv(a, ya, threshold=0.02)
    b = b[a.columns]
    bins = toad.transform.Combiner()
    bins.fit(a, ya, method="dt", n_bins=5, min_samples=0.05)
    if "Age" in a:
        for name in ["年龄自动分箱", "年龄调整分箱"]:
            if name == "年龄调整分箱":
                bins.update({"Age": [36.5, 51.5, 57.5]})
            g = (
                pd.DataFrame({"箱": bins.transform(a, labels=True)["Age"], "违约": ya})
                .groupby("箱", observed=True)["违约"]
                .agg(["size", "mean"])
            )
            c.table(g, name)
            c.bar(g["mean"] * 100, name, "违约率（%）")
    woe = toad.transform.WOETransformer()
    aw = woe.fit_transform(bins.transform(a), ya)
    bw = woe.transform(bins.transform(b))
    aw = toad.selection.drop_corr(aw, ya, threshold=corr)
    # 二分类采用Logit AIC；原默认OLS不适用于违约概率模型。
    with threadpool_limits(1):
        selected, trace = stepwise_logit(aw, ya)
    c.table(trace, "Logit逐步AIC过程")
    if selected.shape[1] == 0:
        raise ValueError("逐步回归未保留特征")
    cols = selected.columns.tolist()
    bw = bw[cols]
    model = LogisticRegression(max_iter=3000).fit(selected, ya)
    evaluate(c, yb, {"WOE逻辑回归": model.predict_proba(bw)[:, 1]}, "独立测试评估")
    c.table(pd.Series(model.coef_[0], index=cols, name="系数"), "入模系数")
    c.bar(
        pd.Series(model.coef_[0], index=cols)
        .sort_values(key=abs, ascending=False)
        .head(20),
        "入模WOE系数",
        "Logit系数",
    )
    vif = toad.selection.VIF(selected)
    psi = toad.metrics.PSI(bw, selected)
    c.table(pd.DataFrame({"VIF": vif, "PSI_测试对训练": psi}), "共线性与稳定性")
    card = toad.ScoreCard(
        combiner=bins, transer=woe, base_score=600, base_odds=35, pdo=50, max_iter=3000
    )
    card.fit(selected, ya)
    # ScoreCard.predict需要原始特征，不能再次给WOE值做分箱。
    scores = card.predict(b[cols])
    p = card.model.predict_proba(bw)[:, 1]
    np.testing.assert_allclose(scores, card.proba_to_score(p), rtol=1e-7, atol=1e-5)
    c.table(card.export(to_frame=True), "评分卡分值表")
    c.table(
        pd.DataFrame({"真实违约": yb, "违约概率": p, "评分": scores}, index=yb.index),
        "测试客户评分",
    )
    perf = pd.DataFrame({"评分": scores, "违约": yb})
    perf["评分区间"] = pd.qcut(perf["评分"], 10, duplicates="drop")
    report = perf.groupby("评分区间", observed=True)["违约"].agg(
        ["size", "sum", "mean"]
    )
    c.table(report, "测试评分分层表现")
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in [0, 1]:
        ax.hist(
            scores[yb.to_numpy() == k],
            bins=25,
            density=True,
            histtype="step",
            linewidth=2,
            label=f"标签{k}",
        )
    ax.set(title="独立测试评分分布", xlabel="评分（越高风险越低）", ylabel="概率密度")
    ax.legend()
    c.figure(fig, "评分分布")
    joblib.dump(
        {"combiner": bins, "woe": woe, "features": cols, "model": model, "card": card},
        c.out / "完整评分卡.joblib",
    )
    c.notes.append(
        f"训练{len(a)}、测试{len(b)}，入模{len(cols)}个特征。分箱、IV、相关性和Logit AIC逐步筛选只拟合训练集。评分基准600分、好坏比35:1、PDO=50，已核对逐项分数之和与概率刻度一致。PSI是随机测试与训练的分布比较，不代表跨期监控结论。"
    )
