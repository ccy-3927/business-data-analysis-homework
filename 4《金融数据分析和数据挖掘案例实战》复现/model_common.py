"""建模共用方法：训练集内预处理、评分和曲线输出。"""

from analysis_common import *
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import joblib


def preprocess(x, onehot=False, scale=False):
    cats = x.select_dtypes(include=["object", "category"]).columns.tolist()
    nums = [col for col in x if col not in cats]
    cat = Pipeline(
        [
            ("fill", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                (
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    if onehot
                    else OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                ),
            ),
        ]
    )
    num = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler() if scale else "passthrough"),
        ]
    )
    return ColumnTransformer(
        [("cat", cat, cats), ("num", num, nums)], verbose_feature_names_out=False
    )


def insurance(c):
    d = data(c, "保险数据_第一期.csv")
    x = d.drop(columns="resp_flag")
    y = d.resp_flag
    a, b, ya, yb = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    c.notes.append(
        f"真实数据{len(d)}条；训练{len(a)}条、测试{len(b)}条。缺失值和类别编码只在训练数据拟合，测试集不参与调参。"
    )
    return a, b, ya, yb


def evaluate(c, y, probs, name, labels=None):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    predtable = pd.DataFrame({"真实标签": y}, index=y.index)
    for i, (key, p) in enumerate(probs.items()):
        p = np.asarray(p)
        pred = (
            np.asarray(labels[key])
            if labels and key in labels
            else (p >= 0.5).astype(int)
        )
        fpr, tpr, th = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        rows.append(
            {
                "模型": key,
                "AUC": roc_auc_score(y, p),
                "AP": average_precision_score(y, p),
                "准确率": accuracy_score(y, pred),
                "precision_1": precision_score(y, pred, zero_division=0),
                "recall_1": recall_score(y, pred, zero_division=0),
                "F1_1": f1_score(y, pred, zero_division=0),
                "KS": max(tpr - fpr),
            }
        )
        predtable[key + "_得分"] = p
        predtable[key + "_标签"] = pred
        style = ["-", "--", ":", "-."][i % 4]
        axes[0].plot(fpr, tpr, ls=style, label=f'{key} {rows[-1]["AUC"]:.3f}')
        axes[1].plot(recall, precision, ls=style, label=key)
        ok = np.isfinite(th)
        axes[2].plot(th[ok], (tpr - fpr)[ok], ls=style, label=key)
    axes[0].plot([0, 1], [0, 1], color="gray", ls=":")
    axes[1].axhline(y.mean(), color="gray", ls=":")
    for ax, title, xlab, ylab in zip(
        axes,
        ["ROC", "PR", "KS随得分阈值"],
        ["假阳性率", "召回率", "阈值"],
        ["真阳性率", "精确率", "TPR−FPR"],
    ):
        ax.set(title=title, xlabel=xlab, ylabel=ylab)
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8, loc="best")
    c.figure(fig, name)
    c.table(pd.DataFrame(rows), name + "_指标")
    c.table(predtable, name + "_明细")
    return pd.DataFrame(rows)


def search(c, estimator, grid, x, y, name, score="roc_auc", cv=3):
    model = GridSearchCV(
        estimator,
        grid,
        scoring=score,
        cv=StratifiedKFold(cv, shuffle=True, random_state=42),
        n_jobs=1,
        error_score="raise",
    )
    with threadpool_limits(1):
        model.fit(x, y)
    tab = pd.DataFrame(model.cv_results_)
    c.table(
        tab[["params", "mean_test_score", "std_test_score", "rank_test_score"]],
        name + "_搜索结果",
    )
    c.notes.append(
        f"{name}：训练集{cv}折{score}最佳值{model.best_score_:.4f}；参数{model.best_params_}。"
    )
    return model.best_estimator_


def treeplot(c, model, features, title):
    from sklearn.tree import plot_tree, export_text

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(
        model,
        feature_names=list(features),
        max_depth=3,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(title + "（最多显示3层）")
    c.figure(fig, title)
    (c.out / (title + "_规则.txt")).write_text(
        export_text(model, feature_names=list(features)), encoding="utf-8"
    )


def forecast_plot(c, observed, forecast, title, lower=None, upper=None):
    f = go.Figure(
        go.Scatter(x=observed.index, y=observed.values, name="观测", mode="lines")
    )
    if lower is not None:
        f.add_trace(
            go.Scatter(
                x=forecast.index,
                y=upper,
                mode="lines",
                line={"width": 0},
                showlegend=False,
            )
        )
        f.add_trace(
            go.Scatter(
                x=forecast.index,
                y=lower,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(0,114,178,.15)",
                name="95%模型预测区间",
            )
        )
    f.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name="预测",
            mode="lines",
            line={"dash": "dash"},
        )
    )
    c.interactive(f, title)
