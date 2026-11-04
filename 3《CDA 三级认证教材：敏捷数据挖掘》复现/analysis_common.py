"""三级教材通用方法。仅加载教材文件或Notebook内明确写出的固定示例。"""

from common import *
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats


def start(script, title, description):
    c = Case(script, title, description)
    rel = Path(script).resolve().relative_to(ROOT).with_suffix(".ipynb")
    c.notebook = c.source(SOURCE / rel)
    return c


def data(c, name, **kwargs):
    p = c.source(c.notebook.parent / name)
    return excel(p, **kwargs) if p.suffix == ".xlsx" else csv(p, **kwargs)


def literal(c, cell, var):
    """从已定位的单元格提取DataFrame固定字面量；拒绝执行随机代码。"""
    cells = json.loads(c.notebook.read_text(encoding="utf-8"))["cells"]
    tree = ast.parse("".join(cells[cell - 1]["source"]))

    class Nulls(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "np"
                and node.attr == "nan"
            ):
                return ast.Constant(None)
            return node

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == var for t in node.targets
        ):
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "DataFrame"
            ):
                return pd.DataFrame(ast.literal_eval(Nulls().visit(call.args[0])))
    raise ValueError(f"未找到固定数据：cell {cell} {var}")


def heat(c, d, title, correlation=False):
    fig, ax = plt.subplots(
        figsize=(max(8, len(d.columns) * 0.95), max(4, len(d) * 0.48))
    )
    values = d.to_numpy(dtype=float)
    signed = correlation or np.nanmin(values) < 0
    limit = 1 if correlation else float(np.nanmax(np.abs(values)))
    im = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r" if signed else "Blues",
        **({"vmin": -limit, "vmax": limit} if signed else {"vmin": 0}),
    )
    ax.set_xticks(range(len(d.columns)), d.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(range(len(d)), d.index.astype(str))
    ax.set_title(title)
    fig.colorbar(
        im,
        ax=ax,
        label="权重" if "权重" in title else ("相关系数" if correlation else "数值"),
    )
    if d.size <= 100:
        for i in range(len(d)):
            for j in range(len(d.columns)):
                v = d.iloc[i, j]
                rgba = im.cmap(im.norm(v))
                light = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                label = (
                    f"{v:,.0f}" if not signed and float(v).is_integer() else f"{v:.2f}"
                )
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if light < 0.5 else "#19344A",
                )
    c.figure(fig, title)


def hist(c, s, title, xlabel=None, kde=False):
    s = s.dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(s, bins=25, density=kde, color="#82B5CC", edgecolor="white")
    if kde:
        xx = np.linspace(s.min(), s.max(), 300)
        ax.plot(xx, stats.gaussian_kde(s)(xx), color=COLORS[0], lw=2)
    ax.set(title=title, xlabel=xlabel or s.name, ylabel="密度" if kde else "观测数")
    ax.grid(axis="y", alpha=0.6)
    c.figure(fig, title)


def box(c, d, group, value, title):
    groups = [(str(k), g[value].dropna()) for k, g in d.groupby(group, observed=True)]
    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 1.2), 5.5))
    for i, (name, s) in enumerate(groups, 1):
        if len(s) >= 10:
            ax.boxplot(
                s,
                positions=[i],
                widths=0.55,
                patch_artist=True,
                boxprops={"facecolor": "#82B5CC"},
                medianprops={"color": "#19344A"},
            )
        else:
            ax.scatter(np.full(len(s), i), s, color=COLORS[0], s=28)
    ax.set_xticks(range(1, len(groups) + 1), [f"{k}\nn={len(s)}" for k, s in groups])
    ax.set(title=title, ylabel=value)
    ax.grid(axis="y", alpha=0.5)
    if value.startswith("CNT_"):
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, float(d[value].max()) * 1.5)
        ax.set_ylabel(value + "（对称对数轴；0–1为线性）")
    fig.text(
        0.5,
        0.01,
        "箱体为四分位距，中线为中位数，须为1.5倍四分位距内；n<10仅列原始点",
        ha="center",
        fontsize=9,
    )
    c.figure(fig, title)


def scatter(c, x, y, title, labels=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.65, s=25, color=COLORS[0])
    if labels is not None:
        for i, (a, b, l) in enumerate(zip(x, y, labels)):
            ax.annotate(
                str(l),
                (a, b),
                xytext=(5, 7 if i % 2 == 0 else -14),
                textcoords="offset points",
                fontsize=10,
            )
    ax.set(title=title, xlabel=x.name, ylabel=y.name)
    ax.grid(alpha=0.4)
    c.figure(fig, title)


def rotated_components(x, n):
    """PCA单位权重正交varimax旋转；匹配教材fa_kit主成分提取路线，不冒称最大似然因子模型。"""
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    pca = PCA(svd_solver="full").fit(z)
    weights = pca.components_[:n].T
    r = np.eye(n)
    old = 0
    for _ in range(1000):
        a = weights @ r
        u, s, v = np.linalg.svd(
            weights.T @ (a**3 - a @ np.diag((a * a).sum(axis=0)) / len(weights))
        )
        r = u @ v
        value = s.sum()
        if old and abs(value - old) < 1e-10:
            break
        old = value
    w = weights @ r
    # 因子符号本就不唯一，最大绝对权重取正，使多次运行便于比较。
    w *= np.sign(w[np.abs(w).argmax(axis=0), np.arange(n)])
    np.testing.assert_allclose(w.T @ w, np.eye(n), atol=1e-8)
    return z @ w, w, pca, z


def pca_tables(c, x, pca):
    t = pd.DataFrame(
        {
            "特征值": pca.explained_variance_,
            "解释率": pca.explained_variance_ratio_,
            "累计解释率": np.cumsum(pca.explained_variance_ratio_),
        },
        index=pd.Index(range(1, len(pca.explained_variance_) + 1), name="主成分"),
    )
    c.table(t, "方差解释")
    c.line(t[["解释率", "累计解释率"]] * 100, "主成分解释比例", "%")
    heat(c, x.corr(), "原始变量相关矩阵", True)


def model_output(c, model, name):
    ci = model.conf_int()
    c.table(
        pd.DataFrame(
            {
                "系数": model.params,
                "标准误": model.bse,
                "p值": model.pvalues,
                "95%下限": ci.iloc[:, 0],
                "95%上限": ci.iloc[:, 1],
            }
        ),
        name + "_系数",
    )
    (c.out / (name + "_摘要.txt")).write_text(
        model.summary().as_text(), encoding="utf-8"
    )
