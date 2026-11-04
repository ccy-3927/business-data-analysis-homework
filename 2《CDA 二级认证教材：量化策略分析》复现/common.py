"""各章共用的数据读取、来源登记与图表输出。只读取教材文件，不生成样本。"""

from pathlib import Path
import os, json, hashlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    os.environ.get(
        "CDA2_SOURCE",
        str(ROOT.parent.parent / "CDA 教材资料2025" / "2《CDA 二级认证教材：量化策略分析》"),
    )
)
plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=COLORS)
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.titlepad": 20,
        "axes.labelcolor": "#526477",
        "text.color": "#19344A",
        "xtick.color": "#526477",
        "ytick.color": "#526477",
        "axes.edgecolor": "#D8E1E8",
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "grid.color": "#E9EEF3",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.dpi": 220,
    }
)


def find(pattern):
    hits = sorted(SOURCE.rglob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def csv(path, **kw):
    for enc in ["utf-8-sig", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"无法解码 {path}")


def excel(path, sheet=0, **kw):
    d = pd.read_excel(path, sheet_name=sheet, **kw)
    d = d.dropna(how="all").dropna(axis=1, how="all")
    d.columns = [str(c).strip() for c in d.columns]
    return d


class Case:
    def __init__(self, script, title, description):
        self.out = Path(script).resolve().parent / "输出" / Path(script).stem
        self.out.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.notes = [description]
        self.sources = []

    def source(self, p):
        p = Path(p)
        self.sources.append(
            {
                "文件": str(p.relative_to(SOURCE)),
                "SHA256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
        return p

    def table(self, d, name):
        if isinstance(d, pd.Series):
            d = d.to_frame()
        d.to_csv(self.out / (name + ".csv"), encoding="utf-8-sig", index=True)

    def figure(self, fig, name):
        fig.tight_layout()
        fig.savefig(self.out / (name + ".png"), bbox_inches="tight")
        (self.out / (name + ".svg")).unlink(missing_ok=True)
        plt.close(fig)

    def interactive(self, fig, name, static_names=()):
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Microsoft YaHei, sans-serif", size=14, color="#19344A"),
            title=dict(text=name, x=0.05, font=dict(size=23)),
            colorway=COLORS,
            margin=dict(t=100, b=75, l=90, r=60),
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=620,
            legend=dict(orientation="h", y=1.12, x=0),
            hoverlabel=dict(bgcolor="white"),
            transition_duration=400,
        )
        # 同目录共享本地 JS，离线打开；不需要服务或外部 CDN。
        fig.write_html(
            str(self.out / (name + "_交互.html")),
            include_plotlyjs="directory",
            full_html=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "toImageButtonOptions": {"format": "svg"},
            },
        )
        # HTML成功写入后移除同图静态版本，每张图仅保留一种格式。
        for static_name in (name, *static_names):
            for suffix in (".png", ".svg"):
                (self.out / (static_name + suffix)).unlink(missing_ok=True)

    def bar(self, s, name, xlabel="数值", interactive=False):
        s = s.dropna()
        fig, ax = plt.subplots(figsize=(10, max(4, len(s) * 0.32)))
        bars = ax.barh(
            s.index.astype(str),
            s.values,
            height=0.62,
            color=[COLORS[0] if i == 0 else "#82B5CC" for i in range(len(s))],
        )
        ax.bar_label(
            bars,
            labels=[
                f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}" for v in s.values
            ],
            padding=7,
            color="#19344A",
            fontsize=10,
        )
        ax.invert_yaxis()
        ax.margins(x=0.18)
        ax.grid(axis="x", alpha=0.8)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set(title=name, xlabel=xlabel)
        if interactive:
            plt.close(fig)
        else:
            self.figure(fig, name)
        if interactive:
            f = go.Figure(
                go.Bar(
                    x=s.values,
                    y=s.index.astype(str),
                    orientation="h",
                    marker_color=COLORS[0],
                    text=s.values,
                    texttemplate="%{text:,.2f}",
                    textposition="outside",
                    hovertemplate="%{y}<br>%{x:,.2f}<extra></extra>",
                )
            )
            f.update_yaxes(autorange="reversed")
            f.update_xaxes(title=xlabel)
            self.interactive(f, name)

    def line(self, d, name, ylabel="数值", interactive=False):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        d.plot(ax=ax, marker="o", markersize=3)
        for i, line in enumerate(ax.lines):
            line.set_linestyle(["-", "--", "-.", ":"][i % 4])
        if pd.api.types.is_integer_dtype(d.index.dtype):
            ax.set_xticks(d.index)
        ax.grid(axis="y", alpha=0.8)
        ax.spines[["left", "bottom"]].set_visible(False)
        if isinstance(d, pd.DataFrame):
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title=d.columns.name)
        ax.set(title=name, ylabel=ylabel)
        if interactive:
            plt.close(fig)
        else:
            self.figure(fig, name)
        if interactive:
            frame = d.to_frame() if isinstance(d, pd.Series) else d
            f = go.Figure()
            for i, col in enumerate(frame):
                f.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=frame[col],
                        name=str(col),
                        mode="lines+markers",
                        line=dict(
                            width=3, dash=["solid", "dash", "dot", "dashdot"][i % 4]
                        ),
                    )
                )
            f.update_layout(hovermode="x unified")
            f.update_yaxes(title=ylabel)
            f.update_xaxes(title=d.index.name or "")
            if pd.api.types.is_integer_dtype(d.index.dtype):
                f.update_xaxes(tickmode="array", tickvals=d.index)
            self.interactive(f, name)

    def heat(self, d, name):
        fig, ax = plt.subplots(
            figsize=(max(8, min(16, len(d.columns) * 0.6)), max(4, len(d) * 0.35))
        )
        im = ax.imshow(d.to_numpy(dtype=float), aspect="auto", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(
            range(len(d.columns)), d.columns.astype(str), rotation=45, ha="right"
        )
        ax.set_yticks(range(len(d)), d.index.astype(str))
        ax.set_title(name)
        self.figure(fig, name)

    def done(self):
        (self.out / "数据来源.json").write_text(
            json.dumps(self.sources, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.out / "分析说明.md").write_text(
            "# " + self.title + "\n\n" + "\n\n".join(self.notes) + "\n",
            encoding="utf-8",
        )
        print("完成：" + self.title)


def profile(c, d, name):
    c.table(
        pd.DataFrame(
            {
                "类型": d.dtypes.astype(str),
                "非空数": d.count(),
                "缺失数": d.isna().sum(),
                "唯一值数": d.nunique(),
            }
        ),
        name + "_数据质量",
    )
    n = d.select_dtypes("number")
    if len(n.columns):
        c.table(n.describe().T, name + "_描述统计")
