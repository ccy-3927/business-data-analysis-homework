"""第二章读取明确单元格区域，避免把工作表中的说明、透视结果混成样本。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import openpyxl

BASE = SOURCE / "第2章 数据分析方法"


def source(c, name):
    paths = sorted(BASE.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"数据路径需唯一：{name}: {paths}")
    return c.source(paths[0])


def block(p, sheet, cells, headers=None):
    w = openpyxl.load_workbook(p, read_only=True, data_only=True)
    min_col, min_row, max_col, max_row = openpyxl.utils.cell.range_boundaries(cells)
    values = list(
        w[sheet].iter_rows(
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
            values_only=True,
        )
    )
    w.close()
    if headers is None:
        headers, values = values[0], values[1:]
    return pd.DataFrame(values, columns=headers).dropna(how="all")


def safe_div(a, b):
    return (
        a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / b if b else np.nan
    )


def compare(c, left, right, name):
    """相同索引数值表比对；保留差异，不能悄悄用参考答案覆盖重算结果。"""
    a, b = left.align(right, join="outer")
    delta = a - b
    c.table(delta, name + "_差异")
    maxdiff = (
        delta.abs().max().max()
        if isinstance(delta, pd.DataFrame)
        else delta.abs().max()
    )
    c.notes.append(
        f"{name}：最大绝对差 {maxdiff:.10g}；仅比对共同有效数值，缺失/额外键另见差异表。"
    )


def newcase(script, title, desc):
    return Case(script, title, desc)


def finish_sub(c, name):
    c.out = c.out / name
    c.out.mkdir(exist_ok=True, parents=True)
    return c


def stacked(c, d, name, percent=False):
    shown = d.div(d.sum(axis=1), axis=0) * 100 if percent else d
    fig, ax = plt.subplots(figsize=(10, 5))
    shown.plot.bar(stacked=True, ax=ax, rot=0, width=0.65)
    ax.set(title=name, ylabel="占比（%）" if percent else "数量", xlabel="")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y")
    c.figure(fig, name)


def scatter(c, d, x, y, name, label=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(d[x], d[y], alpha=0.65, s=30, edgecolor="white", linewidth=0.5)
    ax.set(title=name, xlabel=x, ylabel=y)
    ax.grid(alpha=0.5)
    if label:
        for _, r in d.iterrows():
            ax.annotate(
                str(r[label]),
                (r[x], r[y]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=9,
            )
    c.figure(fig, name)
