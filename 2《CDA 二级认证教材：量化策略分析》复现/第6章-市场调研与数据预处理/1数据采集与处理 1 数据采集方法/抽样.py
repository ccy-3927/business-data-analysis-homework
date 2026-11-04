"""教材Notebook的可运行迁移版；原单元格及变化见同目录对照文档。"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *

import math


def get_sample(df, sampling="simple_random", k=1, stratified_col=None, seed=2025):
    """只抽取原有行；整数k按数量（分层时每层），0<k<1按比例向上取整。"""
    if k <= 0 or (k >= 1 and not isinstance(k, int)):
        raise ValueError("k须为正整数或0至1间比例")
    rng = np.random.default_rng(seed)

    def sample_group(g):
        n = k if k >= 1 else math.ceil(len(g) * k)
        if n > len(g):
            raise ValueError("抽样量超过总体或分层大小")
        return g.iloc[rng.choice(len(g), size=n, replace=False)]

    if sampling == "simple_random":
        return sample_group(df).copy()
    if sampling == "stratified":
        if not stratified_col:
            raise ValueError("须指定分层列")
        return pd.concat(
            [sample_group(g) for _, g in df.groupby(stratified_col, dropna=False)]
        ).copy()
    if sampling == "systematic":
        n = k if k >= 1 else math.ceil(len(df) * k)
        if n > len(df):
            raise ValueError("抽样量超过总体")
        step = len(df) / n
        start_at = rng.uniform(0, step)
        return df.iloc[np.floor(start_at + np.arange(n) * step).astype(int)].copy()
    raise ValueError("未知抽样方式")


c = start(
    __file__,
    "抽样方法",
    "复现简单随机、按地区分层、系统抽样。随机仅用于选择已有记录，不生成样本数值。",
)
d = data(c, "clients.csv")
c.source(c.notebook.parent / "6_1sampling.py")
profile(c, d, "客户数据")
rows = []
for method, k in [
    ("simple_random", 22),
    ("simple_random", 0.1),
    ("stratified", 2),
    ("stratified", 0.1),
    ("systematic", 4),
    ("systematic", 0.1),
]:
    a = get_sample(d, method, k, ["Dist"])
    assert a.index.is_unique and a.index.isin(d.index).all()
    pd.testing.assert_frame_equal(a, d.loc[a.index])
    c.table(a, f"{method}_{k}样本")
    rows.append(
        {
            "方法": method,
            "k": k,
            "总体人数": len(d),
            "抽样人数": len(a),
            "抽样比例": len(a) / len(d),
            "覆盖地区": a.Dist.nunique(),
        }
    )
c.table(pd.DataFrame(rows), "抽样汇总")
s = get_sample(d, "stratified", 0.1, ["Dist"])
compare = pd.DataFrame(
    {
        "总体": d.Dist.value_counts().sort_index(),
        "分层10%样本": s.Dist.value_counts().sort_index(),
    }
)
c.table(compare, "分层样本地区构成")
f = go.Figure(
    go.Scatter(
        x=compare["总体"],
        y=compare["分层10%样本"],
        mode="markers",
        text=compare.index.astype(str),
        hovertemplate="地区%{text}<br>总体%{x}<br>样本%{y}<extra></extra>",
    )
)
f.update_xaxes(title="地区总体人数")
f.update_yaxes(title="分层抽样人数")
c.interactive(f, "地区总体与分层样本数")
c.notes += [
    "修正原函数用is比较字符串、系统抽样步长导致样本数不等于k的问题；系统抽样采用随机起点的等距位置，保证请求数量。随机种子2025用于可重复选行。",
    "分层比例逐层向上取整，因此总样本比例可能略高于10%；整数k表示每层k条。clients.sav是配套SPSS格式，本实现使用同目录CSV，不额外依赖SPSS。",
]
c.done()
