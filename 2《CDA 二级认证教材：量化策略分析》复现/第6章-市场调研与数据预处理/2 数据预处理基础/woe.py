"""根据教材woe.py（原作者Denis Surzhko）的二元WOE思路实现兼容版实操。
保留频数、响应分布和WOE/IV定义；不执行原文件随机main示例。
本文件聚焦fit/transform，不冒称完整复刻原工具的merge/optimize API。
"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *


class BinaryWOE:
    def __init__(self, bins=5):
        self.bins = bins

    def labels(self, x):
        return (
            pd.cut(x, self.edges, include_lowest=True)
            .astype(str)
            .where(x.notna(), "缺失")
        )

    def fit(self, x, y):
        if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
            raise ValueError("要求0/1双类别目标")
        _, edges = pd.qcut(x.dropna(), self.bins, retbins=True, duplicates="drop")
        self.edges = np.r_[-np.inf, edges[1:-1], np.inf]
        q = pd.DataFrame({"箱": self.labels(x), "响应": y})
        tab = (
            q.groupby("箱")
            .响应.agg(["count", "sum"])
            .rename(columns={"count": "记录数", "sum": "响应1数"})
        )
        tab["响应0数"] = tab["记录数"] - tab["响应1数"]
        tab["实际响应率"] = tab["响应1数"] / tab["记录数"]
        # 0.5是估计平滑参数，不增加明细记录；原频数原样保留。
        p1 = (tab["响应1数"] + 0.5) / (tab["响应1数"].sum() + 0.5 * len(tab))
        p0 = (tab["响应0数"] + 0.5) / (tab["响应0数"].sum() + 0.5 * len(tab))
        tab["WOE"] = np.log(p1 / p0)
        tab["IV贡献"] = (p1 - p0) * tab["WOE"]
        self.table = tab
        return self

    def transform(self, x):
        return self.labels(x).map(self.table.WOE)


if __name__ == "__main__":
    c = Case(
        __file__,
        "营销收入WOE编码",
        "用教材营销收入和Suc_flag演示WOE；不生成随机演示样本。",
    )
    base = SOURCE / Path(__file__).resolve().relative_to(ROOT).parent
    c.source(base / "woe.py")
    d = csv(c.source(base / "teleco_camp_orig.csv"))
    x = d.AvgIncome.replace(0, np.nan)
    train = d.sample(frac=0.7, random_state=2025).index
    test = d.index.difference(train)
    model = BinaryWOE(5).fit(x.loc[train], d.loc[train, "Suc_flag"])
    c.table(model.table, "训练集WOE分箱")
    c.table(pd.Series(model.edges, name="边界"), "训练集分箱边界")
    out = d[["ID", "Suc_flag"]].copy()
    out["原收入"] = d.AvgIncome
    out["WOE"] = model.transform(x)
    out["分组"] = np.where(out.index.isin(train), "训练", "测试")
    assert len(out) == len(d) and out.WOE.notna().all()
    assert model.table["记录数"].sum() == len(train)
    c.table(out, "WOE编码结果")
    c.bar(model.table.WOE, "收入分箱WOE", "ln(响应1分布/响应0分布)")
    c.notes += [
        f'训练样本{len(train)}，测试样本{len(test)}；分箱和WOE全部只在训练集拟合。训练IV={model.table["IV贡献"].sum():.6f}。',
        "原工具用“坏/好”命名；本数据Suc_flag是营销响应，因此改为响应1/0，不能称为信用风险。使用每箱每类0.5平滑作为估计参数，原始人数不改写，不新增观测记录。原文件的随机main、连续目标扩展、自动合箱和树优化没有作为本次实操运行。",
    ]
    c.done()
