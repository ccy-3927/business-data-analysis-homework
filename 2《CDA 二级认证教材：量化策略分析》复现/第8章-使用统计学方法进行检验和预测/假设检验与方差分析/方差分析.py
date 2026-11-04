"""教材Notebook的可运行迁移版；原单元格及变化见同目录对照文档。"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

c = start(
    __file__,
    "消费金额方差分析",
    "复现教育、性别和交互项分析；原稿所谓“单因素”实际含两个因素，分别输出真正单因素与原加性规格。",
)
d = data(c, "creditcard_exp.csv")
numeric = pd.to_numeric(d.avg_exp, errors="coerce")
clean = d.assign(avg_exp=numeric)[["avg_exp", "edu_class", "gender"]].dropna().copy()
for col in ["edu_class", "gender"]:
    clean[col] = clean[col].astype(str)
profile(c, clean, "有观测消费金额样本")
c.table(d.loc[numeric.isna()], "未观测消费金额记录")
specs = {
    "单因素教育": "avg_exp~C(edu_class)",
    "教育与性别加性": "avg_exp~C(edu_class)+C(gender)",
    "教育与性别交互": "avg_exp~C(edu_class)*C(gender)",
}
for label, formula in specs.items():
    m = ols(formula, clean).fit()
    model_output(c, m, label)
    c.table(anova_lm(m, typ=2), label + "_ANOVA")
    c.table(
        pd.Series(
            {
                "样本量": m.nobs,
                "残差自由度": m.df_resid,
                "残差Shapiro_p": stats.shapiro(m.resid).pvalue,
            }
        ),
        label + "_诊断",
    )
legacy = d[["avg_exp", "edu_class", "gender"]].dropna().copy()
legacy.avg_exp = pd.to_numeric(legacy.avg_exp.replace(" ", 0), errors="raise")
for col in ["edu_class", "gender"]:
    legacy[col] = legacy[col].astype(str)
m = ols(specs["教育与性别加性"], legacy).fit()
model_output(c, m, "教材空格当0口径_仅对照")
group = clean.groupby(["edu_class", "gender"]).avg_exp.agg(["size", "mean", "std"])
c.table(group, "分组描述")
box(c, clean, "edu_class", "avg_exp", "不同教育组消费金额")
box(c, clean, "gender", "avg_exp", "不同性别消费金额")
tests = stats.levene(
    *[g.avg_exp.to_numpy() for _, g in clean.groupby("edu_class")], center="median"
)
c.table(
    pd.Series({"Levene统计量": tests.statistic, "p值": tests.pvalue}), "方差齐性诊断"
)
c.notes += [
    f"原文件{len(d)}条，消费金额无法解析{numeric.isna().sum()}条。主分析仅使用{len(clean)}条真实可解析金额；原代码将空格写为0的模型单独保存供对照，未把缺失声称为零消费。",
    "ANOVA使用II型平方和；交互模型中主效应应结合交互项解释。正态性与方差齐性结果作为诊断，不将观察性组间差异解释为教育或性别的因果效应。",
]
c.done()
