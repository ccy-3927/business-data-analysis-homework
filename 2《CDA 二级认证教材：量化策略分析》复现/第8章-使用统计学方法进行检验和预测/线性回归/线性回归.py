"""教材Notebook的可运行迁移版；原单元格及变化见同目录对照文档。"""

from pathlib import Path
import sys

for folder in Path(__file__).resolve().parents:
    if (folder / "analysis_common.py").exists():
        sys.path.insert(0, str(folder))
        break
from analysis_common import *

from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

c = start(
    __file__,
    "信用卡消费线性回归",
    "复现清洗、相关、五个逐步规格、共线性与残差诊断。缺失目标值的教材填补口径另作对照。",
)
raw = data(c, "LR_practice.xlsx")
c.table(
    pd.read_excel(c.notebook.parent / "LR_practice.xlsx", sheet_name="数据字典"),
    "教材数据字典",
)
profile(c, raw, "原始数据")
d = raw.drop(columns=["id", "Acc", "edad2"]).drop_duplicates().copy()
d["avg_exp"] = pd.to_numeric(d.avg_exp, errors="coerce")
d["Age原值"] = d.Age
d["Age"] = d.Age.astype(float)
invalid = d.Age.eq(999)
age_mean = d.loc[~invalid, "Age"].mean()
d.loc[invalid, "Age"] = age_mean
d["年龄已填补"] = invalid
d["gender"] = d.gender.map({"Male": 1, "Female": 0})
c.table(d, "清洗数据_目标未填补")
obs = d.dropna().copy()
assert obs.avg_exp.gt(0).all()
obs["ln_avg_exp"] = np.log(obs.avg_exp)
full = "gender+Age+Income+Ownrent+Selfempl+dist_home_val+dist_avg_income+C(edu_class)"
reduced = full.replace("+dist_avg_income", "")
formulas = [
    f"avg_exp~{full}",
    f"avg_exp~{reduced}",
    f"ln_avg_exp~{reduced}",
    f"ln_avg_exp~{reduced}+I(Age**2)",
    f"ln_avg_exp~{reduced}+I(Age**2)+Age:gender",
]
summaries = []
for i, formula in enumerate(formulas, 1):
    m = ols(formula, obs).fit()
    model_output(c, m, f"模型{i}")
    bp = het_breuschpagan(m.resid, m.model.exog)
    summaries.append(
        {
            "模型": i,
            "目标尺度": "金额" if i <= 2 else "对数金额",
            "样本数": m.nobs,
            "R2": m.rsquared,
            "调整R2": m.rsquared_adj,
            "AIC_仅同尺度可比": m.aic,
            "BP检验p值": bp[1],
        }
    )
    c.table(
        pd.DataFrame(
            {"真实目标": m.model.endog, "样本内拟合": m.fittedvalues, "残差": m.resid},
            index=obs.index,
        ),
        f"模型{i}_样本内拟合",
    )
    if i in [2, 3]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(m.fittedvalues, m.resid, s=24, alpha=0.7)
        axes[0].axhline(0, color="#777777", ls="--")
        axes[0].set(title=f"模型{i}残差", xlabel="样本内拟合值", ylabel="残差")
        stats.probplot(m.resid, plot=axes[1])
        axes[1].set_title(f"模型{i}残差Q-Q图")
        c.figure(fig, f"模型{i}残差诊断")
    if i == 1:
        exog = m.model.exog
        c.table(
            pd.Series(
                {
                    name: variance_inflation_factor(exog, j)
                    for j, name in enumerate(m.model.exog_names)
                    if name != "Intercept"
                },
                name="VIF",
            ),
            "实际设计矩阵VIF",
        )
c.table(pd.DataFrame(summaries), "模型比较_不同尺度不可直接比AIC")
legacy = d.copy()
legacy["avg_exp"] = legacy.avg_exp.fillna(legacy.avg_exp.mean())
legacy = legacy.dropna()
m = ols(formulas[0], legacy).fit()
model_output(c, m, "教材目标均值填补_仅对照")
scatter(c, obs.Income, obs.avg_exp, "收入与消费金额")
box(c, obs.assign(样本="有观测值"), "样本", "dist_avg_income", "地区收入分布")
heat(
    c,
    obs[["avg_exp", "Age", "Income", "dist_home_val", "dist_avg_income"]].corr(),
    "连续变量相关矩阵",
    True,
)
c.notes += [
    f"原{len(raw)}行，删除无用列后去重为{len(d)}行；主模型{len(obs)}行，保留目标真实观测。识别到{int(invalid.sum())}个999年龄，用其余年龄均值{age_mean:.4f}替代并留标记，不再按固定行号40修补。",
    "教育用C(edu_class)虚拟变量，避免以unique()出现顺序赋予教育数值距离。五个模型分别为全变量、剔除地区收入、对数目标、年龄平方、年龄与性别交互。VIF使用真正的模型设计矩阵。",
    "教材缺失消费均值填补模型单独保存，不能把填补目标作为真实评价样本。这里是样本内教学拟合，不声称已验证样本外预测能力。原金额与对数金额模型的AIC/R²不能直接比较。",
]
c.done()
