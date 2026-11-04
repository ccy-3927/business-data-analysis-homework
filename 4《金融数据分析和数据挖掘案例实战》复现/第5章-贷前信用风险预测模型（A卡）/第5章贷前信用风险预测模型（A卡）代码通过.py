"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from scorecard_common import scorecard

c = start(
    __file__,
    "贷前申请评分卡A卡",
    "复现信用数据清洗、IV初筛、决策树分箱、WOE、相关性筛选、逐步Logit及分值表。",
)
d = data(c, "creditdata_small.csv").set_index("ID")
d = d.loc[~d.index.duplicated(), ~d.columns.duplicated()].copy()
cats = [
    "GENDER",
    "FLAG_OWN_CAR",
    "HOUSING_TYPE",
    "FLAG_OWN_REALTY",
    "EDULEVEL",
    "MARRIAGE",
]
for col in cats:
    d[col] = d[col].fillna("missing").astype(str)
d["DAYS_EMPLOYED"] = d.DAYS_EMPLOYED.mask(d.DAYS_EMPLOYED < 0)
scorecard(c, d, "TARGET", 1223, 0.7, True)
c.notes.append(
    "使用Notebook指定的creditdata_small.csv，未混用另一份特征口径不同的creditdata.csv。负就业天数按教材设为缺失。Vintage见原子目录中的独立脚本。"
)
c.done()
