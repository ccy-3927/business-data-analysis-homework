"""将已填写的指标设计模板转为结构化字典，不计算缺少数据的指标。"""

from pathlib import Path
import sys, json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

c = Case(
    __file__,
    "指标字典迁移",
    "原模板Step1为空白设计框架，Step2已有指标名称、计算规则、口径、维度；没有实际注册或贷款事件数据。",
)
p = c.source(find("1、指标体系设计模板.xlsx"))
d = excel(p, "Step2")
d = d[d["指标名称（KPI)"].notna()].copy()
d = d.rename(columns={d.columns[0]: "层级"})
d["层级"] = d["层级"].ffill()  # 只展开合并单元格语义，不填业务指标数值。
c.table(d, "教材指标字典")
(c.out / "教材指标字典.json").write_text(
    d.where(d.notna(), None).to_json(orient="records", force_ascii=False, indent=2),
    encoding="utf-8",
)
issues = d[["指标名称（KPI)", "关联指标"]].dropna().copy()
issues["关联名是否存在于字典"] = issues["关联指标"].isin(d["指标名称（KPI)"])
c.table(issues, "关联指标名称核对")
c.notes += [
    "“值”列的“人数/百分比/金额”是类型说明，不是观测数值。仅完成指标字典结构迁移，不画虚构注册转化漏斗。",
    "“通过用户数”“合同结清数”等关联名称未逐字出现在指标名称列中，输出核对表供理解，不自行补造指标记录或修正口径。",
]
c.done()
