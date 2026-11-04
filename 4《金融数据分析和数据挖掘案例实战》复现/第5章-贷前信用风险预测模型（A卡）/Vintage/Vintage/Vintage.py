"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

import re, csv as csvlib, io

c = start(
    __file__,
    "Vintage账龄分析",
    "只读解析教材SQL中的INSERT记录，在Python中重现六期贷款的月度Vintage。",
)


def read_dump(name):
    text = c.source(c.notebook.parent / "sql" / name).read_text(encoding="utf-8-sig")
    ddl = text.split("CREATE TABLE", 1)[1].split("(", 1)[1].split(") ENGINE", 1)[0]
    cols = re.findall(r"^\s*`([^`]+)`", ddl, re.M)
    rows = []
    for line in text.splitlines():
        if line.startswith("INSERT INTO"):
            v = line.split(" VALUES ", 1)[1].strip().removesuffix(";")
            assert v.startswith("(") and v.endswith(")")
            rows.append(
                next(
                    csvlib.reader(
                        io.StringIO(v[1:-1]),
                        delimiter=",",
                        quotechar="'",
                        escapechar="\\",
                        skipinitialspace=True,
                    )
                )
            )
    assert all(len(row) == len(cols) for row in rows)
    return pd.DataFrame(rows, columns=cols).replace("NULL", np.nan)


cust = read_dump("customer_detail.sql")
repay = read_dump("repayment_sum_month.sql")
for col in ["合同期限", "合同金额"]:
    cust[col] = pd.to_numeric(cust[col])
for col in ["分期数", "剩余本金", "当前最大逾期天数"]:
    repay[col] = pd.to_numeric(repay[col])
cust["放款月"] = cust["放款日期"].str[:7]
cust = cust[(cust["合同期限"] == 6) & cust["放款月"].between("2017-11", "2018-04")]
den = cust.groupby("放款月")["合同金额"].sum()
repay = repay[repay["分期数"].eq(6)].copy()
rp = pd.to_datetime(repay["观测月"])
orig = pd.to_datetime(repay["放款月"])
repay["MOB"] = (rp.dt.year - orig.dt.year) * 12 + rp.dt.month - orig.dt.month
repay["逾期剩余本金"] = repay["剩余本金"].where(repay["当前最大逾期天数"] > 0, 0)
num = repay.groupby(["放款月", "MOB"])["逾期剩余本金"].sum().unstack()
vintage = num.reindex(index=den.index, columns=range(1, 7)).div(den, axis=0)
vintage.columns = ["MOB" + str(i) for i in vintage.columns]
c.table(vintage, "Vintage逾期本金率")
c.table(den.rename("放款金额元"), "Vintage分母")
heat(c, vintage * 100, "Vintage逾期本金率（%）")
f = go.Figure()
for cohort, row in vintage.iterrows():
    f.add_trace(
        go.Scatter(
            x=list(range(1, 7)),
            y=row * 100,
            name=cohort,
            mode="lines+markers",
            connectgaps=False,
        )
    )
f.update_xaxes(title="放款后月数 MOB", dtick=1)
f.update_yaxes(title="逾期剩余本金 / 原始放款金额（%）")
c.interactive(f, "Vintage账龄曲线")
c.notes.append(
    "分子为当期逾期贷款的剩余本金，分母为同放款月六期合同原始放款额；不是逾期人数率或累计违约率。未观测账龄保留空值，不补零。SQL仅作为文本解析，未执行DROP或连接数据库。重复checkpoint不另做一份。"
)
c.done()
