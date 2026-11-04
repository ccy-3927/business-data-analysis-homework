"""年度→月度→异常月份→省份→四川月度，复现教材的分析链路。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import openpyxl

c = Case(
    __file__,
    "收入趋势与异常下钻",
    "读取原始数据重新汇总；完整版仅用于核对。销售额沿用源表单位（未标明币种）。",
)
p = c.source(find("案例1：收入趋势分析原始数据.xlsx"))
ref = c.source(find("案例1：收入趋势分析案例-完整.xlsx"))
d = excel(p)
profile(c, d, "收入数据")
annual = d.groupby("订单年份")["销售额"].sum()
w = openpyxl.load_workbook(ref, read_only=True, data_only=True)
assert annual.tolist() == [w["准备区"].cell(r, 2).value for r in range(5, 9)]
c.table(annual, "年度销售额")
c.line(annual, "年度销售趋势", "销售额", True)
m = d.pivot_table(index="订单月份", columns="订单年份", values="销售额", aggfunc="sum")
c.table(m, "各年月度销售额")
c.line(m[[2019, 2020]], "2019与2020月度销售对比", "销售额", True)
yoy = pd.DataFrame(
    {
        "标准同比": (m[2020] - m[2019]) / m[2019],
        "教材原公式结果": (m[2020] - m[2019]) / m[2020],
    }
)
c.table(yoy, "同比口径差异")
c.line(yoy["标准同比"] * 100, "2020月度同比增长率", "同比（%）", True)
nov = d[d["订单月份"].eq(11)].pivot_table(
    index="销售省份", columns="订单年份", values="销售额", aggfunc="sum"
)[[2019, 2020]]
# 原表没有某省某月记录不等同于确认为零。原值留空；增减归因按记录汇总缺项作0，并显式标记。
nov["缺少某期记录"] = nov.isna().any(axis=1)
nov["增减额_缺项按零汇总"] = nov[2020].fillna(0) - nov[2019].fillna(0)
nov["同比_两期均有记录"] = (nov[2020] - nov[2019]) / nov[2019].replace(0, np.nan)
assert nov["增减额_缺项按零汇总"].sum() == m.loc[11, 2020] - m.loc[11, 2019]
c.table(nov, "11月各省下钻")
c.bar(
    nov["增减额_缺项按零汇总"].sort_values(),
    "11月各省销售增减额",
    "销售额变化（缺项按零汇总）",
    True,
)
sc = d[d["销售省份"].eq("四川")].pivot_table(
    index="订单月份", columns="订单年份", values="销售额", aggfunc="sum"
)[[2019, 2020]]
c.table(sc, "四川月度下钻")
c.line(sc, "四川月度销售对比", "销售额", True)
c.notes += [
    f'2020年销售额{annual.loc[2020]:,.0f}，较2019年增长{annual.pct_change().loc[2020]:.2%}；11月同比{yoy.loc[11,"标准同比"]:.2%}，应重点解释该月下滑。',
    "教材准备区 E44:E55 等公式用当期销售额作分母，不是标准同比。本实现输出两种结果供追溯，图表采用（本期－上期）/上期。省份缺项保持缺失标记，不能仅凭缺项断言停业或零销售。",
    "四川下钻沿用教材指定对象；省份差额是数值贡献，不足以证明价格、流量或运营变化等原因，原数据没有这些字段。",
]
c.done()
