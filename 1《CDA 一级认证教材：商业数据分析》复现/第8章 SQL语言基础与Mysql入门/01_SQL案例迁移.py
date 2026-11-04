"""用pandas复现SQL基础语法.sql的有数据查询，免安装数据库。"""

from pathlib import Path
import sys, sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

c = Case(
    __file__,
    "SQL案例迁移",
    "原CSV没有表头，按教材建表语句赋列名。只读原始文件，不连接或修改用户数据库。",
)
base = SOURCE / "第8章 SQL语言基础与Mysql入门"
data = base / "1、Windows系统下将数据导入到MySQL的三种方法" / "方式1" / "mysql_csv"
c.source(base / "SQL基础语法.sql")
c.source(data.parent / "1 create_table.sql")


def read(name, cols):
    hits = [p for p in data.glob("*.csv") if p.stem.lower() == name.lower()]
    assert len(hits) == 1
    return csv(c.source(hits[0]), header=None, names=cols.split(), low_memory=False)


sale = read("sale", "year market sale profit")
trans = read(
    "kdd99_trans",
    "trans_id account_id date type operation amount balance k_symbol bank account",
)
loans = read("kdd99_loans", "loan_id account_id date amount duration payments status")
card = read("kdd99_card", "card_id disp_id issued type")
disp = read("kdd99_disp", "disp_id client_id account_id type")
clients = read("kdd99_clients", "client_id sex birth_date district_id")
for frame, col in [
    (trans, "date"),
    (loans, "date"),
    (card, "issued"),
    (clients, "birth_date"),
]:
    frame[col] = pd.to_datetime(frame[col])
c.table(sale["year"].drop_duplicates(), "01_DISTINCT年份")
c.table(sale[sale["year"].eq(2012)], "02_WHERE筛选2012")
c.table(sale.sort_values("year"), "03_ORDER排序")
rate = sale.assign(rate=sale["profit"] / sale["sale"].replace(0, np.nan))
c.table(rate, "04_表达式利润率")
c.table(rate[rate["rate"].lt(0.07)], "05_计算值筛选")
q = (
    trans[
        trans["date"].between("1998-12-01", "1998-12-31")
        & trans["k_symbol"].eq("房屋贷款")
    ]
    .sort_values(["date", "account_id"])
    .copy()
)
q["weekday"] = q["date"].dt.weekday
q["Weekday_2"] = np.where(q["weekday"].ge(5), "周末", "工作日")
c.table(
    q[["account_id", "date", "bank", "amount", "weekday", "Weekday_2"]],
    "06_房贷流水与CASE重编码",
)
daily = trans.groupby("date").agg(
    COUNT_of_account_id=("account_id", "count"),
    MEAN_of_amount=("amount", "mean"),
    SUM_of_amount=("amount", "sum"),
)
c.table(daily[daily["COUNT_of_account_id"].gt(300)], "07_GROUP_BY与HAVING")
one = read("One", "id name")
two = read("Two", "id name")
t1 = read("Table1", "id name")
t2 = read("Table2", "id name")
for how in ["cross", "inner", "left", "right", "outer"]:
    c.table(
        t1.merge(
            t2,
            how=how,
            **({} if how == "cross" else {"on": "id"}),
            suffixes=("_1", "_2"),
        ),
        "08_JOIN_" + how,
    )
joined = card.merge(
    disp[["disp_id", "client_id"]], on="disp_id", how="left", validate="many_to_one"
).merge(clients, on="client_id", how="left", validate="many_to_one")
assert len(joined) == len(card)
c.table(joined, "09_信用卡客户关联")
# 用真正的日期差实现教材“贷款前一年”的业务意图，避免MySQL date+365的数值加法歧义。
lt = loans.merge(trans, on="account_id", suffixes=("_loan", "_trans"))
delta = (lt["date_loan"] - lt["date_trans"]).dt.days
prior = lt[delta.between(1, 365)].copy()
missing = loans[~loans["loan_id"].isin(prior["loan_id"])]
# LEFT JOIN语义：没有匹配流水的贷款仍保留一行。
none = missing.rename(
    columns={x: x + "_loan" for x in ["date", "amount", "type"] if x in missing}
)
prior = pd.concat([prior, none], ignore_index=True)
c.table(prior.sort_values(["loan_id", "date_trans"]), "10_贷款前一年非等值连接")
bad = loans[loans["status"].eq("B")]
c.table(
    trans[trans["account_id"].isin(bad["account_id"])].sort_values("account_id"),
    "11_违约账户全部流水",
)
badprior = trans.merge(
    bad[["account_id", "date"]].rename(columns={"date": "loan_date"}), on="account_id"
)
badprior = badprior[badprior["date"].lt(badprior["loan_date"])].drop_duplicates(
    "trans_id"
)
c.table(badprior, "12_违约账户贷款前流水")
union = pd.concat([one, two], ignore_index=True)
c.table(union, "13_UNION_ALL")
c.table(union.drop_duplicates(), "14_UNION")
c.table(sale[sale["year"].isin([2010, 2011])], "15_创建表结果")
# 在内存SQLite中验证教材简单查询对应的pandas结果；无磁盘数据库副作用。
con = sqlite3.connect(":memory:")
sale.to_sql("sale", con, index=False)
con.execute("CREATE VIEW aa AS SELECT * FROM sale")
v = pd.read_sql_query("SELECT * FROM aa", con)
pd.testing.assert_frame_equal(v, sale)
c.table(v, "16_视图结果")
expected = pd.read_sql_query(
    "SELECT year,SUM(sale) AS sale FROM sale GROUP BY year ORDER BY year", con
)
actual = sale.groupby("year", as_index=False)["sale"].sum()
pd.testing.assert_frame_equal(actual, expected)
con.close()
c.line(
    sale.pivot(index="year", columns="market", values="sale"),
    "销售年份与市场",
    "销售额",
    True,
)
s = joined.groupby(["type", "sex"]).size().rename("卡数")
s.index = s.index.map(lambda x: " / ".join(x))
c.bar(s, "信用卡类型与性别", "卡数")
c.notes += [
    f"读取{len(trans):,}条交易、{len(card)}张卡、{len(loans)}笔贷款。源CSV首行是数据，未当表头丢弃。",
    "DISTINCT、WHERE、ORDER、CASE、GROUP BY/HAVING、横向连接、非等值连接、子查询、UNION及视图均有计算输出。全连接通过pandas outer实现；未把本地执行声称为MySQL实测。",
    "教材贷款连接中date+365改写为日期差1至365天，并保留没有匹配流水的贷款。SQL安装视频、服务器配置和常见问题不是数据分析项目，不模拟安装截图。开窗函数PDF仅作为参考，未新增虚构数据练习。",
]
c.done()
