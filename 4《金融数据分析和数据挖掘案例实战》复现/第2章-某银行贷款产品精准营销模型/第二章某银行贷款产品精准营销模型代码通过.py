"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from sklearn.linear_model import LogisticRegression

c = start(
    __file__,
    "贷款产品精准营销",
    "复现1998年申请标签、180天交易特征及逻辑回归，并对教材1999年待营销客户评分。",
)
d = data(c, "tran0917.csv")
pred = data(c, "pred0917.csv")
profile(c, d, "训练宽表")
# 原始数据库导出没有表头，必须显式指定，防止丢掉第一条记录。
schemas = {
    "kdd99_accounts": ["account_id", "district_id", "frequency", "date"],
    "Kdd99_loans": [
        "loan_id",
        "account_id",
        "loan_date",
        "amount",
        "duration",
        "payments",
        "status",
    ],
    "Kdd99_disp": ["disp_id", "client_id", "account_id", "type"],
    "Kdd99_clients": ["client_id", "sex", "birth_date", "client_district"],
    "Kdd99_district": [
        "client_district",
        "GDP",
        "A4",
        "A10",
        "A11",
        "A12",
        "A13",
        "A14",
        "A15",
        "A16",
    ],
    "Kdd99_card": ["card_id", "disp_id", "issued", "card_type"],
    "Kdd99_trans": [
        "trans_id",
        "account_id",
        "date",
        "type",
        "operation",
        "amount",
        "balance",
        "K_symbol",
        "bank",
        "account",
    ],
}
tables = {
    n: data(c, "mysql_csv/" + n + ".csv", header=None, names=cols, low_memory=False)
    for n, cols in schemas.items()
}
a = tables["kdd99_accounts"].merge(
    tables["Kdd99_loans"], on="account_id", how="left", validate="one_to_one"
)
a["y"] = np.where(
    a.loan_date.isna(),
    0,
    np.where(pd.to_datetime(a.loan_date) < pd.Timestamp("1998-01-01"), 2, 1),
)
a = a[a.y.ne(2)]
a = (
    a.merge(
        tables["Kdd99_disp"].query("type=='所有者'"),
        on="account_id",
        validate="one_to_one",
    )
    .merge(tables["Kdd99_clients"], on="client_id", validate="many_to_one")
    .merge(tables["Kdd99_district"], on="client_district", validate="many_to_one")
    .merge(tables["Kdd99_card"], on="disp_id", how="left", validate="one_to_one")
)
cut = pd.Timestamp("1998-01-01")
born = pd.to_datetime(a.birth_date)
a["age"] = cut.year - born.dt.year - ((born.dt.month * 100 + born.dt.day) > 101)
a["card"] = (pd.to_datetime(a.issued) < cut).astype(int)
a["avg_GDP"] = a.GDP / a.A4
t = tables["Kdd99_trans"]
t["date"] = pd.to_datetime(t.date)
t = t[(t.date < cut) & (t.date > cut - pd.Timedelta(days=180))].copy()
g = t.groupby("account_id")
agg = g.balance.agg(avg_balance="mean", std_balance=lambda x: x.std(ddof=0))
agg["cv_balance"] = agg.std_balance / agg.avg_balance
agg["sum_income"] = t[t.type.eq("贷")].groupby("account_id").amount.sum()
agg["sum_expenditure"] = t[t.type.eq("借")].groupby("account_id").amount.sum()
agg["in_out_ratio"] = agg.sum_income / agg.sum_expenditure
rebuilt = a[["account_id", "age", "card", "avg_GDP", "y"]].merge(
    agg, on="account_id", how="left"
)
c.table(rebuilt, "原始表重建宽表")
comparison = d.set_index("account_id").join(
    rebuilt.set_index("account_id"), rsuffix="_重建", how="inner"
)
c.table(
    pd.DataFrame(
        {
            "最大绝对差": {
                k: (comparison[k] - comparison[k + "_重建"]).abs().max()
                for k in rebuilt.columns
                if k != "account_id"
            }
        }
    ),
    "重建与教材宽表核对",
)
cols = ["avg_balance", "cv_balance", "sum_expenditure", "card"]
x = d[cols].fillna(0)
y = d.y.astype(int)
tr = d.sample(frac=0.7, random_state=1234).index
te = d.index.difference(tr)
# 教材评分端单独截尾与训练端不一致，统一采用训练集分位点。
lo = x.loc[tr].quantile(0.01)
hi = x.loc[tr].quantile(0.99)
lo["card"] = 0
hi["card"] = 1
xc = x.clip(lo, hi, axis=1)
model = Pipeline(
    [("scale", StandardScaler()), ("logit", LogisticRegression(C=1e6, max_iter=3000))]
).fit(xc.loc[tr], y.loc[tr])
corrected = (
    rebuilt.set_index("account_id")
    .reindex(d.account_id)[cols]
    .reset_index(drop=True)
    .fillna(0)
)
corrected.index = d.index
clo = corrected.loc[tr].quantile(0.01)
chi = corrected.loc[tr].quantile(0.99)
clo["card"] = 0
chi["card"] = 1
corrected = corrected.clip(clo, chi, axis=1)
raw_model = Pipeline(
    [("scale", StandardScaler()), ("logit", LogisticRegression(C=1e6, max_iter=3000))]
).fit(corrected.loc[tr], y.loc[tr])
evaluate(
    c,
    y.loc[te],
    {
        "教材宽表Logit": model.predict_proba(xc.loc[te])[:, 1],
        "原始流水重建Logit": raw_model.predict_proba(corrected.loc[te])[:, 1],
    },
    "独立测试评估",
)
rank = pred[["account_id"]].copy()
rank["贷款响应概率"] = model.predict_proba(pred[cols].fillna(0).clip(lo, hi, axis=1))[
    :, 1
]
c.table(rank.sort_values("贷款响应概率", ascending=False), "待营销客户排序")
fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(
    [x.loc[y.eq(k), "cv_balance"] for k in [0, 1]],
    tick_labels=["未申请", "申请"],
    showfliers=False,
)
ax.set(title="余额变异系数与贷款申请", ylabel="cv_balance")
c.figure(fig, "余额变异系数分组")
pt = pd.DataFrame(
    {"y": y.loc[te], "p": model.predict_proba(xc.loc[te])[:, 1]}
).sort_values("p", ascending=False)
pt["人数占比"] = np.arange(1, len(pt) + 1) / len(pt)
pt["累计申请捕获率"] = pt.y.cumsum() / pt.y.sum()
f = go.Figure(go.Scatter(x=pt["人数占比"], y=pt["累计申请捕获率"], name="模型排序"))
f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="随机基线", line={"dash": "dash"}))
c.interactive(f, "营销累计增益")
joblib.dump(
    {"model": model, "lower": lo, "upper": hi, "features": cols},
    c.out / "营销模型.joblib",
)
counts = {
    k: int(((comparison[k] - comparison[k + "_重建"]).abs() > 1e-6).sum())
    for k in ["sum_income", "sum_expenditure"]
}
c.notes += [
    f"教材训练宽表{len(d)}条、待评分{len(pred)}条。重建使用MySQL STD的总体标准差口径。年龄、卡状态、余额均值和标准差核对一致，但收入/支出与原始流水汇总存在差异，差异行数{counts}。未猜测差异原因，保留教材宽表基准并在相同训练/测试客户上比较重建口径。待营销排序仍是教材宽表模型口径。",
    "未把贷款金额、贷款状态或未来日期作为特征。模型沿教材四个解释变量，使用数值缩放与近似无惩罚Logit；统一训练分位点截尾。测试集只评估，预测表中的y不用于评估未来效果。",
]
c.done()
