"""以客户为事务单位的关联规则，Apriori与FP-growth互相核对。"""

from model_common import *
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def association(c, filename, support):
    d = data(c, filename)
    baskets = d.groupby("CSR_ID").PROD.apply(lambda s: sorted(set(s)))
    enc = TransactionEncoder()
    a = enc.fit_transform(baskets)
    frame = pd.DataFrame(a, columns=enc.columns_)
    freq = apriori(frame, min_support=support, use_colnames=True)
    fp = fpgrowth(frame, min_support=support, use_colnames=True)
    left = freq.set_index("itemsets").support.to_dict()
    right = fp.set_index("itemsets").support.to_dict()
    assert left.keys() == right.keys()
    for key in left:
        assert abs(left[key] - right[key]) < 1e-12
    rules = association_rules(freq, metric="confidence", min_threshold=0.1)
    rules = rules.loc[rules.lift > 1].sort_values("support", ascending=False)
    printable = rules.copy()
    for col in ["antecedents", "consequents"]:
        printable[col] = printable[col].map(lambda s: " + ".join(sorted(s)))
    ft = freq.copy()
    ft.itemsets = ft.itemsets.map(lambda s: " + ".join(sorted(s)))
    c.table(ft, "频繁项集")
    c.table(printable, "筛选后关联规则")
    f = go.Figure(
        go.Scatter(
            x=rules.lift,
            y=rules.support,
            mode="markers",
            text=printable.antecedents + " → " + printable.consequents,
            marker={
                "size": 10,
                "color": rules.confidence,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "置信度"},
            },
            hovertemplate="%{text}<br>提升度=%{x:.3f}<br>支持度=%{y:.3f}<extra></extra>",
        )
    )
    f.update_xaxes(title="提升度")
    f.update_yaxes(title="客户支持度")
    c.interactive(f, "交叉销售规则")
    c.table(
        printable.loc[rules.antecedents.map(lambda x: x == frozenset(["财务通"]))],
        "练习1_仅持有财务通可触发的推荐",
    )
    c.table(
        printable.loc[rules.consequents.map(lambda x: "好享贷" in x)],
        "练习2_推荐好享贷的前提",
    )
    c.notes += [
        f"{len(d)}条交易明细，{len(baskets)}个客户；客户内去重，支持度分母是客户数，修正原FP-growth以明细行数作分母的问题。min_support={support}，置信度≥0.1、提升度>1。Apriori与FP-growth频繁项集及支持度一致。",
        "规则体现同时持有关系而非因果推荐效果。练习1只选前件恰为财务通的规则，多个前件的规则需要同时满足全部条件。",
    ]
