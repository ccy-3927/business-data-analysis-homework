"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from itertools import combinations, product
from collections import Counter

c = start(
    __file__,
    "银行产品序列模式",
    "按客户及OrderNumber升序构造事件序列，以客户计数，避免原AprioriAll缩小支持度分母。",
)
d = data(c, "bank_prod_order.csv")
seqs = []
for _, g in d.groupby("CSR_ID"):
    seqs.append(
        [
            tuple(sorted(set(h.PROD)))
            for _, h in g.sort_values("OrderNumber").groupby("OrderNumber", sort=True)
        ]
    )
# 精确枚举每条短序列的全部非空子序列；事件内无序，同一客户每种模式只计一次。
counts = Counter()
for seq in seqs:
    found = set()
    choices = [
        [tuple(v) for k in range(1, len(event) + 1) for v in combinations(event, k)]
        for event in seq
    ]
    for length in range(1, len(seq) + 1):
        for positions in combinations(range(len(seq)), length):
            found.update(product(*(choices[i] for i in positions)))
    counts.update(found)


def contains(pattern, sequence):
    j = 0
    for event in sequence:
        if set(pattern[j]) <= set(event):
            j += 1
            if j == len(pattern):
                return True
    return False


for title, minimum in [
    ("AprioriAll支持度0.1", int(np.ceil(0.1 * len(seqs)))),
    ("PrefixSpan计数100", 100),
]:
    selected = {k: v for k, v in counts.items() if v >= minimum}
    for pat, n in selected.items():
        assert n == sum(contains(pat, s) for s in seqs)
    tab = pd.DataFrame(
        [
            {
                "序列": " → ".join(" + ".join(e) for e in pat),
                "客户数": n,
                "支持度": n / len(seqs),
                "事件数": len(pat),
            }
            for pat, n in selected.items()
        ]
    )
    if len(tab):
        tab = tab.sort_values(["客户数", "序列"], ascending=[False, True])
        c.bar(tab.head(15).set_index("序列")["客户数"], title + "前15项", "客户数")
    c.table(tab, title)
c.notes += [
    f"共{len(seqs)}个客户。保留教材两种阈值，采用精确子序列枚举替代旧脚本搜索实现；输出全部频繁序列，不仅输出极大序列。每个结果另按子序列匹配逐客户核对。",
    "OrderNumber仅作为教材提供的先后顺序，不臆造日期或购买间隔；同一订单内的产品无先后顺序。",
]
c.done()
