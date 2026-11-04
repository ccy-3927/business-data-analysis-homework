"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from network_common import *

c = start(
    __file__,
    "银行交易圈社区发现",
    "从真实交易对手记录提取无向网络，复现Louvain并补充指定边指标的加权对照。",
)
d = data(c, "data/Archive-社交网络-交易对手明细汇总.csv")
g = nx.Graph()
for row in d[["源客户号", "目标客户号", "交易圈交易边指标值"]].itertuples(
    index=False, name=None
):
    g.add_edge(row[0], row[1], weight=row[2])
groups, p = communities(g)
weighted, pw = communities(g, "weight")
c.table(
    pd.DataFrame({"教材未加权社群": p, "指标加权社群": pw, "度": dict(g.degree())}),
    "客户社群归属",
)
network_plot(c, g, "银行交易圈全网", p)
c.bar(
    pd.Series([len(a) for a in groups]).sort_values(ascending=False).head(20),
    "最大20个交易社群",
    "节点数",
)
stats = pd.DataFrame(
    [
        {
            "口径": "教材未加权",
            "节点": len(g),
            "边": g.number_of_edges(),
            "社群": len(groups),
            "模块度": nx.community.modularity(g, groups, weight=None),
        },
        {
            "口径": "交易边指标加权",
            "节点": len(g),
            "边": g.number_of_edges(),
            "社群": len(weighted),
            "模块度": nx.community.modularity(g, weighted, weight="weight"),
        },
    ]
)
c.table(stats, "社区质量")
c.notes.append(
    f"输入{len(d)}条；教材原社区算法未指定中文权重属性，所以先复现未加权，再给出显式加权结果。简单图重复客户对保留最后一行，与教材一致。全图可缩放、悬停，社群颜色循环使用，精确社群编号见悬停和CSV；聚类关系不等于团伙欺诈。"
)
c.done()
