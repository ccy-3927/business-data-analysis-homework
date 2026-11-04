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
    "无向图与有向图",
    "使用Notebook明确给出的9个节点与固定边，不生成随机网络。",
)
nodes = list("012345678")
edges = [
    ("0", "0", 1),
    ("0", "1", 1),
    ("0", "5", 1),
    ("0", "5", 2),
    ("1", "2", 3),
    ("1", "4", 5),
    ("2", "1", 7),
    ("2", "4", 6),
    ("3", "5", 5),
    ("4", "8", 4),
    ("6", "7", 1),
    ("7", "8", 2),
    ("8", "6", 3),
    ("8", "3", 5),
]
for label, klass in [("无向图", nx.Graph), ("有向图", nx.DiGraph)]:
    g = klass()
    g.add_nodes_from(nodes)
    g.add_weighted_edges_from(edges)
    toy_graph(c, g, label)
    c.table(nx.to_pandas_edgelist(g), label + "边表")
    c.table(pd.Series(dict(g.degree()), name="度"), label + "节点度")
c.notes.append(
    "固定示例直接来自教材代码，不是实际客户数据。简单图重复边以最后权重覆盖；无向自环计两次度。"
)
c.done()
