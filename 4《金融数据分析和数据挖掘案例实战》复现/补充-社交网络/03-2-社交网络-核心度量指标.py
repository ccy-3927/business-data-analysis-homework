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
    "社交网络核心指标",
    "教材有向7节点网络与6节点星形网络：度、紧密、介数、特征向量中心度。",
)
dg = nx.DiGraph()
dg.add_edges_from(
    [
        ("1", "2"),
        ("1", "6"),
        ("1", "7"),
        ("2", "3"),
        ("3", "4"),
        ("4", "5"),
        ("4", "7"),
        ("5", "6"),
        ("6", "7"),
        ("7", "5"),
    ]
)
toy_graph(c, dg, "教材有向网络")
c.table(
    pd.DataFrame({"入度": dict(dg.in_degree()), "出度": dict(dg.out_degree())}),
    "有向节点度",
)
g = nx.Graph([("A", x) for x in "BCDEF"])
toy_graph(c, g, "教材星形网络")
m = metrics_table(g)
c.table(m, "星形中心性")
heat(c, m.drop(columns="度"), "星形中心性对比")
assert m.loc["A", "度中心度"] == 1
c.notes.append(
    "固定图来自教材，不代表真实客户。不同中心性量纲不同，仅在同指标内比较节点排名。"
)
c.done()
