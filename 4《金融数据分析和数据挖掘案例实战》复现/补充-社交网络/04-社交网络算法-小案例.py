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
    "小说人物关系网络",
    "读取教材stormofswords边表，复现关系可视化并输出节点连接统计。",
)
d = data(c, "data/stormofswords.csv")
g = nx.from_pandas_edgelist(d, "Source", "Target", "Weight", create_using=nx.DiGraph)
network_plot(c, g, "人物关系网络")
c.table(
    pd.DataFrame({"入度": dict(g.in_degree()), "出度": dict(g.out_degree())}),
    "人物连接统计",
)
c.notes.append(
    f"{len(g)}个人物、{g.number_of_edges()}条有向边，方向沿教材表Source/Target，不把它解释为现实资金流。HTML线段仅表示连接，方向请查入出度和原始边表。"
)
c.done()
