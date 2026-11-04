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
    "人物关系社区发现",
    "教材无向人物网络的Louvain分割；保留未加权口径并对照按Weight加权。",
)
d = data(c, "data/stormofswords.csv")
g = nx.from_pandas_edgelist(d, "Source", "Target", "Weight")
groups, p = communities(g)
weighted, pw = communities(g, "Weight")
network_plot(c, g, "人物社区分割", p)
c.table(pd.DataFrame({"未加权社群": p, "加权社群": pw}), "人物社群")
c.bar(pd.Series([len(a) for a in groups], name="人数"), "人物社群规模", "人数")
c.table(
    pd.DataFrame(
        [
            {
                "口径": "教材未加权",
                "社群数": len(groups),
                "模块度": nx.community.modularity(g, groups, weight=None),
            },
            {
                "口径": "Weight加权",
                "社群数": len(weighted),
                "模块度": nx.community.modularity(g, weighted, weight="Weight"),
            },
        ]
    ),
    "社区质量",
)
c.notes.append(
    "教材边属性名为Weight，原best_partition默认weight未匹配，因此基准采用未加权。加权结果单列，不暗中改变原口径。社群编号无固有业务含义。"
)
c.done()
