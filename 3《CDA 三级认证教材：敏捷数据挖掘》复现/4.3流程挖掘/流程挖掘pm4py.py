"""教材案例迁移版。输入取自同名教材目录，结果见同级输出。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

import xml.etree.ElementTree as ET
import networkx as nx
from collections import Counter, defaultdict

c = start(
    __file__,
    "事件日志流程发现",
    "解析教材XES，复现直接跟随图的频次与平均耗时；使用标准XML和NetworkX实现原pm4py案例的计算语义。",
)
p = c.source(c.notebook.parent / "running-example.xes")
xml = ET.parse(p)
records = []
traces = []


def attrs(node):
    return {
        el.attrib["key"]: el.attrib.get("value") for el in node if "key" in el.attrib
    }


for trace in xml.getroot():
    if trace.tag.split("}")[-1] != "trace":
        continue
    case = attrs(trace).get("concept:name")
    events = []
    for event in trace:
        if event.tag.split("}")[-1] != "event":
            continue
        a = attrs(event)
        row = {
            "案例": case,
            "活动": a["concept:name"],
            "时间": pd.to_datetime(a["time:timestamp"], utc=True),
            "资源": a.get("org:resource"),
        }
        records.append(row)
        events.append(row)
    events.sort(key=lambda r: r["时间"])
    traces.append(events)
count = Counter()
durations = defaultdict(list)
for trace in traces:
    for left, right in zip(trace, trace[1:]):
        edge = (left["活动"], right["活动"])
        count[edge] += 1
        durations[edge].append((right["时间"] - left["时间"]).total_seconds() / 3600)
table = pd.DataFrame(
    [
        {"前活动": a, "后活动": b, "频次": n, "平均小时": np.mean(durations[(a, b)])}
        for (a, b), n in count.items()
    ]
)
c.table(pd.DataFrame(records), "事件日志")
c.table(table, "直接跟随与耗时")
graph = nx.DiGraph()
graph.add_edges_from(count)
pos = nx.circular_layout(graph)
for name, col in [("流程跟随频次", "频次"), ("流程平均耗时", "平均小时")]:
    fig, ax = plt.subplots(figsize=(15, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=6500, node_color="#deedf4", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=11, ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=20,
        node_size=6500,
        connectionstyle="arc3,rad=.12",
        edge_color="#82a6bb",
    )
    labels = {
        (r["前活动"], r["后活动"]): str(r[col]) if col == "频次" else f"{r[col]:.1f}h"
        for _, r in table.iterrows()
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=labels,
        font_size=8,
        ax=ax,
        connectionstyle="arc3,rad=.12",
        rotate=False,
    )
    ax.set_title(name)
    ax.axis("off")
    c.figure(fig, name)
case_stats = pd.DataFrame(
    [
        {
            "案例": t[0]["案例"],
            "事件数": len(t),
            "历时小时": (t[-1]["时间"] - t[0]["时间"]).total_seconds() / 3600,
        }
        for t in traces
    ]
)
c.table(case_stats, "流程案例汇总")
c.notes.append(
    f"{len(traces)}条案例轨迹、{len(records)}个事件，跟随边总频次{sum(count.values())}。耗时为相邻事件时间差的算术均值，不能单独解释为实际工作处理时长；没有执行无分析作用的XES重复导出。"
)
c.done()
