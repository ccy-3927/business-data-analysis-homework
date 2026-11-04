"""教材案例Python复现；结果保存在同级输出目录。"""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "model_common.py").exists():
        sys.path.insert(0, str(parent))
        break
from model_common import *

from network_common import *
from collections import deque

c = start(
    __file__,
    "广度优先与深度优先遍历",
    "在教材给定的8节点10边上复现BFS与DFS，核对路径有效性。",
)
edges = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)]
g = nx.Graph(edges)


def traverse(source, target, bfs=True):
    todo = deque([source])
    prev = {source: None}
    while todo:
        node = todo.popleft() if bfs else todo.pop()
        if node == target:
            break
        ns = list(g.neighbors(node))
        ns = ns if bfs else ns[::-1]
        for n in ns:
            if n not in prev:
                prev[n] = node
                todo.append(n)
    if target not in prev:
        return []
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


paths = {"BFS": traverse(0, 7), "DFS": list(nx.dfs_edges(g, source=0))}


# 递归DFS准确保留教材的邻居访问次序。
def dfs(node, target, seen, path):
    if node == target:
        return path
    seen.add(node)
    for n in g.neighbors(node):
        if n not in seen:
            found = dfs(n, target, seen, path + [n])
            if found:
                return found
    return []


paths["DFS"] = dfs(0, 7, set(), [0])
assert len(paths["BFS"]) - 1 == nx.shortest_path_length(g, 0, 7)
for name, path in paths.items():
    assert all(g.has_edge(a, b) for a, b in zip(path, path[1:]))
    c.table(pd.Series(path, name="节点"), name + "路径")
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = nx.spring_layout(g, seed=42)
    nx.draw_networkx(
        g,
        pos,
        ax=ax,
        node_size=1000,
        node_color=COLORS[0],
        font_color="white",
        edge_color="#ccd8df",
    )
    nx.draw_networkx_edges(
        g, pos, edgelist=list(zip(path, path[1:])), width=4, edge_color=COLORS[1], ax=ax
    )
    ax.set_title(name + "：" + " → ".join(map(str, path)))
    ax.axis("off")
    c.figure(fig, name + "搜索路径")
assert traverse(0, 0) == [0]
c.notes.append(
    "BFS保证无权图最少边路径，DFS不保证。处理起终点相同及不可达的返回值，修复原递归路径对节点0的真假判断。"
)
c.done()
