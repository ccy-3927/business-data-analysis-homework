"""网络图和原始边表分析；布局位置只用于显示，不是新增业务记录。"""

from model_common import *
import networkx as nx


def network_plot(c, g, name, groups=None):
    if not len(g):
        return
    if len(g) < 600:
        pos = nx.spring_layout(g, seed=42, iterations=100, weight="weight")
    else:
        # 大网络按连通分量分区，保留全部节点和边。
        comps = sorted(
            nx.connected_components(g.to_undirected()), key=len, reverse=True
        )
        pos = {}
        side = int(np.ceil(np.sqrt(len(comps))))
        for i, nodes in enumerate(comps):
            sub = g.subgraph(nodes)
            local = (
                nx.spring_layout(sub, seed=42, iterations=30)
                if len(sub) < 600
                else nx.circular_layout(sub)
            )
            for n, xy in local.items():
                pos[n] = xy * 0.4 + np.array([i % side, -(i // side)])
    xx = []
    yy = []
    for a, b in g.edges:
        xx.extend([pos[a][0], pos[b][0], None])
        yy.extend([pos[a][1], pos[b][1], None])
    f = go.Figure(
        go.Scattergl(
            x=xx,
            y=yy,
            mode="lines",
            line={"width": 0.7, "color": "#b8c8d5"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    nodes = list(g)
    groups = groups or {n: 0 for n in g}
    vals = [groups.get(n, -1) for n in nodes]
    f.add_trace(
        go.Scattergl(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers",
            text=[f"{n}<br>度={g.degree(n)}<br>分组={groups.get(n,-1)}" for n in nodes],
            hovertemplate="%{text}<extra></extra>",
            marker={
                "size": [min(22, 5 + 2 * np.log1p(g.degree(n))) for n in nodes],
                "color": [COLORS[int(v) % len(COLORS)] for v in vals],
            },
            showlegend=False,
        )
    )
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False, scaleanchor="x")
    c.interactive(f, name)


def communities(g, weight=None):
    cc = nx.community.louvain_communities(g, seed=42, weight=weight)
    cc = sorted(cc, key=lambda x: (-len(x), str(min(x))))
    return cc, {node: i for i, group in enumerate(cc) for node in group}


def toy_graph(c, g, name):
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.circular_layout(g)
    nx.draw_networkx(
        g,
        pos,
        ax=ax,
        node_color=COLORS[0],
        font_color="white",
        node_size=1000,
        arrowsize=22,
        edge_color="#82a6bb",
        connectionstyle="arc3,rad=.12" if g.is_directed() else "arc3",
    )
    ax.set_title(name)
    ax.axis("off")
    c.figure(fig, name)


def metrics_table(g):
    return pd.DataFrame(
        {
            "度": dict(g.degree()),
            "度中心度": nx.degree_centrality(g),
            "紧密中心度": nx.closeness_centrality(g),
            "介数中心度": nx.betweenness_centrality(g),
            "特征向量中心度": nx.eigenvector_centrality(g, max_iter=2000),
            "PageRank": nx.pagerank(g, weight="weight"),
        }
    )
