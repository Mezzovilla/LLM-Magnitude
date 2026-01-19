import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def plot_graph(nodes: list[str], dist: list[list[float]],
        digits: int = 2,
        seed: int = 42,
        node_size: float = 300,
        node_color="#1f78b4",
        node_shape="o",
        alpha=None,
        cmap=None,
        vmin=None,
        vmax=None,
        ax=None,
        linewidths=None,
        edgecolors=None,
        label=None,
        margins=None,
        hide_ticks=True,
    ) -> Figure:
    G = nx.Graph()

    for n in nodes:
        G.add_node(n)

    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            weight = round(dist[i][j], digits)

            if weight > 0.0:
                G.add_edge(nodes[i], nodes[j], weight= weight)

    pos = nx.spring_layout(G, seed=seed)

    nx.draw_networkx_nodes(G, pos, 
        node_size=node_size,
        node_color=node_color,
        node_shape=node_shape,
        alpha=alpha,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=linewidths,
        edgecolors=edgecolors,
        label=label,
        margins=margins,
        hide_ticks=hide_ticks,
    )

    weights = [d["weight"] for (_, _, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=weights)

    nx.draw_networkx_labels(G, pos)

    # NOVO: mostrar pesos
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    return plt.gcf()

def main():
    nodes = ["A", "B", "C", "D"]
    dist = [
        [0, 1.2, 3.5, 2.1],
        [1.2, 0, 2.4, 4.0],
        [3.5, 2.4, 0, 1.8],
        [2.1, 4.0, 1.8, 0],
    ]
    fig = plot_graph(nodes, dist)
    figures_path = Path("figures")
    fig.savefig(figures_path / "graph.png")


if __name__ == "__main__":
    main()