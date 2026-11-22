"""Plot the MAP graphs from the five baseline G8 chains in each graph space."""

from collections import Counter
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[5]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.simulation_study.settings import get_simulation_case
from examples.gclm.simulation_study.run_simulation_study import edge_order


HERE = Path(__file__).resolve().parent
RUNS = CODE_ROOT / "examples/gclm/simulation_study/runs"
CASE_IDS = {
    "general": "G8_n800_lambda5_general_base",
    "simple": "G8_n800_lambda5_simple_base",
}


def most_frequent_graph(case):
    """Return the most visited complete graph and its pooled posterior frequency."""
    counts = Counter()
    total = 0
    expected = (case.iterations - case.burn_in + case.thinning - 1) // case.thinning
    for chain_number, seed in enumerate(case.chain_seeds, start=1):
        path = RUNS / case.id / f"chain_{chain_number:02d}.csv"
        metadata = pd.read_csv(
            path, usecols=["configuration_id", "chain_number", "chain_seed"], nrows=1,
        ).iloc[0]
        if (metadata["configuration_id"] != case.id
                or metadata["chain_number"] != chain_number
                or metadata["chain_seed"] != seed):
            raise ValueError(f"Chain metadata do not match settings: {path}")

        chain_draws = 0
        for chunk in pd.read_csv(path, usecols=["graph_bitmask"], chunksize=100_000):
            bitmasks, frequencies = np.unique(
                chunk["graph_bitmask"].to_numpy(dtype=np.int64), return_counts=True,
            )
            counts.update(dict(zip(bitmasks.tolist(), frequencies.tolist())))
            chain_draws += len(chunk)
        if chain_draws != expected:
            raise ValueError(f"{path} has {chain_draws} draws; expected {expected}")
        total += chain_draws

    # In an exact tie, take the smaller canonical bitmask.
    bitmask, visits = max(counts.items(), key=lambda item: (item[1], -item[0]))
    edges = tuple(
        edge for index, edge in enumerate(edge_order(case.generating_model.p))
        if bitmask & (1 << index)
    )
    return edges, visits, total


def plot_graph(edges, p, output):
    """Match the existing GCLM graph style, separating reciprocal arrows."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(1, p + 1))
    graph.add_edges_from((source + 1, target + 1) for source, target in edges)
    angles = np.pi / 2 - 2 * np.pi * np.arange(p) / p
    positions = {
        vertex: (np.cos(angle), np.sin(angle))
        for vertex, angle in enumerate(angles, start=1)
    }

    reciprocal = [(i, j) for i, j in graph.edges if graph.has_edge(j, i)]
    ordinary = [(i, j) for i, j in graph.edges if not graph.has_edge(j, i)]
    figure, axis = plt.subplots(figsize=(5, 5))
    for edgelist, connectionstyle in (
        (ordinary, "arc3,rad=0"),
        (reciprocal, "arc3,rad=0.18"),
    ):
        if edgelist:
            nx.draw_networkx_edges(
                graph, positions, edgelist=edgelist, ax=axis, arrows=True,
                arrowstyle="-|>", arrowsize=22, node_size=1000, width=1.8,
                edge_color="#245b8f", connectionstyle=connectionstyle,
            )
    nx.draw_networkx_nodes(
        graph, positions, ax=axis, node_size=1000, node_color="white",
        edgecolors="#25313d", linewidths=1.5,
    )
    nx.draw_networkx_labels(graph, positions, ax=axis, font_size=12)
    axis.set_aspect("equal")
    axis.set_xlim(-1.10, 1.10)
    axis.set_ylim(-1.00, 1.20)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main():
    for space, case_id in CASE_IDS.items():
        case = get_simulation_case(case_id)
        edges, visits, total = most_frequent_graph(case)
        output = HERE / f"G8_map_graph_{space}.pdf"
        plot_graph(edges, case.generating_model.p, output)
        displayed_edges = tuple((i + 1, j + 1) for i, j in edges)
        print(f"{space}: MAP edges {displayed_edges}; {visits:,}/{total:,} "
              f"draws = {visits / total:.4%}; saved {output}")


if __name__ == "__main__":
    main()
