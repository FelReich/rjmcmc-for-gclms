"""Plot the most frequent graph across the five retained base-general chains."""

from collections import Counter
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.simulation_study.settings import get_simulation_case
from examples.gclm.simulation_study.run_simulation_study import edge_order


BASE_CASE_ID = "G2_n800_lambda5_general_base"
RUNS_DIR = CODE_ROOT / "examples/gclm/simulation_study/runs" / BASE_CASE_ID
OUTPUT = Path(__file__).with_suffix(".pdf")


def most_frequent_graph(case):
    """Count complete graph identities, pooling equally long retained chains."""
    counts = Counter()
    total = 0
    expected_per_chain = (case.iterations - case.burn_in) // case.thinning
    for chain_number in range(1, len(case.chain_seeds) + 1):
        path = RUNS_DIR / f"chain_{chain_number:02d}.csv"
        chain_draws = 0
        for chunk in pd.read_csv(path, usecols=["graph_bitmask"], chunksize=100_000):
            bitmasks, frequencies = np.unique(
                chunk["graph_bitmask"].to_numpy(dtype=np.int64),
                return_counts=True,
            )
            counts.update(dict(zip(bitmasks.tolist(), frequencies.tolist())))
            chain_draws += len(chunk)
        if chain_draws != expected_per_chain:
            raise ValueError(f"{path} has {chain_draws} draws; expected {expected_per_chain}")
        total += chain_draws

    # Resolve an exact count tie by choosing the smaller canonical bitmask.
    bitmask, visits = max(counts.items(), key=lambda item: (item[1], -item[0]))
    edges = tuple(edge for index, edge in enumerate(edge_order(case.generating_model.p))
                  if bitmask & (1 << index))
    return edges, visits / total


def main():
    case = get_simulation_case(BASE_CASE_ID)
    edges, probability = most_frequent_graph(case)
    p = case.generating_model.p
    graph = nx.DiGraph()
    graph.add_nodes_from(range(1, p + 1))
    # The diagonal drift terms are implicit; only off-diagonal edges are drawn.
    graph.add_edges_from((source + 1, target + 1) for source, target in edges)

    angles = np.pi / 2 - 2 * np.pi * np.arange(p) / p
    positions = {
        vertex: (np.cos(angle), np.sin(angle))
        for vertex, angle in enumerate(angles, start=1)
    }
    figure, axis = plt.subplots(figsize=(5, 5))
    nx.draw_networkx_edges(
        graph, positions, ax=axis, arrows=True, arrowstyle="-|>",
        arrowsize=22, node_size=1000, width=1.8, edge_color="#245b8f",
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
    figure.savefig(OUTPUT, bbox_inches="tight")
    plt.close(figure)
    print(f"MAP graph: {edges}; estimated posterior probability: {probability:.6f}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
