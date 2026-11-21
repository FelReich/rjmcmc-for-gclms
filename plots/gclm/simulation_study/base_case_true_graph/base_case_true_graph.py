"""Plot the generating graph for the G2, n=800 base case."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.simulation_study.settings import get_simulation_case


BASE_CASE_ID = "G2_n800_lambda5_general_base"
OUTPUT = Path(__file__).with_suffix(".pdf")


def main():
    model = get_simulation_case(BASE_CASE_ID).generating_model
    graph = nx.DiGraph()
    graph.add_nodes_from(range(1, model.p + 1))
    # The diagonal drift terms are implicit; only off-diagonal edges are drawn.
    graph.add_edges_from((source + 1, target + 1)
                         for source, target in model.ordered_edges)

    angles = np.pi / 2 - 2 * np.pi * np.arange(model.p) / model.p
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


if __name__ == "__main__":
    main()
