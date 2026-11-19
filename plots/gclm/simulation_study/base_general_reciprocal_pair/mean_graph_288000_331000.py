"""Summarize chain 1's reciprocal-pair regime using saved retained draws.

Run from the Code repository root:
    python3 plots/gclm/simulation_study/base_general_reciprocal_pair/mean_graph_288000_331000.py

The interval [288000, 331000) uses zero-based retained indices, corresponding
to original sampler iterations 788001–831000. Vertices in outputs are one-based.
The graph is summarized by empirical edge-inclusion frequencies; the displayed
majority graph contains edges with inclusion probability strictly above 0.5.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = (HERE.parents[3] / "examples/gclm/simulation_study/runs/"
          "G2_n800_lambda5_general_base/chain_01.csv")
START, END = 288_000, 331_000
P = 5
EDGES = tuple((i, j) for i in range(P) for j in range(P) if i != j)
EDGE_COLUMNS = [f"edge_{i}_{j}" for i, j in EDGES]
M_COLUMNS = [f"M_{i}_{j}" for i in range(P) for j in range(P)]
READ_COLUMNS = ["iteration", "graph_bitmask", "k", "eta", "log_likelihood",
                *EDGE_COLUMNS, *M_COLUMNS]


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def summarize():
    counts = Counter()
    edge_sum = np.zeros(len(EDGES), dtype=np.int64)
    m_sum = np.zeros((P, P), dtype=float)
    k_sum = eta_sum = ll_sum = 0.0
    n = pair_draws = 0
    for chunk_number, chunk in enumerate(pd.read_csv(
            SOURCE, usecols=READ_COLUMNS, chunksize=50_000)):
        chunk_start = chunk_number * 50_000
        left = max(START - chunk_start, 0)
        right = min(END - chunk_start, len(chunk))
        if left >= right:
            if chunk_start >= END:
                break
            continue
        selected = chunk.iloc[left:right]
        global_indices = chunk_start + np.arange(left, right)
        if not np.array_equal(selected["iteration"].to_numpy(), 500_001 + global_indices):
            raise ValueError("Saved iteration numbers do not match the retained interval")
        indicators = selected[EDGE_COLUMNS].to_numpy(dtype=np.uint8)
        if not np.isin(indicators, [0, 1]).all():
            raise ValueError("An edge indicator is not binary")
        mask = selected["graph_bitmask"].to_numpy(dtype=np.uint32)
        reconstructed = indicators.astype(np.uint32) @ (1 << np.arange(len(EDGES), dtype=np.uint32))
        if not np.array_equal(mask, reconstructed):
            raise ValueError("Graph bitmask and edge indicators disagree")
        k = selected["k"].to_numpy(dtype=np.uint8)
        if not np.array_equal(k, indicators.sum(axis=1)):
            raise ValueError("Graph size and edge indicators disagree")
        matrix = selected[M_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite drift matrix entry")
        edge_sum += indicators.sum(axis=0).astype(np.int64)
        m_sum += matrix.sum(axis=0).reshape(P, P)
        k_sum += float(k.sum())
        eta_sum += float(selected["eta"].sum())
        ll_sum += float(selected["log_likelihood"].sum())
        pair_draws += int(np.sum(indicators[:, EDGES.index((0, 1))]
                                 & indicators[:, EDGES.index((1, 0))]))
        counts.update(mask.tolist())
        n += len(selected)
    if n != END - START:
        raise ValueError(f"Expected {END - START} draws, found {n}")
    probabilities = edge_sum / n
    mean_m = m_sum / n
    return n, pair_draws, probabilities, mean_m, counts, k_sum / n, eta_sum / n, ll_sum / n


def graph_edges(mask):
    return [f"{i + 1}→{j + 1}" for index, (i, j) in enumerate(EDGES)
            if int(mask) & (1 << index)]


def plot(probabilities, mean_m, n):
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "savefig.bbox": "tight"})
    adjacency = np.full((P, P), np.nan)
    for value, (source, target) in zip(probabilities, EDGES):
        adjacency[target, source] = value

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5),
                             gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]},
                             constrained_layout=True)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f0f0f0")
    image = axes[0].imshow(adjacency, vmin=0, vmax=1, cmap=cmap)
    axes[0].set(title="Edge inclusion in window", xlabel="Source vertex", ylabel="Target vertex")
    axes[0].set_xticks(range(P), range(1, P + 1))
    axes[0].set_yticks(range(P), range(1, P + 1))
    for target in range(P):
        for source in range(P):
            if target != source:
                value = adjacency[target, source]
                axes[0].text(source, target, f"{value:.2f}", ha="center", va="center",
                             fontsize=8, color="white" if value > .6 else "black")
    fig.colorbar(image, ax=axes[0], fraction=.045, pad=.03)

    ax = axes[1]
    angles = np.pi / 2 - 2 * np.pi * np.arange(P) / P
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    for node, (x, y) in enumerate(positions):
        ax.add_patch(Circle((x, y), .12, facecolor="#e8edf1", edgecolor="#26384a", lw=1.3))
        ax.text(x, y, str(node + 1), ha="center", va="center", weight="bold")
    majority = [(edge, float(probability)) for edge, probability in zip(EDGES, probabilities)
                if probability > .5]
    for (source, target), probability in majority:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        reciprocal = any(edge == (target, source) for edge, _ in majority)
        arc = .23 if reciprocal else .0
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle="-|>", mutation_scale=14, lw=2.0,
                                color="#a74432", shrinkA=15, shrinkB=17,
                                connectionstyle=f"arc3,rad={arc}")
        ax.add_patch(arrow)
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.25, 1.25),
           title="Majority graph ($P>0.5$)")
    ax.set_aspect("equal")
    ax.axis("off")

    max_abs = max(1e-12, float(np.abs(mean_m).max()))
    image = axes[2].imshow(mean_m, cmap="RdBu_r",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs))
    axes[2].set(title=r"Mean sampled drift $\overline{M}$",
                xlabel="Source vertex (column)", ylabel="Target vertex (row)")
    axes[2].set_xticks(range(P), range(1, P + 1))
    axes[2].set_yticks(range(P), range(1, P + 1))
    for row in range(P):
        for col in range(P):
            axes[2].text(col, row, f"{mean_m[row, col]:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if abs(mean_m[row, col]) > .55 * max_abs else "black")
    fig.colorbar(image, ax=axes[2], fraction=.045, pad=.03)
    fig.suptitle(f"G2 general chain 1 · retained indices [{START:,}, {END:,}) · {n:,} draws",
                 fontsize=11)
    fig.savefig(HERE / "mean_graph_and_drift_288000_331000.pdf")
    plt.close(fig)


def main():
    n, pair_draws, probabilities, mean_m, counts, mean_k, mean_eta, mean_ll = summarize()
    majority = [edge for edge, value in zip(EDGES, probabilities) if value > .5]
    write_csv(HERE / "edge_inclusion_288000_331000.csv",
              ("source_vertex", "target_vertex", "inclusion_frequency", "included_at_50_percent"),
              ((i + 1, j + 1, format(float(value), ".17g"), int(value > .5))
               for (i, j), value in zip(EDGES, probabilities)))
    write_csv(HERE / "mean_drift_288000_331000.csv",
              ("target_vertex", *(f"source_{i}" for i in range(1, P + 1))),
              ((i + 1, *(format(float(value), ".17g") for value in mean_m[i]))
               for i in range(P)))
    top_mask, top_count = counts.most_common(1)[0]
    lines = [
        "# Mean graph structure and drift, chain 1", "",
        f"Retained indices **[{START:,}, {END:,})** contain **{n:,}** draws "
        f"(original sampler iterations {500_001 + START:,}–{500_000 + END:,}, inclusive). "
        "The source CSV uses zero-based vertices; this report and its figure use one-based vertices.", "",
        f"Both **1→2** and **2→1** occur in {pair_draws:,}/{n:,} draws "
        f"({100 * pair_draws / n:.2f}%). The graph formed by edges with empirical inclusion "
        f"strictly above 50% contains: **{', '.join(graph_edges(sum(1 << EDGES.index(edge) for edge in majority)))}**. "
        f"Mean graph size is **{mean_k:.3f}**, because additional edges vary across draws. "
        f"The most frequent *complete* graph is {', '.join(graph_edges(top_mask))} "
        f"({100 * top_count / n:.2f}% of draws); it is not a representation of every draw.", "",
        f"Mean stability margin: **{mean_eta:.5f}**. Mean total log-likelihood: **{mean_ll:.5f}**.", "",
        "Mean sampled drift matrix (rows are targets, columns are sources):", "",
        "| target \\ source | 1 | 2 | 3 | 4 | 5 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(mean_m, 1):
        lines.append("| " + str(i) + " | " + " | ".join(f"{value:.4f}" for value in row) + " |")
    lines += [
        "", "This is the average of the draw-wise drift matrices, not the drift obtained by "
        "fitting the 50%-threshold graph. Small nonzero mean entries outside the two majority "
        "edges arise from edges visited in fewer than half the draws. "
        "Full-precision values are in `mean_drift_288000_331000.csv`; edge probabilities are in "
        "`edge_inclusion_288000_331000.csv`.", "",
    ]
    (HERE / "mean_graph_288000_331000.md").write_text("\n".join(lines), encoding="utf-8")
    plot(probabilities, mean_m, n)
    print(f"Saved interval analysis in {HERE}")


if __name__ == "__main__":
    main()
