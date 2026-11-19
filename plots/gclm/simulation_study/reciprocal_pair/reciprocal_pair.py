"""Recreate only the focused reciprocal-pair trace for the base general case.

Run from the Code directory:
    python3 plots/gclm/simulation_study/reciprocal_pair/reciprocal_pair.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.simulation_study.settings import get_simulation_case


CASE = get_simulation_case("G2_n800_lambda5_general_base")
SOURCE = (CODE_ROOT / "examples/gclm/simulation_study/runs" / CASE.id
          / "chain_01.csv")
OUTPUT = Path(__file__).with_name("focus_240000_350000.pdf")
START, END = 240_000, 350_000
HIGHLIGHT_START, HIGHLIGHT_END = 288_000, 331_000


def main():
    draws = pd.read_csv(SOURCE, usecols=["graph_bitmask", "k", "eta"])
    if len(draws) != (CASE.iterations - CASE.burn_in) // CASE.thinning:
        raise ValueError("The first chain does not contain all retained draws")

    mask = draws["graph_bitmask"].to_numpy(dtype=np.int64)
    k = draws["k"].to_numpy(dtype=np.int64)
    eta = draws["eta"].to_numpy(dtype=float)
    p = CASE.generating_model.p
    edges = tuple((i, j) for i in range(p) for j in range(p) if i != j)
    edge_index = {edge: index for index, edge in enumerate(edges)}
    pairs = tuple((i, j) for i in range(p) for j in range(i + 1, p))
    pair_flags = np.column_stack([
        ((mask >> edge_index[(i, j)]) & 1).astype(bool)
        & ((mask >> edge_index[(j, i)]) & 1).astype(bool)
        for i, j in pairs
    ])
    reciprocal_count = pair_flags.sum(axis=1)
    leading_pair = int(np.argmax(
        pair_flags[HIGHLIGHT_START:HIGHLIGHT_END].mean(axis=0)
    ))
    i, j = pairs[leading_pair]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "pdf.fonttype": 42,
                         "savefig.bbox": "tight"})
    figure, axes = plt.subplots(4, 1, figsize=(9, 7.7), sharex=True,
                               constrained_layout=True)
    x = np.arange(START, END)
    series = (
        (k, "Off-diagonal edges $k$", "#254b70"),
        (reciprocal_count, "Reciprocal pairs $c(G)$", "#b34e38"),
        (eta, "Stability margin $\\eta$", "#4a806b"),
        (pair_flags[:, leading_pair].astype(int),
         f"Pair ${i + 1}\\leftrightarrow{j + 1}$", "#946c21"),
    )
    for axis, (values, ylabel, color) in zip(axes, series):
        axis.plot(x, values[START:END], lw=.35, color=color, rasterized=True)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.18)
        axis.axvspan(HIGHLIGHT_START, HIGHLIGHT_END,
                     color="#cfa82f", alpha=.12)
    for axis in (axes[0], axes[1], axes[3]):
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel("Retained iteration after burn-in (zero-based)")
    axes[-1].set_xlim(START, END)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
