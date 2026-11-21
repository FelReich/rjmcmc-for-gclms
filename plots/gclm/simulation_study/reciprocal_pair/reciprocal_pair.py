import csv
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


HERE = Path(__file__).resolve().parent
CASE = get_simulation_case("G2_n800_lambda5_general_base")
CHAIN = (CODE_ROOT / "examples/gclm/simulation_study/runs" / CASE.id
         / "chain_01.csv")
VIEW_START, VIEW_END = 260_000, 340_000
ROLLING_WINDOW = 1_000
HIGH_K_THRESHOLD = 3.3
EDGES = tuple((i, j) for i in range(CASE.generating_model.p)
              for j in range(CASE.generating_model.p) if i != j)
PAIRS = tuple((i, j) for i in range(CASE.generating_model.p)
              for j in range(i + 1, CASE.generating_model.p))


def load_chain():
    if CASE.proposal.sigma_r != 0.2:
        raise ValueError("These figures are for the sigma_r=0.2 baseline only")
    with CHAIN.open(newline="", encoding="utf-8") as file:
        first = next(csv.DictReader(file), None)
    if first is None or (
        first["configuration_id"] != CASE.id
        or first["chain_number"] != "1"
        or first["chain_seed"] != str(CASE.chain_seeds[0])
        or first["adaptive_margin"] != "0"
        or first["cycle_moves"] != "0"
    ):
        raise ValueError(f"Chain metadata do not match the baseline settings: {CHAIN}")
    data = pd.read_csv(
        CHAIN, usecols=["iteration", "k", "eta", "log_likelihood", "graph_bitmask"]
    )
    expected = CASE.iterations - CASE.burn_in
    if len(data) != expected or not np.array_equal(
        data["iteration"].to_numpy(), np.arange(CASE.burn_in + 1, CASE.iterations + 1)
    ):
        raise ValueError("Chain 1 has missing or misordered retained draws")
    if not np.isfinite(data[["eta", "log_likelihood"]].to_numpy()).all():
        raise ValueError("Chain 1 has non-finite trace values")
    return data


def episodes(indicator):
    boundaries = np.flatnonzero(np.diff(np.r_[False, indicator, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(boundaries[::2], boundaries[1::2])]


def identify_regime(k):
    smooth = np.convolve(k, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="same")
    candidates = [(a, b) for a, b in episodes(smooth >= HIGH_K_THRESHOLD)
                  if a < VIEW_END and b > VIEW_START]
    if not candidates:
        raise ValueError("No elevated-k run overlaps the selected view")
    return max(candidates, key=lambda interval: interval[1] - interval[0])


def reciprocal_pairs(graphs):
    bits = {edge: 1 << index for index, edge in enumerate(EDGES)}
    return np.column_stack([
        ((graphs & bits[i, j]) != 0) & ((graphs & bits[j, i]) != 0)
        for i, j in PAIRS
    ])


def plot_focused_traces(data, presence, regime, pair_episode, important_pairs):
    a, b = regime
    x = np.arange(VIEW_START + 1, VIEW_END + 1)
    view = slice(VIEW_START, VIEW_END)
    figure, axes = plt.subplots(5, 1, figsize=(11, 9.5), sharex=True,
                                constrained_layout=True)
    values = (data["k"].to_numpy()[view], presence[view].sum(axis=1),
              data["eta"].to_numpy()[view],
              data["log_likelihood"].to_numpy()[view])
    labels = ("Edges $k$", "Reciprocal pairs $c(G)$", "Margin $\\eta$",
              "Log-likelihood")
    for axis, value, label in zip(axes[:4], values, labels):
        axis.plot(x, value, lw=0.5, color="#245b8f", rasterized=True)
        axis.set_ylabel(label)
        axis.grid(alpha=0.18)
    for pair_index in important_pairs:
        i, j = PAIRS[pair_index]
        axes[4].step(x, presence[view, pair_index].astype(int), where="post",
                     lw=0.75, label=f"{i + 1}↔{j + 1}", rasterized=True)
    axes[4].set(ylabel="Pair present", yticks=[0, 1],
                xlabel="Retained iteration after burn-in")
    axes[4].legend(loc="upper right", ncol=len(important_pairs), fontsize=8)
    for axis in axes[:2]:
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in axes:
        axis.axvspan(a + 1, b, color="#ffca74", alpha=0.18)
        axis.axvline(pair_episode[0] + 1, color="#a33b2b", lw=0.8, ls="--")
        axis.axvline(pair_episode[1], color="#a33b2b", lw=0.8, ls="--")
        axis.set_xlim(VIEW_START + 1, VIEW_END)
    path = HERE / "focused_traces.pdf"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_regime_distributions(data, regime):
    a, b = regime
    local_flanks = np.r_[VIEW_START:a, b:VIEW_END]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, variable, title in zip(
        axes, ("k", "eta", "log_likelihood"),
        ("Number of edges", "Stability margin", "Total log-likelihood"),
    ):
        inside = data[variable].to_numpy()[a:b]
        outside = data[variable].to_numpy()[local_flanks]
        if variable == "k":
            bins = np.arange(min(inside.min(), outside.min()) - 0.5,
                             max(inside.max(), outside.max()) + 1.5)
        else:
            bins = np.histogram_bin_edges(np.r_[inside, outside], bins=45)
        axis.hist(inside, bins=bins, density=True, alpha=0.55,
                  label="Elevated regime", color="#245b8f")
        axis.hist(outside, bins=bins, density=True, alpha=0.55,
                  label="Local flanks", color="#d58232")
        axis.set(xlabel=title, ylabel="Density")
        axis.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    path = HERE / "regime_distributions.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path


def main():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    data = load_chain()
    regime = identify_regime(data["k"].to_numpy())
    presence = reciprocal_pairs(data["graph_bitmask"].to_numpy(dtype=np.int64))
    a, b = regime
    important_pairs = np.argsort(presence[a:b].mean(axis=0))[::-1][:3]
    dominant = int(important_pairs[0])
    candidates = [(start, end) for start, end in episodes(presence[:, dominant])
                  if start < b and end > a]
    pair_episode = max(candidates, key=lambda interval: interval[1] - interval[0])
    focused = plot_focused_traces(data, presence, regime, pair_episode,
                                  important_pairs)
    distributions = plot_regime_distributions(data, regime)
    print(f"Saved {focused} and {distributions}")


if __name__ == "__main__":
    main()
