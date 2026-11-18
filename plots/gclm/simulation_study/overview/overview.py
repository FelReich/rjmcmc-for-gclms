"""Summarize the saved GCLM runs and plot the first chain of each case.

From the Code directory:
    python3 plots/gclm/simulation_study/overview/overview.py

Each case gets overview.csv (five chains and a pooled row) and trace_plots.pdf.
Acceptance rates and all summaries use retained, post-burn-in draws only.
The stored log-likelihood is the total log-likelihood up to a constant.
ESS is ArviZ's bulk ESS; R-hat is its rank-normalized split/folded diagnostic.
Each chain has its own ESS; pooled ESS and R-hat use all five chains.
"""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import arviz as az
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.simulation_study.settings import get_simulation_cases


HERE = Path(__file__).resolve().parent
RUNS = CODE_ROOT / "examples/gclm/simulation_study/runs"
QUANTITIES = ("k", "eta", "log_likelihood")
READ_COLUMNS = list(QUANTITIES) + ["graph_bitmask", "accepted", "graph_changed"]


def empty_totals():
    return dict(draws=0, sums=np.zeros(3), squares=np.zeros(3),
                accepted=0, graph_changed=0, true_k=0, true_graph=0)


def add_totals(target, source):
    for name in ("draws", "accepted", "graph_changed", "true_k", "true_graph"):
        target[name] += source[name]
    target["sums"] += source["sums"]
    target["squares"] += source["squares"]


def true_graph_bitmask(case):
    edges = [(i, j) for i in range(case.generating_model.p)
             for j in range(case.generating_model.p) if i != j]
    return sum(1 << index for index, edge in enumerate(edges)
               if edge in case.generating_model.ordered_edges)


def read_chain(path, case, chain_number, seed):
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed chain: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        first = next(csv.DictReader(file), None)
    if first is None or (first["configuration_id"], first["chain_number"],
                         first["chain_seed"]) != (case.id, str(chain_number), str(seed)):
        raise ValueError(f"Chain metadata do not match settings: {path}")

    totals = empty_totals()
    trace = {name: [] for name in QUANTITIES}
    true_k = len(case.generating_model.ordered_edges)
    bitmask = true_graph_bitmask(case)
    for chunk in pd.read_csv(path, usecols=READ_COLUMNS, chunksize=100_000):
        values = chunk[list(QUANTITIES)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite trace value in {path}")
        totals["draws"] += len(chunk)
        totals["sums"] += values.sum(axis=0)
        totals["squares"] += np.square(values).sum(axis=0)
        totals["accepted"] += int(chunk["accepted"].sum())
        totals["graph_changed"] += int(chunk["graph_changed"].sum())
        totals["true_k"] += int((chunk["k"] == true_k).sum())
        totals["true_graph"] += int((chunk["graph_bitmask"] == bitmask).sum())
        for index, name in enumerate(QUANTITIES):
            trace[name].append(values[:, index].copy())

    expected = (case.iterations - case.burn_in + case.thinning - 1) // case.thinning
    if totals["draws"] != expected:
        raise ValueError(f"{path} has {totals['draws']} draws; expected {expected}")
    trace = {name: np.concatenate(parts) for name, parts in trace.items()}
    return totals, trace


def summary_row(case, label, seed, totals):
    n = totals["draws"]
    means = totals["sums"] / n
    variances = np.maximum(
        (totals["squares"] - totals["sums"] ** 2 / n) / (n - 1), 0.0
    )
    row = {
        "configuration_id": case.id,
        "chain": label,
        "chain_seed": seed,
        "iterations": case.iterations,
        "burn_in": case.burn_in,
        "retained_draws": n,
        "acceptance_rate_retained": totals["accepted"] / n,
        "graph_change_rate_retained": totals["graph_changed"] / n,
        "posterior_p_true_k": totals["true_k"] / n,
        "posterior_p_true_graph": totals["true_graph"] / n,
    }
    for index, name in enumerate(QUANTITIES):
        row[f"mean_{name}"] = means[index]
        row[f"sd_{name}"] = np.sqrt(variances[index])
        row[f"ess_bulk_{name}"] = ""
        row[f"rhat_rank_{name}"] = ""  # Requires multiple chains.
    return row


def diagnostic_value(value):
    """Leave an undefined diagnostic blank in the CSV."""
    value = float(value)
    return value if np.isfinite(value) else ""


def plot_trace(case, trace, destination):
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                                constrained_layout=True)
    x = np.arange(1, len(trace["k"]) + 1)
    for axis, name, ylabel in zip(
        axes, QUANTITIES, ("Number of edges $k$", "Stability margin $\\eta$",
                            "Total log-likelihood"),
    ):
        axis.plot(x, trace[name], color="#245b8f", linewidth=0.4,
                  rasterized=True)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    axes[-1].set_xlim(0, len(x))
    axes[-1].set_xlabel("Retained iteration after burn-in")
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def process_case(case):
    folder = HERE / case.id
    rows = []
    pooled = empty_totals()
    chain_traces = []
    for chain_number, seed in enumerate(case.chain_seeds, start=1):
        path = RUNS / case.id / f"chain_{chain_number:02d}.csv"
        totals, trace = read_chain(path, case, chain_number, seed)
        add_totals(pooled, totals)
        row = summary_row(case, chain_number, seed, totals)
        for name in QUANTITIES:
            row[f"ess_bulk_{name}"] = diagnostic_value(
                az.ess(trace[name][None, :], method="bulk")
            )
        rows.append(row)
        chain_traces.append(trace)

    pooled_row = summary_row(case, "pooled", "", pooled)
    for name in QUANTITIES:
        values = np.stack([trace[name] for trace in chain_traces])
        pooled_row[f"ess_bulk_{name}"] = diagnostic_value(
            az.ess(values, method="bulk")
        )
        pooled_row[f"rhat_rank_{name}"] = diagnostic_value(
            az.rhat(values, method="rank")
        )
    rows.append(pooled_row)

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "overview.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_trace(case, chain_traces[0], folder / "trace_plots.pdf")
    print(f"Saved {folder.name}/overview.csv and trace_plots.pdf", flush=True)


def main():
    cases = get_simulation_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.id for case in cases],
                        help="Process one case (default: all 11)")
    args = parser.parse_args()
    for case in cases:
        if args.case is None or case.id == args.case:
            process_case(case)


if __name__ == "__main__":
    main()
