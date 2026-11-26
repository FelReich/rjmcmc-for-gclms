"""Summarize the saved protein-data chains and plot each case's first chain.

Each case gets overview.csv (five chains and a pooled row) and trace_plots.pdf.
All summaries use retained draws. The stored log-likelihood is the total
log-likelihood up to a constant. ESS is ArviZ bulk ESS; R-hat is the
rank-normalized split/folded diagnostic and is reported only for pooled chains.
"""

import argparse
import csv
import gzip
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

from examples.gclm.protein_data.settings import get_simulation_cases


HERE = Path(__file__).resolve().parent
RUNS = CODE_ROOT / "examples/gclm/protein_data/runs"
QUANTITIES = ("k", "eta", "log_likelihood")
READ_COLUMNS = list(QUANTITIES) + ["accepted", "graph_changed"]


def empty_totals():
    return dict(draws=0, sums=np.zeros(3), squares=np.zeros(3),
                accepted=0, graph_changed=0)


def add_totals(target, source):
    for name in ("draws", "accepted", "graph_changed"):
        target[name] += source[name]
    target["sums"] += source["sums"]
    target["squares"] += source["squares"]


def read_chain(path, case, chain_number, seed, n, p):
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed chain: {path}")
    with gzip.open(path, "rt", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        first = next(reader, None)
        if first is None or not set(READ_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Missing data columns in {path}")
    expected_metadata = {
        "configuration_id": case.id,
        "n": str(n),
        "p": str(p),
        "inference_graph_space": case.prior.graph_space,
        "lambda_p": format(case.prior.lambda_p, ".17g"),
        "adaptive_margin": str(int(case.proposal.adaptive_margin)),
        "cycle_moves": str(int(case.proposal.cycle_moves)),
        "chain_number": str(chain_number),
        "chain_seed": str(seed),
        "iteration": str(case.burn_in + 1),
    }
    if any(first.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError(f"Chain metadata do not match settings: {path}")

    totals = empty_totals()
    trace_parts = {name: [] for name in QUANTITIES}
    for chunk in pd.read_csv(path, usecols=READ_COLUMNS, chunksize=100_000):
        values = chunk[list(QUANTITIES)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite trace value in {path}")
        totals["draws"] += len(chunk)
        totals["sums"] += values.sum(axis=0)
        totals["squares"] += np.square(values).sum(axis=0)
        totals["accepted"] += int(chunk["accepted"].sum())
        totals["graph_changed"] += int(chunk["graph_changed"].sum())
        for index, name in enumerate(QUANTITIES):
            trace_parts[name].append(values[:, index].copy())

    expected_draws = (case.iterations - case.burn_in + case.thinning - 1) // case.thinning
    if totals["draws"] != expected_draws:
        raise ValueError(f"{path} has {totals['draws']} draws; expected {expected_draws}")
    return totals, {name: np.concatenate(parts) for name, parts in trace_parts.items()}


def summary_row(case, label, seed, totals, n_observations, p):
    draws = totals["draws"]
    means = totals["sums"] / draws
    variances = np.maximum(
        (totals["squares"] - totals["sums"] ** 2 / draws) / (draws - 1), 0.0
    )
    row = {
        "configuration_id": case.id,
        "graph_space": case.prior.graph_space,
        "n": n_observations,
        "p": p,
        "lambda_p": case.prior.lambda_p,
        "chain": label,
        "chain_seed": seed,
        "iterations": case.iterations,
        "burn_in": case.burn_in,
        "retained_draws": draws,
        "acceptance_rate_retained": totals["accepted"] / draws,
        "graph_change_rate_retained": totals["graph_changed"] / draws,
    }
    for index, name in enumerate(QUANTITIES):
        row[f"mean_{name}"] = means[index]
        row[f"sd_{name}"] = np.sqrt(variances[index])
        row[f"ess_bulk_{name}"] = ""
        row[f"rhat_rank_{name}"] = ""
    return row


def diagnostic_value(value):
    value = float(value)
    return value if np.isfinite(value) else ""


def plot_trace(trace, destination):
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                                constrained_layout=True)
    x = np.arange(1, len(trace["k"]) + 1)
    for axis, name, ylabel in zip(
        axes, QUANTITIES,
        ("Number of edges $k$", "Stability margin $\\eta$", "log-likelihood"),
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


def process_case(case, *, csv_only=False):
    data = pd.read_csv(case.data_path)
    n_observations, p = data.shape
    rows = []
    pooled = empty_totals()
    chain_traces = []
    for chain_number, seed in enumerate(case.chain_seeds, start=1):
        path = RUNS / case.id / f"chain_{chain_number:02d}.csv.gz"
        totals, trace = read_chain(path, case, chain_number, seed, n_observations, p)
        add_totals(pooled, totals)
        row = summary_row(case, chain_number, seed, totals, n_observations, p)
        for name in QUANTITIES:
            row[f"ess_bulk_{name}"] = diagnostic_value(
                az.ess(trace[name][None, :], method="bulk")
            )
        rows.append(row)
        chain_traces.append(trace)
        print(f"{case.id}: loaded chain {chain_number}/{len(case.chain_seeds)}",
              flush=True)

    pooled_row = summary_row(case, "pooled", "", pooled, n_observations, p)
    for name in QUANTITIES:
        values = np.stack([trace[name] for trace in chain_traces])
        pooled_row[f"ess_bulk_{name}"] = diagnostic_value(
            az.ess(values, method="bulk")
        )
        pooled_row[f"rhat_rank_{name}"] = diagnostic_value(
            az.rhat(values, method="rank")
        )
    rows.append(pooled_row)

    folder = HERE / case.id
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "overview.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not csv_only:
        plot_trace(chain_traces[0], folder / "trace_plots.pdf")
    print(f"Saved {folder / 'overview.csv'}" +
          ("" if csv_only else f" and {folder / 'trace_plots.pdf'}"), flush=True)


def main():
    cases = get_simulation_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.id for case in cases],
                        help="Process one case (default: both cases)")
    parser.add_argument("--csv-only", action="store_true",
                        help="Regenerate overview CSVs without replacing trace PDFs")
    args = parser.parse_args()
    for case in cases:
        if args.case is None or case.id == args.case:
            process_case(case, csv_only=args.csv_only)


if __name__ == "__main__":
    main()
