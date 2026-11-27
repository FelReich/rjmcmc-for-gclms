"""Summarize the five Raf-removed simple-LI chains and plot the first chain.

Writes overview.csv and trace_plots.pdf beside this script. The summaries use
retained draws, matching the full protein-data overview.
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import arviz as az
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[5]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.protein_data.settings import get_simulation_case  # noqa: E402
from plots.gclm.protein_data.overview.overview import (  # noqa: E402
    QUANTITIES, add_totals, diagnostic_value, empty_totals, plot_trace,
    read_chain, summary_row,
)


HERE = Path(__file__).resolve().parent
PROTEIN_DIR = CODE_ROOT / "examples/gclm/protein_data/remove_protein"
RUNS = PROTEIN_DIR / "runs"
DATA_PATH = PROTEIN_DIR / "preprocessed_remove_protein.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-only", action="store_true",
        help="Regenerate the overview CSV without replacing the trace PDF",
    )
    args = parser.parse_args()

    case = replace(
        get_simulation_case("simple_li"),
        id="simple_li_remove_raf",
        data_path=DATA_PATH,
    )
    data = pd.read_csv(case.data_path)
    n_observations, p = data.shape
    if p != 10 or "Raf" in data.columns:
        raise ValueError("Expected ten preprocessed protein columns without Raf")

    rows = []
    pooled = empty_totals()
    chain_traces = []
    for chain_number, seed in enumerate(case.chain_seeds, start=1):
        path = RUNS / f"chain_{chain_number:02d}.csv.gz"
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

    with (HERE / "overview.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.csv_only:
        plot_trace(chain_traces[0], HERE / "trace_plots.pdf")
    print(f"Saved {HERE / 'overview.csv'}" +
          ("" if args.csv_only else f" and {HERE / 'trace_plots.pdf'}"),
          flush=True)


if __name__ == "__main__":
    main()
