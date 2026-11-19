"""Brooks-Giudici plots for eta in the G2, n=800, lambda_P=5 base runs.

Run from the Code directory:
    python3 plots/gclm/simulation_study/brooks_giudici/brooks_giudici_base.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[4]
RUNS = CODE_ROOT / "examples/gclm/simulation_study/runs"
HERE = Path(__file__).resolve().parent
REFERENCE = 160_000
CHECKPOINTS = (160_000, 200_000, 250_000, 300_000,
               350_000, 400_000, 450_000, 500_000)
COMPARISONS = (
    ("Overall", "V_hat", "W_c", "V_hat_over_W_c"),
    ("Within model", "W_m", "W_m_W_c", "W_m_over_W_m_W_c"),
    ("Between model", "B_m", "B_m_W_c", "B_m_over_B_m_W_c"),
)


def load_chains(space):
    """Read only graph identities and margins; verify five distinct chains."""
    case_id = f"G2_n800_lambda5_{space}_base"
    folder = RUNS / case_id
    chains, seeds, dataset_seeds = [], set(), set()
    for number in range(1, 6):
        path = folder / f"chain_{number:02d}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            first = next(csv.DictReader(stream), None)
        if first is None or any((
            first["configuration_id"] != case_id,
            first["generating_model"] != "G2",
            first["n"] != "800",
            first["inference_graph_space"] != space,
            float(first["lambda_p"]) != 5.0,
            first["adaptive_margin"] != "0",
            first["cycle_moves"] != "0",
            first["chain_number"] != str(number),
            first["iteration"] != "500001",
        )):
            raise ValueError(f"Unexpected chain metadata in {path}")
        seeds.add(first["chain_seed"])
        dataset_seeds.add(first["dataset_seed"])
        frame = pd.read_csv(path, usecols=["graph_bitmask", "eta"])
        if (len(frame) != CHECKPOINTS[-1]
                or not np.isfinite(frame["eta"].to_numpy()).all()
                or (frame["eta"] <= 0).any()):
            raise ValueError(f"Incomplete chain or invalid eta in {path}")
        chains.append((frame["graph_bitmask"].to_numpy(dtype=np.int64),
                       frame["eta"].to_numpy(dtype=float)))
    if len(seeds) != 5 or len(dataset_seeds) != 1:
        raise ValueError(f"{case_id} needs five distinct chain seeds on one dataset")
    return chains


def select_graphs(chains):
    """Fix the graph set before computing the progression."""
    eligible = []
    for masks, _ in chains:
        labels, counts = np.unique(masks[:REFERENCE], return_counts=True)
        eligible.append(set(labels[counts >= 2].tolist()))
    models = np.array(sorted(set.intersection(*eligible)), dtype=np.int64)
    if len(models) < 2:
        raise ValueError("Fewer than two graphs have two visits in every chain")
    return models


def variance_estimates(chains, models, n, center):
    """Compute the six estimators in BaseRJMCMC.two_way_anova from cell sums.

    Centering eta by one constant improves numerical accuracy without changing
    any variance or ratio. This uses the same unweighted B_m and B_mW_c and the
    same corrected variance denominators as the base implementation.
    """
    n_chains, n_models = len(chains), len(models)
    counts = np.zeros((n_chains, n_models), dtype=np.int64)
    sums = np.zeros((n_chains, n_models))
    squares = np.zeros((n_chains, n_models))
    for i, (masks, margins) in enumerate(chains):
        prefix = masks[:n]
        indices = np.searchsorted(models, prefix)
        selected = (indices < n_models) & (models[np.minimum(indices, n_models - 1)] == prefix)
        cells = indices[selected]
        values = margins[:n][selected] - center
        counts[i] = np.bincount(cells, minlength=n_models)
        sums[i] = np.bincount(cells, weights=values, minlength=n_models)
        squares[i] = np.bincount(cells, weights=values * values, minlength=n_models)
    if (counts < 2).any():
        raise ValueError(f"Selected graph has fewer than two visits at {n} draws")

    chain_counts = counts.sum(axis=1)
    model_counts = counts.sum(axis=0)
    total = int(chain_counts.sum())
    chain_sums, model_sums = sums.sum(axis=1), sums.sum(axis=0)
    chain_squares, model_squares = squares.sum(axis=1), squares.sum(axis=0)
    grand_mean = sums.sum() / total
    chain_means = chain_sums / chain_counts
    model_means = model_sums / model_counts
    cell_means = sums / counts

    # These are exactly the variance components used by two_way_anova().
    v_hat = max(squares.sum() - sums.sum() ** 2 / total, 0.0) / (total - 1)
    w_c = np.mean(np.maximum(chain_squares - chain_sums**2 / chain_counts, 0.0)
                  / (chain_counts - 1))
    w_m = np.mean(np.maximum(model_squares - model_sums**2 / model_counts, 0.0)
                  / (model_counts - 1))
    w_m_w_c = np.mean(np.maximum(squares - sums**2 / counts, 0.0) / (counts - 1))
    b_m = np.sum((model_means - grand_mean)**2) / (n_models - 1)
    b_m_w_c = np.mean(np.sum((cell_means - chain_means[:, None])**2, axis=1)
                      / (n_models - 1))
    result = dict(V_hat=v_hat, W_c=w_c, W_m=w_m, W_m_W_c=w_m_w_c,
                  B_m=b_m, B_m_W_c=b_m_w_c, coverage=total / (n_chains * n))
    result.update(
        V_hat_over_W_c=v_hat / w_c,
        W_m_over_W_m_W_c=w_m / w_m_w_c,
        B_m_over_B_m_W_c=b_m / b_m_w_c,
    )
    return result


def plot_case(space):
    chains = load_chains(space)
    models = select_graphs(chains)
    center = float(chains[0][1].mean())
    results = [variance_estimates(chains, models, n, center)
               for n in CHECKPOINTS]

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharex=True)
    for axis, (title, left, right, ratio) in zip(axes, COMPARISONS):
        axis.plot(CHECKPOINTS, [row[left] for row in results], "o-",
                  markersize=3.5, linewidth=1.4, label=left.replace("_", " "))
        axis.plot(CHECKPOINTS, [row[right] for row in results], "s--",
                  markersize=3.2, linewidth=1.4, label=right.replace("_", " "))
        axis.set_title(f"{title}\nfinal ratio = {results[-1][ratio]:.3f}")
        axis.set_xlabel("Retained iterations per chain")
        axis.set_xticks((160_000, 250_000, 350_000, 500_000),
                        ("160k", "250k", "350k", "500k"))
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel(r"Variance estimate for $\eta$")
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    output = HERE / f"brooks_giudici_base_{space}.pdf"
    figure.savefig(output)
    plt.close(figure)
    print(f"Saved {output} ({len(models)} graphs, "
          f"{results[-1]['coverage']:.1%} final coverage)")


def main():
    for space in ("general", "simple"):
        plot_case(space)


if __name__ == "__main__":
    main()
