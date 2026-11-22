"""Plot eta conditional on the 2-5 edge orientation in all five G8 simple chains."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[5]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.simulation_study.settings import get_simulation_case


CASE = get_simulation_case("G8_n800_lambda5_simple_base")
RUNS = CODE_ROOT / "examples/gclm/simulation_study/runs" / CASE.id
OUTPUT = Path(__file__).with_suffix(".pdf")
COLUMNS = [
    "configuration_id", "chain_number", "chain_seed", "eta",
    "edge_1_4", "edge_4_1",  # Zero-based: 2->5 and 5->2.
]


def conditional_samples():
    """Pool retained draws; keep the two conditional samples separate."""
    forward, reverse = [], []
    expected = (CASE.iterations - CASE.burn_in + CASE.thinning - 1) // CASE.thinning
    for chain_number, seed in enumerate(CASE.chain_seeds, start=1):
        path = RUNS / f"chain_{chain_number:02d}.csv"
        draws = 0
        for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=100_000):
            if (not (chunk["configuration_id"] == CASE.id).all()
                    or not (chunk["chain_number"] == chain_number).all()
                    or not (chunk["chain_seed"] == seed).all()):
                raise ValueError(f"Chain metadata do not match settings: {path}")
            edges = chunk[["edge_1_4", "edge_4_1"]].to_numpy(dtype=bool)
            if np.any(edges[:, 0] & edges[:, 1]):
                raise ValueError(f"Reciprocal 2-5 pair in a simple graph: {path}")
            eta = chunk["eta"].to_numpy(dtype=float)
            if not np.isfinite(eta).all() or np.any(eta <= 0):
                raise ValueError(f"Invalid eta in {path}")
            forward.append(eta[edges[:, 0]])
            reverse.append(eta[edges[:, 1]])
            draws += len(chunk)
        if draws != expected:
            raise ValueError(f"{path} has {draws} draws; expected {expected}")

    forward = np.concatenate(forward)
    reverse = np.concatenate(reverse)
    if not len(forward) or not len(reverse):
        raise ValueError("Both orientations need retained observations")
    return forward, reverse


def main():
    forward, reverse = conditional_samples()
    bins = np.linspace(min(forward.min(), reverse.min()),
                       max(forward.max(), reverse.max()), 100)
    figure, axis = plt.subplots(figsize=(8.3, 4.8))
    for values, label, color in (
        (forward, r"$2\to5$ present", "#245b8f"),
        (reverse, r"$5\to2$ present", "#c26b31"),
    ):
        density, _ = np.histogram(values, bins=bins, density=True)
        axis.stairs(density, bins, color=color, linewidth=1.6,
                    label=f"{label} ($N={len(values):,}$)")
    axis.set(xlabel=r"Stability margin $\eta$", ylabel="Conditional density")
    axis.set_xlim(bins[0], bins[-1])
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT)
    plt.close(figure)
    print(f"Saved {OUTPUT}")
    print(f"2->5: {len(forward):,} draws, mean eta {forward.mean():.4f}")
    print(f"5->2: {len(reverse):,} draws, mean eta {reverse.mean():.4f}")


if __name__ == "__main__":
    main()
