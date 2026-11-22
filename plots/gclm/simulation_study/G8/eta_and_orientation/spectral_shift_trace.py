"""Plot phi(H), eta, and phi(H)+eta for the first G8 general baseline chain."""

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


CASE = get_simulation_case("G8_n800_lambda5_general_base")
CHAIN_NUMBER = 1
INPUT = (CODE_ROOT / "examples/gclm/simulation_study/runs" / CASE.id
         / f"chain_{CHAIN_NUMBER:02d}.csv")
OUTPUT = Path(__file__).with_suffix(".pdf")
COLUMNS = ["configuration_id", "chain_number", "chain_seed", "iteration",
           "phi_H", "eta", "stability_shift"]


def main():
    data = pd.read_csv(INPUT, usecols=COLUMNS)
    expected = (CASE.iterations - CASE.burn_in + CASE.thinning - 1) // CASE.thinning
    if len(data) != expected:
        raise ValueError(f"{INPUT} has {len(data)} draws; expected {expected}")
    if (not (data["configuration_id"] == CASE.id).all()
            or not (data["chain_number"] == CHAIN_NUMBER).all()
            or not (data["chain_seed"] == CASE.chain_seeds[CHAIN_NUMBER - 1]).all()):
        raise ValueError(f"Chain metadata do not match settings: {INPUT}")
    iterations = data["iteration"].to_numpy(dtype=np.int64)
    if not np.array_equal(iterations, np.arange(CASE.burn_in + 1,
                                                CASE.iterations + 1, CASE.thinning)):
        raise ValueError("Unexpected retained iteration numbers")
    phi_h = data["phi_H"].to_numpy(dtype=float)
    eta = data["eta"].to_numpy(dtype=float)
    shift = data["stability_shift"].to_numpy(dtype=float)
    if (not np.isfinite(np.column_stack((phi_h, eta, shift))).all()
            or not np.allclose(phi_h + eta, shift, rtol=1e-12, atol=1e-12)):
        raise ValueError("Invalid or inconsistent spectral shift values")

    x = np.arange(expected)  # Zero-based retained iteration after burn-in.
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                                constrained_layout=True)
    for axis, values, label in zip(
        axes, (phi_h, eta, shift),
        (r"Spectral abscissa $\phi(H)$", r"Stability margin $\eta$",
         r"Stability shift $\phi(H)+\eta$"),
    ):
        axis.plot(x, values, color="#245b8f", linewidth=0.4, rasterized=True)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlim(0, expected)
    axes[-1].set_xlabel("Retained iteration after burn-in")
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
