"""Compare posterior mean drift matrices with the true G2 drift matrix.

For each baseline graph space, pool the five retained G2, n=800, lambda_P=5
chains. The MAP-conditional mean uses only draws of the most frequent complete
graph. The overall mean uses every retained draw. Matrix rows are targets and
columns are sources, as in the sampler's stored M_0_0, ..., M_4_4 columns.
"""

from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.simulation_study.settings import get_simulation_case


HERE = Path(__file__).resolve().parent
RUNS = CODE_ROOT / "examples/gclm/simulation_study/runs"
OUTPUT = HERE / "drift_comparison.csv"
CASE_IDS = (
    "G2_n800_lambda5_general_base",
    "G2_n800_lambda5_simple_base",
)


def summarize_case(case):
    """Accumulate graph frequencies and drift-matrix sums in one CSV pass."""
    p = case.generating_model.p
    matrix_columns = [f"M_{i}_{j}" for i in range(p) for j in range(p)]
    counts = Counter()
    sums_by_graph = {}
    overall_sum = np.zeros(p * p)
    total = 0
    expected = (case.iterations - case.burn_in + case.thinning - 1) // case.thinning

    for chain_number, seed in enumerate(case.chain_seeds, start=1):
        path = RUNS / case.id / f"chain_{chain_number:02d}.csv"
        with path.open(newline="", encoding="utf-8") as file:
            first = next(csv.DictReader(file), None)
        if first is None or (
            first["configuration_id"], first["chain_number"],
            first["chain_seed"], first["dataset_seed"],
            first["adaptive_margin"], first["cycle_moves"],
        ) != (
            case.id, str(chain_number), str(seed),
            str(case.generating_model.dataset_seed), "0", "0",
        ):
            raise ValueError(f"Chain metadata do not match the baseline case: {path}")

        chain_draws = 0
        for chunk in pd.read_csv(path, usecols=["graph_bitmask", *matrix_columns],
                                 chunksize=100_000):
            matrices = chunk[matrix_columns].to_numpy(dtype=float)
            if not np.isfinite(matrices).all():
                raise ValueError(f"Non-finite drift matrix in {path}")
            overall_sum += matrices.sum(axis=0)

            grouped = chunk.groupby("graph_bitmask", sort=False)
            group_counts = grouped.size()
            for mask, values in grouped[matrix_columns].sum().iterrows():
                mask = int(mask)
                counts[mask] += int(group_counts.loc[mask])
                if mask not in sums_by_graph:
                    sums_by_graph[mask] = np.zeros(p * p)
                sums_by_graph[mask] += values.to_numpy(dtype=float)
            chain_draws += len(chunk)

        if chain_draws != expected:
            raise ValueError(f"{path} has {chain_draws} draws; expected {expected}")
        total += chain_draws

    if sum(counts.values()) != total:
        raise ValueError("Graph counts do not match the retained-draw count")
    # Resolve an exact tie by choosing the smaller canonical graph bitmask.
    map_mask, map_visits = max(counts.items(), key=lambda item: (item[1], -item[0]))
    edge_order = [(i, j) for i in range(p) for j in range(p) if i != j]
    map_edges = [[i + 1, j + 1] for index, (i, j) in enumerate(edge_order)
                 if map_mask & (1 << index)]
    true_drift = np.asarray(case.generating_model.drift, dtype=float)
    true_norm = float(np.linalg.norm(true_drift, ord="fro"))

    rows = []
    for estimate, matrix_sum, draws_used in (
        ("MAP-conditional mean", sums_by_graph[map_mask], map_visits),
        ("Overall mean", overall_sum, total),
    ):
        mean_drift = (matrix_sum / draws_used).reshape(p, p)
        absolute_error = float(np.linalg.norm(mean_drift - true_drift, ord="fro"))
        row = {
            "configuration_id": case.id,
            "graph_space": case.prior.graph_space,
            "estimate": estimate,
            "draws_used": draws_used,
            "total_retained_draws": total,
            "map_graph_bitmask": map_mask,
            "map_graph_edges_one_based": json.dumps(map_edges, separators=(",", ":")),
            "map_graph_visits": map_visits,
            "map_graph_posterior_probability": map_visits / total,
            "true_drift_frobenius_norm": true_norm,
            "absolute_frobenius_error": absolute_error,
            "relative_frobenius_error": absolute_error / true_norm,
        }
        row.update(zip((f"mean_{name}" for name in matrix_columns), mean_drift.flat))
        rows.append(row)
    return rows


def main():
    rows = []
    for case_id in CASE_IDS:
        case = get_simulation_case(case_id)
        rows.extend(summarize_case(case))
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
