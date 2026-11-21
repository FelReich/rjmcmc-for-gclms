"""Compare drift and covariance estimates for the current chain-1 pair episode.

The sigma_r=0.2 baseline has an uninterrupted 2<->4 reciprocal-pair episode
at zero-based retained indices [274549, 328284). The MAP graph and overall
posterior covariance use all retained draws from all five base general
chains. Matrix rows are target vertices and columns are sources.
The covariance likelihoods use the common fitted mean y_bar and omit the
Gaussian factor (2*pi)^(-n*p/2), matching the sampler's log-likelihood
convention. They evaluate each reported covariance matrix; they are not
posterior averages of the draw-wise likelihoods.
"""

import csv
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_continuous_lyapunov


CODE_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.gclm import GCLMCalculations
from examples.gclm.simulation_study.run_simulation_study import generate_dataset
from examples.gclm.simulation_study.settings import get_simulation_case


CASE = get_simulation_case("G2_n800_lambda5_general_base")
RUNS = CODE_ROOT / "examples/gclm/simulation_study/runs" / CASE.id
OUTPUT = Path(__file__).with_suffix(".csv")
START, END = 274_549, 328_284
PAIR_EDGES = ((1, 3), (3, 1))  # Zero-based: 2->4 and 4->2.
CHUNK_SIZE = 50_000


def check_chain(path, number, seed):
    """Confirm that a file belongs to the intended independent simulation."""
    with path.open(newline="", encoding="utf-8") as file:
        first = next(csv.DictReader(file))
    expected = (CASE.id, str(number), str(seed),
                str(CASE.generating_model.dataset_seed))
    actual = tuple(first[name] for name in (
        "configuration_id", "chain_number", "chain_seed", "dataset_seed"
    ))
    if actual != expected:
        raise ValueError(f"Chain metadata disagree in {path}: {actual}")


def summarize_chains():
    """Stream the matrices, accumulating graph-conditional covariance sums."""
    p = CASE.generating_model.p
    edges = tuple((i, j) for i in range(p) for j in range(p) if i != j)
    pair_mask = sum(1 << edges.index(edge) for edge in PAIR_EDGES)
    m_columns = [f"M_{i}_{j}" for i in range(p) for j in range(p)]
    sigma_columns = [f"Sigma_{i}_{j}" for i in range(p) for j in range(p)]
    counts = Counter()
    sigma_by_graph = {}
    posterior_sigma_sum = np.zeros((p, p))
    window_m_sum = np.zeros((p, p))
    window_sigma_sum = np.zeros((p, p))
    posterior_draws = window_draws = 0
    boundary_pair_present = {}

    for number, seed in enumerate(CASE.chain_seeds, start=1):
        path = RUNS / f"chain_{number:02d}.csv"
        check_chain(path, number, seed)
        columns = ["iteration", "graph_bitmask", *sigma_columns]
        if number == 1:
            columns += m_columns
        offset = 0
        for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK_SIZE):
            size = len(chunk)
            expected_iterations = CASE.burn_in + 1 + np.arange(offset, offset + size)
            if not np.array_equal(chunk["iteration"].to_numpy(), expected_iterations):
                raise ValueError(f"Retained iterations are not consecutive in {path}")
            sigma = chunk[sigma_columns].to_numpy(dtype=float)
            if not np.isfinite(sigma).all():
                raise ValueError(f"Nonfinite covariance in {path}")
            posterior_sigma_sum += sigma.sum(axis=0).reshape(p, p)

            grouped = chunk.groupby("graph_bitmask", sort=False)
            group_counts = grouped.size()
            group_sums = grouped[sigma_columns].sum()
            for mask, values in group_sums.iterrows():
                mask = int(mask)
                counts[mask] += int(group_counts.loc[mask])
                if mask not in sigma_by_graph:
                    sigma_by_graph[mask] = np.zeros((p, p))
                sigma_by_graph[mask] += values.to_numpy(dtype=float).reshape(p, p)

            if number == 1:
                graph_masks = chunk["graph_bitmask"].to_numpy(dtype=np.int64)
                for boundary in (START - 1, END):
                    if offset <= boundary < offset + size:
                        boundary_pair_present[boundary] = bool(
                            graph_masks[boundary - offset] & pair_mask == pair_mask
                        )
                left = max(START - offset, 0)
                right = min(END - offset, size)
                if left < right:
                    if not np.all(graph_masks[left:right] & pair_mask == pair_mask):
                        raise ValueError("The selected window is not entirely the 2<->4 episode")
                    m = chunk.iloc[left:right][m_columns].to_numpy(dtype=float)
                    if not np.isfinite(m).all():
                        raise ValueError(f"Nonfinite drift in {path}")
                    window_m_sum += m.sum(axis=0).reshape(p, p)
                    window_sigma_sum += sigma[left:right].sum(axis=0).reshape(p, p)
                    window_draws += right - left
            offset += size

        if offset != (CASE.iterations - CASE.burn_in) // CASE.thinning:
            raise ValueError(f"Incomplete retained chain in {path}: {offset} draws")
        posterior_draws += offset
        print(f"Read chain {number}/{len(CASE.chain_seeds)}", flush=True)

    if window_draws != END - START or sum(counts.values()) != posterior_draws:
        raise ValueError("Window or graph counts do not match the retained draws")
    if boundary_pair_present != {START - 1: False, END: False}:
        raise ValueError("The selected 2<->4 episode does not have the expected boundaries")
    map_mask, map_draws = max(counts.items(), key=lambda item: (item[1], -item[0]))
    return (window_m_sum / window_draws, window_sigma_sum / window_draws,
            window_draws, sigma_by_graph[map_mask] / map_draws,
            map_mask, map_draws, posterior_sigma_sum / posterior_draws,
            posterior_draws)


def stationary_covariance(drift):
    """Solve M Sigma + Sigma M.T = -2I, as in GCLMCalculations."""
    p = len(drift)
    if np.max(np.linalg.eigvals(drift).real) >= 0:
        raise ValueError("The mean drift is not stable")
    covariance = solve_continuous_lyapunov(drift, -2.0 * np.eye(p))
    covariance = (covariance + covariance.T) / 2.0
    residual = drift @ covariance + covariance @ drift.T + 2.0 * np.eye(p)
    if (not np.isfinite(covariance).all()
            or np.linalg.norm(residual, ord="fro") > 1e-8
            or np.linalg.eigvalsh(covariance).min() <= 0):
        raise ValueError("The mean drift did not yield a valid stationary covariance")
    return covariance


def relative_error(estimate, reference):
    return float(np.linalg.norm(estimate - reference, ord="fro")
                 / np.linalg.norm(reference, ord="fro"))


def covariance_log_likelihood(covariance, empirical_covariance, n):
    """Gaussian log-likelihood at mu=y_bar, omitting the shared normalizer."""
    factor = cho_factor(covariance, lower=True, check_finite=True)
    log_determinant = 2.0 * np.log(np.diag(factor[0])).sum()
    empirical_trace = np.trace(cho_solve(factor, empirical_covariance))
    return float(-0.5 * n * (log_determinant + empirical_trace))


def main():
    if CASE.proposal.sigma_r != 0.2:
        raise ValueError("This comparison is for the sigma_r=0.2 baseline")
    (mean_m, mean_sigma, window_draws, map_sigma, map_mask,
     map_draws, posterior_sigma, posterior_draws) = summarize_chains()
    model = CASE.generating_model
    true_m = np.asarray(model.drift, dtype=float)
    true_sigma = np.asarray(model.covariance, dtype=float)
    observations = generate_dataset(model)[slice(*CASE.dataset_rows)]
    empirical_sigma = GCLMCalculations(observations).empirical_covariance
    sigma_at_mean_m = stationary_covariance(mean_m)

    p = model.p
    edges = tuple((i, j) for i in range(p) for j in range(p) if i != j)
    map_edges = ";".join(f"{i + 1}->{j + 1}" for bit, (i, j) in enumerate(edges)
                         if map_mask & (1 << bit))
    matrices = (
        ("window_mean_drift", "drift", mean_m, window_draws),
        ("window_covariance_of_mean_drift", "covariance", sigma_at_mean_m, window_draws),
        ("window_mean_covariance", "covariance", mean_sigma, window_draws),
        ("map_graph_mean_covariance", "covariance", map_sigma, map_draws),
        ("overall_posterior_mean_covariance", "covariance", posterior_sigma, posterior_draws),
        ("empirical_covariance", "covariance", empirical_sigma, len(observations)),
        ("true_drift", "drift", true_m, ""),
        ("true_covariance", "covariance", true_sigma, ""),
    )
    entry_columns = [f"entry_{i + 1}_{j + 1}" for i in range(p) for j in range(p)]
    columns = ["matrix", "kind", "draws", "chain_number", "proposal_sigma_r",
               "window_pair_one_based",
               "retained_start_zero_based", "retained_end_exclusive",
               "relative_error_to_true",
               "relative_error_to_empirical_covariance",
               "log_likelihood_at_ybar_up_to_constant",
               "likelihood_at_ybar_up_to_constant", "graph_bitmask",
               "graph_edges_one_based", "posterior_graph_probability", *entry_columns]
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for name, kind, matrix, draws in matrices:
            log_likelihood = (
                covariance_log_likelihood(matrix, empirical_sigma, CASE.n)
                if kind == "covariance" else None
            )
            row = {
                "matrix": name,
                "kind": kind,
                "draws": draws,
                "chain_number": 1 if name.startswith("window_") else "",
                "proposal_sigma_r": CASE.proposal.sigma_r,
                "window_pair_one_based": "2<->4" if name.startswith("window_") else "",
                "retained_start_zero_based": START if name.startswith("window_") else "",
                "retained_end_exclusive": END if name.startswith("window_") else "",
                "relative_error_to_true": relative_error(
                    matrix, true_m if kind == "drift" else true_sigma),
                "relative_error_to_empirical_covariance": (
                    relative_error(matrix, empirical_sigma) if kind == "covariance" else ""),
                "log_likelihood_at_ybar_up_to_constant": (
                    log_likelihood if log_likelihood is not None else ""),
                "likelihood_at_ybar_up_to_constant": (
                    float(np.exp(log_likelihood)) if log_likelihood is not None else ""),
                "graph_bitmask": map_mask if name == "map_graph_mean_covariance" else "",
                "graph_edges_one_based": map_edges if name == "map_graph_mean_covariance" else "",
                "posterior_graph_probability": (
                    map_draws / posterior_draws if name == "map_graph_mean_covariance" else ""),
            }
            row.update({column: format(float(value), ".17g")
                        for column, value in zip(entry_columns, matrix.flat)})
            writer.writerow(row)
    print(f"MAP graph: {map_edges} ({map_draws}/{posterior_draws} draws)")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
