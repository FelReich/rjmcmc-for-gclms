"""Run the protein-data cases, saving one compressed CSV per chain.

The CSV columns follow the simulation-study runner. Compression limits the
size of the stored drift and covariance matrices for the 11-protein data.
"""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from examples.gclm.gclm import (
    GCLM, GCLMCalculations, GCLMPriorConfig, GCLMProposalConfig,
)
from examples.gclm.protein_data.settings import ProteinDataCase, get_simulation_cases
from examples.gclm.simulation_study.run_simulation_study import (
    MOVE_CLASSES, edge_order, number,
)


RUNS_DIR = Path(__file__).resolve().parent / "runs"


def columns(p: int) -> list[str]:
    """Columns needed to reconstruct each state and inspect its fitted matrices."""
    edges = edge_order(p)
    names = [
        "configuration_id", "n", "p", "inference_graph_space", "lambda_p",
        "adaptive_margin", "cycle_moves", "chain_number", "chain_seed",
        "iteration", "move_type", "accepted", "graph_changed", "k",
        "graph_bitmask", "graph_edges_zero_based",
    ]
    names += [f"edge_{i}_{j}" for i, j in edges]
    names += [f"mu_{i}" for i in range(p)]
    names += [f"r_{i}" for i in range(p - 1)]
    names += ["eta"]
    names += [f"b_{i}_{j}" for i, j in edges]
    names += [
        "phi_H", "stability_shift", "phi_M",
        "relative_lyapunov_residual", "log_likelihood",
    ]
    names += [f"M_{i}_{j}" for i in range(p) for j in range(p)]
    names += [f"Sigma_{i}_{j}" for i in range(p) for j in range(p)]
    return names


def load_data(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read the existing standardized data without transforming it again."""
    frame = pd.read_csv(path)
    if len(frame) < 2 or frame.shape[1] < 2 or frame.columns.has_duplicates:
        raise ValueError(f"Invalid protein-data dimensions or labels in {path}")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f"Protein columns must be numeric in {path}")
    observations = frame.to_numpy(dtype=float)
    if not np.isfinite(observations).all():
        raise ValueError(f"Protein data contain non-finite values in {path}")
    np.testing.assert_allclose(observations.mean(axis=0), 0.0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(observations.std(axis=0, ddof=1), 1.0,
                               rtol=0, atol=1e-10)
    return tuple(frame.columns), observations


def make_row(case: ProteinDataCase, chain_number: int, seed: int, iteration: int,
             move: str, accepted: bool, previous_graph, state,
             calculations: GCLMCalculations, edges: tuple[tuple[int, int], ...]):
    """Serialize the retained state using the sampler's cached calculation."""
    graph = state.graph
    indicators = [int(edge in graph.edges) for edge in edges]
    bitmask = sum(bit << index for index, bit in enumerate(indicators))
    coefficients = state.coefficients_by_edge
    calculation = calculations.drift(state)
    shift = calculation.spectral_abscissa + state.eta
    row = [
        case.id, calculations.n, calculations.p, case.prior.graph_space,
        number(case.prior.lambda_p), int(case.proposal.adaptive_margin),
        int(case.proposal.cycle_moves), chain_number, seed, iteration, move,
        int(accepted), int(graph != previous_graph), graph.k, bitmask,
        json.dumps(graph.ordered_edges, separators=(",", ":")),
    ]
    row += indicators
    row += [number(value) for value in state.mu]
    row += [number(value) for value in state.r]
    row += [number(state.eta)]
    row += [number(coefficients[edge]) if edge in coefficients else "" for edge in edges]
    row += [
        number(calculation.spectral_abscissa), number(shift),
        number(-state.eta), number(calculation.relative_lyapunov_residual),
        number(calculations.log_likelihood(state)),
    ]
    row += [number(value) for value in calculation.M.flat]
    row += [number(value) for value in calculation.covariance.flat]
    return row


def run_chain(case: ProteinDataCase, chain_number: int, seed: int,
              observations: np.ndarray, destination: Path) -> None:
    """Write a chain to a partial file and publish it only when complete."""
    prior = GCLMPriorConfig(**case.prior.as_kwargs())
    proposal = GCLMProposalConfig(**case.proposal.as_kwargs())
    calculations = GCLMCalculations(observations)
    moves = {
        name: MOVE_CLASSES[name](calculations, prior, proposal)
        for name, _ in case.proposal.move_weights
    }
    sampler = GCLM(moves, calculations, prior, proposal, seed=seed)
    edges = edge_order(calculations.p)
    partial = destination.with_name(destination.name + ".partial")

    with gzip.open(partial, "wt", newline="", encoding="utf-8", compresslevel=1) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(columns(calculations.p))
        for iteration in range(1, case.iterations + 1):
            previous_graph = sampler.state.graph
            state, move, accepted = sampler.step()
            if iteration <= case.burn_in:
                continue
            if (iteration - case.burn_in - 1) % case.thinning:
                continue
            writer.writerow(make_row(
                case, chain_number, seed, iteration, move, accepted,
                previous_graph, state, calculations, edges,
            ))
    os.replace(partial, destination)


def verify_completed(path: Path, case: ProteinDataCase,
                     chain_number: int, seed: int, n: int, p: int) -> None:
    """Check a completed chain before skipping it during --resume."""
    expected_rows = (case.iterations - case.burn_in + case.thinning - 1) // case.thinning
    count = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != columns(p):
            raise ValueError(f"Unexpected CSV columns in {path}")
        for row in reader:
            count += 1
            expected_iteration = case.burn_in + 1 + (count - 1) * case.thinning
            if (None in row or any(value is None for value in row.values())
                    or row["configuration_id"] != case.id
                    or row["n"] != str(n) or row["p"] != str(p)
                    or row["chain_number"] != str(chain_number)
                    or row["chain_seed"] != str(seed)
                    or row["iteration"] != str(expected_iteration)):
                raise ValueError(f"Invalid row {count} in {path}")
    if count != expected_rows:
        raise ValueError(f"Incomplete CSV {path}: {count} rows, expected {expected_rows}")


def main() -> None:
    cases = get_simulation_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.id for case in cases],
                        help="Run only this case (default: all cases)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip verified completed chains and restart interrupted ones")
    args = parser.parse_args()
    selected = [case for case in cases if args.case is None or case.id == args.case]
    datasets = {case.data_path: load_data(case.data_path) for case in selected}
    tasks = [
        (case, chain_number, seed)
        for case in selected
        for chain_number, seed in enumerate(case.chain_seeds, start=1)
    ]

    started = perf_counter()
    for case, chain_number, seed in tasks:
        progress = f"[{chain_number}/{len(case.chain_seeds)}]"
        names, observations = datasets[case.data_path]
        folder = RUNS_DIR / case.id
        destination = folder / f"chain_{chain_number:02d}.csv.gz"
        partial = destination.with_name(destination.name + ".partial")
        if destination.exists():
            if not args.resume:
                raise FileExistsError(f"{destination} already exists; use --resume")
            verify_completed(destination, case, chain_number, seed,
                             len(observations), len(names))
            print(f"{progress} Skipped completed {destination}", flush=True)
            continue
        if partial.exists() and not args.resume:
            raise FileExistsError(f"{partial} exists; use --resume to restart this chain")

        folder.mkdir(parents=True, exist_ok=True)
        print(f"{progress} Running {case.id} chain {chain_number} "
              f"(seed {seed})", flush=True)
        chain_started = perf_counter()
        run_chain(case, chain_number, seed, observations, destination)
        print(f"{progress} Saved {destination} "
              f"in {perf_counter() - chain_started:.1f}s", flush=True)
    print(f"Done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
