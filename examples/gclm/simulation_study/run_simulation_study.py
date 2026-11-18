"""Run the cases in settings.py, saving one CSV per independent chain.

From the Code directory:
    python3 examples/gclm/simulation_study/run_simulation_study.py
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter


CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np

from examples.gclm.gclm import (
    GCLM, GCLMCalculations, GCLMPriorConfig, GCLMProposalConfig,
)
from examples.gclm.gclm_moves import (
    BirthMove, CoefficientMove, ContrastMove, CycleBirthMove,
    CycleDeathMove, DeathMove, MarginMove, MeanMove, ReversalMove,
)
from examples.gclm.simulation_study.settings import (
    SimulationCase, get_simulation_cases,
)


RUNS_DIR = Path(__file__).resolve().parent / "runs"
MOVE_CLASSES = {
    "mean": MeanMove,
    "contrast": ContrastMove,
    "margin": MarginMove,
    "coefficient": CoefficientMove,
    "reversal": ReversalMove,
    "birth": BirthMove,
    "death": DeathMove,
    "cycle_birth": CycleBirthMove,
    "cycle_death": CycleDeathMove,
}


def edge_order(p: int) -> tuple[tuple[int, int], ...]:
    """Canonical order for indicator and coefficient columns."""
    return tuple((i, j) for i in range(p) for j in range(p) if i != j)


def columns(p: int) -> list[str]:
    """Keep each row sufficient to reconstruct its state and fitted matrices."""
    edges = edge_order(p)
    names = [
        "configuration_id", "generating_model", "n", "inference_graph_space",
        "lambda_p", "adaptive_margin", "cycle_moves", "chain_number",
        "chain_seed", "dataset_seed", "iteration", "move_type", "accepted",
        "graph_changed", "k", "graph_bitmask", "graph_edges_zero_based",
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


def number(value: float) -> str:
    """Serialize float64 values with enough digits to round-trip exactly."""
    return format(float(value), ".17g")


def make_row(case, chain_number, seed, iteration, move, accepted,
             previous_graph, state, calculations, edges):
    """Serialize the current state without another Lyapunov solve."""
    graph = state.graph
    indicators = [int(edge in graph.edges) for edge in edges]
    bitmask = sum(bit << index for index, bit in enumerate(indicators))
    coefficients = state.coefficients_by_edge
    calculation = calculations.drift(state)  # Already cached by sampler.step().
    shift = calculation.spectral_abscissa + state.eta
    row = [
        case.id, case.generating_model.name, case.n, case.prior.graph_space,
        number(case.prior.lambda_p), int(case.proposal.adaptive_margin),
        int(case.proposal.cycle_moves), chain_number, seed,
        case.generating_model.dataset_seed, iteration, move, int(accepted),
        int(graph != previous_graph), graph.k, bitmask,
        json.dumps(graph.ordered_edges, separators=(",", ":")),
    ]
    row += indicators
    row += [number(value) for value in state.mu]
    row += [number(value) for value in state.r]
    row += [number(state.eta)]
    row += [number(coefficients[edge]) if edge in coefficients else "" for edge in edges]
    row += [
        number(calculation.spectral_abscissa), number(shift),
        number(calculation.spectral_abscissa - shift),
        number(calculation.relative_lyapunov_residual),
        number(calculations.log_likelihood(state)),
    ]
    row += [number(value) for value in calculation.M.flat]
    row += [number(value) for value in calculation.covariance.flat]
    return row


def generate_dataset(model):
    """Generate the established 800-row dataset once per generating model."""
    mean = np.asarray(model.mean, dtype=float)
    covariance = np.asarray(model.covariance, dtype=float)
    return np.random.default_rng(model.dataset_seed).multivariate_normal(
        mean, covariance, size=model.dataset_draws,
    )


def run_chain(case: SimulationCase, chain_number: int, seed: int,
              observations: np.ndarray, destination: Path) -> None:
    """Stream retained states to a partial CSV, then publish it atomically."""
    prior = GCLMPriorConfig(**case.prior.as_kwargs())
    proposal = GCLMProposalConfig(**case.proposal.as_kwargs())
    calculations = GCLMCalculations(observations)
    moves = {
        name: MOVE_CLASSES[name](calculations, prior, proposal)
        for name, _ in case.proposal.move_weights
    }
    sampler = GCLM(
        moves, calculations, prior, proposal,
        initial_state=case.initial_state, seed=seed,
    )

    edges = edge_order(calculations.p)
    partial = destination.with_name(destination.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as file:
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
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, destination)


def verify_completed(path: Path, case: SimulationCase,
                     chain_number: int, seed: int) -> None:
    """Check a saved CSV before skipping it during --resume."""
    expected_columns = columns(case.generating_model.p)
    expected_rows = (case.iterations - case.burn_in + case.thinning - 1) // case.thinning
    count = 0
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected CSV columns in {path}")
        for row in reader:
            count += 1
            expected_iteration = case.burn_in + 1 + (count - 1) * case.thinning
            if (None in row or any(value is None for value in row.values())
                    or row["configuration_id"] != case.id
                    or row["chain_number"] != str(chain_number)
                    or row["chain_seed"] != str(seed)
                    or row["dataset_seed"] != str(case.generating_model.dataset_seed)
                    or row["iteration"] != str(expected_iteration)):
                raise ValueError(f"Invalid row {count} in {path}")
    if count != expected_rows:
        raise ValueError(f"Incomplete CSV {path}: {count} rows, expected {expected_rows}")


def main() -> None:
    cases = get_simulation_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=[case.id for case in cases],
        help="Run just this configuration (default: all 11)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed CSVs and restart any interrupted chain",
    )
    args = parser.parse_args()
    selected = [case for case in cases if args.case is None or case.id == args.case]

    # A model's n=50 observations are the first 50 of the same n=800 dataset.
    models = {case.generating_model.name: case.generating_model for case in selected}
    datasets = {name: generate_dataset(model) for name, model in models.items()}
    tasks = [
        (case, chain_number, seed)
        for case in selected
        for chain_number, seed in enumerate(case.chain_seeds, start=1)
    ]
    started = perf_counter()
    for index, (case, chain_number, seed) in enumerate(tasks, start=1):
        folder = RUNS_DIR / case.id
        destination = folder / f"chain_{chain_number:02d}.csv"
        partial = destination.with_name(destination.name + ".partial")
        if destination.exists():
            if not args.resume:
                raise FileExistsError(f"{destination} already exists; use --resume")
            verify_completed(destination, case, chain_number, seed)
            print(f"[{index}/{len(tasks)}] Skipped completed {destination}", flush=True)
            continue
        if partial.exists() and not args.resume:
            raise FileExistsError(f"{partial} exists; use --resume to restart this chain")

        folder.mkdir(parents=True, exist_ok=True)
        first, last = case.dataset_rows
        observations = datasets[case.generating_model.name][first:last]
        if observations.shape != (case.n, case.generating_model.p):
            raise ValueError(f"Dataset dimensions do not match {case.id}")
        print(
            f"[{index}/{len(tasks)}] Running {case.id} chain {chain_number} "
            f"(seed {seed})", flush=True,
        )
        chain_started = perf_counter()
        run_chain(case, chain_number, seed, observations, destination)
        print(
            f"[{index}/{len(tasks)}] Saved {destination} "
            f"in {perf_counter() - chain_started:.1f}s", flush=True,
        )
    print(f"Done in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
