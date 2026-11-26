"""Configurations for the GCLM protein-data experiment.

This module defines settings only; importing it does not start any runs.
"""

from dataclasses import dataclass
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "preprocessing" / "preprocessed_protein_data.csv"

PRIOR = {
    "lambda_p": 7.0,
    "s_r": 2.0,
    "mu_eta": -0.7,
    "s_eta": 0.5,
    "tau": 2.5,
    "m_0": 0.0,
    "s_mu": 1.0,
}

PROPOSAL = {
    "sigma_r": 0.02,
    "sigma_eta": 0.04,
    "sigma_b": 0.12,
    "gamma": 0.2,
    "adaptive_margin": False,
}

BASE_MOVE_WEIGHTS = {
    "mean": 0.05,
    "contrast": 0.15,
    "margin": 0.15,
    "coefficient": 0.40,
    "reversal": 0.10,
    "birth": 0.10,
    "death": 0.10,
}

CYCLE_MOVE_WEIGHTS = {
    "cycle_birth": 0.10,
    "cycle_death": 0.10,
}


@dataclass(frozen=True, slots=True)
class PriorSettings:
    """Prior arguments for GCLMPriorConfig."""

    graph_space: str
    lambda_p: float
    s_r: float
    mu_eta: float
    s_eta: float
    tau: float
    m_0: float
    s_mu: float

    def as_kwargs(self) -> dict[str, str | float]:
        return {
            "graph_space": self.graph_space,
            "lambda_p": self.lambda_p,
            "s_r": self.s_r,
            "mu_eta": self.mu_eta,
            "s_eta": self.s_eta,
            "tau": self.tau,
            "m_0": self.m_0,
            "s_mu": self.s_mu,
        }


@dataclass(frozen=True, slots=True)
class ProposalSettings:
    """Proposal arguments for GCLMProposalConfig."""

    sigma_r: float
    sigma_eta: float
    sigma_b: float
    gamma: float
    adaptive_margin: bool
    cycle_moves: bool
    move_weights: tuple[tuple[str, float], ...]
    likelihood_informed: bool = False

    def as_kwargs(self) -> dict[str, object]:
        return {
            "sigma_r": self.sigma_r,
            "sigma_eta": self.sigma_eta,
            "sigma_b": self.sigma_b,
            "gamma": self.gamma,
            "adaptive_margin": self.adaptive_margin,
            "cycle_moves": self.cycle_moves,
            "move_weights": dict(self.move_weights),
            "likelihood_informed": self.likelihood_informed,
        }


@dataclass(frozen=True, slots=True)
class ProteinDataCase:
    """One protein-data configuration with five independent chains."""

    id: str
    data_path: Path
    prior: PriorSettings
    proposal: ProposalSettings
    chain_seeds: tuple[int, ...]
    iterations: int = 1_000_000
    burn_in: int = 500_000
    thinning: int = 1


def _case(case_id: str, graph_space: str, cycle_moves: bool,
          first_seed: int, likelihood_informed: bool = False) -> ProteinDataCase:
    weights = BASE_MOVE_WEIGHTS | (CYCLE_MOVE_WEIGHTS if cycle_moves else {})
    return ProteinDataCase(
        id=case_id,
        data_path=DATA_PATH,
        prior=PriorSettings(graph_space=graph_space, **PRIOR),
        proposal=ProposalSettings(
            **PROPOSAL,
            cycle_moves=cycle_moves,
            move_weights=tuple(weights.items()),
            likelihood_informed=likelihood_informed,
        ),
        chain_seeds=tuple(range(first_seed, first_seed + 5)),
    )


def get_simulation_cases() -> tuple[ProteinDataCase, ...]:
    """Return the simple base, simple informed, and general cycle cases."""
    return (
        _case("simple_base", "simple", False, 2026005),
        _case("simple_li", "simple", False, 2026010, likelihood_informed=True),
        _case("general_cycle", "general", True, 2026055),
    )


def get_simulation_case(case_id: str) -> ProteinDataCase:
    """Return one case by ID; raise KeyError if it does not exist."""
    for case in get_simulation_cases():
        if case.id == case_id:
            return case
    raise KeyError(f"Unknown protein-data case: {case_id}")
