"""One-dimensional, deterministic coefficient proposals for GCLM graph moves.

The approximation changes a proposal, never the posterior target. Numerical
failures return the coefficient prior, so the reverse density is reproducible.
"""

from dataclasses import dataclass
from math import log, pi

import numpy as np
from scipy.optimize import minimize_scalar

from examples.gclm.state import State


def normal_logpdf(value: float, mean: float, standard_deviation: float) -> float:
    """Log density of a one-dimensional normal distribution."""
    standardized = (value - mean) / standard_deviation
    return -0.5 * standardized**2 - log(standard_deviation) - 0.5 * log(2 * pi)


@dataclass(frozen=True)
class CoefficientProposal:
    mean: float
    standard_deviation: float
    informed: bool

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, self.standard_deviation))

    def logpdf(self, value: float) -> float:
        return normal_logpdf(value, self.mean, self.standard_deviation)


def build_coefficient_proposal(reduced: State, edge, calculations, prior, tuning):
    """Approximate p(b_edge | other coordinates, graph with edge).

    A local bounded optimization follows a coarse search. All objective
    evaluations within this construction are memoized. The only persistent
    likelihood cache remains the two-slot cache in GCLMCalculations.
    """
    if edge not in reduced.graph.birth_edges(simple=prior.graph_space == "simple"):
        raise ValueError("The proposed edge is not admissible from the reduced graph")

    fallback = CoefficientProposal(0.0, prior.tau, False)
    added_graph = reduced.graph.add_edge(
        edge, simple=prior.graph_space == "simple"
    )
    coefficients = reduced.coefficients_by_edge
    evaluations = {}

    def objective(value):
        value = float(value)
        if value in evaluations:
            return evaluations[value]
        try:
            coefficients[edge] = value
            candidate = State.from_coefficients(
                added_graph, reduced.mu, reduced.r, reduced.eta, coefficients
            )
            score = calculations.log_likelihood(candidate, proposed=True)
            score -= 0.5 * (value / prior.tau) ** 2
            if not np.isfinite(score):
                score = -np.inf
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError):
            score = -np.inf
        evaluations[value] = float(score)
        return evaluations[value]

    radius = 3.0 * prior.tau # start search over [-3*tau, 3*tau]
    for expansion in range(tuning.li_max_expansions + 1):
        grid = np.linspace(-radius, radius, tuning.li_grid_points)
        scores = np.array([objective(value) for value in grid])
        if not np.isfinite(scores).any():
            return fallback
        best = int(np.argmax(scores))
        if 0 < best < len(grid) - 1:
            break
        if expansion == tuning.li_max_expansions:
            return fallback  # No bracketed maximum within the search range.
        radius *= 2.0

    try:
        result = minimize_scalar(
            lambda value: -objective(value),
            bounds=(float(grid[best - 1]), float(grid[best + 1])),
            method="bounded",
            options={"xatol": tuning.li_xatol, "maxiter": tuning.li_maxiter},
        )
    except (ValueError, FloatingPointError, OverflowError, RuntimeError):
        return fallback
    if not result.success or not np.isfinite(result.x):
        return fallback
    mode = float(result.x)
    center = objective(mode)
    if not np.isfinite(center) or center < scores[best]:
        return fallback

    step = max(10.0 * tuning.li_xatol, 1e-3 * prior.tau,
               1e-5 * max(1.0, abs(mode)))
    left, right = objective(mode - step), objective(mode + step)
    if not np.isfinite(left) or not np.isfinite(right):
        return fallback
    curvature = (2.0 * center - left - right) / step**2
    if not np.isfinite(curvature) or curvature <= 0:
        return fallback
    standard_deviation = tuning.li_inflation / np.sqrt(curvature)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return fallback
    return CoefficientProposal(mode, float(standard_deviation), True)
