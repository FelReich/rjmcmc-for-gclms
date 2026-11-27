"""GCLM moves with optional informed, adaptive-margin, and cycle proposals.

Each move implements its simplified acceptance ratio in the coordinates
(G, mu, r, eta, b_G). The sampler supplies move-selection probabilities.
"""

from dataclasses import replace
from math import log

import numpy as np

from examples.gclm.gclm import (
    GCLMCalculations,
    GCLMPriorConfig,
    GCLMProposalConfig,
)
from examples.gclm.likelihood_informed import (
    build_coefficient_proposal, normal_logpdf,
)
from examples.gclm.state import State
from rjmcmc.rjmcmc import BaseMove, Proposal


def _accept(log_ratio, rng):
    """Accept with probability min(1, exp(log_ratio))."""
    # -Exp(1) has the same distribution as log(U), U ~ Uniform(0, 1).
    return bool(-rng.exponential() < log_ratio)


class _GCLMMove(BaseMove):
    """Shared data calculations and prior/proposal settings for GCLM moves."""

    def __init__(
        self,
        calculations: GCLMCalculations,
        prior_config: GCLMPriorConfig,
        proposal_config: GCLMProposalConfig,
        beta: float = 1.0,
    ):
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        self.calculations = calculations
        self.prior = prior_config
        self.proposal = proposal_config
        self.beta = float(beta)

    def _li_distribution(self, reduced: State, edge):
        return build_coefficient_proposal(
            reduced, edge, self.calculations, self.prior, self.proposal,
            beta=self.beta,
        )

    def _remember_li_forward(self, state, proposed, edge, distribution):
        # One proposal only; no likelihood approximations accumulate with a run.
        self._li_forward = (state, proposed, edge, distribution)

    def _li_forward_distribution(self, state, proposed, selected_edge,
                                 reduced, proposed_edge):
        cached = getattr(self, "_li_forward", None)
        if (cached is not None and cached[0] is state
                and cached[1] is proposed and cached[2] == selected_edge):
            return cached[3]
        return self._li_distribution(reduced, proposed_edge)

    @staticmethod
    def _without_edge(state, edge):
        coefficients = state.coefficients_by_edge
        del coefficients[edge]
        return State.from_coefficients(
            state.graph.remove_edge(edge), state.mu, state.r, state.eta,
            coefficients,
        )

    def _log_likelihood_ratio(self, state: State, proposed: State) -> float:
        """Return log L(proposed) - log L(state); reject a failed candidate solve."""
        old = self.calculations.log_likelihood(state)
        try:
            new = self.calculations.log_likelihood(proposed, proposed=True)
        except np.linalg.LinAlgError:
            return -np.inf
        return self.beta * (new - old)

    def _adjust_margin(self, state: State, proposed: State) -> State | None:
        """Apply the reversible softplus margin shift, if enabled."""
        if not self.proposal.adaptive_margin:
            return proposed
        try:
            old_phi = self.calculations.spectral_abscissa(state)
            new_phi = self.calculations.spectral_abscissa(proposed)
        except np.linalg.LinAlgError:
            return None
        eta = state.eta
        # softplus^{-1}(eta) = eta + log(1 - exp(-eta)).
        z = eta + np.log(-np.expm1(-eta))
        new_z = z + old_phi - new_phi
        # softplus(new_z) = log(1 + exp(new_z)).
        new_eta = float(np.logaddexp(0.0, new_z))
        if not np.isfinite(new_eta) or new_eta <= 0.0:
            return None
        return replace(proposed, eta=new_eta)

    def _log_margin_proposal_ratio(self, old_eta: float, new_eta: float) -> float:
        """Log prior-density ratio plus log softplus Jacobian."""
        if not self.proposal.adaptive_margin:
            return 0.0
        old_log = log(old_eta)
        new_log = log(new_eta)
        log_prior_ratio = old_log - new_log - (
            (new_log - self.prior.mu_eta) ** 2
            - (old_log - self.prior.mu_eta) ** 2
        ) / (2.0 * self.prior.s_eta ** 2)
        # s'(softplus^{-1}(eta)) = 1 - exp(-eta).
        log_jacobian = (
            np.log(-np.expm1(-new_eta)) - np.log(-np.expm1(-old_eta))
        )
        return float(log_prior_ratio + log_jacobian)


class MeanMove(_GCLMMove):
    """Draw the entire mean vector from its Gaussian full conditional."""

    def __init__(self, calculations, prior_config, proposal_config, beta=1.0):
        super().__init__(calculations, prior_config, proposal_config, beta)
        self._eigen_key = None
        self._eigenvalues = None
        self._eigenvectors = None

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        calculation = self.calculations.drift(state)
        # One bounded eigendecomposition cache: mean moves leave Sigma fixed.
        if calculation.key != self._eigen_key:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(
                calculation.covariance
            )
            if np.any(self._eigenvalues <= 0):
                raise np.linalg.LinAlgError("Covariance eigenvalues must be positive")
            self._eigen_key = calculation.key

        eigenvalues = self._eigenvalues
        eigenvectors = self._eigenvectors
        prior_variance = self.prior.s_mu**2
        n_prior_variance = self.beta * self.calculations.n * prior_variance
        denominator = eigenvalues + n_prior_variance
        mean_weights = n_prior_variance / denominator
        posterior_std = self.prior.s_mu * np.sqrt(eigenvalues / denominator)

        centered_data = self.calculations.y_bar - self.prior.m_0
        posterior_mean = (
            self.prior.m_0
            + eigenvectors @ (mean_weights * (eigenvectors.T @ centered_data))
        )
        new_mu = posterior_mean + eigenvectors @ (
            posterior_std * rng.standard_normal(self.calculations.p)
        )
        return Proposal(replace(state, mu=new_mu), "mean")

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        # The proposal is the full conditional, so this is a Gibbs step.
        return True


class ContrastMove(_GCLMMove):
    """Gaussian random walk on all p-1 diagonal contrast coordinates."""

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        new_r = state.r + rng.normal(0.0, self.proposal.sigma_r, size=state.r.size)
        if not np.all(np.isfinite(new_r)):
            return Proposal(state, "contrast")
        return Proposal(replace(state, r=new_r), "contrast", u=True)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio -= (
            np.dot(proposal.state.r, proposal.state.r) - np.dot(state.r, state.r)
        ) / (2.0 * self.prior.s_r**2)
        return _accept(log_ratio, rng)


class MarginMove(_GCLMMove):
    """Gaussian random walk on log(eta), preserving a positive margin."""

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        log_eta = log(state.eta)
        with np.errstate(over="ignore", under="ignore"):
            new_eta = np.exp(log_eta + rng.normal(0.0, self.proposal.sigma_eta))
        if not np.isfinite(new_eta) or new_eta <= 0:
            return Proposal(state, "margin")
        return Proposal(replace(state, eta=new_eta), "margin", u=True)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        old_log_eta = log(state.eta) - self.prior.mu_eta
        new_log_eta = log(proposal.state.eta) - self.prior.mu_eta
        # The 1/eta term in the lognormal prior cancels the proposal ratio.
        log_ratio -= (new_log_eta**2 - old_log_eta**2) / (
            2.0 * self.prior.s_eta**2
        )
        return _accept(log_ratio, rng)


class CoefficientMove(_GCLMMove):
    """Update a Bernoulli-selected subset of the existing edge coefficients."""

    def available(self, state):
        return state.graph.k > 0

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        selected = np.flatnonzero(
            rng.random(state.graph.k) < self.proposal.gamma
        )
        if selected.size == 0:
            # The empty subset is a no-op, not an accepted coefficient change.
            return Proposal(state, "coefficient")
        new_b = state.b.copy()
        new_b[selected] += rng.normal(
            0.0, self.proposal.sigma_b, size=selected.size
        )
        if not np.all(np.isfinite(new_b)):
            return Proposal(state, "coefficient")
        return Proposal(replace(state, b=new_b), "coefficient", u=selected)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        selected = proposal.u
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio -= (
            np.dot(proposal.state.b[selected], proposal.state.b[selected])
            - np.dot(state.b[selected], state.b[selected])
        ) / (2.0 * self.prior.tau**2)
        return _accept(log_ratio, rng)


class ReversalMove(_GCLMMove):
    """Reverse one eligible edge and propose its new coefficient."""

    def available(self, state):
        return bool(state.graph.reversible_edges())

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        reversible = state.graph.reversible_edges()
        edge = reversible[int(rng.integers(len(reversible)))]
        if self.proposal.likelihood_informed:
            reduced = self._without_edge(state, edge)
            distribution = self._li_distribution(reduced, (edge[1], edge[0]))
            coefficient = distribution.sample(rng)
        else:
            coefficient = rng.normal(0.0, self.prior.tau)
        if not np.isfinite(coefficient):
            return Proposal(state, "reversal")
        graph = state.graph.reverse_edge(edge)
        coefficients = state.coefficients_by_edge
        del coefficients[edge]
        coefficients[(edge[1], edge[0])] = coefficient
        proposed = State.from_coefficients(
            graph, state.mu, state.r, state.eta, coefficients
        )
        proposed = self._adjust_margin(state, proposed)
        if proposed is None:
            return Proposal(state, "reversal")
        if self.proposal.likelihood_informed:
            self._remember_li_forward(state, proposed, edge, distribution)
        return Proposal(proposed, "reversal", u=edge)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        li_correction = 0.0
        if self.proposal.likelihood_informed:
            edge = proposal.u
            reverse_edge = (edge[1], edge[0])
            old_coefficient = state.coefficients_by_edge[edge]
            new_coefficient = proposal.state.coefficients_by_edge[reverse_edge]
            forward = self._li_forward_distribution(
                state, proposal.state, edge, self._without_edge(state, edge),
                reverse_edge,
            )
            reverse = self._li_distribution(
                self._without_edge(proposal.state, reverse_edge), edge
            )
            li_correction = (
                normal_logpdf(new_coefficient, 0.0, self.prior.tau)
                - normal_logpdf(old_coefficient, 0.0, self.prior.tau)
                + reverse.logpdf(old_coefficient)
                - forward.logpdf(new_coefficient)
            )
        # With prior proposals the coefficient factors cancel exactly.
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio += self._log_margin_proposal_ratio(
            state.eta, proposal.state.eta
        )
        log_ratio += li_correction
        return _accept(log_ratio, rng)


class BirthMove(_GCLMMove):
    """Add one admissible edge and propose its coefficient."""

    def _birth_edges(self, state):
        return state.graph.birth_edges(simple=self.prior.graph_space == "simple")

    def available(self, state):
        return bool(self._birth_edges(state))

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        birth_edges = self._birth_edges(state)
        edge = birth_edges[int(rng.integers(len(birth_edges)))]
        if self.proposal.likelihood_informed:
            distribution = self._li_distribution(state, edge)
            coefficient = distribution.sample(rng)
        else:
            coefficient = rng.normal(0.0, self.prior.tau)
        if not np.isfinite(coefficient):
            return Proposal(state, "death")
        graph = state.graph.add_edge(edge, simple=self.prior.graph_space == "simple")
        coefficients = state.coefficients_by_edge
        coefficients[edge] = coefficient
        proposed = State.from_coefficients(
            graph, state.mu, state.r, state.eta, coefficients
        )
        proposed = self._adjust_margin(state, proposed)
        if proposed is None:
            return Proposal(state, "birth")
        if self.proposal.likelihood_informed:
            self._remember_li_forward(state, proposed, edge, distribution)
        return Proposal(proposed, "death", u=edge)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        li_correction = 0.0
        if self.proposal.likelihood_informed:
            coefficient = proposal.state.coefficients_by_edge[proposal.u]
            distribution = self._li_forward_distribution(
                state, proposal.state, proposal.u, state, proposal.u
            )
            li_correction = (
                normal_logpdf(coefficient, 0.0, self.prior.tau)
                - distribution.logpdf(coefficient)
            )
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio += log(self.prior.lambda_p) - log(state.graph.k + 1)
        log_ratio += log(reverse_probability) - log(forward_probability)
        log_ratio += self._log_margin_proposal_ratio(
            state.eta, proposal.state.eta
        )
        log_ratio += li_correction
        return _accept(log_ratio, rng)


class DeathMove(_GCLMMove):
    """Delete one uniformly chosen off-diagonal edge and its coefficient."""

    def available(self, state):
        return state.graph.k > 0

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        edges = state.graph.ordered_edges
        edge = edges[int(rng.integers(len(edges)))]
        graph = state.graph.remove_edge(edge)
        coefficients = state.coefficients_by_edge
        del coefficients[edge]
        proposed = State.from_coefficients(
            graph, state.mu, state.r, state.eta, coefficients
        )
        proposed = self._adjust_margin(state, proposed)
        if proposed is None:
            return Proposal(state, "death")
        return Proposal(proposed, "birth", u=edge)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        li_correction = 0.0
        if self.proposal.likelihood_informed:
            coefficient = state.coefficients_by_edge[proposal.u]
            reverse = self._li_distribution(proposal.state, proposal.u)
            li_correction = (
                reverse.logpdf(coefficient)
                - normal_logpdf(coefficient, 0.0, self.prior.tau)
            )
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio += log(state.graph.k) - log(self.prior.lambda_p)
        log_ratio += log(reverse_probability) - log(forward_probability)
        log_ratio += self._log_margin_proposal_ratio(
            state.eta, proposal.state.eta
        )
        log_ratio += li_correction
        return _accept(log_ratio, rng)


class CycleBirthMove(_GCLMMove):
    """Add both directions of an unconnected pair in the general graph space."""

    def available(self, state):
        return self.prior.graph_space == "general" and bool(
            state.graph.cycle_birth_pairs()
        )

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        pairs = state.graph.cycle_birth_pairs()
        i, j = pairs[int(rng.integers(len(pairs)))]
        new_b = rng.normal(0.0, self.prior.tau, size=2)
        if not np.all(np.isfinite(new_b)):
            return Proposal(state, "cycle_birth")
        graph = state.graph.add_edge((i, j)).add_edge((j, i))
        coefficients = state.coefficients_by_edge
        coefficients[i, j], coefficients[j, i] = new_b
        proposed = State.from_coefficients(
            graph, state.mu, state.r, state.eta, coefficients
        )
        proposed = self._adjust_margin(state, proposed)
        if proposed is None:
            return Proposal(state, "cycle_birth")
        return Proposal(proposed, "cycle_death", u=(i, j))

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        available_edges = state.graph.p * (state.graph.p - 1) - state.graph.k
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        # Poisson graph-size prior, uniform graphs at each size; the two
        # coefficient priors cancel their independent proposal densities.
        log_ratio += 2 * log(self.prior.lambda_p)
        log_ratio -= log(available_edges) + log(available_edges - 1)
        log_ratio += log(reverse_probability) - log(forward_probability)
        log_ratio += (
            log(len(state.graph.cycle_birth_pairs()))
            - log(len(proposal.state.graph.cycle_death_pairs()))
        )
        log_ratio += self._log_margin_proposal_ratio(
            state.eta, proposal.state.eta
        )
        return _accept(log_ratio, rng)


class CycleDeathMove(_GCLMMove):
    """Remove both directions of a two-cycle in the general graph space."""

    def available(self, state):
        return self.prior.graph_space == "general" and bool(
            state.graph.cycle_death_pairs()
        )

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        pairs = state.graph.cycle_death_pairs()
        i, j = pairs[int(rng.integers(len(pairs)))]
        graph = state.graph.remove_edge((i, j)).remove_edge((j, i))
        coefficients = state.coefficients_by_edge
        del coefficients[i, j]
        del coefficients[j, i]
        proposed = State.from_coefficients(
            graph, state.mu, state.r, state.eta, coefficients
        )
        proposed = self._adjust_margin(state, proposed)
        if proposed is None:
            return Proposal(state, "cycle_death")
        return Proposal(proposed, "cycle_birth", u=(i, j))

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        available_edges = state.graph.p * (state.graph.p - 1) - state.graph.k
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        # Reciprocal graph-prior and pair-selection factors of cycle birth.
        log_ratio += log(available_edges + 1) + log(available_edges + 2)
        log_ratio -= 2 * log(self.prior.lambda_p)
        log_ratio += log(reverse_probability) - log(forward_probability)
        log_ratio += (
            log(len(state.graph.cycle_death_pairs()))
            - log(len(proposal.state.graph.cycle_birth_pairs()))
        )
        log_ratio += self._log_margin_proposal_ratio(
            state.eta, proposal.state.eta
        )
        return _accept(log_ratio, rng)
