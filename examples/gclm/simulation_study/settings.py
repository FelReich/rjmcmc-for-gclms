"""Immutable settings for the GCLM simulation study."""

from dataclasses import dataclass


BASE_MOVE_WEIGHTS = (
    ("mean", 1.0),
    ("contrast", 1.0),
    ("margin", 1.0),
    ("coefficient", 1.0),
    ("reversal", 5.0),
    ("birth", 10.0),
    ("death", 10.0),
)
CYCLE_MOVE_WEIGHTS = (("cycle_birth", 5.0), ("cycle_death", 5.0))


@dataclass(frozen=True, slots=True)
class GeneratingModel:
    """Fixed data-generating model and the seed for its shared dataset."""

    name: str
    p: int
    ordered_edges: tuple[tuple[int, int], ...]
    coefficients: tuple[float, ...]
    r_star: tuple[float, ...]
    eta_star: float
    mean: tuple[float, ...]
    drift: tuple[tuple[float, ...], ...]
    covariance: tuple[tuple[float, ...], ...]
    generating_seed: int
    dataset_seed: int
    dataset_draws: int
    true_tau: float
    diffusion_scale: float


@dataclass(frozen=True, slots=True)
class PriorSettings:
    """Arguments accepted by GCLMPriorConfig."""

    graph_space: str
    lambda_p: float
    s_r: float = 1.0
    mu_eta: float = 0.0
    s_eta: float = 0.5
    tau: float = 2.0
    m_0: float = 0.0
    s_mu: float = 5.0

    def as_kwargs(self) -> dict[str, str | float]:
        """Return a fresh mapping for GCLMPriorConfig(**case.prior.as_kwargs())."""
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
    """Arguments accepted by GCLMProposalConfig, with immutable weights."""

    adaptive_margin: bool
    cycle_moves: bool
    move_weights: tuple[tuple[str, float], ...]
    sigma_r: float = 1.25
    sigma_eta: float = 0.125
    sigma_b: float = 0.5
    gamma: float = 0.5

    def as_kwargs(self) -> dict[str, object]:
        """Return fresh arguments, including a fresh move-weight dictionary."""
        return {
            "sigma_r": self.sigma_r,
            "sigma_eta": self.sigma_eta,
            "sigma_b": self.sigma_b,
            "gamma": self.gamma,
            "move_weights": dict(self.move_weights),
            "adaptive_margin": self.adaptive_margin,
            "cycle_moves": self.cycle_moves,
        }


@dataclass(frozen=True, slots=True)
class SimulationCase:
    """One inference configuration, including five independent chain seeds."""

    id: str
    generating_model: GeneratingModel
    n: int
    prior: PriorSettings
    proposal: ProposalSettings
    chain_seeds: tuple[int, ...]
    iterations: int = 1_000_000
    burn_in: int = 500_000
    thinning: int = 1
    initial_state: None = None

    @property
    def dataset_rows(self) -> tuple[int, int]:
        """Use this prefix of the shared 800-observation dataset."""
        return (0, self.n)


_R_STAR = (
    -0.7931224751578991,
    0.24057128353827487,
    -1.8963263495990657,
    1.3957717101086942,
)
_ETA_STAR = 0.9684446308087681
_MEAN = (0.0, 0.0, 0.0, 0.0, 0.0)

_G2 = GeneratingModel(
    name="G2",
    p=5,
    ordered_edges=((0, 1), (0, 2)),
    coefficients=(1.2765894818564305, -0.5840949704387269),
    r_star=_R_STAR,
    eta_star=_ETA_STAR,
    mean=_MEAN,
    drift=(
        (-3.546867199213283, 0.0, 0.0, 0.0, 0.0),
        (1.2765894818564305, -2.451788471780133, 0.0, 0.0, 0.0),
        (-0.5840949704387269, 0.0, -4.277920243171159, 0.0, 0.0),
        (0.0, 0.0, 0.0, -2.942363913221171, 0.0),
        (0.0, 0.0, 0.0, 0.0, -0.968444630808768),
    ),
    covariance=(
        (0.28193894607100206, 0.06000015883897592, -0.021045826686467182, 0.0, 0.0),
        (0.06000015883897592, 0.43910622146856104, -0.009199903682419047, 0.0, 0.0),
        (-0.021045826686467182, -0.009199903682419047, 0.23663198563185298, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.3398627870286935, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0325835553085565),
    ),
    generating_seed=2026,
    dataset_seed=2_026_002,
    dataset_draws=800,
    true_tau=2.0,
    diffusion_scale=2.0,
)

_G8 = GeneratingModel(
    name="G8",
    p=5,
    ordered_edges=(
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4), (2, 3),
    ),
    coefficients=(
        1.2765894818564305, -0.5840949704387269,
        -0.6238986563803515, 0.6076707355576247,
        -0.5353206044086792, -0.4518177223264272,
        1.4401356400182261, 1.0294104115475398,
    ),
    r_star=_R_STAR,
    eta_star=_ETA_STAR,
    mean=_MEAN,
    drift=(
        (-3.546867199213283, 0.0, 0.0, 0.0, 0.0),
        (1.2765894818564305, -2.451788471780133, 0.0, 0.0, 0.0),
        (-0.5840949704387269, -0.5353206044086792, -4.277920243171159, 0.0, 0.0),
        (-0.6238986563803515, -0.4518177223264272, 1.0294104115475398, -2.942363913221171, 0.0),
        (0.6076707355576247, 1.4401356400182261, 0.0, 0.0, -0.968444630808768),
    ),
    covariance=(
        (0.28193894607100206, 0.06000015883897592, -0.025150643786075146, -0.03527394776751554, 0.05708009182852217),
        (0.06000015883897592, 0.43910622146856104, -0.04490765633396968, -0.06063765494037027, 0.2168567704453056),
        (-0.025150643786075146, -0.04490765633396968, 0.24281199723817784, 0.04695087341426871, -0.04372256297255497),
        (-0.03527394776751554, -0.06063765494037027, 0.04695087341426871, 0.37307970259936435, -0.07347886678481645),
        (0.05708009182852217, 0.2168567704453056, -0.04372256297255497, -0.07347886678481645, 1.3908787579933943),
    ),
    generating_seed=2026,
    dataset_seed=2_026_008,
    dataset_draws=800,
    true_tau=2.0,
    diffusion_scale=2.0,
)


def _case(
    model: GeneratingModel,
    n: int,
    lambda_p: float,
    graph_space: str,
    chain_seeds: tuple[int, ...],
    *,
    adaptive_margin: bool = False,
    cycle_moves: bool = False,
) -> SimulationCase:
    suffix = "adaptive_cycle" if cycle_moves else "adaptive" if adaptive_margin else "base"
    return SimulationCase(
        id=f"{model.name}_n{n}_lambda{lambda_p:g}_{graph_space}_{suffix}",
        generating_model=model,
        n=n,
        prior=PriorSettings(graph_space=graph_space, lambda_p=lambda_p),
        proposal=ProposalSettings(
            adaptive_margin=adaptive_margin,
            cycle_moves=cycle_moves,
            move_weights=BASE_MOVE_WEIGHTS + (CYCLE_MOVE_WEIGHTS if cycle_moves else ()),
        ),
        chain_seeds=chain_seeds,
    )


def _five_seeds(first_seed: int) -> tuple[int, ...]:
    return tuple(range(first_seed, first_seed + 5))


def get_simulation_cases() -> tuple[SimulationCase, ...]:
    """Return all 11 immutable configurations in the requested order."""
    return (
        _case(_G2, 800, 5.0, "general", _five_seeds(202610066)),
        _case(_G2, 800, 5.0, "simple", _five_seeds(202610061)),
        _case(_G2, 800, 2.0, "general", _five_seeds(202611001)),
        _case(_G2, 800, 2.0, "simple", _five_seeds(202611006)),
        _case(_G2, 50, 5.0, "general", _five_seeds(202610006)),
        _case(_G2, 50, 5.0, "simple", _five_seeds(202610001)),
        _case(_G8, 800, 5.0, "general", _five_seeds(202610086)),
        _case(_G8, 800, 5.0, "simple", _five_seeds(202610081)),
        _case(
            _G2, 800, 5.0, "general", _five_seeds(202611011),
            adaptive_margin=True,
        ),
        _case(
            _G2, 800, 5.0, "simple", _five_seeds(202611016),
            adaptive_margin=True,
        ),
        _case(
            _G2, 800, 5.0, "general", _five_seeds(202611021),
            adaptive_margin=True, cycle_moves=True,
        ),
    )


def get_simulation_case(case_id: str) -> SimulationCase:
    """Return one case by ID, raising KeyError for an unknown ID."""
    for case in get_simulation_cases():
        if case.id == case_id:
            return case
    raise KeyError(f"Unknown simulation case: {case_id}")
