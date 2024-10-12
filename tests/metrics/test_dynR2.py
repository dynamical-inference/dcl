import pytest
from dcl.datasets.synthetic import GenericDynamicsDataset
from dcl.datasets.synthetic import SLDSDynamicsDataset
from dcl.distributions.time_distributions import OffsetTimeDistribution
from dcl.distributions.time_distributions import \
    UniformDiscreteTimeDistribution
from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.metrics.identifiability import DynamicsR2
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.models.dynamics.switching_dynamics import FixedMarkovSwitchingModel
from dcl.models.dynamics.utils import RotationLDSParameters
from dcl.models.encoder import IdentityEncoder
from dcl.models.mixing import IdentityMixingModel
from dcl.solver.contrastive_solver import DynamicsContrastiveLearningSolver
from dcl.solver.contrastive_solver import SLDSContrastiveLearningSolver
from dcl.utils.rotation_matrix import MinMaxRotationSampler


@pytest.mark.parametrize("num_modes", [5, 20])
@pytest.mark.parametrize("transition_probability", [1e-2, 1e-1])
def test_dynR2_slds(
    num_modes: int,
    transition_probability: float,
):
    data_dynamics = GumbelSLDS(
        linear_dynamics=LinearDynamicsModel(
            seed=42,
            dim=6,
            num_systems=num_modes,
            noise_std=0.0,
            initializer=RotationLDSParameters(
                rotation_sampler=MinMaxRotationSampler(
                    min_angle=10,
                    max_angle=20,
                )),
        ),
        switching_model=FixedMarkovSwitchingModel(
            num_modes=num_modes, transition_probability=transition_probability),
    )

    dataset = SLDSDynamicsDataset(
        num_trials=10,
        num_steps=200,
        dynamics_model=data_dynamics,
        mixing_model=IdentityMixingModel(
            input_dim=data_dynamics.dim,
            output_dim=data_dynamics.dim,
        ),
        seed=42,
    )

    # Create data loader
    loader = DiscreteTimeContrastiveDataLoader(
        reference_distribution=UniformDiscreteTimeDistribution(),
        positive_distribution=OffsetTimeDistribution(offset=1),
        negative_distribution=UniformDiscreteTimeDistribution(),
    )
    loader.lazy_init(dataset)

    # Create solver
    solver = SLDSContrastiveLearningSolver(
        model=IdentityEncoder(
            input_dim=dataset.observed_dim,
            output_dim=dataset.latent_dim,
        ),
        dynamics_model=data_dynamics.to_gt_dynamics(),
        silence_metric_errors=False,
    )

    dynR2 = DynamicsR2(
        bias=True,
        direction="forward",
        n_steps=1,
    )

    metrics = solver.validate(
        loader=loader,
        metrics=[dynR2],
    )

    r2_score = metrics[dynR2.name]
    assert r2_score == 1.0


def test_dynR2_lds():
    data_dynamics = LinearDynamicsModel(
        seed=42,
        dim=6,
        noise_std=0.0,
        initializer=RotationLDSParameters(
            rotation_sampler=MinMaxRotationSampler(
                min_angle=10,
                max_angle=20,
            )),
    )

    dataset = GenericDynamicsDataset(
        num_trials=10,
        num_steps=200,
        dynamics_model=data_dynamics,
        mixing_model=IdentityMixingModel(
            input_dim=data_dynamics.dim,
            output_dim=data_dynamics.dim,
        ),
        seed=42,
    )

    # Create data loader
    loader = DiscreteTimeContrastiveDataLoader(
        reference_distribution=UniformDiscreteTimeDistribution(),
        positive_distribution=OffsetTimeDistribution(offset=1),
        negative_distribution=UniformDiscreteTimeDistribution(),
    )
    loader.lazy_init(dataset)

    # Create solver
    solver = DynamicsContrastiveLearningSolver(
        model=IdentityEncoder(
            input_dim=dataset.observed_dim,
            output_dim=dataset.latent_dim,
        ),
        dynamics_model=data_dynamics.to_gt_dynamics(),
        silence_metric_errors=False,
    )

    dynR2 = DynamicsR2(
        bias=True,
        direction="forward",
        n_steps=1,
    )

    metrics = solver.validate(
        loader=loader,
        metrics=[dynR2],
    )

    r2_score = metrics[dynR2.name]
    assert r2_score == 1.0
