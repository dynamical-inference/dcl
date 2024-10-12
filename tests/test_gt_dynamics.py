import itertools

import pytest
import torch
from dcl.datasets.synthetic import SLDSDynamicsDataset
from dcl.distributions.time_distributions import OffsetTimeDistribution
from dcl.distributions.time_distributions import \
    UniformDiscreteTimeDistribution
from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.models.dynamics.switching_dynamics import FixedMarkovSwitchingModel
from dcl.models.dynamics.utils import RotationLDSParameters
from dcl.models.mixing import IdentityMixingModel
from dcl.utils.datatypes import GumbelSLDSInput
from dcl.utils.rotation_matrix import MinMaxRotationSampler


@pytest.mark.parametrize("dynamics_noise, num_modes, transition_probability",
                         itertools.product(
                             [0.0, 1e-4],
                             [1, 5, 10],
                             [1e-1, 1e-3],
                         ))
def test_slds_gt_dynamics(
    dynamics_noise: float,
    num_modes: int,
    transition_probability: float,
):
    data_dynamics = GumbelSLDS(
        linear_dynamics=LinearDynamicsModel(
            seed=42,
            dim=6,
            num_systems=num_modes,
            noise_std=dynamics_noise,
            initializer=RotationLDSParameters(
                rotation_sampler=MinMaxRotationSampler(
                    min_angle=0,
                    max_angle=10,
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

    data = dataset.ground_truth_data

    # this model should be able to predict the data perfectly
    slds_gt_dynamics = dataset.dynamics_model.to_gt_dynamics()

    slds_gt_dynamics.train()
    slds_gt_dynamics.to(data.latents.device)

    # Create data loader
    loader = DiscreteTimeContrastiveDataLoader(
        reference_distribution=UniformDiscreteTimeDistribution(),
        positive_distribution=OffsetTimeDistribution(offset=1),
        negative_distribution=UniformDiscreteTimeDistribution(),
    )
    loader.lazy_init(dataset)

    data_input = loader.validation_data
    # drop encoder_offset dimension
    embeddings = data_input.observed
    model_input = GumbelSLDSInput(
        x=embeddings[data_input.reference_index],
        x_index=data_input.reference_index,
        x_next=embeddings[data_input.positive_index],
    )

    model_prediction = slds_gt_dynamics(model_input)
    model_prediction = model_prediction.to_SLDSPrediction()

    assert torch.all(model_prediction.modes == data.modes[
        model_prediction.x_index]), "modes are not predicted correctly"
    if dynamics_noise == 0.0:
        assert torch.allclose(model_prediction.x, data.latents[
            model_prediction.x_index]), "latents are not predicted correctly"
    else:
        pred = model_prediction.x
        true = data.latents[model_prediction.x_index]
        mse = torch.mean((pred - true)**2)
        assert mse < dynamics_noise, "latents are not predicted correctly"
