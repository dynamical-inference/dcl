import torch
from dcl.criterions.contrastive import MseInfoNCE
from dcl.datasets.splits import TrialSplit
from dcl.datasets.synthetic import SLDSDynamicsDataset
from dcl.datasets.timeseries import TensorDataset
from dcl.datasets.timeseries import TensorDatasetWithLatents
from dcl.distributions.time_distributions import OffsetTimeDistribution
from dcl.distributions.time_distributions import \
    UniformDiscreteTimeDistribution
from dcl.experiments.experiments import Experiment
from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.metrics.dynamics import AccuracyViaHungarian
from dcl.metrics.dynamics import PredictiveMSE
from dcl.metrics.identifiability import R2
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.models.dynamics.switching_dynamics import FixedMarkovSwitchingModel
from dcl.models.dynamics.switching_dynamics import MSESwitchingModel
from dcl.models.dynamics.utils import IdentityLDSParameters
from dcl.models.dynamics.utils import RotationLDSParameters
from dcl.models.encoder import MLP
from dcl.models.mixing import NonlinearLinearMixingModel
from dcl.solver.contrastive_solver import SLDSContrastiveLearningSolver
from dcl.solver.optimizer import DCLAdamOptimizer
from dcl.utils.rotation_matrix import MinMaxRotationSampler


def get_synthetic_data(num_modes: int = 5):
    data_slds = GumbelSLDS(
        linear_dynamics=LinearDynamicsModel(
            dim=6,
            num_systems=num_modes,
            initializer=RotationLDSParameters(
                rotation_sampler=MinMaxRotationSampler(
                    min_angle=0,
                    max_angle=10,
                )),
            noise_std=1e-4),
        switching_model=FixedMarkovSwitchingModel(num_modes=num_modes,
                                                  transition_probability=1e-1),
    )

    mixing_model = NonlinearLinearMixingModel(
        input_dim=data_slds.dim,
        output_dim=50,
        n_layers=4,
        n_iter_cond_thresh=int(10),
        cond_thresh_ratio=1e-4,
    )

    # Create smaller dataset for testing
    dataset = SLDSDynamicsDataset(
        num_trials=100,
        num_steps=100,
        dynamics_model=data_slds,
        mixing_model=mixing_model,
    )

    return dataset


def get_loader():
    return DiscreteTimeContrastiveDataLoader(
        num_iterations=10,
        batch_size=128,
        batch_size_neg=512,
        reference_distribution=UniformDiscreteTimeDistribution(),
        positive_distribution=OffsetTimeDistribution(offset=1),
        negative_distribution=UniformDiscreteTimeDistribution(),
    )


def get_dynamics_model(dim, num_modes):
    return GumbelSLDS(
        linear_dynamics=LinearDynamicsModel(
            dim=dim,
            num_systems=num_modes,
            initializer=IdentityLDSParameters(),
        ),
        switching_model=MSESwitchingModel(num_modes=num_modes,),
    )


def get_encoder_model(observed_dim, latent_dim):
    return MLP(
        input_dim=observed_dim,
        output_dim=latent_dim,
        hidden_dim=180,
        num_layers=3,
        normalize=False,
    )


def get_solver(dynamics_model, encoder_model):
    return SLDSContrastiveLearningSolver(
        model=encoder_model,
        dynamics_model=dynamics_model,
        optimizer=DCLAdamOptimizer(encoder_learning_rate=1e-3,
                                   dynamics_learning_rate=1e-2),
        criterion=MseInfoNCE(
            temperature=1.0,
            infonce_type="infonce_full_denominator",
        ),
        silence_metric_errors=True,
    )


def train_slds_contrastive(dataset,
                           num_modes,
                           latent_dim=None,
                           observed_dim=None):
    loader = get_loader()
    loader.lazy_init(dataset)

    # by default use dataset info
    if hasattr(dataset, "latent_dim"):
        latent_dim = dataset.latent_dim
    if hasattr(dataset, "observed_dim"):
        observed_dim = dataset.observed_dim

    assert latent_dim is not None, "latent_dim must be provided"
    assert observed_dim is not None, "observed_dim must be provided"

    # Create SLDS model
    slds = get_dynamics_model(
        dim=latent_dim,
        num_modes=num_modes,
    )
    encoder = get_encoder_model(
        observed_dim=observed_dim,
        latent_dim=latent_dim,
    )
    # Create solver
    solver = get_solver(dynamics_model=slds, encoder_model=encoder)

    # Create and run experiment
    exp = Experiment(
        dataset=dataset,
        dataset_split=TrialSplit(),
        solver=solver,
        train_loader=loader,
        name="test_integration",
    )

    exp.run(eval_frequency=5,
            eval_kwargs=dict(metrics=[
                AccuracyViaHungarian(),
                PredictiveMSE(),
                R2(bias=True, direction="backward"),
            ]))
    return exp


def test_dcl_solver():
    num_modes = 5

    dataset = get_synthetic_data(num_modes=num_modes)
    exp = train_slds_contrastive(dataset, num_modes)
    # Get final metrics
    final_metrics = exp.evaluate_test(metrics=[
        AccuracyViaHungarian(),
        PredictiveMSE(),
        R2(bias=True, direction="backward"),
    ])

    # Assertions to verify model performance
    assert final_metrics[
        'R2_Backward_bias'] > 0.5, "R2 score should be above 0.5"
    assert final_metrics['PredictiveMSE'] < 1.0, "MSE should be below 1.0"
    assert final_metrics[
        'AccuracyViaHungarian'] > 0.265, "Accuracy should be above 0.27"

    # Verify model produces reasonable predictions
    predictions = exp.solver.predictions(exp.val_loader)
    assert predictions.embeddings.shape[
        1] == dataset.latent_dim, "Incorrect embedding dimension"
    assert not torch.isnan(
        predictions.embeddings).any(), "Predictions contain NaN values"

    # Verify the dynamics model parameters
    assert exp.solver.dynamics_model.num_systems == num_modes, "Incorrect number of modes"
    assert exp.solver.dynamics_model.dim == dataset.latent_dim, "Incorrect latent dimension"


def test_dcl_solver_via_tensordataset():
    num_modes = 5

    synthetic_dataset = get_synthetic_data(num_modes=num_modes)

    exp_synthetic = train_slds_contrastive(synthetic_dataset.clone(), num_modes)

    dataset = TensorDataset(
        data=synthetic_dataset.get_observed_data(synthetic_dataset.index),
        trial_id=synthetic_dataset.auxilary_variables.trial_id,
        trial_time=synthetic_dataset.auxilary_variables.trial_time,
    )

    exp_tensor = train_slds_contrastive(
        dataset,
        num_modes,
        latent_dim=synthetic_dataset.latent_dim,
        observed_dim=synthetic_dataset.observed_dim,
    )

    # tensor dataset doesn't have true latents, so we can't check metrics
    # but the models should be the same after training...

    synthetic_encoder_params = exp_synthetic.solver.model.state_dict()
    tensor_encoder_params = exp_tensor.solver.model.state_dict()

    for key in synthetic_encoder_params:
        assert torch.allclose(synthetic_encoder_params[key],
                              tensor_encoder_params[key])


def test_dcl_solver_via_tensordatasetwithlatents():
    num_modes = 5

    synthetic_dataset = get_synthetic_data(num_modes=num_modes)

    exp_synthetic = train_slds_contrastive(synthetic_dataset.clone(), num_modes)

    dataset = TensorDatasetWithLatents(
        data=synthetic_dataset.get_observed_data(synthetic_dataset.index),
        latents=synthetic_dataset.get_latent_data(synthetic_dataset.index),
        trial_id=synthetic_dataset.auxilary_variables.trial_id,
        trial_time=synthetic_dataset.auxilary_variables.trial_time,
    )

    exp_tensor = train_slds_contrastive(
        dataset,
        num_modes,
        latent_dim=synthetic_dataset.latent_dim,
        observed_dim=synthetic_dataset.observed_dim,
    )

    # tensor dataset doesn't have true latents, so we can't check metrics
    # but the models should be the same after training...

    synthetic_encoder_params = exp_synthetic.solver.model.state_dict()
    tensor_encoder_params = exp_tensor.solver.model.state_dict()

    for key in synthetic_encoder_params:
        assert torch.allclose(synthetic_encoder_params[key],
                              tensor_encoder_params[key])

    # also check that we get some metrics
    metrics = exp_tensor.evaluate_test(metrics=[
        PredictiveMSE(),
        R2(bias=True, direction="backward"),
    ])
    assert metrics['R2_Backward_bias'] > 0.5, "R2 score should be above 0.5"
    assert metrics['PredictiveMSE'] < 1.0, "MSE should be below 1.0"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
