from pathlib import Path

import torch
from dcl.datasets.synthetic import GenericDynamicsDataset
from dcl.models.dynamics.nonlinear_dynamics import LorenzAttractorDynamicsModel
from dcl.models.mixing import NonlinearLinearMixingModel


def lorenz_dataset_config(seed: int = 42):

    data_lorenz = LorenzAttractorDynamicsModel(
        sigma=10.0,
        rho=28.0,
        beta=8.0 / 3.0,
        dt=0.01,
        noise_std=1e-16,
    )

    mixing_model = NonlinearLinearMixingModel(
        input_dim=data_lorenz.dim,
        output_dim=10,  # Smaller dimension for faster tests
        n_layers=2,  # Fewer layers for faster tests
        n_iter_cond_thresh=int(10),
        cond_thresh_ratio=1e-4,
        seed=seed + 1,
        lazy=True,  # Don't initialize yet
    )

    # Create a lazy dataset to get its configuration
    lazy_dataset = GenericDynamicsDataset(
        num_trials=200,
        num_steps=20,
        dynamics_model=data_lorenz,
        mixing_model=mixing_model,
        seed=seed,
        lazy=True,  # Don't initialize yet
    )
    return lazy_dataset.to_dict()


def assert_datasets_equal(dataset1: GenericDynamicsDataset,
                          dataset2: GenericDynamicsDataset):

    assert torch.all(dataset1.index == dataset2.index), "Indices are not equal"
    assert torch.all(
        dataset1.get_observed_data(dataset1.index) ==
        dataset2.get_observed_data(dataset2.index)), "Data is not equal"

    assert dataset1.auxilary_variables == dataset2.auxilary_variables, "Auxilary variables are not equal"


def assert_datasets_model_equal(dataset1: GenericDynamicsDataset,
                                dataset2: GenericDynamicsDataset):
    assert dataset1.dynamics_model == dataset2.dynamics_model, "Dynamics models are not equal"
    assert dataset1.mixing_model == dataset2.mixing_model, "Mixing models are not equal"


def test_lorenz_dataset_consistent_generation(tmp_path):
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()

    dataset_config = lorenz_dataset_config(seed=123)
    # try loading the same dataset again:
    dataset_1 = GenericDynamicsDataset.from_dict(
        dataset_config,
        kwargs={
            "GenericDynamicsDataset":
                dict(
                    root=tmp_data_root,
                    force_regenerate=True,
                )
        },
    )

    # this time we force regeneration to check the generation is consistent
    dataset_2 = GenericDynamicsDataset.from_dict(
        dataset_config,
        kwargs={
            "GenericDynamicsDataset":
                dict(
                    root=tmp_data_root,
                    force_regenerate=True,
                )
        },
    )

    assert_datasets_equal(dataset_1, dataset_2)
    assert_datasets_model_equal(dataset_1, dataset_2)
