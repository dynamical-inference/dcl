import multiprocessing
import traceback
from pathlib import Path

import pytest
import torch

from dcl.datasets.synthetic import GenericDynamicsDataset
from dcl.datasets.synthetic import SLDSDynamicsDataset
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.models.dynamics.switching_dynamics import FixedMarkovSwitchingModel
from dcl.models.dynamics.utils import RotationLDSParameters
from dcl.models.mixing import NonlinearLinearMixingModel
from dcl.utils.rotation_matrix import MinMaxRotationSampler


def slds_dataset_config(seed: int = 42, num_modes: int = 3):
    # Create a single dataset configuration
    data_slds = GumbelSLDS(
        linear_dynamics=LinearDynamicsModel(
            seed=seed,
            dim=4,  # Smaller dimension for faster tests
            num_systems=num_modes,
            initializer=RotationLDSParameters(
                rotation_sampler=MinMaxRotationSampler(
                    min_angle=0,
                    max_angle=10,
                )),
            noise_std=1e-4),
        switching_model=FixedMarkovSwitchingModel(num_modes=num_modes,
                                                  transition_probability=1e-4),
    )

    mixing_model = NonlinearLinearMixingModel(
        input_dim=data_slds.dim,
        output_dim=10,  # Smaller dimension for faster tests
        n_layers=2,  # Fewer layers for faster tests
        n_iter_cond_thresh=int(10),
        cond_thresh_ratio=1e-4,
        seed=seed + 1,
        lazy=True,  # Don't initialize yet
    )

    # Create a lazy dataset to get its configuration
    lazy_dataset = SLDSDynamicsDataset(
        num_trials=200,
        num_steps=20,
        dynamics_model=data_slds,
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


def test_dataset_consistent_storage(tmp_path):
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()

    dataset_config = slds_dataset_config(seed=1958, num_modes=3)
    # try loading the same dataset again:
    dataset_1 = SLDSDynamicsDataset.from_dict(
        dataset_config,
        kwargs={
            "SLDSDynamicsDataset":
                dict(
                    root=tmp_data_root,
                    force_regenerate=True,
                )
        },
    )

    # try loading the same dataset again:
    dataset_2 = SLDSDynamicsDataset.from_dict(
        dataset_config,
        kwargs={"SLDSDynamicsDataset": dict(root=tmp_data_root,)},
    )

    assert_datasets_equal(dataset_1, dataset_2)
    assert_datasets_model_equal(dataset_1, dataset_2)


def test_dataset_consistent_generation(tmp_path):
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()
    # clean up any existing data
    if tmp_data_root.exists():
        import shutil
        shutil.rmtree(tmp_data_root,)
    tmp_data_root.mkdir()

    dataset_config = slds_dataset_config(seed=123, num_modes=5)
    # try loading the same dataset again:
    dataset_1 = SLDSDynamicsDataset.from_dict(
        dataset_config,
        kwargs={
            "SLDSDynamicsDataset":
                dict(
                    root=tmp_data_root,
                    force_regenerate=True,
                )
        },
    )

    # this time we force regeneration to check the generation is consistent
    dataset_2 = SLDSDynamicsDataset.from_dict(
        dataset_config,
        kwargs={
            "SLDSDynamicsDataset":
                dict(
                    root=tmp_data_root,
                    force_regenerate=True,
                )
        },
    )

    assert_datasets_equal(dataset_1, dataset_2)
    assert_datasets_model_equal(dataset_1, dataset_2)


def test_concurrent_dataset_generation(tmp_path):
    """Test creating multiple dataset instances concurrently.

    This test creates n dataset instances in parallel and verifies that
    they all initialize properly without race conditions.

    All instances use the EXACT SAME configuration, including the same seed,
    to test how the system handles concurrent access to the same dataset files.
    """

    # TODO(stes): Skipped for now, very slow
    pytest.skip(
        "Skipping, test too slow; See https://github.com/dynamical-inference/dyncl-dev/issues/25"
    )

    num_processes = 20  # Number of concurrent dataset instantiations

    # Create a shared data directory
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()

    # Get the configuration dictionary
    config_dict = slds_dataset_config(seed=864, num_modes=2)

    def create_dataset(tmp_data_root, config_dict, event=None):
        # Wait for event if provided (used in staggered test)
        if event is not None:
            event.wait()

        # Create dataset from the same config dict
        dataset = SLDSDynamicsDataset.from_dict(
            config_dict,
            kwargs={"SLDSDynamicsDataset": dict(root=tmp_data_root,)})
        # Verify the dataset was created properly
        assert len(dataset) > 0
        return dataset

    # Define a wrapper function for multiprocessing
    def create_dataset_wrapper(tmp_data_root, config_dict, results):
        try:
            dataset = create_dataset(tmp_data_root, config_dict)
            results.append(dataset)
        except Exception as e:
            print(f"Process failed with exception: {e}")
            # print full traceback
            traceback.print_exc()
            results.append(None)

    # Start multiple processes to create datasets concurrently
    processes = []
    results = multiprocessing.Manager().list()

    for i in range(num_processes):
        p = multiprocessing.Process(target=create_dataset_wrapper,
                                    args=(tmp_data_root, config_dict, results))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Verify all processes successfully created their datasets
    assert len(results) == num_processes
    assert all(r is not None for r in results), "Some dataset creations failed"

    # Compare all datasets to ensure they are equal
    for i in range(1, len(results)):
        assert_datasets_equal(results[0], results[i])
        assert_datasets_model_equal(results[0], results[i])


def test_staggered_dataset_generation(tmp_path):
    """Test creating multiple dataset instances with staggered starts.

    This test creates n dataset instances in parallel but starts each one
    with a small delay to test how the system handles near-concurrent access.

    All instances use the EXACT SAME configuration, including the same seed,
    to test how the system handles concurrent access to the same dataset files.
    """

    pytest.skip(
        "Skipping, test too slow; See https://github.com/dynamical-inference/dyncl-dev/issues/25"
    )

    import multiprocessing
    import time

    num_processes = 15  # Number of dataset instantiations
    stagger_delay = 1  # Seconds between starts

    # Create a shared data directory
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()

    # Get the configuration dictionary
    config_dict = slds_dataset_config(seed=346, num_modes=2)

    # Create events for staggered starts
    events = [multiprocessing.Event() for _ in range(num_processes)]
    processes = []
    results = multiprocessing.Manager().list()

    def create_dataset_with_result(tmp_data_root, config_dict, event, results):
        try:
            dataset = create_dataset(tmp_data_root, config_dict, event)
            results.append(dataset)
        except Exception as e:
            print(f"Process failed with exception: {e}")
            results.append(None)

    # Define the dataset creation function
    def create_dataset(tmp_data_root, config_dict, event=None):
        # Wait for event
        if event is not None:
            event.wait()

        try:
            # Create dataset from the same config dict
            dataset = SLDSDynamicsDataset.from_dict(
                config_dict,
                kwargs={"SLDSDynamicsDataset": dict(root=tmp_data_root,)})
            # Verify the dataset was created properly
            assert len(dataset) > 0
            # Return the dataset
            return dataset
        except Exception as e:
            print(f"Process failed: {e}")
            return None

    # Start multiple processes
    for i in range(num_processes):
        p = multiprocessing.Process(target=create_dataset_with_result,
                                    args=(tmp_data_root, config_dict, events[i],
                                          results))
        processes.append(p)
        p.start()

    # Trigger the events with staggered delays
    for i, event in enumerate(events):
        time.sleep(stagger_delay)
        event.set()

    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=60)  # Set a timeout for safety

    # Verify all processes completed
    for i, p in enumerate(processes):
        assert not p.is_alive(), f"Process {i} did not complete in time"

    # Verify all datasets were created successfully
    assert len(results) == num_processes
    assert all(d is not None for d in results), "Some dataset creations failed"

    # Compare all datasets against the first one
    reference_dataset = results[0]
    for i, dataset in enumerate(results[1:], 1):
        assert_datasets_equal(reference_dataset, dataset)
        assert_datasets_model_equal(reference_dataset, dataset)


if __name__ == "__main__":
    pytest.main([__file__])
