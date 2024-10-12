from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING, TypeVar, Union

import numpy as np
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import TrialMetadata

if TYPE_CHECKING:
    from dcl.datasets.base import BaseDataset
    from dcl.datasets.timeseries import MultiTrialDatasetMixin

T = TypeVar("T", bound="BaseDataset")


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DatasetSplit(Configurable, ABC):
    """Handles dataset splitting into train, val and test sets with reproducible seeding."""

    seed: int = config_field(default=42)

    train_ratio: float = config_field(default=0.7)
    test_ratio: float = config_field(default=0.15)

    @property
    def val_ratio(self) -> float:
        _val_ratio = 1 - self.train_ratio - self.test_ratio
        if _val_ratio <= 0:
            raise ValueError(
                f"Validation ratio is less than 0. Got {_val_ratio} with train_ratio={self.train_ratio} and test_ratio={self.test_ratio}."
            )
        return _val_ratio

    def reseed(self):
        self.rng = np.random.RandomState(self.seed)

    def create_split(
        self,
        dataset: T,
    ) -> Tuple[T, T, T]:
        """Create reproducible train/val/test splits."""
        self.reseed()
        train_idx, val_idx, test_idx = self._create_split(dataset)
        return (
            dataset.split(train_idx),
            dataset.split(val_idx),
            dataset.split(test_idx),
        )

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "BaseDataset",
    ) -> Tuple[
            Integer[Union[Tensor, np.ndarray], " train_size"],
            Integer[Union[Tensor, np.ndarray], " val_size"],
            Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        pass

    def absolut_split_sizes(self, total_size: int) -> Tuple[int, int, int]:
        """Calculate the absolute sizes of the splits."""
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        assert train_size > 0, "Absolute train size is less than 0"
        assert val_size > 0, "Absolute validation size is less than 0"
        assert test_size > 0, "Absolute test size is less than 0"
        return train_size, val_size, test_size


@dataclass(kw_only=True)
class RandomSplit(DatasetSplit):
    """Handles dataset splitting with reproducible seeding."""

    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "BaseDataset",
    ) -> Tuple[
            Integer[Union[Tensor, np.ndarray], " train_size"],
            Integer[Union[Tensor, np.ndarray], " val_size"],
            Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        """Create reproducible train/val/test splits."""

        dataset_index = dataset.index

        rand_permutation = np.arange(len(dataset_index))
        self.rng.shuffle(rand_permutation)

        (
            train_size,
            val_size,
            test_size,
        ) = self.absolut_split_sizes(len(dataset_index))

        train_rand_permutation = rand_permutation[:train_size]
        val_rand_permutation = rand_permutation[train_size:train_size +
                                                val_size]
        test_rand_permutation = rand_permutation[train_size + val_size:]

        return (
            dataset_index[train_rand_permutation],
            dataset_index[val_rand_permutation],
            dataset_index[test_rand_permutation],
        )


@dataclass(kw_only=True)
class SequentialSplit(DatasetSplit):
    """For splitting data into sequential train/val/test splits."""

    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "BaseDataset",
    ) -> Tuple[
            Integer[Union[Tensor, np.ndarray], " train_size"],
            Integer[Union[Tensor, np.ndarray], " val_size"],
            Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        """Create reproducible train/val/test splits."""

        dataset_index = dataset.index

        (
            train_size,
            val_size,
            test_size,
        ) = self.absolut_split_sizes(len(dataset_index))

        train_index = dataset_index[:train_size]
        val_index = dataset_index[train_size:train_size + val_size]
        test_index = dataset_index[train_size + val_size:]

        return (
            train_index,
            val_index,
            test_index,
        )


@dataclass(kw_only=True)
class TrialSplit(DatasetSplit):
    """For splitting data into train/val/test by sampling random trials for each split"""

    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "MultiTrialDatasetMixin",
    ) -> Tuple[
            Integer[Union[Tensor, np.ndarray], " train_size"],
            Integer[Union[Tensor, np.ndarray], " val_size"],
            Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        """Create reproducible train/val/test splits."""

        aux_variables: TrialMetadata = dataset.auxilary_variables
        assert isinstance(
            aux_variables, TrialMetadata
        ), "TrialSplit only works for datasets with tiral meta data"

        trial_ids = aux_variables.trial_id
        unique_trial_ids = np.unique(trial_ids)

        (
            train_size,
            val_size,
            test_size,
        ) = self.absolut_split_sizes(len(unique_trial_ids))

        unique_trial_ids_permutation = self.rng.permutation(unique_trial_ids)
        train_trial_ids = unique_trial_ids_permutation[:train_size]
        val_trial_ids = unique_trial_ids_permutation[train_size:train_size +
                                                     val_size]
        test_trial_ids = unique_trial_ids_permutation[train_size + val_size:]

        # assert that there's no overlap between the splits
        assert len(np.intersect1d(train_trial_ids, val_trial_ids)) == 0
        assert len(np.intersect1d(train_trial_ids, test_trial_ids)) == 0
        assert len(np.intersect1d(val_trial_ids, test_trial_ids)) == 0

        dataset_index = dataset.index
        train_index = dataset_index[np.isin(trial_ids, train_trial_ids)]
        val_index = dataset_index[np.isin(trial_ids, val_trial_ids)]
        test_index = dataset_index[np.isin(trial_ids, test_trial_ids)]

        return (train_index, val_index, test_index)
