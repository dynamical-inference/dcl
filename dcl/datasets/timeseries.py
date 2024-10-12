from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar, Union

import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import Tensor
from typeguard import typechecked

from dcl.datasets.base import BaseDataset
from dcl.utils.configurable import check_initialized
from dcl.utils.datatypes import AuxilaryVariables
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import GroundTruthData
from dcl.utils.datatypes import TrialMetadata

T = TypeVar("T")


@dataclass(kw_only=True)
class TimeSeriesDataset(BaseDataset, ABC):
    """Dataset of timeseries data."""

    @property
    @check_initialized
    def auxilary_variables(self) -> AuxilaryVariables:
        """Get the auxilary variables of the dataset."""
        return AuxilaryVariables()

    @property
    def ground_truth_data(self) -> GroundTruthData:
        """Get the ground truth data for evaluation purposes."""
        return GroundTruthData(
            observed=self.get_observed_data(self.index),
            index=self.index,
            auxilary=self.auxilary_variables,
            latents=None,
            dynamics_model=None,
        )

    @abstractmethod
    def get_observed_data(
        self, indices: Integer[Tensor, " *batch_shape"]
    ) -> Float[Tensor, "*batch_shape feature_dim"]:
        """Get a batch of data from the dataset."""
        raise NotImplementedError("Data batch access not implemented")

    @property
    def is_multi_trial(self) -> bool:
        """Check if the dataset is multi-trial."""
        auxilary = self.auxilary_variables
        if hasattr(auxilary, "trial_id"):
            unique_trials = auxilary.trial_id.unique()
            return len(unique_trials) > 1
        return False


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TensorDataset(TimeSeriesDataset):
    """In Memory Timeseries Dataset"""

    data: Float[Tensor, "num_samples feature_dim"]
    trial_id: Optional[Integer[Tensor, " num_samples"]] = None
    trial_time: Optional[Union[Float[Tensor, " num_samples"],
                               Integer[Tensor, " num_samples"]]] = None

    def __post_init__(self):
        if self.trial_id is None:
            self.trial_id = torch.zeros(self.data.shape[0], dtype=torch.int64)
        if self.trial_time is None:
            self.trial_time = torch.zeros(self.data.shape[0],
                                          dtype=torch.float32)
        super().__post_init__()

    @property
    def auxilary_variables(self) -> TrialMetadata:
        """Get the auxilary variables of the dataset."""
        return TrialMetadata(
            trial_id=self.trial_id,
            trial_time=self.trial_time,
        )

    @property
    def observed_dim(self) -> int:
        return self.data.shape[-1]

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(
        self, indices: Integer[Tensor, " *batch_shape"]
    ) -> Float[Tensor, "*batch_shape feature_dim"]:
        """Get a batch of data from the dataset."""
        return self.data[indices]

    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Union[
            Shaped[Tensor, " batch_shape ..."],
    ]]:
        return dict(
            data=self.get_observed_data(indices),
            trial_id=self.trial_id[indices],
            trial_time=self.trial_time[indices],
        )

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        """Split the dataset into a new dataset with the given indices."""

        data_dict = self[indices]
        return self.__class__(**data_dict)

    def __len__(self) -> int:
        return self.data.shape[0]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TensorDatasetWithLatents(TensorDataset):
    """TensorDataset with ground truth data"""

    latents: Float[Tensor, "num_samples latent_dim"]

    @property
    def latent_dim(self) -> int:
        return self.latents.shape[-1]

    def get_latent_data(
        self, index: Integer[Tensor, " *batch_shape"]
    ) -> Float[Tensor, "*batch_shape {self.latent_dim}"]:
        """Get the latent data of the dataset."""
        return self.latents[index]

    @property
    @check_initialized
    def ground_truth_data(self) -> GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.latents = self.get_latent_data(gt_batch.index)
        return GroundTruthData.from_batch(gt_batch)

    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Union[
            Shaped[Tensor, " batch_shape ..."],
    ]]:
        batch_dict = super().__getitems__(indices)
        batch_dict["latents"] = self.get_latent_data(indices)
        return batch_dict


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TensorDatasetFromFile(TensorDataset):
    """TensorDataset from a file"""

    data_path: str = config_field(default="data/timeseries_data.pt")
    observed_key: str = config_field(default="y")
    trial_id_key: str = config_field(default="trial_id")
    trial_time_key: str = config_field(default="trial_time")

    # NOTE: make these optional, so the init doesn't require them
    data: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    trial_id: Optional[Integer[Tensor, " num_samples"]] = None
    trial_time: Optional[Union[Float[Tensor, " num_samples"],
                               Integer[Tensor, " num_samples"]]] = None

    def __post_init__(self):
        self.data_dict = torch.load(self.data_path)
        self.data = self.data_dict[self.observed_key]
        self.trial_id = self.data_dict[self.trial_id_key]
        self.trial_time = self.data_dict[self.trial_time_key]
        super().__post_init__()

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        data_dict = self[indices]
        return TensorDataset(**data_dict)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TensorDatasetWithLatentsFromFile(TensorDatasetWithLatents):
    """TensorDatasetWithLatents from a file"""

    data_path: str = config_field(default="data/timeseries_data.pt")
    observed_key: str = config_field(default="y")
    trial_id_key: str = config_field(default="trial_id")
    trial_time_key: str = config_field(default="trial_time")
    latents_key: str = config_field(default="x")

    # NOTE: make these optional, so the init doesn't require them
    data: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    trial_id: Optional[Integer[Tensor, " num_samples"]] = None
    trial_time: Optional[Union[Float[Tensor, " num_samples"],
                               Integer[Tensor, " num_samples"]]] = None
    latents: Optional[Float[Tensor, "num_samples latent_dim"]] = None

    def __post_init__(self):
        self.data_dict = torch.load(self.data_path)
        self.data = self.data_dict[self.observed_key]
        self.trial_id = self.data_dict[self.trial_id_key]
        self.trial_time = self.data_dict[self.trial_time_key]
        self.latents = self.data_dict[self.latents_key]
        super().__post_init__()

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        data_dict = self[indices]
        return TensorDatasetWithLatents(**data_dict)
