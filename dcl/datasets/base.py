from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, TypeVar, Union

import torch
from jaxtyping import Integer
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import Tensor
from typeguard import typechecked

from dcl.utils.configurable import check_initialized
from dcl.utils.configurable import Configurable

T = TypeVar("T")


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class BaseDataset(Configurable, ABC):
    """Abstract base class for dataset"""

    @property
    @check_initialized
    def index(self) -> Integer[Tensor, " num_samples"]:
        """Get the index of the dataset."""
        return torch.arange(len(self))

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        raise NotImplementedError("Data length not implemented")

    # no typechecking here, jaxtyping just for docs
    def __getitem__(
        self, idx: Union[int, List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Union[
            Shaped[Tensor, " batch_shape ..."],
            any,
    ]]:
        """Returns a dictionary with all the available data for the given index."""
        if isinstance(idx, int):
            return self.__getitems__(torch.tensor([idx]))
        else:
            return self.__getitems__(idx)

    # no typechecking here, jaxtyping just for docs
    @abstractmethod
    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Union[
            Shaped[Tensor, " batch_shape ..."],
            any,
    ]]:
        """Get a list of samples from dataset."""
        raise NotImplementedError("Data item access not implemented")

    @abstractmethod
    def split(self: T, indices: Integer[Tensor, " batch_shape"]) -> T:
        """Split the dataset into a new dataset with the given indices."""
        raise NotImplementedError("Split not implemented")
