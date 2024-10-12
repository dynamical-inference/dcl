import importlib
import pkgutil
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Dict, Iterator, Optional

import torch
from jaxtyping import jaxtyped
from typeguard import typechecked

from dcl.datasets.base import BaseDataset
from dcl.utils.configurable import check_initialized
from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field


@dataclass(kw_only=True)
class BaseDataLoader(Configurable, ABC):
    """Base class for iterative data loaders that can be used for training.

    This class uses lazy initialization, meaning the dataset is not loaded immediately
    upon instantiation. Instead, you must call lazy_init() with the dataset after creating
    the loader object:

    Example:
        >>> loader = BaseDataLoader(batch_size=32)  # Create loader without dataset
        >>> loader.lazy_init(dataset)  # Initialize with dataset when ready

    Args:
        batch_size: Size of each batch (default: 32)
        seed: Random seed for shuffling (default: 42)
        lazy: Whether to use lazy initialization (default: True)

    Yields:
        Dict[str, Tensor]: Batches of the specified size from the dataset, with exact
            contents depending on the dataset and loader implementation.

    Note:
        The loader will automatically move batches to the specified device (default: 'cpu').
        Use the .to() method to change devices.
    """

    # By default dataloaders need lazy initialization
    lazy: bool = True

    batch_size: int = config_field(default=32)
    seed: int = config_field(default=42)

    _device: torch.device = field(
        default_factory=lambda: torch.device('cuda')
        if torch.cuda.is_available() else torch.device('cpu'),
        init=False,
        repr=False)

    _dataset: Optional[BaseDataset] = field(default=None,
                                            init=False,
                                            repr=False)

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "loader"

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(self, dataset: BaseDataset):
        """Initialize the data loader with a dataset."""
        self._dataset = dataset
        torch.manual_seed(self.seed)
        self.reset()
        return self

    def validate_config(self):
        """Validate the configuration of the data loader."""
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(
                f"Batch size has to be None, or a non-negative value. Got {self.batch_size}."
            )
        super().validate_config()

    @property
    @check_initialized
    def dataset(self) -> BaseDataset:
        """Get the dataset of the data loader."""
        return self._dataset

    @property
    def device(self) -> torch.device:
        """Get the device of the data loader."""
        return self._device

    def to(self, device: torch.device) -> 'BaseDataLoader':
        """Move the data loader to the specified device."""
        self._device = device
        return self

    @abstractmethod
    def reset(self):
        raise NotImplementedError(
            "reset method must be implemented by subclass.")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "__len__ method must be implemented by subclass.")

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        raise NotImplementedError(
            "__iter__ method must be implemented by subclass.")


# Dynamically import all submodules in this package
for _, module_name, _ in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)

# Optionally define __all__
__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]
