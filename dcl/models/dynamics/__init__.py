import importlib
import pkgutil
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar

from dcl.models import BaseModel
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import DynamicsData

T = TypeVar("T", bound="BaseDynamicsModel")


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class BaseDynamicsModel(BaseModel, ABC):
    """Base class for all dynamics models. Mostly for type checking."""

    index_forward_offset: int = config_field(default=1)

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "dynamics_model"

    @property
    def input_type(self) -> Type[DynamicsData]:
        return super().input_type

    @abstractmethod
    def forward(
        self,
        data: DynamicsData,
        **kwargs,
    ):
        pass

    def to_gt_dynamics(self: T) -> T:
        """Create a new dynamics model that represents the same dynamics and can be used for training"""
        return self.clone()


# Dynamically import all submodules in this package
for _, module_name, _ in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)

# Optionally define __all__
__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]
