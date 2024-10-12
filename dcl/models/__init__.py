import importlib
import pkgutil
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Type

import torch

from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import Batch


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class BaseModel(Configurable, torch.nn.Module, ABC):
    """Base class for all models."""

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "model"

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    @property
    def input_type(self) -> Type[Batch]:
        """Returns the data type the model expects to be passed to the forward method."""
        # Get the type annotation from the forward method's data parameter
        return self.fn_input_type("forward", "data")

    def fn_input_type(
        self,
        function_name: str,
        arg_name: str,
    ) -> Type[Batch]:
        """Returns the annoated type of the argument of the function with name function_name."""
        fn = getattr(self, function_name)
        attr_type = fn.__annotations__[arg_name]
        return attr_type

    @abstractmethod
    def forward(
        self,
        data: Batch,
        **kwargs,
    ):
        pass

    @property
    def time_offset(self) -> int:
        """
        Specifies the number of consecutive time steps the model expects to be passed to the forward method to make a single prediction.
        """
        return 1


# Dynamically import all submodules in this package
for _, module_name, _ in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)

# Optionally define __all__
__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]
