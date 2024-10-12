from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import torch

from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class MSEToLogitFunction(Configurable, ABC):
    """Base class for MSE to logit conversion functions."""

    @abstractmethod
    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        """Convert MSE to logits."""


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class NegativeMSE(MSEToLogitFunction):
    """Converts MSE to logits by taking the negative."""

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return -mse


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class ScaledNegativeMSE(MSEToLogitFunction):
    """Converts MSE to logits by taking the negative with scaling."""
    scale: float = config_field(default=10.0)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return -mse * self.scale


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class InverseMSE(MSEToLogitFunction):
    """Converts MSE to logits by taking the inverse."""

    eps: float = config_field(default=1e-16)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return 1 / (mse + self.eps)


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class SoftLogMSE(MSEToLogitFunction):
    """Converts MSE to logits using a soft log transformation."""
    eps: float = config_field(default=1e-8)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return -torch.log(mse + self.eps)


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class TemperatureScaledMSE(MSEToLogitFunction):
    """Converts MSE to logits using temperature scaling."""
    tau: float = config_field(default=1.0)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return -mse / self.tau


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class ExponentialMSE(MSEToLogitFunction):
    """Converts MSE to logits using an exponential transformation."""
    alpha: float = config_field(default=1.0)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.alpha * mse)


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class IdentityMSE(MSEToLogitFunction):
    """Returns MSE unchanged."""

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return mse
