from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

from dcl.models.dynamics import BaseDynamicsModel
from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field
from dcl.utils.rotation_matrix import MinMaxRotationSampler
from dcl.utils.rotation_matrix import RotationSampler


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class LDSParameterInitializer(Configurable, ABC):
    """
    A class for initializing the parameters of a LinearDynamicsModel.
    """

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        pass


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class NormalLDSParameters(LDSParameterInitializer):
    """
    Initializes Linear Dynamical System parameters from normal distribution.
    """
    dynamics_matrix_mean: float = config_field(default=0.0)
    dynamics_matrix_std: float = config_field(default=1.0)
    dynamics_bias_mean: float = config_field(default=0.0)
    dynamics_bias_std: float = config_field(default=1.0)

    @jaxtyped(typechecker=typechecked)
    def dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        return torch.normal(mean=self.dynamics_matrix_mean,
                            std=self.dynamics_matrix_std,
                            size=(num_systems, dim, dim))

    @jaxtyped(typechecker=typechecked)
    def dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        return torch.normal(mean=self.dynamics_bias_mean,
                            std=self.dynamics_bias_std,
                            size=(num_systems, dim))


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class IdentityLDSParameters(LDSParameterInitializer):
    """
    LinearDynamicsModel initialized with identity matrices.
    """

    @jaxtyped(typechecker=typechecked)
    def dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        A = torch.stack([torch.eye(dim) for _ in range(num_systems)])
        return A

    @jaxtyped(typechecker=typechecked)
    def dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class RotationLDSParameters(LDSParameterInitializer):
    """
    LinearDynamicsModel initialized with rotation matrices.
    """
    rotation_sampler: RotationSampler = config_field(
        default_factory=MinMaxRotationSampler)

    @jaxtyped(typechecker=typechecked)
    def dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        A = torch.stack(
            [self.rotation_sampler.sample(dim) for _ in range(num_systems)])
        return A

    @jaxtyped(typechecker=typechecked)
    def dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
def freeze_model(model: BaseDynamicsModel) -> None:
    """
    Freezes the given PyTorch model by disabling gradient updates for its parameters
    and setting it to evaluation mode.
    """
    # Disable gradient computation for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Set the model to evaluation mode (important if the model has BatchNorm or Dropout layers)
    model.eval()
