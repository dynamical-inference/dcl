"""
Portions of this code are based on https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/cebra/models/model.py
which is distributed under Apache License, Version 2.0
"""

from abc import abstractmethod
from dataclasses import dataclass

import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import nn
from torch import Tensor
from typeguard import typechecked

from dcl.models import BaseModel
from dcl.models.utils import ModelStorageMixin
from dcl.utils.datatypes import config_field


class NormLayer(nn.Module):

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp / torch.norm(inp, dim=-1, keepdim=True)


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class EncoderModel(ModelStorageMixin, BaseModel):
    """ Base class for all encoder models. """

    input_dim: int = config_field(default=5)
    output_dim: int = config_field(default=2)
    normalize: bool = config_field(default=False)

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "encoder_model"

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    def __lazy_post_init__(self):
        ModelStorageMixin.__lazy_post_init__(self)
        BaseModel.__lazy_post_init__(self)
        self.init_parameters()
        if self.normalize:
            self.norm_layer = NormLayer()

    @abstractmethod
    def init_parameters(self):
        pass

    @abstractmethod
    def _forward(
        self, x: Float[Tensor, "*batch encoder_offset {self.input_dim}"]
    ) -> Float[Tensor, "*batch {self.output_dim}"]:
        pass

    @jaxtyped(typechecker=typechecked)
    def forward(
        self, x: Float[Tensor, "*batch encoder_offset {self.input_dim}"]
    ) -> Float[Tensor, "*batch {self.output_dim}"]:
        x = self._forward(x)
        if self.normalize:
            x = self.norm_layer(x)
        return x


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class MLP(EncoderModel):
    """ Simple MLP encoder model. """

    hidden_dim: int = config_field(default=128)
    num_layers: int = config_field(default=3)

    def init_parameters(self):
        first_layers = [
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(
                self.input_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
        ]
        middle_layers = []
        for _ in range(self.num_layers - 1):
            middle_layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            ])
        last_layers = [
            nn.Linear(self.hidden_dim, self.output_dim),
        ]
        layers = first_layers + middle_layers + last_layers
        self.net = nn.Sequential(*layers)

    @jaxtyped(typechecker=typechecked)
    def _forward(
        self, x: Float[Tensor, "*batch encoder_offset {self.input_dim}"]
    ) -> Float[Tensor, "*batch {self.output_dim}"]:
        return self.net(x)


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class Offset1ModelMLP(EncoderModel):
    """ Simple MLP encoder model. """

    hidden_dim: int = config_field(default=128)
    num_layers: int = config_field(default=3)

    def init_parameters(self):
        print("Ignoring hidden_dim and num_layers with Offset1ModelMLP")

        layers = [
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(
                self.input_dim,
                self.output_dim * 30,
            ),
            nn.GELU(),
            nn.Linear(self.output_dim * 30, self.output_dim * 30),
            nn.GELU(),
            nn.Linear(self.output_dim * 30, self.output_dim * 10),
            nn.GELU(),
            nn.Linear(int(self.output_dim * 10), self.output_dim),
        ]
        self.net = nn.Sequential(*layers)

    @jaxtyped(typechecker=typechecked)
    def _forward(
        self, x: Float[Tensor, "*batch encoder_offset {self.input_dim}"]
    ) -> Float[Tensor, "*batch {self.output_dim}"]:
        return self.net(x)


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class IdentityEncoder(EncoderModel):
    """Identity model for debugging purposes."""

    def init_parameters(self):
        # add a scalar parameter to be optimized
        # we only add this here so we technically have a parameter to optimize so pytorch doesn't complain
        # but we only add and subtract it in the forward pass to still have an identity transform
        self.bias = nn.Parameter(torch.zeros(1, 1))

    @jaxtyped(typechecker=typechecked)
    def _forward(
        self, x: Float[Tensor, "*batch encoder_offset {self.input_dim}"]
    ) -> Float[Tensor, "*batch {self.output_dim}"]:
        """Compute the embedding given the input signal.

        Args:
            inp: The input tensor of shape `num_samples x self.num_input x time`

        Returns:
            The output tensor of shape `num_samples x self.num_output x time`.

        """
        emb = x.squeeze(-2)
        emb += self.bias
        emb -= self.bias
        return emb


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class LinearEncoderModel(EncoderModel):
    """Simple linear encoder model."""

    def init_parameters(self):
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    @jaxtyped(typechecker=typechecked)
    def _forward(
        self, x: Float[Tensor, "*batch encoder_offset {self.input_dim}"]
    ) -> Float[Tensor, "*batch {self.output_dim}"]:
        """Linear transformation of input.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Transformed tensor of shape (batch_size, output_dim)
        """
        return self.linear(x.squeeze(-2))
