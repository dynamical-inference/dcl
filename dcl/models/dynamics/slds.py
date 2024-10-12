from dataclasses import dataclass
from typing import TypeVar

import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

from dcl.models.dynamics import BaseDynamicsModel
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.switching_dynamics import GumbelSwitchingModel
from dcl.models.dynamics.switching_dynamics import MarkovSwitchingModel
from dcl.models.dynamics.switching_dynamics import MSESwitchingModel
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import GumbelSLDSInput
from dcl.utils.datatypes import GumbelSLDSPrediction
from dcl.utils.datatypes import LinearDynamicsInput

T = TypeVar("T", bound="GumbelSLDS")


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class GumbelSLDS(BaseDynamicsModel):
    """Switching Linear Dynamical System that uses MSE to determine switching state."""

    linear_dynamics: LinearDynamicsModel = config_field(
        default_factory=LinearDynamicsModel)
    switching_model: GumbelSwitchingModel = config_field(
        default_factory=MSESwitchingModel)

    @property
    def tau(self) -> float:
        """Temperature parameter for Gumbel-Softmax sampling."""
        return self.switching_model.tau

    @tau.setter
    def tau(self, value: float):
        """Set temperature parameter for Gumbel-Softmax sampling."""
        self.switching_model.tau = value

    @property
    def num_systems(self) -> int:
        """Number of linear dynamical systems."""
        return self.linear_dynamics.num_systems

    @property
    def num_modes(self) -> int:
        """Alias for num_systems."""
        return self.num_systems

    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        return self.linear_dynamics.dim

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        data: GumbelSLDSInput,
        num_samples: int = 1,
    ) -> GumbelSLDSPrediction:
        """Forward pass of the SLDS model.
        Uses Gumbel-Softmax sampling for the mode prediction.

        Args:
            x: Input state tensor of shape (batch, dim)
            x_next: Next state tensor of shape (batch, dim)
            previous_mode: Previous mode indices of shape (batch)
            num_samples: Number of samples to draw from mode distribution

        Returns:
            Tuple containing:
            - Predicted next states of shape (batch, num_samples, dim)
            - Mode log probabilities (logits) of shape (batch, num_modes)
            - Mode gumbel samples of shape (batch, num_samples, num_modes)
        """
        data = data.clone()
        # Predict next state for all possible modes
        # Expand x to (batch, num_modes, dim)
        data.x_next_pred = self.forward_all_modes(data)

        # Sample modes using Gumbel-Softmax
        # mode_samples shape (batch, num_samples, num_modes)
        mode_samples = self.switching_model(
            self.switching_model.input_type.from_batch(data),
            num_samples=num_samples,
        )

        # Expand predictions to include samples dimension
        # x_pred shape (batch, num_samples, num_modes, dim)
        x_next_pred = data.x_next_pred.unsqueeze(1).expand(
            -1, num_samples, -1, -1)

        # when in training mode, mode_samples are probabilities and this is a weighted average
        # when in evaluation mode, mode_samples are one-hot encoded and this is a selection (think argmax)
        # x_next_pred shape (batch, num_samples, dim)
        x_next_pred = (x_next_pred * mode_samples.unsqueeze(-1)).sum(dim=-2)

        data.x = x_next_pred
        data.modes = mode_samples
        # update the index to reflect the forward prediction
        data.x_index = data.x_index[..., -1] + self.index_forward_offset
        # also expand to include num_samples dimension
        data.x_index = data.x_index.unsqueeze(1).expand(-1, num_samples)

        return GumbelSLDSPrediction.from_batch(data)

    @jaxtyped(typechecker=typechecked)
    def forward_all_modes(
        self,
        data: GumbelSLDSInput,
    ) -> Float[Tensor, "batch {self.num_modes} dim"]:
        batch_size, dim = data.batch, data.dim

        # Expand x to predict with all possible modes
        x = data.x.unsqueeze(1).expand(-1, self.num_systems, -1, -1)
        x_index = data.x_index.unsqueeze(1).expand(-1, self.num_systems, -1)
        # create a tensor of all modes
        system_idx = torch.arange(self.num_systems, device=x.device)
        # expand to match batch
        system_idx = system_idx.expand(batch_size, self.num_systems)

        # and now we need to flatten the batch and systems dimension
        x = x.flatten(start_dim=0, end_dim=1)
        system_idx = system_idx.flatten(start_dim=0, end_dim=1)
        x_index = x_index.flatten(start_dim=0, end_dim=1)

        # Get predictions for all modes
        lds_pred = self.linear_dynamics(
            LinearDynamicsInput(
                x=x,
                x_index=x_index,
                system_idx=system_idx,
            ))

        # unflatten the result and reshape to (batch, num_modes, dim)
        x_next_pred = lds_pred.x.reshape(batch_size, self.num_systems, dim)
        return x_next_pred

    #@jaxtyped(typechecker=typechecked)
    def to_gt_dynamics(self: T, set_noise_to_zero: bool = True) -> T:
        """Create a new dynamics model that represents the same dynamics and can be used for training"""
        if isinstance(self.switching_model, MarkovSwitchingModel):
            # we can't use markov switching model for training, because
            # a) it's forward pass requires knowledge about the previous modes
            # b) the model is stochastic
            # So instead we replace the MarkowSwitching with MSESwitchingModel
            return GumbelSLDS(
                linear_dynamics=self.linear_dynamics.to_gt_dynamics(
                    set_noise_to_zero),
                switching_model=MSESwitchingModel(num_modes=self.num_modes),
            ).clone()
        else:
            return super().to_gt_dynamics()
