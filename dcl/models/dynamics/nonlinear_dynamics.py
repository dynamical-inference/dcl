import random
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

from dcl.models.dynamics import BaseDynamicsModel
from dcl.models.utils import ModelStorageMixin
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import DynamicsData
from dcl.utils.datatypes import LorenzPrediction


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class LorenzAttractorDynamicsModel(ModelStorageMixin, BaseDynamicsModel):

    sigma: float = config_field(default=10.0)
    rho: float = config_field(default=28.0)
    beta: float = config_field(default=8.0 / 3.0)
    dt: float = config_field(default=0.01)
    noise_std: float = config_field(default=1e-16)

    def __lazy_post_init__(self):
        ModelStorageMixin.__lazy_post_init__(self)
        BaseDynamicsModel.__lazy_post_init__(self)
        self.dim = 3
        self.num_systems = 1
        self.init_parameters()

    def init_parameters(self):
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    @jaxtyped(typechecker=typechecked)
    def _lorenz_derivatives(
            self, x: Float[Tensor, "batch 3"]) -> Float[Tensor, "batch 3"]:
        """
        Compute the time derivatives for the Lorenz system.
        x: (*batch, 3), where the last dimension represents [x, y, z] coordinates.
        Returns the derivatives dx/dt, dy/dt, dz/dt.
        """
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        dx1 = self.sigma * (x2 - x1)
        dx2 = x1 * (self.rho - x3) - x2
        dx3 = x1 * x2 - self.beta * x3
        return torch.stack([dx1, dx2, dx3], dim=-1)

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        data: DynamicsData,
    ) -> LorenzPrediction:
        data = data.clone()
        # we predict based on a single (current) time step
        # hence, selecting on the dynamics_offset dimension
        x = data.x[..., -1, :]
        data.x_index = data.x_index[..., -1]
        derivatives = self._lorenz_derivatives(x)
        x = x + self.dt * derivatives

        if self.noise_std > 0:
            noise_dist = torch.distributions.normal.Normal(
                loc=torch.zeros_like(torch.tensor(self.noise_std)),
                scale=torch.tensor(self.noise_std),
            )
            x += noise_dist.rsample()

        data.x = x
        data.x_index += 1
        return LorenzPrediction.from_batch(data)

    def to_gt_dynamics(self, set_noise_to_zero: bool = True):
        lorenz_dynamics_model = self.clone()
        if set_noise_to_zero:
            lorenz_dynamics_model.noise_std = 0.
        return lorenz_dynamics_model
