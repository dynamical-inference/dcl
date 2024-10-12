from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Type

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from torch.nn import functional as F
from typeguard import typechecked

from dcl.models.dynamics import BaseDynamicsModel
from dcl.models.dynamics.mse_logits import InverseMSE
from dcl.models.dynamics.mse_logits import MSEToLogitFunction
from dcl.models.utils import ModelStorageMixin
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import DynamicsData
from dcl.utils.datatypes import ErrorBasedSwitchingDynamicsData
from dcl.utils.datatypes import ModeBasedSwitchingInput
from dcl.utils.datatypes import state_field
from dcl.utils.datatypes import TimeBasedSwitchingInput


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class SwitchingModel(ModelStorageMixin, BaseDynamicsModel, ABC):
    """Models the switching behavior of a switching linear dynamical system.
    This model predicts the mode (the lds system to be used for the next timestep) of the SLDS for the given input."""

    num_modes: int = config_field(default=1)

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    def __lazy_post_init__(self):
        ModelStorageMixin.__lazy_post_init__(self)
        BaseDynamicsModel.__lazy_post_init__(self)
        self.init_parameters()

    def init_parameters(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._init_parameters()

    @abstractmethod
    def _init_parameters(self):
        pass


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class GumbelSwitchingModel(SwitchingModel, ABC):
    """
    These types of switching models don't predict the mode directly,
    but rather model the distribution of the modes and sample via gumbel_softmax in the forward pass.
    """

    tau: float = state_field(
        default=1.0,
        init=False,
    )

    @property
    def input_type(self) -> Type[DynamicsData]:
        """
        Returns the data type the model expects to be passed to the forward method.
        For GumbelSwitchingModel, the forward type is determined by the logits function instead.
        """
        # Get the type annotation from the forward method's data parameter
        return self.fn_input_type("logits", "data")

    def forward(
        self,
        data: DynamicsData,
        num_samples: int,
    ) -> Float[Tensor, "batch {num_samples} {self.num_modes}"]:
        """
        Returns the logits used for sampling via gumbel_softmax as well as the sampled modes.
        """

        if self.training:
            tau = self.tau
            hard_sampling = False
        else:
            # during evaluation we use a lower temperature
            tau = 1e-16
            hard_sampling = True

        logits = self.logits(data)
        # expand logits to sample num_samples times
        logits = logits.unsqueeze(-2).expand(-1, num_samples, -1)

        # use gumple softmax for training
        gumbel_sampels = F.gumbel_softmax(
            logits,
            tau=tau,
            hard=hard_sampling,
        )

        return gumbel_sampels

    @abstractmethod
    def logits(
        self,
        data: DynamicsData,
    ) -> Float[Tensor, "batch {self.num_modes}"]:
        """
        Returns the logits used for sampling via gumbel_softmax.
        """


@jaxtyped(typechecker=typechecked)
def probabilities_to_logits(probabilities, epsilon=1e-16):
    # Adjust probabilities to avoid log(0)
    adjusted_probabilities = torch.clamp(probabilities, epsilon, 1)
    logits = torch.log(adjusted_probabilities / torch.clamp(
        (1 - adjusted_probabilities), epsilon, float('inf')))
    return logits


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class TimeBasedSwitchingModel(GumbelSwitchingModel, ABC):
    """
    Switching model that predicts the mode based on the current time step.
    """

    @abstractmethod
    def logits(
        self,
        data: TimeBasedSwitchingInput,
    ) -> Float[Tensor, "batch {self.num_modes}"]:
        """
        Returns the logits used for sampling via gumbel_softmax.
        """


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class ModeBasedSwitchingModel(GumbelSwitchingModel, ABC):
    """
    Switching model that predicts the current mode based on the previous mode.
    """

    @abstractmethod
    def logits(
        self,
        data: ModeBasedSwitchingInput,
        **kwargs,
    ) -> Float[Tensor, "batch {self.num_modes}"]:
        """
        Returns the logits used for sampling via gumbel_softmax.
        """


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class MarkovSwitchingModel(ModeBasedSwitchingModel):
    """
    Models the switching behavior via a Markov chain parametrized by a transition matrix.
    """

    def _init_parameters(self):
        super()._init_parameters()
        self._init_transition_matrix()

    def _init_transition_matrix(self):
        self.transition_matrix = torch.nn.Parameter(
            torch.randn(self.num_modes, self.num_modes))

    @jaxtyped(typechecker=typechecked)
    def logits(
        self,
        data: ModeBasedSwitchingInput,
        **kwargs,
    ) -> Float[Tensor, "batch {self.num_modes}"]:
        """
        Returns the logits used for sampling via gumbel_softmax.
        """
        return torch.log(self.transition_matrix[data.modes[:, -1]])


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class FixedMarkovSwitchingModel(MarkovSwitchingModel):
    """
    Initializes the transition matrix that has a fixed transition probability.
    The transition probability determines the probability of transitioning to a different mode.
    I.e. stay probability = 1 - transition probability.
    """

    transition_probability: float = config_field(default=0.1)

    def _init_transition_matrix(self):
        stay_prob = 1 - self.transition_probability
        transition_matrix = torch.rand((self.num_modes, self.num_modes))
        # first we fill the diagonal with 0s
        transition_matrix.fill_diagonal_(0)
        # normalize the rows to sum up to transition_prob
        transition_matrix = transition_matrix / transition_matrix.sum(
            dim=1, keepdim=True)
        transition_matrix = transition_matrix * self.transition_probability
        # now we fill the diagonal with the remaining probability mass
        transition_matrix.fill_diagonal_(stay_prob)
        self.transition_matrix = torch.nn.Parameter(transition_matrix)


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class SymmetricFixedMarkovSwitchingModel(FixedMarkovSwitchingModel):
    """
    Additionally makes sure the transition matrix is symmetric.
    """

    def _init_transition_matrix(self):
        stay_prob = 1 - self.transition_probability

        transition_prob_off_diagonal = self.transition_probability / (
            self.num_modes - 1)

        transition_matrix = torch.full(
            (self.num_modes, self.num_modes),
            fill_value=transition_prob_off_diagonal,
        )

        transition_matrix.fill_diagonal_(stay_prob)
        self.transition_matrix = torch.nn.Parameter(transition_matrix)


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class PredictionErrorSwitchingModel(GumbelSwitchingModel, ABC):
    """
    Switching model that predicts the mode based on the prediction error.
    """

    def _init_parameters(self):
        super()._init_parameters()
        # this is actually a parameter free model
        # however because of the inheritance from torch.nn.Module we need a parameter
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1))


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class MSESwitchingModel(PredictionErrorSwitchingModel):
    """
    Switching model that predicts the mode based on the prediction error.
    Uses the MSE between the ground truth and the prediction to determine the logits.
    """

    mse_to_logits: MSEToLogitFunction = config_field(default_factory=InverseMSE)

    @jaxtyped(typechecker=typechecked)
    def logits(
        self,
        data: ErrorBasedSwitchingDynamicsData,
        **kwargs,
    ) -> Float[Tensor, "batch {self.num_modes}"]:
        # Compute MSE between predictions and targets
        mse = torch.mean((data.x_next.unsqueeze(1) - data.x_next_pred)**2,
                         dim=-1)

        return self.mse_to_logits(mse)
