from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from jaxtyping import jaxtyped
from typeguard import typechecked

from dcl.models.utils import ModelStorageMixin
from dcl.utils.configurable import check_initialized
from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class Optimizer(ModelStorageMixin, Configurable, ABC):
    """Configurable wrapper around torch.optim optimizers with lazy initialization."""

    # By default optimizers need lazy initialization
    lazy: bool = True
    # for lazy execution of _load_additional
    _optimizer_load_path: Optional[Path] = field(default=None,
                                                 init=False,
                                                 repr=False)
    _optimizer: Optional[torch.optim.Optimizer] = field(default=None,
                                                        init=False,
                                                        repr=False)

    def __lazy_post_init__(self, parameters):
        super().__lazy_post_init__()
        self._lazy_init_optimizer(parameters)
        if self._optimizer_load_path:
            ModelStorageMixin._load_additional(self, self._optimizer_load_path)

    @abstractmethod
    def _lazy_init_optimizer(self, parameters):
        pass

    @check_initialized
    def zero_grad(self):
        """Zero gradients of the optimizer."""
        self._optimizer.zero_grad()

    @check_initialized
    def step(self):
        """Perform optimization step."""
        self._optimizer.step()

    @check_initialized
    def state_dict(self):
        """Get optimizer state dict."""
        return self._optimizer.state_dict()

    @check_initialized
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self._optimizer.load_state_dict(state_dict)

    def _load_additional(self, path: Path):
        """
        Custom handling lazy initialization of optimizer.
        Instead of trying to load the optimizer directly (which would fail, since it is not initialized yet)
        we store the path to indicate the optimizer should load its state when lazy_init is called.
        """
        self._optimizer_load_path = path


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DCLAdamOptimizer(Optimizer):
    """Configurable wrapper around torch.optim.Adam for DCL."""

    # Adam parameters
    encoder_learning_rate: float = config_field(default=1e-3)
    dynamics_learning_rate: float = config_field(default=1e-2)
    beta1: float = config_field(default=0.9)
    beta2: float = config_field(default=0.999)
    eps: float = config_field(default=1e-8)
    weight_decay: float = config_field(default=0.0)

    def _lazy_init_optimizer(self, parameters: Dict[str, nn.Parameter]):
        # NOTE: we build the params list for the optimizer where we allow different
        # learning rates for the encoder and dynamics model.
        params = [{
            "params": parameters["encoder_model"],
            "lr": self.encoder_learning_rate
        }, {
            "params": parameters["dynamics_model"],
            "lr": self.dynamics_learning_rate
        }]
        self._optimizer = torch.optim.Adam(params,
                                           betas=(self.beta1, self.beta2),
                                           eps=self.eps,
                                           weight_decay=self.weight_decay)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DynamicsModelAdamOptimizer(Optimizer):
    """Configurable wrapper around torch.optim.Adam for dynamics model. It
    should be used when we want to train the dynamics model on its own."""

    # Adam parameters
    dynamics_learning_rate: float = config_field(default=1e-3)
    beta1: float = config_field(default=0.9)
    beta2: float = config_field(default=0.999)
    eps: float = config_field(default=1e-8)
    weight_decay: float = config_field(default=0.0)

    def _lazy_init_optimizer(self, parameters: List[nn.Parameter]):
        self._optimizer = torch.optim.Adam(parameters,
                                           lr=self.dynamics_learning_rate,
                                           betas=(self.beta1, self.beta2),
                                           eps=self.eps,
                                           weight_decay=self.weight_decay)
