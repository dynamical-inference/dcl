from dataclasses import dataclass
from typing import Any, Dict

import torch

from dcl.models.dynamics.slds import GumbelSLDS
from dcl.solver import BaseSolver
from dcl.solver.optimizer import Optimizer
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import state_field
from dcl.utils.temperature_scheduler import ConstantTemperatureScheduler
from dcl.utils.temperature_scheduler import TemperatureScheduler


@dataclass(kw_only=True)
class GumbelDynamicsSolver(BaseSolver):
    """Solver for training SLDS models using Gumbel-Softmax relaxation."""

    # Model
    model: GumbelSLDS = config_field(default_factory=GumbelSLDS)

    # Optimizer
    optimizer: Optimizer = config_field(default_factory=Optimizer)

    # Temperature scheduler
    temperature_scheduler: TemperatureScheduler = config_field(
        default_factory=ConstantTemperatureScheduler)

    # state
    current_temp: float = state_field(default=0.0, init=False)
    current_epoch: int = state_field(default=0, init=False)

    def __lazy_post_init__(self):
        """Initialize solver with optimizer and temperature."""
        super().__lazy_post_init__()
        self.optimizer.lazy_init(self.model.parameters())
        self.current_temp = self.temperature_scheduler.initial_temp
        self.model.tau = self.current_temp

    def reset(self):
        """Reset the solver."""
        super().reset()
        self.current_temp = self.temperature_scheduler.initial_temp
        self.model.tau = self.current_temp

    def _update_temperature(self, epoch: int):
        """Update Gumbel-Softmax temperature using scheduler."""
        self.current_temp = self.temperature_scheduler.get_temperature(
            epoch=epoch)
        self.model.tau = self.current_temp

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform single training step using Gumbel-Softmax relaxation.

        Args:
            batch: Dictionary containing:
                sample_1: Initial points 1
                sample_2: Initial points 2
                transformed_1: Transformed points 1
                transformed_2: Transformed points 2
                state_indices: True indicies of group actions (i.e. which transformation was applied) (optional)

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        transformed_1_pred, transformed_2_pred, selected_modes = self.model(
            batch["sample_1"],
            batch["sample_2"],
            batch["transformed_1"],
            batch["transformed_2"],
            system_idx=batch.get("state_indices", None))

        # Compute losses
        loss_1 = torch.mean(
            (transformed_1_pred - batch["transformed_1"].unsqueeze(1))**2)
        loss_2 = torch.mean(
            (transformed_2_pred - batch["transformed_2"].unsqueeze(1))**2)
        loss = loss_1 + loss_2

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "loss_1": loss_1.item(),
            "loss_2": loss_2.item(),
            "temperature": self.current_temp,
        }

    def start_epoch(self, epoch: int):
        """At start of epoch, update temperature."""
        self._update_temperature(epoch=epoch)

    def _validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform a single validation step.

        Args:
            batch: Dictionary containing the current batch of data

        Returns:
            Dictionary containing validation metrics
        """
        # Forward pass
        transformed_1_pred, transformed_2_pred, selected_modes = self.model(
            batch["sample_1"],
            batch["sample_2"],
            batch["transformed_1"],
            batch["transformed_2"],
            num_gumbel_samples=1,
        )
        metrics = {}

        # Compute losses
        metrics["loss_1"] = torch.mean(
            (transformed_1_pred -
             batch["transformed_1"].unsqueeze(1))**2).item()
        metrics["loss_2"] = torch.mean(
            (transformed_2_pred -
             batch["transformed_2"].unsqueeze(1))**2).item()
        metrics["loss"] = metrics["loss_1"] + metrics["loss_2"]

        return metrics
