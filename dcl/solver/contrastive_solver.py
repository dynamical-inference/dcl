from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

from dcl.criterions.contrastive import ContrastiveCriterion
from dcl.criterions.contrastive import MseInfoNCE
from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.models.dynamics import BaseDynamicsModel
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.models.dynamics.utils import freeze_model
from dcl.models.encoder import EncoderModel
from dcl.models.encoder import MLP
from dcl.solver import BaseSolver
from dcl.solver.optimizer import DCLAdamOptimizer
from dcl.solver.optimizer import Optimizer
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import DynamicsPrediction
from dcl.utils.datatypes import DynamicsSolverValidationPrediction
from dcl.utils.datatypes import GumbelSLDSInput
from dcl.utils.datatypes import GumbelSLDSPrediction
from dcl.utils.datatypes import SLDSDynamicsSolverValidationPrediction
from dcl.utils.datatypes import state_field
from dcl.utils.datatypes import TimeContrastiveBatch
from dcl.utils.datatypes import TimeContrastiveLatentBatch
from dcl.utils.datatypes import TimeContrastiveLossBatch
from dcl.utils.temperature_scheduler import ConstantTemperatureScheduler
from dcl.utils.temperature_scheduler import TemperatureScheduler


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DynamicsContrastiveLearningSolver(BaseSolver):
    """Solver for training encoder model jointly with slds model using contrastive learning."""

    # Models
    # encoder model
    model: EncoderModel = config_field(default_factory=MLP)
    dynamics_model: BaseDynamicsModel = config_field(
        default_factory=LinearDynamicsModel)

    # Optimizer
    optimizer: Optimizer = config_field(default_factory=DCLAdamOptimizer)

    # Criterion
    criterion: ContrastiveCriterion = config_field(default_factory=MseInfoNCE)

    # state
    current_epoch: int = state_field(default=0, init=False)

    # Whether to freeze the dynamics model
    freeze_dynamics_model: bool = config_field(default=False)

    @property
    def dynamics_model_offset(self) -> int:
        return self.dynamics_model.time_offset

    def __lazy_post_init__(self):
        """Initialize solver with optimizer and temperature."""
        super().__lazy_post_init__()
        self.optimizer.lazy_init(
            parameters=dict(encoder_model=self.model.parameters(),
                            dynamics_model=self.dynamics_model.parameters()))
        self.reset()

        if self.freeze_dynamics_model:
            print("Freezing dynamics model for training.")
            freeze_model(self.dynamics_model)

    def validate_config(self):
        super().validate_config()
        if isinstance(self.dynamics_model, GumbelSLDS) and not isinstance(
                self, SLDSContrastiveLearningSolver):
            raise ValueError(
                "SLDSContrastiveLearningSolver is required for GumbelSLDS dynamics model"
            )

    @jaxtyped(typechecker=typechecked)
    def fit(self,
            train_loader: DiscreteTimeContrastiveDataLoader,
            val_loader: Optional[DiscreteTimeContrastiveDataLoader] = None,
            **kwargs):
        train_loader = self.sync_dynamics_offset(train_loader)
        if val_loader is not None:
            val_loader = self.sync_dynamics_offset(val_loader)

        super().fit(
            train_loader,
            val_loader=val_loader,
            **kwargs,
        )

    @jaxtyped(typechecker=typechecked)
    def train_step(self, batch: TimeContrastiveBatch) -> Dict[str, Any]:
        """Perform single training step using contrastive learning.

        Args:
            batch: TimeContrastiveBatch

        Returns:
            Dictionary containing training metrics
        """
        self.set_train()
        self.optimizer.zero_grad()

        embedding_batch = self.predict_step(batch)

        loss, loss_metrics = self.compute_loss(embedding_batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return dict(**loss_metrics, loss=loss.item())

    @jaxtyped(typechecker=typechecked)
    def predict_step(self,
                     batch: TimeContrastiveBatch) -> TimeContrastiveLossBatch:
        """Takes an input batch and returns a batch of embeddings with predictions."""
        embedding_batch = self.encode_batch(batch)
        dynamics_prediction = self.dynamics_step(embedding_batch)
        return self.include_dynamics_prediction(embedding_batch,
                                                dynamics_prediction)

    @jaxtyped(typechecker=typechecked)
    def encode_batch(self,
                     batch: TimeContrastiveBatch) -> TimeContrastiveLatentBatch:
        """Encode the batch into embeddings."""
        # TODO: To improve performance, we should instead concatenate all
        # of these tensors and then pass them through the model in one forward pass

        # reference has an additional dynamics_offset dimension, we need to flatten
        batch_size = batch.batch
        dynamics_offset = batch.dynamics_offset
        batch.reference = self.model(
            batch.reference.flatten(
                start_dim=0,
                end_dim=1,
            )).reshape(
                batch_size,
                dynamics_offset,
                -1,
            )
        batch.positive = self.model(batch.positive)
        batch.negative = self.model(batch.negative)

        # update indices, drop encoder_offset dimension
        batch.indices = batch.indices[..., -1]

        return TimeContrastiveLatentBatch.from_batch(batch)

    def embeddings(
        self, batch: Float[Tensor, "batch encoder_offset obs_dim"]
    ) -> Float[Tensor, "batch latent_dim"]:
        """Encode the batch into embeddings."""
        return self.model(batch)

    @jaxtyped(typechecker=typechecked)
    def dynamics_step(self,
                      batch: TimeContrastiveLatentBatch) -> DynamicsPrediction:
        """Perform a single dynamics step."""
        dynamics_input = self.dynamics_model.input_type(
            x=batch.reference,
            x_index=batch.reference_indices,
        )
        dynamics_prediction = self.dynamics_model(dynamics_input)
        assert torch.all(
            dynamics_prediction.x_index == batch.positive_indices
        ), "Dynamics model did not predict for the correct time steps"

        return dynamics_prediction

    @jaxtyped(typechecker=typechecked)
    def include_dynamics_prediction(
        self,
        batch: TimeContrastiveLatentBatch,
        dynamics_prediction: DynamicsPrediction,
    ) -> TimeContrastiveLossBatch:
        batch.reference = dynamics_prediction.x
        batch.reference_indices = batch.reference_indices[..., -1]
        return TimeContrastiveLossBatch.from_batch(batch)

    @jaxtyped(typechecker=typechecked)
    def compute_loss(
        self, batch: TimeContrastiveLossBatch
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        total, align, uniformity = self.criterion(batch)

        return total, dict(
            loss_align=align.item(),
            loss_uniformity=uniformity.item(),
            loss_total=total.item(),
        )

    @jaxtyped(typechecker=typechecked)
    def filter_tqdm_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out keys that are not meant to be displayed in the progress bar."""
        key_to_keep = [
            "loss",
            "loss_align",
            "loss_uniformity",
        ]
        return {k: v for k, v in stats.items() if k not in key_to_keep}

    @jaxtyped(typechecker=typechecked)
    def validate_step(
        self,
        batch: TimeContrastiveBatch,
    ) -> Dict[str, Any]:
        """Does a single prediction step, computes metrics for the batch and returns both the predictions and the metrics"""

        self.set_eval()
        embedding_batch = self.predict_step(batch)
        loss, loss_metrics = self.compute_loss(embedding_batch)

        return dict(**loss_metrics, loss=loss.item())

    def sync_dynamics_offset(
        self, loader: DiscreteTimeContrastiveDataLoader
    ) -> DiscreteTimeContrastiveDataLoader:
        if loader.dynamics_offset != self.dynamics_model_offset:
            loader.dynamics_offset = self.dynamics_model_offset
        return loader

    @jaxtyped(typechecker=typechecked)
    def predictions(
        self,
        loader: DiscreteTimeContrastiveDataLoader,
    ) -> DynamicsSolverValidationPrediction:
        """Perform predictions for metrics."""
        loader = self.sync_dynamics_offset(loader)
        val_data = loader.validation_data
        embeddings = self.embeddings(val_data.observed[val_data.encoder_index])
        dynamics_input = self.dynamics_model.input_type(
            x=embeddings[val_data.reference_index],
            x_index=val_data.reference_index,
        )
        dynamics_predictions = self.dynamics_model(dynamics_input)
        predictions = DynamicsSolverValidationPrediction(
            embeddings=embeddings,
            embeddings_index=val_data.encoder_index[..., -1],
            dynamics=dynamics_predictions,
        )
        return predictions

    def to(self, device: torch.device):
        """Move the model to a specific device."""
        super().to(device)
        self.dynamics_model.to(device)
        return self

    def set_eval(self):
        """Set the model to evaluation mode."""
        super().set_eval()
        self.dynamics_model.eval()
        return self

    def set_train(self):
        """Set the model to training mode."""
        super().set_train()
        self.dynamics_model.train()
        return self


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class SLDSContrastiveLearningSolver(DynamicsContrastiveLearningSolver):
    """Solver for SLDS dynamics model"""

    dynamics_model: GumbelSLDS = config_field(default_factory=GumbelSLDS)

    # Temperature scheduler
    temperature_scheduler: TemperatureScheduler = config_field(
        default_factory=ConstantTemperatureScheduler)

    # state
    current_temp: float = state_field(default=0.0, init=False)

    def __lazy_post_init__(self):
        super().__lazy_post_init__()

    @jaxtyped(typechecker=typechecked)
    def train_step(self, batch: TimeContrastiveBatch) -> Dict[str, Any]:
        metrics = super().train_step(batch)
        metrics["temperature"] = self.current_temp
        return metrics

    @jaxtyped(typechecker=typechecked)
    def dynamics_step(
            self, batch: TimeContrastiveLatentBatch) -> GumbelSLDSPrediction:
        """Perform a single dynamics step."""
        dynamics_input = GumbelSLDSInput(
            x=batch.reference,
            x_index=batch.reference_indices,
            x_next=batch.positive,
        )
        dynamics_prediction = self.dynamics_model(dynamics_input)
        assert torch.all(
            dynamics_prediction.x_index == batch.positive_indices.unsqueeze(
                1)), "Dynamics model did not predict for the correct time steps"

        return dynamics_prediction

    @jaxtyped(typechecker=typechecked)
    def include_dynamics_prediction(
        self,
        batch: TimeContrastiveLatentBatch,
        dynamics_prediction: GumbelSLDSPrediction,
    ) -> TimeContrastiveLossBatch:
        # With gumbel samples, we predict multiple samples for each time step
        # We need to a) flatten the batch and gumbel samples dimensions for reference
        # and b) repeat the positive and negative samples to match the new reference
        batch = batch.repeat_interleave(
            repeats=dynamics_prediction.gumbel_samples,
            dim=0,
        )

        dynamics_prediction.x = dynamics_prediction.x.flatten(
            start_dim=0,
            end_dim=1,
        )

        return super().include_dynamics_prediction(batch, dynamics_prediction)

    @jaxtyped(typechecker=typechecked)
    def predictions(
        self,
        loader: DiscreteTimeContrastiveDataLoader,
    ) -> DynamicsSolverValidationPrediction:
        """Perform predictions for metrics."""
        loader = self.sync_dynamics_offset(loader)
        self.to(loader.device)
        val_data = loader.validation_data
        embeddings = self.embeddings(val_data.observed[val_data.encoder_index])
        dynamics_input = GumbelSLDSInput(
            x=embeddings[val_data.reference_index],
            x_index=val_data.reference_index,
            x_next=embeddings[val_data.positive_index],
        )
        dynamics_predictions = self.dynamics_model(
            dynamics_input).to_SLDSPrediction()
        predictions = SLDSDynamicsSolverValidationPrediction(
            embeddings=embeddings,
            embeddings_index=val_data.encoder_index[..., -1],
            dynamics=dynamics_predictions,
        )
        return predictions

    def reset(self):
        """Reset the solver."""
        super().reset()
        self.current_temp = self.temperature_scheduler.initial_temp
        self.dynamics_model.tau = self.current_temp

    def _update_temperature(self, epoch: int):
        """Update Gumbel-Softmax temperature using scheduler."""
        self.current_temp = self.temperature_scheduler.get_temperature(
            epoch=epoch)
        self.dynamics_model.tau = self.current_temp

    @jaxtyped(typechecker=typechecked)
    def start_epoch(self, epoch: int):
        """At start of epoch, update temperature."""
        self._update_temperature(epoch=epoch)
