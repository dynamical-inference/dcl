import importlib
import json
import pkgutil
import random
import traceback
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import jaxtyped
from tqdm.auto import tqdm
from typeguard import typechecked

from dcl.loader import BaseDataLoader
from dcl.metrics import Metric
from dcl.models import BaseModel
from dcl.utils.checkpoints import CheckpointSavingCallback
from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import Batch
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import DynamicsSolverValidationPrediction
from dcl.utils.datatypes import state_field


@dataclass(kw_only=True)
class BaseSolver(Configurable, ABC):
    """Base class for all solvers."""

    model: BaseModel = config_field(default_factory=BaseModel)
    # Training parameters
    num_epochs: int = config_field(default=1)
    seed: int = config_field(default=42)

    # State
    logs: Dict[str, List[float]] = state_field(
        default_factory=lambda: defaultdict(list),
        init=False,
    )
    current_step: int = state_field(
        default=0,
        init=False,
    )

    silence_metric_errors: bool = field(default=True)
    skip_metrics: List[str] = field(default_factory=list,
                                    init=False,
                                    repr=False)

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "solver"

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        """Initialize solver."""
        self.reset()

    def reset(self):
        """Reset the solver."""
        self.logs = defaultdict(list)
        self.current_step = 0

    @property
    def encoder_offset(self) -> int:
        return self.model.time_offset()

    @property
    def state(self) -> Dict[str, Any]:
        """Get the current state of the solver."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("type") == "state"
        }

    def _save_state(self, path: Path) -> None:
        """Save the current state of the solver."""
        state_path = path / "solver_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, 'w') as f:
            json.dump(self.state, f, indent=4)

    def _load_state(self, path: Path) -> bool:
        """Load the current state of the solver."""
        state_path = path / "solver_state.json"
        if not state_path.exists():
            return False
        with open(state_path, 'r') as f:
            state = json.load(f)
            for f in fields(self):
                if f.name in state:
                    setattr(self, f.name, state[f.name])

    def _save_additional(self, path: Path) -> None:
        """Save additional state to disk."""
        self._save_state(path)

    def _load_additional(self, path: Path) -> bool:
        """Load additional state from disk."""
        return self._load_state(path)

    @abstractmethod
    def train_step(self, batch: Batch) -> Dict[str, Any]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing the current batch of data

        Returns:
            Dictionary containing training metrics
        """

    @torch.no_grad()
    def validate(
        self,
        loader: BaseDataLoader,
        eval_batches: int = 1,
        metrics: List[Metric] = [],
    ) -> Dict[str, Any]:
        """Validate the model.

        Args:
            loader: data loader
            eval_batches: number of batches to evaluate loss metricson
        Returns:
            Dictionary containing metrics
        """

        self.to(loader.device)
        self.set_eval()

        loss_metrics = defaultdict(list)
        for i, batch in enumerate(loader):
            if i >= eval_batches:
                break
            losses = self.validate_step(batch=batch)
            for k, v in losses.items():
                loss_metrics[k].append(v)

        loss_metrics = {
            k: torch.mean(torch.tensor(v)).item()
            for k, v in loss_metrics.items()
        }

        global_metrics = self.compute_metrics(loader=loader, metrics=metrics)
        return {**global_metrics, **loss_metrics}

    @abstractmethod
    def predictions(
        self,
        loader: BaseDataLoader,
    ) -> DynamicsSolverValidationPrediction:
        pass

    @jaxtyped(typechecker=typechecked)
    def compute_metrics(
        self,
        loader: BaseDataLoader,
        metrics: List[Metric] = [],
    ) -> Dict[str, Any]:
        """Compute global metrics."""

        predictions = self.predictions(loader)
        ground_truth = loader.ground_truth_data
        metric_results = {}
        for m in metrics:
            try:
                if m.name in self.skip_metrics:
                    continue
                metric_results[m.name] = m.compute(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    solver=self,
                    loader=loader,
                )
            except Exception as e:
                # skiping metrics so we don't print the same error over and over during training
                self.skip_metrics.append(m.name)
                if self.silence_metric_errors:
                    print(f"Error computing metric {m.name}:")
                    print(e)
                    traceback.print_exc()
                else:
                    raise e
        return metric_results

    @abstractmethod
    def validate_step(self, batch: Batch,
                      gt_batch: Batch) -> Tuple[Batch, Dict[str, Any]]:
        """Does a single prediction step, computes metrics for the batch and returns both the predictions and the metrics"""

    def evaluate(
        self,
        loader: BaseDataLoader,
        metrics: List[Metric] = [],
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate the model after training is complete.
        This may be used to compute additional metrics, besides the one computed by validate.

        Args:
            loader: data loader

        Returns:
            Dictionary containing metrics
        """
        # by default, just call validate
        return self.validate(loader, metrics=metrics, **kwargs)

    def filter_tqdm_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out keys that are not meant to be displayed in the progress bar."""
        return stats

    @property
    def priority_keys(self) -> List[str]:
        """Prioritiy keys for tqdm progress bar"""
        return [
            'epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
        ]

    def update_tqdm_stats(self, old_postfix: Dict[str, Any],
                          **kwargs) -> Dict[str, Any]:
        """Update progress bar statistics by combining old and new data.

        Args:
            old_postfix: Previous postfix dictionary
            **kwargs: Keyword arguments to update postfix with. Special keys:
                - epoch: Current epoch number
                Other keys will be formatted with default formatting

        Returns:
            Updated postfix dictionary with formatted strings
        """
        postfix = old_postfix.copy()

        for k, v in kwargs.items():
            try:
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.item()
            except Exception as e:
                print(f"Error converting tensor for key {k}: {e}")

        # Handle epoch/num_epochs if present
        epoch = kwargs.pop('epoch', None)
        if epoch is not None:
            postfix['epoch'] = f"{epoch}/{self.num_epochs}"

        loss_keys = [k for k in kwargs.keys() if "loss" in k]
        for loss_key in loss_keys:
            loss_value = kwargs.pop(loss_key)
            postfix[loss_key] = f"{loss_value:.4e}"

        # Handle remaining kwargs with default formatting
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                postfix[k] = f"{v:.4f}"
            else:
                postfix[k] = str(v)

        # Create ordered dict with priority keys first
        ordered_postfix = OrderedDict()
        for key in self.priority_keys:
            if key in postfix:
                ordered_postfix[key] = postfix[key]

        # Add remaining keys
        for key in postfix:
            if key not in self.priority_keys:
                ordered_postfix[key] = postfix[key]

        return ordered_postfix

    @jaxtyped(typechecker=typechecked)
    def fit(self,
            train_loader: BaseDataLoader,
            val_loader: Optional[BaseDataLoader] = None,
            save_hook: Optional[CheckpointSavingCallback] = None,
            eval_frequency: Optional[int] = None,
            eval_kwargs: Optional[Dict[str, Any]] = None):
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            save_hook: Optional checkpoint saving callback
            eval_frequency: Optional evaluation frequency
            eval_batches: Optional number of batches to evaluate loss metrics on
        """
        eval_kwargs = eval_kwargs or {}
        print(f"Training on device: {train_loader.device}")
        self.to(train_loader.device)
        if val_loader is not None:
            val_loader.to(train_loader.device)

        print(f"Training with random seed: {self.seed}")
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Calculate total iterations
        total_iters = self.num_epochs * len(train_loader)

        # Single progress bar for all iterations
        progress_bar = tqdm(
            initial=self.current_step,
            total=total_iters,
            desc="Training",
            unit="iter",
            # min and max control how often the pbar is updated (in seconds)
            miniters=50,
            maxinterval=float('inf'),
        )
        postfix = {}

        val_metrics = None
        done_training = False
        for epoch in range(self.num_epochs):
            if done_training:
                break
            self.start_epoch(epoch)
            for batch in train_loader:
                self.set_train()
                done_training = self.current_step >= total_iters
                if done_training:
                    break
                # Training step
                metrics = self.train_step(batch)

                self._log_metrics(metrics, prefix="train_")
                # Logging
                postfix = self.update_tqdm_stats(postfix, **metrics)

                # Validation
                if eval_frequency is not None and val_loader is not None and self.current_step % eval_frequency == 0:
                    val_metrics = self.validate(val_loader, **eval_kwargs)
                    val_metrics["step"] = self.current_step
                    self._log_metrics(val_metrics, prefix="val_")
                    # Update progress bar with validation metrics
                    postfix = self.update_tqdm_stats(postfix, **val_metrics)

                progress_bar.update(1)
                progress_bar.set_postfix(**postfix, refresh=False)
                self.current_step += 1

                if save_hook is not None:
                    save_hook.maybe_save(
                        solver=self,
                        step=self.current_step,
                    )

            self.end_epoch(epoch)

        # final validation after training end
        if val_loader is not None and eval_frequency is not None:
            self._log_metrics(val_metrics, prefix="val_")
            postfix = self.update_tqdm_stats(postfix, **val_metrics)
        postfix = self.update_tqdm_stats(postfix, epoch=epoch + 1)
        if save_hook is not None:
            save_hook(
                solver=self,
                ckpt_dir_name="last_checkpoint",
            )
        progress_bar.close()

    def start_epoch(self, epoch: int):
        """Hook for start of epoch."""

    def end_epoch(self, epoch: int):
        """Hook for end of epoch."""

    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log metrics to the logs dictionary.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            log_name = f"{prefix}{name}"
            self.logs[log_name].append(value)

    def to(self, device: torch.device):
        """Move the model to a specific device."""
        self.model.to(device)
        return self

    def set_eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        return self

    def set_train(self):
        """Set the model to training mode."""
        self.model.train()


# Dynamically import all submodules in this package
for _, module_name, _ in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)

# Optionally define __all__
__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]
