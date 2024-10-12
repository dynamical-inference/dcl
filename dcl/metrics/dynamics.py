from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

from dcl.metrics import GlobalMetric
from dcl.metrics.utils import fit_linear_regression
from dcl.metrics.utils import linear_assignment
from dcl.metrics.utils import one_to_one_assignment
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.solver.contrastive_solver import DynamicsContrastiveLearningSolver
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import DynamicsSolverValidationPrediction
from dcl.utils.datatypes import GroundTruthData
from dcl.utils.datatypes import SLDSDynamicsSolverValidationPrediction


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class AccuracyViaHungarian(GlobalMetric):
    """Metric that computes accuracy taking into account possible permutations of labels."""

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        predictions: SLDSDynamicsSolverValidationPrediction,
        ground_truth: GroundTruthData,
        **kwargs,
    ) -> float:
        """Compute accuracy taking into account possible permutations of labels.
        """
        gt = ground_truth.clone()
        gt = gt[predictions.dynamics.x_index]
        num_classes = ground_truth.modes.max().item() + 1
        assert torch.all(gt.index == predictions.dynamics.x_index
                        ), "Index mismatch between ground truth and predictions"
        if gt.modes is None:
            raise ValueError("Ground truth modes are not provided")
        if predictions.dynamics.modes is None:
            raise ValueError("Predictions modes are not provided")
        return self._compute(
            true=gt.modes,
            pred=predictions.dynamics.modes,
            num_classes=num_classes,
        )

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        true: Integer[Tensor, " num_samples"],
        pred: Integer[Tensor, " num_samples"],
        num_classes: int,
        **kwargs,
    ) -> float:
        """Compute accuracy taking into account possible permutations of labels.
        """
        # Convert to numpy arrays
        true = true.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        label_mapping = one_to_one_assignment(true,
                                              pred,
                                              num_classes=num_classes)
        # Apply mapping to predictions
        pred = np.vectorize(label_mapping.get)(pred)

        # Calculate accuracy
        accuracy = (pred == true).mean().item()

        return accuracy


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class PredictiveMSE(GlobalMetric):
    """Metric that computes MSE for dynamics model predictions  ."""

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        predictions: DynamicsSolverValidationPrediction,
        **kwargs,
    ) -> float:
        """Compute MSE for dynamics model predictions."""

        emb_index = predictions.embeddings_index[predictions.dynamics.x_index]
        assert torch.all(
            emb_index == predictions.dynamics.x_index
        ), "Index mismatch between embeddings and dynamics predictions"
        embeddings = predictions.embeddings[emb_index]

        return self._compute(
            true=embeddings,
            pred=predictions.dynamics.x,
        )

    def _compute(
        self,
        true: Float[Tensor, "batch latent_dim"],
        pred: Float[Tensor, "batch latent_dim"],
        **kwargs,
    ) -> float:
        """Compute MSE for dynamics model predictions."""
        mse = (pred - true).pow(2).mean()
        return mse.item()


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class LDSError(GlobalMetric):
    """Metric that computes the error between true and estimated dynamics matrices."""

    inverse_type: Literal["explicit",
                          "implicit"] = config_field(default="explicit")
    bias: bool = config_field(default=True)

    @property
    def name(self) -> str:
        bias_str = "bias" if self.bias else "no_bias"
        inverse_type_str = "inv_exp" if self.inverse_type == "explicit" else "inv_imp"
        return f"LDSError_{bias_str}_{inverse_type_str}"

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        ground_truth: GroundTruthData,
        solver: DynamicsContrastiveLearningSolver,
        predictions: DynamicsSolverValidationPrediction,
        **kwargs,
    ) -> float:
        """Compute the error between true and estimated dynamics matrices."""

        assert torch.all(ground_truth.index == predictions.embeddings_index
                        ), "Ground truth and predictions indices do not match"

        return self._compute(
            dynamics_true=ground_truth.dynamics_model,
            dynamics_pred=solver.dynamics_model,
            z_true=ground_truth.latents.cpu(),
            z_pred=predictions.embeddings.cpu(),
        )

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        dynamics_true: Union[GumbelSLDS, LinearDynamicsModel],
        dynamics_pred: Union[GumbelSLDS, LinearDynamicsModel],
        z_true: Float[Tensor, "batch latent_dim"],
        z_pred: Float[Tensor, "batch latent_dim"],
        **kwargs,
    ) -> float:
        """Compute the error between true and estimated dynamics matrices."""

        lr_model = fit_linear_regression(z_true, z_pred, bias=self.bias)
        L = lr_model.coef_
        if self.inverse_type == "explicit":
            L_inv = np.linalg.inv(L)
        elif self.inverse_type == "implicit":
            lr_model_inv = fit_linear_regression(z_pred, z_true, bias=self.bias)
            L_inv = lr_model_inv.coef_
        else:
            raise ValueError(f"Invalid inverse type: {self.inverse_type}")

        return self.score(
            dynamics_true=dynamics_true,
            dynamics_pred=dynamics_pred,
            L=torch.from_numpy(L),
            L_inv=torch.from_numpy(L_inv),
        )

    @jaxtyped(typechecker=typechecked)
    def score(
        self,
        dynamics_true: Union[GumbelSLDS, LinearDynamicsModel],
        dynamics_pred: Union[GumbelSLDS, LinearDynamicsModel],
        L: Float[Tensor, "latent_dim latent_dim"],
        L_inv: Float[Tensor, "latent_dim latent_dim"],
        **kwargs,
    ) -> float:
        # Only supports SDLS models with linear dynamics that are purely linear (not affine),
        # i.e. they are of the form x_{t+1} = A x_t

        def get_lds(
            dynamics_model: Union[GumbelSLDS, LinearDynamicsModel]
        ) -> LinearDynamicsModel:
            if isinstance(dynamics_model, GumbelSLDS):
                return dynamics_model.linear_dynamics
            elif isinstance(dynamics_model, LinearDynamicsModel):
                return dynamics_model
            else:
                raise ValueError(
                    f"Unsupported dynamics model: {type(dynamics_model)}")

        lds_true = get_lds(dynamics_true)
        lds_pred = get_lds(dynamics_pred)

        # make sure the dynamics are purely linear
        assert lds_true.use_bias is False, "Only linear dynamics are supported"
        assert lds_pred.use_bias is False, "Only linear dynamics are supported"
        assert lds_true.num_systems == lds_pred.num_systems, "Number of systems must match for comparing dynamics parameters"

        return float(
            self._score(
                A_true=lds_true.A,
                A_pred=lds_pred.A,
                L=L,
                L_inv=L_inv,
            ).mean())

    @jaxtyped(typechecker=typechecked)
    def _score(
        self,
        L: Float[Tensor, "latent_dim latent_dim"],
        L_inv: Float[Tensor, "latent_dim latent_dim"],
        A_true: Float[Tensor, "num_modes latent_dim latent_dim"],
        A_pred: Float[Tensor, "num_modes latent_dim latent_dim"],
    ) -> Float[np.ndarray, " num_modes"]:

        A_pred = A_pred.cpu()
        A_true = A_true.cpu()
        L = L.cpu()
        L_inv = L_inv.cpu()

        A_pred_ordered, _ = linear_assignment(A_true, A_pred)

        # torch einsum for A_pred @ L
        self.WA = torch.einsum('...ij,...jk->...ik', A_pred_ordered, L)

        # torch einsum for L_inv @ (A_pred @ L)
        self.AWA = torch.einsum('...ij,...jk->...ik', L_inv, self.WA)

        error = torch.linalg.matrix_norm(A_true - self.AWA)
        return np.array(error.detach().cpu())
