from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import scipy as sp
import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from scipy.optimize import linear_sum_assignment
from sklearn.cross_decomposition import CCA as cca_skl
from sklearn.metrics import r2_score
from torch import Tensor
from typeguard import typechecked

from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.metrics import GlobalMetric
from dcl.metrics.utils import apply_identifiability_constant
from dcl.metrics.utils import fit_linear_regression
from dcl.models.dynamics import BaseDynamicsModel
from dcl.solver.contrastive_solver import DynamicsContrastiveLearningSolver
from dcl.utils.datatypes import AuxilaryVariables
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import DynamicsData
from dcl.utils.datatypes import DynamicsSolverValidationPrediction
from dcl.utils.datatypes import GroundTruthData
from dcl.utils.datatypes import GumbelSLDSPrediction


class IdentifiabilityMetric(GlobalMetric, ABC):

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        predictions: DynamicsSolverValidationPrediction,
        ground_truth: GroundTruthData,
        **kwargs,
    ) -> float:
        assert torch.all(ground_truth.index == predictions.embeddings_index
                        ), "Ground truth and predictions indices do not match"
        return self._compute(z_true=ground_truth.latents.cpu(),
                             z_pred=predictions.embeddings.cpu())

    @jaxtyped(typechecker=typechecked)
    @abstractmethod
    def _compute(
        self,
        z_true: Float[Tensor, "batch latents_dim"],
        z_pred: Float[Tensor, "batch latents_dim"],
    ) -> float:
        pass


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class MCC(IdentifiabilityMetric):
    corr_method: Literal["Pearson",
                         "Spearman"] = config_field(default="Pearson")

    @property
    def name(self):
        corr_method_name = f"_{self.corr_method.capitalize()}" if self.corr_method == "Spearman" else ""
        return f"{self.__class__.__name__}{corr_method_name}"

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        z_true: Float[Tensor, "batch latents_dim"],
        z_pred: Float[Tensor, "batch latents_dim"],
    ) -> float:
        x = z_true.numpy().copy().T
        y = z_pred.numpy().copy().T
        dim = x.shape[0]

        if self.corr_method == "Pearson":
            corr = np.corrcoef(y, x)
            corr = corr[0:dim, dim:]
        elif self.corr_method == "Spearman":
            corr, _ = sp.stats.spearmanr(y.T, x.T)
            corr = corr[0:dim, dim:]
        else:
            raise ValueError(f"Invalid correlation method: {self.corr_method}")

        row, cols = linear_sum_assignment(-np.absolute(corr))
        indexes = list(zip(row.tolist(), cols.tolist()))

        sort_idx = np.zeros(dim, dtype=int)
        for i in range(dim):
            sort_idx[i] = indexes[i][1]

        x_sort = x[sort_idx]
        if self.corr_method == "Pearson":
            corr_sort = np.corrcoef(y, x_sort)
            corr_sort = corr_sort[0:dim, dim:]
        elif self.corr_method == "Spearman":
            corr_sort, _ = sp.stats.spearmanr(y.T, x_sort.T)
            corr_sort = corr_sort[0:dim, dim:]

        mcc = np.mean(np.abs(np.diag(corr_sort)))
        return float(mcc)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class CCA(IdentifiabilityMetric):
    n_components: Optional[int] = config_field(default=None)

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        z_true: Float[Tensor, "batch latents_dim"],
        z_pred: Float[Tensor, "batch latents_dim"],
    ) -> float:
        n_components = self.n_components or z_true.shape[-1]
        cca = cca_skl(n_components=n_components)
        cca.fit(z_true, z_pred)

        z_true_c, z_pred_c = cca.transform(z_true, z_pred)
        correlation_matrix = np.corrcoef(z_true_c.T, z_pred_c.T)[:n_components,
                                                                 n_components:]
        mean_corr = correlation_matrix.diagonal().mean()
        return float(mean_corr)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class R2(IdentifiabilityMetric):
    bias: bool = config_field(default=True)
    use_inverse: bool = config_field(default=False)
    direction: Literal["forward", "backward"] = config_field(default="backward")

    def validate_config(self):
        super().validate_config()
        if self.bias and self.use_inverse:
            raise ValueError(
                "bias=True and use_inverse=True is not supported for R2 metric")

    @property
    def name(self):
        bias_name = "_bias" if self.bias else ""
        direction_name = f"_{self.direction.capitalize()}"
        use_inverse_name = "_ViaInverse" if self.use_inverse else ""
        return f"{self.__class__.__name__}{direction_name}{bias_name}{use_inverse_name}"

    @jaxtyped(typechecker=typechecked)
    def _directional_params(
        self,
        z_true: Float[Tensor, "batch latents_dim"],
        z_pred: Float[Tensor, "batch latents_dim"],
    ) -> Union[
            Tuple[Float[Tensor, "batch latents_dim"], Float[
                Tensor, "batch latents_dim"]],
            Tuple[Float[Tensor, "batch latents_dim"], Float[
                Tensor, "batch latents_dim"]],
    ]:

        if self.direction == "forward":
            return z_true, z_pred
        elif self.direction == "backward":
            return z_pred, z_true
        else:
            raise ValueError(f"Invalid direction: {self.direction}")

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        z_true: Float[Tensor, "batch latents_dim"],
        z_pred: Float[Tensor, "batch latents_dim"],
    ) -> float:

        x, y = self._directional_params(z_true, z_pred)
        if self.use_inverse:

            lr_model = fit_linear_regression(y, x, self.bias)
            y_pred = apply_identifiability_constant(A=np.linalg.inv(
                lr_model.coef_),
                                                    x=x)
            r2 = r2_score(y, y_pred)
        else:
            lr_model = fit_linear_regression(x, y, self.bias)
            r2 = lr_model.score(x, y)

        return float(r2)


@jaxtyped(typechecker=typechecked)
def predict_with_dynamics(
    latents: Float[Tensor, "time latents_dim"],
    dynamics_model: BaseDynamicsModel,
    loader: DiscreteTimeContrastiveDataLoader,
    auxilary: AuxilaryVariables,
    # num_steps: int,
) -> Tuple[
        Float[Tensor, "batch latents_dim"],
        Integer[Tensor, " batch"],
]:
    # setup the loader
    loader.dynamics_offset = dynamics_model.time_offset
    # TODO: to support num_steps > 1 we need to tell the loader somehow how many steps to predict

    # use the loader to get the dynamics index
    loader_data = loader.validation_data

    latents = latents.to(loader.device)
    dynamics_model_input = DynamicsData(
        x=latents[loader_data.reference_index],
        x_index=loader_data.reference_index,
    )

    # in case we have an SLDS model at hand, also add x_next
    dynamics_model_input.x_next = latents[loader_data.positive_index]

    # cast to correct model input type
    dynamics_model_input = dynamics_model.input_type.from_batch(
        dynamics_model_input)

    # predict
    dynamics_model.eval()
    dynamics_model.to(loader.device)
    dynamics_prediction = dynamics_model(dynamics_model_input)
    # handle GumbelSLDSPrediction which contains gumbel_samples dimensions
    if isinstance(dynamics_prediction, GumbelSLDSPrediction):
        dynamics_prediction = dynamics_prediction.to_SLDSPrediction()

    # make sure we didn't predict across trial boundaries
    if auxilary.trial_id is not None:
        trial_ids_input = auxilary.trial_id[loader_data.reference_index]
        trial_ids_prediction = auxilary.trial_id[dynamics_prediction.x_index]
        assert torch.all(trial_ids_input == trial_ids_prediction.unsqueeze(
            -1)), "Dynamics model predicted across trial boundaries"

    # return the predicted latents and the index
    return dynamics_prediction.x, dynamics_prediction.x_index


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DynamicsR2(GlobalMetric):

    bias: bool = config_field(default=True)
    direction: Literal["forward", "backward"] = config_field(default="backward")
    n_steps: int = config_field(default=1)

    def validate_config(self):
        super().validate_config()
        if self.n_steps > 1:
            raise ValueError(
                f"n_steps > 1 is not supported yet, got n_steps={self.n_steps}")

    @property
    def name(self):
        bias_name = "_bias" if self.bias else ""
        direction_name = f"_{self.direction.capitalize()}"
        n_steps_name = f"_nsteps{self.n_steps}" if self.n_steps > 1 else ""
        return f"{self.__class__.__name__}{direction_name}{bias_name}{n_steps_name}"

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        predictions: DynamicsSolverValidationPrediction,
        ground_truth: GroundTruthData,
        loader: DiscreteTimeContrastiveDataLoader,
        solver: DynamicsContrastiveLearningSolver,
        **kwargs,
    ) -> float:
        assert torch.all(ground_truth.index == predictions.embeddings_index
                        ), "Ground truth and predictions indices do not match"
        return self._compute(
            z_true=ground_truth.latents,
            z_pred=predictions.embeddings,
            dynamics_true=ground_truth.dynamics_model.to_gt_dynamics(),
            dynamics_pred=solver.dynamics_model,
            loader=loader,
            auxilary=ground_truth.auxilary)

    @jaxtyped(typechecker=typechecked)
    def _directional_params(
        self,
        z_true: Float[Tensor, "batch_time latents_dim"],
        z_pred: Float[Tensor, "batch_time emb_dim"],
        dynamics_true: BaseDynamicsModel,
        dynamics_pred: BaseDynamicsModel,
    ) -> Tuple[
            Float[Tensor, "batch_time latents_dim"],
            BaseDynamicsModel,
            Float[Tensor, "batch_time latents_dim"],
            BaseDynamicsModel,
    ]:
        if self.direction == "forward":
            # target == True space
            # source == Predicted space
            z_target = z_true
            dynamics_target = dynamics_true
            z_source = z_pred
            dynamics_source = dynamics_pred
        elif self.direction == "backward":
            # target == Predicted space
            # source == True space
            z_target = z_pred
            dynamics_target = dynamics_pred

            z_source = z_true
            dynamics_source = dynamics_true
        else:
            raise ValueError(f"Invalid direction: {self.direction}")

        return z_target, dynamics_target, z_source, dynamics_source

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        z_true: Float[Tensor, "batch_time latents_dim"],
        z_pred: Float[Tensor, "batch_time emb_dim"],
        dynamics_true: BaseDynamicsModel,
        dynamics_pred: BaseDynamicsModel,
        loader: DiscreteTimeContrastiveDataLoader,
        auxilary: AuxilaryVariables,
    ) -> float:
        """
        Computes an R² score on the transferability of dynamics between true and predicted space.
        In the paper this is defined as:
        dynR2(f, f_pred) = r2_score(f_pred(x_pred), L_1 f(L_2 x_pred + b_2) + b_1)
        where L_1, L_2, b_1, b_2 are the parameters of the linear regression model.

        For our code base this means we need to compute
        1. prep:
            a) lr_model_1.fit(z_true, z_pred)
            b) lr_model_2.fit(z_pred, z_true)
        2. LHS: dynamics_pred.forward(z_pred)
        3. RHS: lr_model_1.predict(dynamics_true.forward(lr_model_2.predict(z_pred)))
        4. r2_score(LHS, RHS)

        The direction parameter controls which space is considered source vs target:
        - "forward": true latents = target, predicted embeddings = source
        - "backward": predicted embeddings = target, true latents = source

        A high R² score indicates that applying the dynamics in both spaces still results in embeddings
        that are linearly related (i.e. the encoder & dynamics are identifiable)

        Args:
            z_true: Ground truth latent trajectories [batch_time, latents_dim]
            z_pred: Predicted embedding trajectories [batch_time, emb_dim]
            dynamics_true: Dynamics model in true latent space
            dynamics_pred: Dynamics model in predicted embedding space

        Returns:
            dynR² score between target dynamics and source dynamics
        """

        z_target, dynamics_target, z_source, dynamics_source = self._directional_params(
            z_true=z_true,
            z_pred=z_pred,
            dynamics_true=dynamics_true,
            dynamics_pred=dynamics_pred,
        )

        # In terms of target / source naming:
        # LHS: dynamics_target.forward(z_target)
        # RHS: lr_model_to_target.predict(dynamics_source.forward(lr_model_to_source.predict(z_target)))

        # 1. first we fit lr_model
        lr_model_to_target = fit_linear_regression(x=z_source.detach().cpu(),
                                                   y=z_target.detach().cpu(),
                                                   bias=self.bias)
        lr_model_to_source = fit_linear_regression(x=z_target.detach().cpu(),
                                                   y=z_source.detach().cpu(),
                                                   bias=self.bias)

        # 2. compute lhs
        # 2a) predict with dynamics
        z_target_dynamics_pred, z_target_dynamics_pred_index = predict_with_dynamics(
            latents=z_target,
            dynamics_model=dynamics_target,
            loader=loader,
            auxilary=auxilary,
        )

        # 3. compute rhs
        # 3a) we don't use z_source but instead approx_z_source = lr_model.predict(z_target)
        approx_z_source = torch.from_numpy(
            lr_model_to_source.predict(z_target.detach().cpu().numpy()))
        # 3b) predict with dynamics
        z_source_dynamics_pred, z_source_dynamics_pred_index = predict_with_dynamics(
            latents=approx_z_source,
            dynamics_model=dynamics_source,
            loader=loader,
            auxilary=auxilary,
        )

        # 3c) finally transform back from source space to target space
        approx_z_target_dynamics_pred = lr_model_to_target.predict(
            z_source_dynamics_pred.detach().cpu())

        # 4. compute r2 score
        r2 = r2_score(z_target_dynamics_pred.detach().cpu(),
                      approx_z_target_dynamics_pred)

        # check that the indices of the predictions match
        assert torch.all(z_target_dynamics_pred_index.detach().cpu(
        ) == z_source_dynamics_pred_index.detach().cpu(
        )), "Indices of the dynamics model predictions do not match"
        print('dynR2', r2, flush=True)
        return r2
