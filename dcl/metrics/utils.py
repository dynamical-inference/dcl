from typing import Dict, Tuple

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from torch import Tensor
from typeguard import typechecked


def decision_tree_mapping(
    true: Integer[np.ndarray, " num_samples"],
    pred: Integer[np.ndarray, " num_samples"],
):
    num_modes = pred.max() + 1
    # create a one hot encoding of mode_sequence
    X = np.eye(num_modes)[pred]
    clf = DecisionTreeClassifier()

    clf.fit(X, true)
    dt_predictions = clf.predict(X)

    # there should always be a mapping from mode_sequence to prediction
    # let's find it
    mapping = {}
    for mode, pred in zip(pred, dt_predictions):
        mode = mode.item()
        pred = pred.item()
        if mode not in mapping:
            mapping[mode] = pred
        else:
            assert mapping[mode] == pred

    return dt_predictions, mapping


def one_to_many_assignment(
        true: Integer[np.ndarray, " num_samples"],
        pred: Integer[np.ndarray, " num_samples"]) -> Dict[int, int]:
    _, mapping = decision_tree_mapping(true, pred)
    return mapping


@jaxtyped(typechecker=typechecked)
def one_to_one_assignment(true: Integer[np.ndarray, " num_samples"],
                          pred: Integer[np.ndarray, " num_samples"],
                          num_classes: int) -> Dict[int, int]:

    # Construct the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        confusion_matrix[t, p] += 1

    # Convert the confusion matrix to a cost matrix
    cost_matrix = confusion_matrix.max() - confusion_matrix

    # indexes = Munkres().compute(cost_matrix)
    row, cols = linear_sum_assignment(cost_matrix)
    indexes = list(zip(row.tolist(), cols.tolist()))
    return {shuffled: original for original, shuffled in indexes}


@jaxtyped(typechecker=typechecked)
def apply_identifiability_constant(
    A: Float[Tensor, "n_features n_features"],
    x: Float[Tensor, "batch_size n_features"],
) -> Float[Tensor, "batch_size n_features"]:
    """
    Apply the identifiability constant to estimate y from x via y_t = A x_t.
    Args:
        A (torch.tensor): Linear transformation matrix of shape (num_features, num_features).
        x (torch.tensor): Input tensor of shape (num_samples, num_features).
    """
    y = A @ x.unsqueeze(-1)
    y = y.squeeze(-1)
    return y


@jaxtyped(typechecker=typechecked)
def linear_assignment(
    matrix_true: Float[Tensor, "num_modes num_latent num_latent"],
    matrix_hat: Float[Tensor, "num_modes num_latent num_latent"],
) -> Tuple[
        Float[Tensor, "num_modes num_latent num_latent"],
        Dict,
]:
    """Matches estimated matrices to true matrices based on minimum cost assignment.

    This function computes the optimal assignment between estimated and true matrices that
    minimizes the overall distance between them. It flattens the matrices and uses the
    Hungarian algorithm to find the minimum cost matching. The primary use case is in
    evaluating the performance of dynamics models by comparing their estimated transition
    matrices with the true transition matrices.

    Args:
        matrix_true: A tensor containing the true matrices.
            Shape should be `[num_modes, num_latent, num_latent]`, where `num_modes` is the
            number of modes/matrices, and `num_latent` is the dimensionality of each matrix.
        matrix_hat: A tensor containing the estimated matrices,
            with the same shape as `matrix_true`.

    Returns:
        A tuple containing:
            - the reordered estimated matrices to best match the true matrices
            - a dictionary mapping estimated matrix indices to true matrix indices.
    """
    matrix_true = matrix_true.detach().cpu()
    matrix_hat = matrix_hat.detach().cpu()
    num_modes = matrix_true.shape[0]
    matrix_hat_flat = matrix_hat.reshape(num_modes, -1)
    matrix_true_flat = matrix_true.reshape(num_modes, -1)

    row_ind, col_ind = linear_sum_assignment(
        np.array(torch.cdist(matrix_true_flat, matrix_hat_flat)))
    assert np.all(row_ind[:-1] <= row_ind[1:])

    # order mappings
    true2estimate_mapping = dict(zip(row_ind, col_ind))
    estimate2true_mapping = dict(zip(col_ind, row_ind))

    # reordering estimated centroids to match W_true order
    matrix_hat_ordered = torch.stack([
        matrix_hat[true2estimate_mapping[true_idx]]
        for true_idx in range(num_modes)
    ])

    return matrix_hat_ordered, estimate2true_mapping


def fit_linear_regression(x, y, bias=True):
    lr_model = LinearRegression(fit_intercept=bias)
    lr_model.fit(x, y)
    return lr_model
