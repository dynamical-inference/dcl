from typing import Tuple

import torch
from dcl.criterions.contrastive import MseInfoNCE
from dcl.utils.datatypes import TimeContrastiveLossBatch
from torch import nn
from torch import Tensor


@torch.jit.script
def ref_dot_similarity(ref: Tensor, pos: Tensor,
                       neg: Tensor) -> Tuple[Tensor, Tensor]:
    """Cosine similarity the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    pos_dist = torch.einsum("ni,ni->n", ref, pos)
    neg_dist = torch.einsum("ni,mi->nm", ref, neg)
    return pos_dist, neg_dist


def ref_euclidean_similarity(ref: Tensor, pos: Tensor, neg: Tensor,
                             normalize: bool,
                             mode: str) -> Tuple[Tensor, Tensor]:
    """Negative L2 distance between the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    ref_sq = torch.einsum("ni->n", ref**2)
    pos_sq = torch.einsum("ni->n", pos**2)
    neg_sq = torch.einsum("ni->n", neg**2)

    pos_cosine, neg_cosine = ref_dot_similarity(ref, pos, neg)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)

    if normalize:
        if mode == "only_positives":
            D_pos = abs(pos_dist).max()
            pos_dist = (1 / D_pos) * pos_dist
        elif mode == "only_negatives":
            D_neg = abs(neg_dist).max()
            neg_dist = (1 / D_neg) * neg_dist
        elif mode == "both":
            D_pos = abs(pos_dist).max()
            pos_dist = (1 / D_pos) * pos_dist

            D_neg = abs(neg_dist).max()
            neg_dist = (1 / D_neg) * neg_dist

        elif mode == "both_together":
            D = torch.max(
                torch.cat([abs(neg_dist.view(-1)),
                           abs(pos_dist.view(-1))]))
            pos_dist = (1 / D) * pos_dist
            neg_dist = (1 / D) * neg_dist
        else:
            raise ValueError(f"{mode} is not implemented")

    return pos_dist, neg_dist


def ref_infonce(
        pos_dist: Tensor,  # nxd
        neg_dist: Tensor,  # nxd
):

    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()

    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    pos = (-pos_dist).mean()
    neg = torch.logsumexp(neg_dist, dim=1).mean()

    c_mean = c.mean()
    numerator = pos - c_mean
    denominator = neg + c_mean
    return numerator + denominator, numerator, denominator


def ref_infonce_full_denominator(pos_dist, neg_dist):

    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()

    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    numerator = (-pos_dist).mean()
    denominator = torch.logsumexp(
        torch.concatenate([
            pos_dist.unsqueeze(1),
            neg_dist,
        ], dim=1),
        dim=1,
    ).mean()

    c_mean = c.mean()
    numerator = numerator - c_mean
    denominator = denominator + c_mean

    return numerator + denominator, numerator, denominator


class ref_MseInfoNCE(nn.Module):

    def __init__(
        self,
        temperature=1.0,
        normalize=False,
        mode=None,
        infonce_type="infonce",
    ):

        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.mode = mode
        self.infonce_type = infonce_type

    @torch.jit.export
    def _distance(self, ref, pos, neg):
        pos_dist, neg_dist = ref_euclidean_similarity(ref,
                                                      pos,
                                                      neg,
                                                      normalize=self.normalize,
                                                      mode=self.mode)
        return pos_dist / self.temperature, neg_dist / self.temperature

    def forward(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the InfoNCE loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.
        """
        pos_dist, neg_dist = self._distance(
            batch.reference,
            batch.positive,
            batch.negative,
        )

        if self.infonce_type == "infonce":
            return ref_infonce(pos_dist, neg_dist)
        elif self.infonce_type == "infonce_full_denominator":
            return ref_infonce_full_denominator(pos_dist, neg_dist)
        else:
            raise ValueError


def test_mse_infonce_implementations_match():
    """Test that the MseInfoNCE and ref_MseInfoNCE implementations produce identical results."""
    mse_infonce = MseInfoNCE(
        temperature=1.0,
        infonce_type="infonce_full_denominator",
    )

    ref_mse_infonce = ref_MseInfoNCE(
        temperature=1.0,
        infonce_type="infonce_full_denominator",
    )

    batch_size = 1000
    dim = 10

    # Set a fixed seed for reproducibility
    torch.manual_seed(42)

    # Create inputs that require gradients
    reference = torch.randn(batch_size, dim, requires_grad=True)
    positive = torch.randn(batch_size, dim, requires_grad=True)
    negative = torch.randn(batch_size, dim, requires_grad=True)

    # Create copies for the reference implementation
    reference_ref = reference.clone().detach().requires_grad_(True)
    positive_ref = positive.clone().detach().requires_grad_(True)
    negative_ref = negative.clone().detach().requires_grad_(True)

    batch = TimeContrastiveLossBatch(
        reference=reference,
        positive=positive,
        negative=negative,
    )

    batch_ref = TimeContrastiveLossBatch(
        reference=reference_ref,
        positive=positive_ref,
        negative=negative_ref,
    )

    # Forward pass
    loss, numerator, denominator = mse_infonce(batch)
    ref_loss, ref_numerator, ref_denominator = ref_mse_infonce(batch_ref)

    # Check forward pass results
    assert torch.allclose(loss,
                          ref_loss), f"Loss values differ: {loss} vs {ref_loss}"
    assert torch.allclose(
        numerator, ref_numerator
    ), f"Numerator values differ: {numerator} vs {ref_numerator}"
    assert torch.allclose(
        denominator, ref_denominator
    ), f"Denominator values differ: {denominator} vs {ref_denominator}"

    # Backward pass
    loss.backward()
    ref_loss.backward()

    # Check gradients
    assert torch.allclose(
        reference.grad, reference_ref.grad
    ), f"Reference gradients differ: {reference.grad} vs {reference_ref.grad}"

    assert torch.allclose(
        positive.grad, positive_ref.grad
    ), f"Positive gradients differ: {positive.grad} vs {positive_ref.grad}"

    assert torch.allclose(
        negative.grad, negative_ref.grad
    ), f"Negative gradients differ: {negative.grad} vs {negative_ref.grad}"
