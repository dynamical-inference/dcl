"""
Portions of this code are based on https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/cebra/models/criterions.py
which is distributed under Apache License, Version 2.0
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import Tensor

from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import TimeContrastiveLossBatch


@torch.jit.script
def dot_similarity(ref: Tensor, pos: Tensor,
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


def euclidean_similarity(
    ref: Tensor,
    pos: Tensor,
    neg: Tensor,
) -> Tuple[Tensor, Tensor]:
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

    pos_cosine, neg_cosine = dot_similarity(ref, pos, neg)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)

    return pos_dist, neg_dist


def infonce(
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


def infonce_ratio(pos_dist: Tensor, neg_dist: Tensor, log_q_pos: Tensor,
                  log_q_neg: Tensor):

    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()

    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    log_ratio = log_q_pos.unsqueeze(1) - log_q_neg.unsqueeze(0)

    pos = (-pos_dist).mean()
    neg = torch.logsumexp(log_ratio + neg_dist, dim=1).mean()

    c_mean = c.mean()
    numerator = pos - c_mean
    denominator = neg + c_mean
    return numerator + denominator, numerator, denominator


def infonce_full_denominator_old(pos_dist, neg_dist):
    """
    NOTE:
    This is the old implementation of the full denominator, where the
    constant c is not added back, which makes the interpretation of the
    alignment/uniformity approach not possible.
    """

    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()

    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    n_pos = pos_dist.shape[0]
    n_neg = neg_dist.shape[1]

    nominator = pos_dist  # (n_pos,)
    denominator = neg_dist  # (n_pos, n_neg)

    denominator = torch.concatenate([nominator.unsqueeze(1), denominator],
                                    dim=1)

    assert denominator.shape == (n_pos, n_neg + 1)

    denominator = torch.logsumexp(denominator, dim=1)
    loss = denominator - nominator
    assert loss.shape == (n_pos,)

    return loss.mean(), (-nominator).mean(), denominator.mean()


@torch.jit.script
def infonce_full_denominator(pos_dist, neg_dist):

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


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class ContrastiveCriterion(Configurable, torch.nn.Module, ABC):
    """Contrastive criterion for training encoder model jointly with slds model using contrastive learning."""

    temperature: float = config_field(default=1.0)
    infonce_type: Literal["infonce", "infonce_full_denominator"] = config_field(
        default="infonce")

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    @torch.jit.export
    @abstractmethod
    def _distance(self,
                  batch: TimeContrastiveLossBatch) -> Tuple[Tensor, Tensor]:
        """Compute distances between reference, positive and negative samples.

        Args:
            batch: Batch containing reference, positive and negative samples

        Returns:
            Tuple of:
                pos_dist: Distance between reference and positive samples
                neg_dist: Distance between reference and negative samples
        """

    def forward(
        self,
        batch: TimeContrastiveLossBatch,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the InfoNCE loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.
        """
        pos_dist, neg_dist = self._distance(batch)

        if self.infonce_type == "infonce":
            return infonce(pos_dist, neg_dist)
        elif self.infonce_type == "infonce_full_denominator":
            return infonce_full_denominator(pos_dist, neg_dist)
        else:
            raise ValueError


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class DotInfoNCE(ContrastiveCriterion):

    @torch.jit.export
    def _distance(self, batch: TimeContrastiveLossBatch):
        pos_dist, neg_dist = dot_similarity(batch.reference, batch.positive,
                                            batch.negative)
        return pos_dist / self.temperature, neg_dist / self.temperature


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class MseInfoNCE(ContrastiveCriterion):

    @torch.jit.export
    def _distance(self, batch: TimeContrastiveLossBatch):
        pos_dist, neg_dist = euclidean_similarity(
            ref=batch.reference,
            pos=batch.positive,
            neg=batch.negative,
        )
        return pos_dist / self.temperature, neg_dist / self.temperature
