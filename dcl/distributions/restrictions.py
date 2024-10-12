from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from jaxtyping import Bool
from jaxtyping import Integer
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import Tensor
from typeguard import typechecked

from dcl.utils.datatypes import config_field

if TYPE_CHECKING:
    from dcl.distributions.time_distributions import TimeDistribution


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class Restriction(ABC):
    """
    A restriction on the support of a distribution.
    """

    @abstractmethod
    def get_restriction_mask(
        self,
        distribution: "TimeDistribution",
    ) -> Bool[Tensor, " {distribution.support_length}"]:
        """
        Returns a subset of the support that should be filtered out.
        """


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class SequenceModelRestriction(Restriction):
    """
    A restriction based on the sequence length expected by the model.
    """
    sequence_length: int = config_field(default=1)

    @jaxtyped(typechecker=typechecked)
    def get_restriction_mask(
        self,
        distribution: "TimeDistribution",
    ) -> Bool[Tensor, " {distribution.support_length}"]:
        """
        Sequence models require previous samples to be passed.
        We therefore need to filter out any samples that don't allow for this.
        So this function returns index that doesn't have the minium of consecutive preceeding samples.
        """

        return self.is_supported(
            self.get_sequence_indices(distribution),
            distribution,
        )

    @jaxtyped(typechecker=typechecked)
    def get_sequence_indices(
        self,
        distribution: "TimeDistribution",
    ) -> Integer[Tensor,
                 "{distribution.support_length} {self.sequence_length}"]:
        """
        Construct a tensor that contains all indices that would be required for the specified sequence length and current support.
        """
        return distribution.support.unsqueeze(1) - torch.arange(
            self.sequence_length).unsqueeze(0)

    @jaxtyped(typechecker=typechecked)
    def is_supported(
        self,
        sequence_indices: Shaped[Tensor, "num_samples {self.sequence_length}"],
        distribution: "TimeDistribution",
    ) -> Bool[Tensor, " num_samples"]:
        """
        Check if the sequence indices are supported by the distribution.
        """
        return (distribution.check_supported(sequence_indices)).all(dim=1)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TrialSequenceModelRestriction(SequenceModelRestriction):
    """
    A restriction based on the sequence length expected by the model but also taking into account trial boundaries.
    """

    @jaxtyped(typechecker=typechecked)
    def get_restriction_mask(
        self,
        distribution: "TimeDistribution",
    ) -> Bool[Tensor, " {distribution.support_length}"]:
        """
        Sequence models require previous samples to be passed.
        We therefore need to filter out any samples that don't allow for this.
        So this function returns index that doesn't have the minium of consecutive preceeding samples.
        """

        sequence_indices = self.get_sequence_indices(distribution)

        return (self.is_supported(sequence_indices, distribution) &
                self.is_same_trial(sequence_indices, distribution))

    @jaxtyped(typechecker=typechecked)
    def is_same_trial(
        self,
        sequence_indices: Shaped[Tensor, "num_samples {self.sequence_length}"],
        distribution: "TimeDistribution",
    ) -> Bool[Tensor, " num_samples"]:
        """
        Check if the sequence indices are part of the same trial.
        """
        original_trial_id = distribution.auxilary_variables.trial_id[
            sequence_indices[:, -1]].unsqueeze(-1)
        sequence_trial_ids = distribution.auxilary_variables.trial_id[
            sequence_indices]

        return (sequence_trial_ids == original_trial_id).all(dim=1)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ConditionalRestriction():
    """
    A restriction on the support of **prior** of the given distribution.

    Conditional distributions may require to restrict the support of their prior distribution
    to make sure, that the conditional distribution's support is not violated.
    """

    @jaxtyped(typechecker=typechecked)
    def get_restriction_mask(
        self,
        prior_distribution: "TimeDistribution",
        conditional_distribution: "TimeDistribution",
    ) -> Bool[Tensor, " {prior_distribution.support_length}"]:
        """
        To find the restrictions, we simulate a conditional sampling given the current support of the prior distribution.
        Then we check whether the samples are supported by the conditional distribution.
        """
        return self.is_supported(
            self.simulated_conditional_sampling(
                prior_distribution=prior_distribution,
                conditional_distribution=conditional_distribution,
            ),
            prior_distribution,
        )

    @jaxtyped(typechecker=typechecked)
    def simulated_conditional_sampling(
        self,
        prior_distribution: "TimeDistribution",
        conditional_distribution: "TimeDistribution",
    ) -> Shaped[Tensor, " {prior_distribution.support_length}"]:
        """
        Sample from the conditional distribution.
        """
        conditional_index = conditional_distribution.sample_conditional(
            ref=prior_distribution.support)
        return conditional_index

    @jaxtyped(typechecker=typechecked)
    def is_supported(
        self,
        conditional_index: Integer[Tensor,
                                   " {prior_distribution.support_length}"],
        prior_distribution: "TimeDistribution",
    ) -> Bool[Tensor, " {prior_distribution.support_length}"]:
        return prior_distribution.check_supported(conditional_index)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TrialConditionalRestriction(ConditionalRestriction):

    @jaxtyped(typechecker=typechecked)
    def is_supported(
        self,
        conditional_index: Integer[Tensor,
                                   " {prior_distribution.support_length}"],
        prior_distribution: "TimeDistribution",
    ) -> Bool[Tensor, " {prior_distribution.support_length}"]:
        """
        Additionally check whether the conditional samples are part of the same trial as the corresponding prior samples.
        """
        supported = super().is_supported(
            conditional_index=conditional_index,
            prior_distribution=prior_distribution,
        )
        # only apply additional checks for supported samples
        return self.is_same_trial(
            supported=supported,
            conditional_index=conditional_index,
            prior_distribution=prior_distribution,
        )

    @jaxtyped(typechecker=typechecked)
    def is_same_trial(
        self,
        supported: Bool[Tensor, " {prior_distribution.support_length}"],
        conditional_index: Integer[Tensor,
                                   " {prior_distribution.support_length}"],
        prior_distribution: "TimeDistribution",
    ) -> Bool[Tensor, " {prior_distribution.support_length}"]:

        trials_cond = prior_distribution.auxilary_variables.trial_id[
            conditional_index[supported]]
        trials_prior = prior_distribution.auxilary_variables.trial_id[
            prior_distribution.support[supported]]
        supported[supported.clone()] = (trials_cond == trials_prior)
        return supported
