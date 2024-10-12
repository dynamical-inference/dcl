from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Iterator, Optional, TypeVar, Union

import torch
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

from dcl.datasets.timeseries import TimeSeriesDataset
from dcl.distributions.restrictions import ConditionalRestriction
from dcl.distributions.restrictions import SequenceModelRestriction
from dcl.distributions.restrictions import TrialConditionalRestriction
from dcl.distributions.restrictions import TrialSequenceModelRestriction
from dcl.distributions.time_distributions import \
    ConditionalDiscreteTimeDistribution
from dcl.distributions.time_distributions import OffsetTimeDistribution
from dcl.distributions.time_distributions import \
    UnconditionalDiscreteTimeDistribution
from dcl.distributions.time_distributions import \
    UniformDiscreteTimeDistribution
from dcl.loader import BaseDataLoader
from dcl.utils.configurable import check_initialized
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import ContrastiveBatch
from dcl.utils.datatypes import ContrastiveBatchIndex
from dcl.utils.datatypes import DynamicsSolverValidationData
from dcl.utils.datatypes import GroundTruthData
from dcl.utils.datatypes import TimeContrastiveBatch
from dcl.utils.datatypes import TrialMetadata

B = TypeVar('B')


@dataclass(kw_only=True)
class ContrastiveDataLoader(BaseDataLoader, ABC):
    """Data loader for contrastive learning.

    This class samples reference, positive, and negative samples from the dataset for contrastive learning.
    """

    num_iterations: int = config_field(default=1)
    batch_size_neg: Optional[int] = config_field(default=None)

    @property
    def num_negatives(self) -> int:
        """Number of negative samples to sample."""
        if self.batch_size_neg is None:
            return self.batch_size
        else:
            return self.batch_size_neg

    def __len__(self) -> int:
        """Return number of iterations per epoch."""
        return self.num_iterations

    def reset(self):
        # ContrastiveDataLaoders are stateless, so we don't need to reset anything
        pass

    @check_initialized
    def __iter__(self) -> Iterator[
            Any,
    ]:
        """Return iterator over batches."""
        self.reset()

        for _ in range(self.num_iterations):
            yield self.sample_batch().to(self.device)

    @abstractmethod
    def sample_batch(self) -> ContrastiveBatch:
        """Sample a batch of contrastive samples."""


@dataclass(kw_only=True)
class TimeContrastiveDataLoader(ContrastiveDataLoader, ABC):
    """Data loader for time contrastive learning."""

    def __lazy_post_init__(self, *args, **kwargs):
        super().__lazy_post_init__(*args, **kwargs)
        self._init_distributions()

    @abstractmethod
    def _init_distributions(self):
        """Initialize the distributions."""
        raise NotImplementedError("init_distributions not implemented")


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DiscreteTimeContrastiveDataLoader(TimeContrastiveDataLoader):
    """Data loader for time contrastive learning with discrete (and equidistant) time steps."""

    reference_distribution: UnconditionalDiscreteTimeDistribution = config_field(
        default_factory=UniformDiscreteTimeDistribution)
    positive_distribution: ConditionalDiscreteTimeDistribution = config_field(
        default_factory=OffsetTimeDistribution)
    negative_distribution: Union[
        ConditionalDiscreteTimeDistribution,
        UnconditionalDiscreteTimeDistribution,
    ] = config_field(default_factory=UniformDiscreteTimeDistribution,)

    encoder_offset: int = field(default=1, init=False)
    _dynamics_offset: int = field(default=1, init=False)

    @property
    def dynamics_offset(self) -> int:
        return self._dynamics_offset

    @dynamics_offset.setter
    def dynamics_offset(self, value: int):
        self._dynamics_offset = value
        # changing the dynamics offset effects the support of the distributions, reinitialize them
        self._init_distributions()
        print(f"Updated dynamics offset to {self.dynamics_offset}")

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(self, dataset: TimeSeriesDataset):
        super().__lazy_post_init__(dataset)

    @property
    @jaxtyped(typechecker=typechecked)
    def dataset(self) -> TimeSeriesDataset:
        return super().dataset

    def _init_distributions(self):

        index = self.dataset.index
        auxilary_variables = self.dataset.auxilary_variables

        restrictions = []
        conditional_restrictions = []
        if isinstance(auxilary_variables, TrialMetadata):
            restrictions += [
                TrialSequenceModelRestriction(
                    sequence_length=self.dynamics_offset),
            ]
            conditional_restrictions += [
                TrialConditionalRestriction(),
            ]
        else:
            restrictions += [
                SequenceModelRestriction(sequence_length=self.dynamics_offset),
            ]
            conditional_restrictions += [
                ConditionalRestriction(),
            ]

        self.reference_distribution.lazy_init(
            time_index=index,
            auxilary_variables=auxilary_variables,
            restrictions=restrictions,
        )
        self.positive_distribution.lazy_init(
            time_index=index,
            auxilary_variables=auxilary_variables,
            conditional_restrictions=conditional_restrictions,
        )

        self.negative_distribution.lazy_init(
            time_index=index,
            auxilary_variables=auxilary_variables,
            # no restrictions on negative distribution
            restrictions=[],
            # expect for conditional_restrictions
            conditional_restrictions=conditional_restrictions,
        )

        prior_restrictions = self.positive_distribution.get_prior_restrictions(
            prior_distribution=self.reference_distribution)
        self.reference_distribution.restrict_support(prior_restrictions)

    @jaxtyped(typechecker=typechecked)
    def sample_batch(self) -> TimeContrastiveBatch:

        reference_indices = self.reference_distribution.sample(
            num_samples=self.batch_size,)

        positive_indices = self.positive_distribution.sample_conditional(
            ref=reference_indices,)

        negative_indices = self._sample_negative_indices(reference_indices)

        batch_data = self.batch_from_indices(
            reference_indices=reference_indices,
            negative_indices=negative_indices,
            positive_indices=positive_indices,
        )

        return batch_data

    @jaxtyped(typechecker=typechecked)
    def _sample_negative_indices(
        self, reference_indices: Integer[Tensor, " {self.batch_size}"]
    ) -> Integer[Tensor, " {self.num_negatives}"]:

        if isinstance(self.negative_distribution,
                      ConditionalDiscreteTimeDistribution):
            assert self.num_negatives == self.batch_size, (
                f"When using a conditional time distribution, num_negatives must be equal to batch_size, "
                f"but got num_negatives={self.num_negatives} and batch_size={self.batch_size}."
            )
            negative_indices = self.negative_distribution.sample_conditional(
                ref=reference_indices,)
        elif isinstance(self.negative_distribution,
                        UnconditionalDiscreteTimeDistribution):
            negative_indices = self.negative_distribution.sample(
                num_samples=self.num_negatives,)
        else:
            raise ValueError(
                f"Negative distribution {self.negative_distribution} is not a valid time distribution."
            )
        return negative_indices

    @jaxtyped(typechecker=typechecked)
    def expand_dynamics_offset(
        self, index: Integer[Tensor, " *batch_shape"]
    ) -> Integer[Tensor, "*batch_shape {self.dynamics_offset}"]:
        """Expand index to a sequence of length dynamics_offset."""
        offset = torch.arange(self.dynamics_offset, device=index.device).view(
            *([1] * len(index.shape)), -1)
        return index.unsqueeze(-1) - offset

    @jaxtyped(typechecker=typechecked)
    def expand_index_offset(
        self,
        index: Integer[Tensor, " *batch_shape"],
        offset: int,
    ) -> Integer[Tensor, "*batch_shape {offset}"]:
        """Expand index to a sequence of length dynamics_offset."""
        return index.unsqueeze(-1) - torch.arange(offset, device=index.device)

    def expand_index_encoder_offset(
        self,
        index: Integer[Tensor, " *batch_shape"],
    ) -> Integer[Tensor, "*batch_shape {self.encoder_offset}"]:
        """Expand index to a sequence of length dynamics_offset."""
        return self.expand_index_offset(index, offset=self.encoder_offset)

    def expand_index_dynamics_offset(
        self,
        index: Integer[Tensor, " *batch_shape"],
    ) -> Integer[Tensor, "*batch_shape {self.dynamics_offset}"]:
        """Expand index to a sequence of length dynamics_offset."""
        return self.expand_index_offset(index, offset=self.dynamics_offset)

    @jaxtyped(typechecker=typechecked)
    def expand_indices(
        self,
        reference_indices: Integer[Tensor, " {self.batch_size}"],
        positive_indices: Integer[Tensor, " {self.batch_size}"],
        negative_indices: Integer[Tensor, " {self.num_negatives}"],
    ) -> ContrastiveBatchIndex:
        """Expand indices to a sequence of length dynamics_offset."""

        reference_indices = self.expand_index_encoder_offset(reference_indices)
        positive_indices = self.expand_index_encoder_offset(positive_indices)
        negative_indices = self.expand_index_encoder_offset(negative_indices)

        return ContrastiveBatchIndex(
            reference_indices=self.expand_index_dynamics_offset(
                reference_indices),
            positive_indices=positive_indices,
            negative_indices=negative_indices,
        )

    @jaxtyped(typechecker=typechecked)
    def batch_from_indices(
        self,
        reference_indices: Integer[Tensor, " {self.batch_size}"],
        positive_indices: Integer[Tensor, " {self.batch_size}"],
        negative_indices: Integer[Tensor, " {self.num_negatives}"],
    ) -> TimeContrastiveBatch:
        """Get a batch of data from the dataset for given indices."""
        batch = self.expand_indices(
            reference_indices=reference_indices,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
        )
        batch.reference = self.dataset.get_observed_data(
            batch.reference_indices)
        batch.negative = self.dataset.get_observed_data(batch.negative_indices)
        batch.positive = self.dataset.get_observed_data(batch.positive_indices)

        batch = TimeContrastiveBatch.from_batch(batch)

        return batch

    @property
    def validation_data(self) -> DynamicsSolverValidationData:
        """Get a batch of data from the dataset for validation."""

        reference_index = self.reference_distribution.support
        positive_index = self.positive_distribution.sample_conditional(
            ref=reference_index)

        enc_index = self.expand_index_encoder_offset(self.dataset.index)
        reference_index = self.expand_index_dynamics_offset(reference_index)

        batch = DynamicsSolverValidationData(
            auxilary=self.dataset.auxilary_variables,
            observed=self.dataset.get_observed_data(self.dataset.index),
            encoder_index=enc_index,
            reference_index=reference_index,
            positive_index=positive_index,
        )

        return batch.to(self.device)

    @property
    def ground_truth_data(self) -> GroundTruthData:
        """Get the ground truth data for the dataset."""
        return self.dataset.ground_truth_data.to(self.device)
