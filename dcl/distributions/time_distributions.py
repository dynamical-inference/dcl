from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import List, Optional

import torch
from jaxtyping import Bool
from jaxtyping import Integer
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import Tensor
from typeguard import typechecked

from dcl.distributions.restrictions import ConditionalRestriction
from dcl.distributions.restrictions import Restriction
from dcl.utils.configurable import check_initialized
from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import AuxilaryVariables
from dcl.utils.datatypes import config_field


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TimeDistribution(Configurable, ABC):
    """
    Distribution across discrete time steps.
    """
    lazy: bool = True
    _support: Optional[Tensor] = field(default=None, init=False)
    _auxilary_variables: Optional[AuxilaryVariables] = field(default=None,
                                                             init=False)

    restrictions: List[Restriction] = field(default_factory=list, init=False)

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        *args,
        time_index: Integer[Tensor, " time"],
        auxilary_variables: AuxilaryVariables,
        restrictions: List[Restriction] = [],
        **kwargs,
    ):
        super().__lazy_post_init__(*args, **kwargs)
        self.restrictions = restrictions
        self._initialize_support(time_index, auxilary_variables)

    @property
    @check_initialized
    def support(self) -> Tensor:
        """Support of the distribution."""
        return self._support

    @property
    @check_initialized
    def auxilary_variables(self) -> AuxilaryVariables:
        """Auxilary variables of the distribution."""
        return self._auxilary_variables

    @property
    def support_length(self) -> int:
        """Length of the support of the distribution."""
        return len(self.support)

    def _initialize_support(
        self,
        time_index: Tensor,
        auxilary_variables: AuxilaryVariables,
    ):
        """Initialize the support of the distribution."""
        self._support = time_index
        self._auxilary_variables = auxilary_variables
        self._apply_restrictions()

    def _apply_restrictions(self):
        """
        Apply the restrictions to the support of the distribution.
        """
        for restriction in self.restrictions:
            self.restrict_support(restriction.get_restriction_mask(self))

    @jaxtyped(typechecker=typechecked)
    def restrict_support(
        self,
        mask: Bool[Tensor, " {self.support_length}"],
    ):
        """Restrict the support of the distribution."""
        self._support = self._support[mask]

    def filter_support(self, filter_indices: Tensor):
        """Removes any elements from the support that are in the filter_indices."""
        mask = ~torch.isin(self.support, filter_indices)
        self.mask_support(mask)

    @jaxtyped(typechecker=typechecked)
    def check_supported(
        self,
        samples: Shaped[Tensor, " *shape"],
    ) -> Bool[Tensor, " *shape"]:
        """Checks if the samples are supported by the distribution."""
        return torch.isin(samples, self.support)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ConditionalDistributionMixin(ABC):

    conditional_restrictions: List[ConditionalRestriction] = field(
        default_factory=list,
        init=False,
    )

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        *args,
        conditional_restrictions: List[ConditionalRestriction],
        **kwargs,
    ):
        super().__lazy_post_init__(*args, **kwargs)
        self.conditional_restrictions = conditional_restrictions

    @abstractmethod
    def sample_conditional(
        self,
        ref: Tensor,
    ) -> Tensor:
        """Sample from the distribution."""
        raise NotImplementedError("sample_conditional not implemented")

    @jaxtyped(typechecker=typechecked)
    def get_prior_restrictions(
        self,
        *,
        prior_distribution: TimeDistribution,
    ) -> Bool[Tensor, " {prior_distribution.support_length}"]:
        """
        To prevent the conditional distribution from sampling outside its support,
        we need to restrict the support of the prior distribution.

        This function returns a tensor of indices to filter the support of the prior distribution.
        """

        supported_mask = torch.ones_like(self.support, dtype=torch.bool)

        for restriction in self.conditional_restrictions:
            supported_mask = supported_mask & restriction.get_restriction_mask(
                prior_distribution=prior_distribution,
                conditional_distribution=self,
            )

        return supported_mask


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class UnconditionalDistributionMixin(ABC):

    @jaxtyped(typechecker=typechecked)
    @abstractmethod
    def sample(
        self,
        num_samples: int,
        aux: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample from the distribution."""
        raise NotImplementedError("sample not implemented")


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DiscreteTimeDistribution(TimeDistribution):
    pass


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ContinuousTimeDistribution(TimeDistribution):
    pass


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ConditionalDiscreteTimeDistribution(ConditionalDistributionMixin,
                                          DiscreteTimeDistribution):
    pass


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class UnconditionalDiscreteTimeDistribution(UnconditionalDistributionMixin,
                                            DiscreteTimeDistribution):
    pass


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class UniformDiscreteTimeDistribution(UnconditionalDiscreteTimeDistribution):
    """
    Uniform distribution across discrete time steps.
    """

    @jaxtyped(typechecker=typechecked)
    def sample(
        self,
        num_samples: int,
        **kwargs,
    ) -> Integer[Tensor, " {num_samples}"]:
        """Samples uniformly across support of the distribution."""

        rand_indices = torch.randint(
            low=0,
            high=len(self.support),
            size=(num_samples,),
            device=self.support.device,
        )
        return self.support[rand_indices]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class OffsetTimeDistribution(ConditionalDiscreteTimeDistribution):
    """Conditional time distribution."""

    offset: int = config_field(default=1)

    def __lazy_post_init__(self, *args, **kwargs):
        super().__lazy_post_init__(*args, **kwargs)

    def validate_config(self):
        assert self.offset > 0, "Time offset has to be positive."
        super().validate_config()

    @jaxtyped(typechecker=typechecked)
    def sample_conditional(
        self,
        ref: Tensor,
    ) -> Tensor:
        return ref + self.offset
