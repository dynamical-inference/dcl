from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

from jaxtyping import jaxtyped
from typeguard import typechecked

from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import DynamicsSolverValidationPrediction
from dcl.utils.datatypes import GroundTruthData

if TYPE_CHECKING:
    from dcl.loader import BaseDataLoader
    from dcl.solver import BaseSolver


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class Metric(Configurable, ABC):
    """Base class for all metrics."""

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the metric."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class GlobalMetric(Metric):
    """Global metric."""

    @jaxtyped(typechecker=typechecked)
    @abstractmethod
    def compute(
        self,
        predictions: DynamicsSolverValidationPrediction,
        ground_truth: GroundTruthData,
        solver: "BaseSolver",
        loader: "BaseDataLoader",
    ) -> Dict[str, Any]:
        """Compute the metric."""


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class BatchMetric(Metric):
    """Batch metric."""

    @jaxtyped(typechecker=typechecked)
    @abstractmethod
    def compute(
        self,
        predictions: DynamicsSolverValidationPrediction,
        ground_truth: GroundTruthData,
        solver: "BaseSolver",
    ) -> Dict[str, Any]:
        """Compute the metric."""
