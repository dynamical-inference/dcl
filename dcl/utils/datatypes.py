import warnings
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

B = TypeVar('B')


def _make_field(field_type: str, help: Optional[str] = None, **kwargs) -> Any:
    """Helper function to create a field with specified metadata type.

    Args:
        field_type: Type of field ("meta", "config", or "derived_meta")
        default: Default value for the field
        **kwargs: Additional field arguments to pass to field()

    Returns:
        Field with metadata type set to specified type
    """
    metadata = {"type": field_type}
    if "metadata" in kwargs:
        metadata.update(kwargs.pop("metadata"))
    if help:
        metadata["help"] = help
    return field(metadata=metadata, **kwargs)


def meta_field(**kwargs) -> Any:
    """Field decorator that marks a dataclass field as metadata."""
    return _make_field("meta", **kwargs)


def config_field(help: Optional[str] = None, **kwargs) -> Any:
    """Field decorator that marks a dataclass field as configuration."""
    return _make_field("config", help=help, **kwargs)


def derived_meta_field(**kwargs) -> Any:
    """Field decorator that marks a dataclass field as derived metadata."""
    return _make_field("derived_meta", **kwargs)


def state_field(**kwargs) -> Any:
    """Field decorator that marks a dataclass field as state."""
    return _make_field("state", **kwargs)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class Batch():

    def clone(self) -> "Batch":
        return self.__class__.from_batch(self, clone_tensors=True)

    def view(self: B) -> B:
        """
        Returns a new batch object where all tensors are a view of the original tensors.
        This may be used when we want to adjust the shapes of tensors in the batch object
        without chaning the original batch object but also without actually copying the underlying data.
        """
        kwargs = {}
        self = self.clone()
        for f in fields(self):
            field_value = getattr(self, f.name, None)
            if isinstance(field_value, Tensor):
                # no change in shape just yet
                kwargs[f.name] = field_value.view(field_value.shape)
            elif isinstance(field_value, Batch):
                kwargs[f.name] = field_value.view()
            else:
                kwargs[f.name] = field_value
        return self

    def unsqueeze(self: B, dim: int) -> B:
        for f in fields(self):
            data = getattr(self, f.name)
            if isinstance(data, Tensor) or isinstance(data, Batch):
                setattr(self, f.name, data.unsqueeze(dim))
        return self

    def repeat_interleave(self: B, dim: int, repeats: int) -> B:
        for f in fields(self):
            data = getattr(self, f.name)
            if isinstance(data, Tensor) or isinstance(data, Batch):
                setattr(self, f.name, data.repeat_interleave(repeats, dim=dim))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to(self: B, device: torch.device) -> B:
        for f in fields(self):
            field_value = getattr(self, f.name, None)
            if isinstance(field_value, Tensor) or isinstance(
                    field_value, Batch):
                setattr(self, f.name, field_value.to(device))
        return self

    def append(self,
               batch: "Batch",
               dim: int = 0,
               dim_named: Optional[Dict[str, int]] = None):
        """Append another batch to this batch.

        This method concatenates the tensors from another batch object to this batch object.
        For each field, it concatenates along the specified dimension.

        Args:
            batch: The batch to append to this batch
            dim: The default dimension to concatenate along (default: 0, typically batch dimension)
            dim_named: Optional dictionary mapping field names to dimensions for field-specific concatenation
                       This allows different fields to be concatenated along different dimensions

        Returns:
            None, modifies the batch in place

        Raises:
            AssertionError: If trying to append incompatible tensor types
            NotImplementedError: If trying to append fields that are not tensors or Batch objects
        """
        for f in fields(self):
            # concat along the batch dimension
            existing_data = getattr(self, f.name, None)
            if existing_data is None:
                continue
            additional_data = getattr(batch, f.name, None)

            # Determine which dimension to use for this field
            field_dim = dim
            if dim_named is not None and f.name in dim_named:
                field_dim = dim_named[f.name]

            if isinstance(existing_data, Tensor):
                assert isinstance(
                    additional_data, Tensor
                ), f"Cannot append {f.name} of type {type(existing_data)} with {type(additional_data)}"
                new_data = torch.cat(
                    [existing_data, additional_data],
                    dim=field_dim,
                )
                setattr(self, f.name, new_data)
            elif isinstance(existing_data, Batch):
                assert isinstance(
                    additional_data, Batch
                ), f"Cannot append {f.name} of type {type(existing_data)} with {type(additional_data)}"
                existing_data.append(additional_data,
                                     dim=field_dim,
                                     dim_named=dim_named)
            else:
                raise NotImplementedError(
                    f"Appending {f.name} of type {type(existing_data)} is not implemented"
                )

    def add_fields(self, batch: "Batch"):
        """
        Adds the fields from another batch to this batch.
        """
        for f in fields(batch):
            # make sure we don't overwrite existing fields
            # but we do allow overwriting fields that are None
            if getattr(self, f.name, None) is not None:
                # warning if we are overwriting a field
                warnings.warn(
                    f"Tried overwriting field {f.name} of Batch type {type(self)} with value of Batch type {type(batch)}."
                )
                continue
            setattr(self, f.name, getattr(batch, f.name))

    def __getitem__(self: B, idx) -> B:
        self = self.clone()
        for f in fields(self):
            field_value = getattr(self, f.name, None)
            if field_value is not None and hasattr(field_value, "__getitem__"):
                setattr(self, f.name, field_value[idx])
        return self

    @property
    def shape(self) -> Dict[str, Tuple[int, ...]]:
        return {
            f.name: getattr(self, f.name).shape
            for f in fields(self)
            if isinstance(getattr(self, f.name), Tensor) or
            isinstance(getattr(self, f.name), Batch)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Batch":
        return cls(**data)

    @classmethod
    def from_batch(cls,
                   batch: "Batch",
                   clone_tensors: bool = False,
                   keep_all_tensors: bool = False):
        kwargs = {f.name: getattr(batch, f.name, None) for f in fields(cls)}

        if clone_tensors:
            for k, v in kwargs.items():
                if isinstance(v, Tensor) or isinstance(v, Batch):
                    kwargs[k] = v.clone()

        return cls(**kwargs)

    @classmethod
    def concat(cls,
               batches: List[B],
               dim: int = 0,
               dim_named: Optional[Dict[str, int]] = None) -> B:
        """Concatenate a list of batches along a specified dimension.

        This method concatenates a list of batch objects along the specified dimension.
        For each field, it concatenates along the specified dimension.

        Args:
            batches: List of batches to concatenate
            dim: The dimension to concatenate along (default: 0, typically batch dimension)
            dim_named: Optional dictionary mapping field names to dimensions for field-specific concatenation

        Returns:
            A new batch object with concatenated data
        """
        if not batches:
            raise ValueError("Cannot concatenate empty list of batches")

        # Get all field names from the first batch
        field_names = [f.name for f in fields(batches[0])]

        batch_class = type(batches[0])

        # Initialize dict to store concatenated tensors
        concat_data = {}

        # For each field, concatenate the tensors from all batches
        for field_name in field_names:
            field_values = [
                getattr(batch, field_name, None) for batch in batches
            ]
            # check all field values have same type
            if not all(
                    isinstance(value, type(field_values[0]))
                    for value in field_values):
                raise ValueError(
                    f"All field values must have the same type of {type(field_values[0])}. Found types: {set(type(value) for value in field_values)}"
                )
            if field_values:
                # Get concatenation dimension for this field
                field_dim = dim_named.get(field_name, dim) if dim_named else dim

                if isinstance(field_values[0], Tensor):
                    # Concatenate tensors
                    concat_data[field_name] = torch.cat(field_values,
                                                        dim=field_dim)
                elif isinstance(field_values[0], Batch):
                    # Recursively concatenate nested batch objects
                    concat_data[field_name] = type(field_values[0]).concat(
                        field_values, dim=field_dim, dim_named=dim_named)
                elif field_values[0] is None:
                    concat_data[field_name] = None
                else:
                    # For non-tensor/batch fields, use the first value
                    concat_data[field_name] = field_values[0]
            else:
                concat_data[field_name] = None

        return batch_class(**concat_data)

    def __eq__(self, other: B) -> bool:
        if not isinstance(other, type(self)):
            return False
        for f in fields(self):
            if isinstance(getattr(self, f.name), Tensor):
                if not torch.all(
                        getattr(self, f.name) == getattr(other, f.name)):
                    return False
            elif getattr(self, f.name) != getattr(other, f.name):
                return False
        return True


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class AuxilaryVariables(Batch):
    """Collection of auxilary variables of a dataset"""
    trial_id: Optional[Integer[Tensor, " batch"]] = None
    trial_time: Optional[Union[Float[Tensor, " batch"],
                               Integer[Tensor, " batch"]]] = None

    def __eq__(self, other: B) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
class TrialMetadata(AuxilaryVariables):
    """Metadata for a trial"""
    trial_id: Integer[Tensor, " batch"]
    trial_time: Union[Float[Tensor, " batch"], Integer[Tensor, " batch"]]

    def __eq__(self, other: B) -> bool:
        return super().__eq__(other)


@dataclass(kw_only=True)
class ContrastiveGTBatch(Batch):
    """Additional ground truth data for contrastive batch"""
    reference_gt: Float[Tensor, "batch latent_dim"]
    positive_gt: Float[Tensor, "batch latent_dim"]
    negative_gt: Float[Tensor, "neg_batch latent_dim"]


@dataclass(kw_only=True)
class ContrastiveTrialIdBatch(Batch):
    """Additional trial id data for contrastive batch"""
    reference_trial_id: Integer[Tensor, " batch"]
    positive_trial_id: Integer[Tensor, " batch"]
    negative_trial_id: Integer[Tensor, " neg_batch"]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ContrastiveBatch(Batch):
    """Batch of latent variables."""
    reference: Float[Tensor, "batch dynamics_offset encoder_offset dim"]
    positive: Float[Tensor, "batch encoder_offset dim"]
    # there may be more negatives than positive/reference
    negative: Float[Tensor, "neg_batch encoder_offset dim"]

    @property
    def batch(self) -> int:
        return self.reference.shape[0]

    @property
    def dynamics_offset(self) -> int:
        return self.reference.shape[1]

    @property
    def encoder_offset(self) -> int:
        return self.reference.shape[2]

    @property
    def dim(self) -> int:
        return self.reference.shape[3]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ContrastiveBatchIndex(Batch):
    """Batch of indices."""
    reference_indices: Integer[Tensor, "batch dynamics_offset encoder_offset"]
    positive_indices: Integer[Tensor, "batch encoder_offset"]
    negative_indices: Integer[Tensor, "neg_batch encoder_offset"]

    @property
    def indices(self) -> "ContrastiveBatchIndex":
        """Get indices from reference indices."""
        return ContrastiveBatchIndex.from_batch(self)

    @indices.setter
    def indices(self, value: "ContrastiveBatchIndex"):
        """Set reference indices."""
        self.reference_indices = value.reference_indices
        self.positive_indices = value.positive_indices
        self.negative_indices = value.negative_indices


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TimeContrastiveBatch(ContrastiveBatch, ContrastiveBatchIndex):
    """Batch of latent variables with indices."""


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TimeContrastiveLatentBatch(Batch):
    """Batch of latent variables with indices."""

    reference: Float[Tensor, "batch dynamics_offset dim"]
    positive: Float[Tensor, "batch dim"]
    negative: Float[Tensor, "neg_batch dim"]
    reference_indices: Integer[Tensor, "batch dynamics_offset"]
    positive_indices: Integer[Tensor, "batch"]
    negative_indices: Integer[Tensor, " neg_batch"]

    @property
    def batch(self) -> int:
        return self.reference.shape[0]

    @property
    def dynamics_offset(self) -> int:
        return self.reference.shape[1]

    @property
    def dim(self) -> int:
        return self.reference.shape[-1]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TimeContrastiveLossBatch(Batch):
    """Batch of latent variables with indices."""

    reference: Float[Tensor, "batch dim"]
    positive: Float[Tensor, "batch dim"]
    negative: Float[Tensor, "neg_batch dim"]
    reference_indices: Optional[Integer[Tensor, "batch"]] = None
    positive_indices: Optional[Integer[Tensor, "batch"]] = None
    negative_indices: Optional[Integer[Tensor, " neg_batch"]] = None

    @property
    def batch(self) -> int:
        return self.reference.shape[0]

    @property
    def dim(self) -> int:
        return self.reference.shape[-1]


class ConsecutiveIndexMixin():

    def __post_init__(self):
        super().__post_init__()
        ## make sure the index is a sequence of consecutive integers
        assert torch.all(self.x_index.diff(dim=0) == 1)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DynamicsData(Batch):
    """Base class for all dynamics data."""

    x: Float[Tensor, "batch dynamics_offset dim"]
    x_index: Integer[Tensor, "batch dynamics_offset"]

    @property
    def batch(self) -> int:
        return self.x.shape[-3]

    @property
    def dynamics_offset(self) -> int:
        return self.x.shape[-2]

    @property
    def dim(self) -> int:
        return self.x.shape[-1]

    @jaxtyped(typechecker=typechecked)
    @classmethod
    def from_start_points(
        cls,
        start_points: Float[Tensor, "batch dynamics_offset dim"],
        **kwargs,
    ) -> "DynamicsData":
        dynamics_offset = start_points.shape[1]
        if dynamics_offset != 1:
            raise NotImplementedError(
                "from_start_points only supports single time step input.")
        return cls(**cls._init_start_points(start_points, **kwargs))

    @staticmethod
    def _init_start_points(
        start_points: Float[Tensor, "batch dynamics_offset dim"],
        **kwargs,
    ):
        batch = start_points.shape[0]
        dynamics_offset = start_points.shape[1]
        return dict(
            x=start_points,
            x_index=torch.zeros((batch, dynamics_offset),
                                device=start_points.device,
                                dtype=torch.long),
        )


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DynamicsPrediction(Batch, ABC):

    x: Float[Tensor, "batch dim"]
    x_index: Integer[Tensor, "batch"]

    @property
    def batch(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[-1]

    @abstractmethod
    def to_input(self) -> DynamicsData:
        raise NotImplementedError(
            f"to_input not implemented for class {self.__class__}")


################################################
# Define model input and prediction types
################################################


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ErrorBasedStateData(Batch):
    x: Float[Tensor, "batch dynamics_offset dim"]
    x_next: Float[Tensor, "batch dim"]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ModeBasedSwitchingInput(Batch):
    modes: Integer[Tensor, "batch dynamics_offset"]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TimeBasedSwitchingInput(Batch):
    x_index: Integer[Tensor, "batch dynamics_offset"]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ErrorBasedSwitchingDynamicsData(Batch):
    x_next: Float[Tensor, "batch dim"]
    x_next_pred: Float[Tensor, "batch num_modes dim"]


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class LinearDynamicsInput(DynamicsData):
    system_idx: Optional[Integer[Tensor, " batch"]] = None


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class LinearDynamicsPrediction(DynamicsPrediction):

    def to_input(self) -> LinearDynamicsInput:
        return LinearDynamicsInput.from_batch(self.unsqueeze(1))


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class LorenzPrediction(DynamicsPrediction):

    def to_input(self) -> DynamicsData:
        return DynamicsData.from_batch(self.unsqueeze(1))


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class GumbelSLDSInput(DynamicsData):
    modes: Optional[Integer[Tensor, "batch dynamics_offset"]] = None
    x_next: Optional[Float[Tensor, "batch dim"]] = None

    @jaxtyped(typechecker=typechecked)
    @classmethod
    def from_start_points(
        cls,
        start_points: Float[Tensor, "batch dynamics_offset dim"],
        **kwargs,
    ) -> "GumbelSLDSInput":
        return super().from_start_points(start_points, **kwargs)

    @classmethod
    def _init_start_points(
        cls,
        start_points: Float[Tensor, "batch dynamics_offset dim"],
        num_modes: int = 1,
        **kwargs,
    ):
        start_points_kwargs = super()._init_start_points(start_points, **kwargs)
        batch = start_points.shape[0]
        dynamics_offset = start_points.shape[1]
        start_points_kwargs.update(
            dict(modes=torch.randint(0,
                                     num_modes, (batch,),
                                     device=start_points.device,
                                     dtype=torch.long).unsqueeze(1).expand(
                                         -1, dynamics_offset),))
        return start_points_kwargs


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class SLDSPrediction(DynamicsPrediction):
    modes: Integer[Tensor, " batch"]

    def to_input(self) -> GumbelSLDSInput:
        return GumbelSLDSInput.from_batch(self.unsqueeze(1))


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class GumbelSLDSPrediction(SLDSPrediction):
    """Adds a gumbel sample dimension to the data."""
    x: Float[Tensor, "batch gumbel_samples dim"]
    x_index: Integer[Tensor, "batch gumbel_samples"]
    modes: Float[Tensor, "batch gumbel_samples num_modes"]

    @property
    def gumbel_samples(self) -> int:
        return self.x.shape[1]

    def to_SLDSPrediction(self) -> SLDSPrediction:
        self = self.view()
        """Combines the batch dimension into the batch dimension."""
        # TODO: handle arbitrary auxilary variables or gt fields,
        # they may need to be expanded and then flattened as well

        # flatten first two dimensions
        self.x = self.x.flatten(start_dim=0, end_dim=1)
        self.x_index = self.x_index.flatten(start_dim=0, end_dim=1)
        self.modes = self.modes.flatten(start_dim=0, end_dim=1)
        # SLDS Prediction also assumes that modes are provided as Integers
        # not as one hot vectors or log probabilities
        # so we also need to argmax over the modes
        self.modes = self.modes.argmax(dim=-1)
        return SLDSPrediction.from_batch(self)

    def to_input(self) -> GumbelSLDSInput:
        return self.to_SLDSPrediction().to_input()


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class GroundTruthData(Batch):
    """Batch of ground truth data."""

    index: Integer[Tensor, " batch"]
    observed: Float[Tensor, "batch observed_dim"]
    auxilary: AuxilaryVariables = field(default_factory=AuxilaryVariables)
    latents: Optional[Float[Tensor, "batch latent_dim"]] = None
    modes: Optional[Integer[Tensor, " batch"]] = None
    # Should be BaseDynamicsModel but can't import it here because of circular import
    # also can't use TYPE_CHECKING because jaxtyping doesn't like forward references
    dynamics_model: Optional[torch.nn.Module] = None


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DynamicsSolverValidationData(Batch):
    """Batch of validation data for dynamics solver."""
    observed: Float[Tensor, "batch observed_dim"]
    encoder_index: Integer[Tensor, "batch encoder_offset"]
    reference_index: Integer[Tensor, "batch_reference dynamics_offset"]
    positive_index: Integer[Tensor, " batch_reference"]
    auxilary: AuxilaryVariables = field(default_factory=AuxilaryVariables)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DynamicsSolverValidationPrediction(Batch):
    """Batch of validation data for dynamics solver."""
    embeddings: Float[Tensor, "batch latent_dim"]
    embeddings_index: Integer[Tensor, " batch"]
    dynamics: DynamicsPrediction
    auxilary: AuxilaryVariables = field(default_factory=AuxilaryVariables)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class SLDSDynamicsSolverValidationPrediction(DynamicsSolverValidationPrediction
                                            ):
    dynamics: SLDSPrediction
