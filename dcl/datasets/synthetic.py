from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import Tensor
from tqdm import tqdm
from typeguard import typechecked

from dcl.datasets.timeseries import TimeSeriesDataset
from dcl.models.dynamics import BaseDynamicsModel
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.models.mixing import IdentityMixingModel
from dcl.models.mixing import MixingModel
from dcl.utils.configurable import check_initialized
from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field
from dcl.utils.datatypes import DynamicsData
from dcl.utils.datatypes import GroundTruthData
from dcl.utils.datatypes import GumbelSLDSInput
from dcl.utils.datatypes import TrialMetadata

T = TypeVar("T")


@dataclass(kw_only=True)
class StartingPointsSampler(Configurable, ABC):

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        dim: int,
    ) -> Float[Tensor, "num_samples 1 dim"]:
        pass


@dataclass(kw_only=True)
class FixedPointsSampler(StartingPointsSampler):
    """Sampler that uses a fixed set of points provided by the user."""

    points: List[Float[Tensor, " dim"]] = config_field(
        default_factory=list, help="List of starting points to use")

    def sample(
        self,
        num_samples: int,
        dim: int,
    ) -> Float[Tensor, "num_samples 1 dim"]:
        """Return the fixed points provided by the user.

        If num_samples > len(points), points will be repeated.
        If num_samples < len(points), only the first num_samples points will be used.
        """
        assert num_samples == 1, "FixedPointsSampler only supports num_samples = 1"
        points = torch.tensor(self.points).unsqueeze(0).float()
        if points.shape[-1] != dim:
            raise ValueError(
                f"Dimension of points {points.shape[-1]} does not match dimension of dataset {dim}"
            )
        return points.unsqueeze(1)


@dataclass(kw_only=True)
class HypersphereSampler(StartingPointsSampler):

    def sample(
        self,
        num_samples: int,
        dim: int,
    ) -> Float[Tensor, "num_samples 1 dim"]:
        # generate random points from a standard normal distribution and then project them onto the unit sphere
        points = torch.randn(num_samples, dim)
        points = points / points.norm(dim=-1, keepdim=True)
        return points.unsqueeze(1)


@dataclass(kw_only=True)
class BoxSampler(StartingPointsSampler):

    box_size: float = config_field(default=1.0,
                                   help="Size of the box to sample points from")

    def sample(
        self,
        num_samples: int,
        dim: int,
    ) -> Float[Tensor, "num_samples 1 dim"]:
        # generate random points from a uniform distribution and then scale them to the box size
        points = torch.rand(num_samples, dim) * self.box_size
        return points.unsqueeze(1)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class SyntheticTimeSeriesDataset(
        TimeSeriesDataset,
        ABC,
):
    root: Union[str, Path] = "./data"
    force_regenerate: bool = False

    # Base fields about the configuration of the dataset
    version: str = config_field(default="2.0")
    seed: int = config_field(default=42)
    # Internal state (not part of metadata/config)
    _data: Dict[str, Tensor] = field(default_factory=dict,
                                     init=False,
                                     repr=False)

    def __lazy_post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

        self._initialize_dataset()

    @property
    def dataset_id(self) -> str:
        """Generate unique dataset ID from configuration fields."""
        return self.config_hash

    @property
    @check_initialized
    def dataset_path(self) -> Path:
        """Get base path for dataset storage."""
        return self.root / self.__class__.__name__ / self.dataset_id

    @property
    def data_path(self) -> Path:
        return self.dataset_path / "data.pt"

    def _load_data(self):
        """Load data  from disk"""
        data_path = self.data_path
        self._data = torch.load(data_path, weights_only=True)

    def _acquire_data(self) -> None:
        """Implement data acquisition"""
        # Set all random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self._data = self._generate_data()

        self.save()

    def _initialize_dataset(self) -> None:
        """Initialize dataset ensuring proper seeding and storage."""
        if not self.force_regenerate:
            try:
                self.locked_recursive_load_additional(self.dataset_path,
                                                      timeout=60)
                self.validate_data()
                print(f"Loaded {self.__class__.__name__} from disk", flush=True)
                return
            except Exception as e:
                print(
                    f"Could not load{self.__class__.__name__} from {self.dataset_path}: {e}",
                    flush=True)

        print(
            f"Could not load {self.__class__.__name__} from disk, acquiring data...",
            flush=True)
        # Acquire and validate data
        try:
            self._acquire_data()
        except Exception as e:
            # Clean up any partially created files
            if self.dataset_path.exists():
                import shutil
                shutil.rmtree(self.dataset_path)
            raise e

        # Load from disk for consistency
        # longer time out, since dataset may be large
        self.locked_recursive_load_additional(self.dataset_path, timeout=60)
        self.validate_data()

    def save(self, path: Optional[Path] = None) -> Path:
        """We ignore the path argument and always use the self.dataset_path"""
        return super().save(self.dataset_path)

    def _save_additional(self, path: Path) -> None:
        data_path = self.data_path
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._data, data_path)

    def _load_additional(self, path: Path) -> None:
        data_path = self.data_path
        self._data = torch.load(data_path, weights_only=True)

    def validate_data(self):
        try:
            self._validate_data()
        except Exception as e:
            raise ValueError(f"Data validation failed: {e}") from e

    @property
    def _data_keys(self) -> List[str]:
        """
        Returns a list of keys of the self._data dictionary that can be indexed (and are required to be present).
        The first key is used to determine the length of the dataset.
        All keys are passed to _validate_types for type checking.
        """
        return [
            "observed",
            "latents",
            "trial_id",
            "trial_time",
        ]

    def _validate_data(self) -> bool:
        """Validate acquired data."""
        # check that all required keys are present
        missing_keys = [key for key in self._data_keys if key not in self._data]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in self._data: {missing_keys}")
        return True

    @abstractmethod
    def _generate_data(
            self
    ) -> Dict[str, Union[Float[Tensor, "..."], Integer[Tensor, "..."]]]:
        """Generate data."""
        raise NotImplementedError("Data generation not implemented")

    @property
    @check_initialized
    def auxilary_variables(self) -> TrialMetadata:
        """Get the auxilary variables of the dataset."""
        return TrialMetadata(
            trial_id=self._data["trial_id"],
            trial_time=self._data["trial_time"],
        )

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(
        self, indices: Integer[Tensor, " *batch_shape"]
    ) -> Float[Tensor, "*batch_shape  {self.observed_dim}"]:
        """Get a batch of data from the dataset."""
        return self._data["observed"][indices]

    @jaxtyped(typechecker=typechecked)
    def get_latent_data(
        self, index: Integer[Tensor, " *batch_shape"]
    ) -> Float[Tensor, "*batch_shape {self.latent_dim}"]:
        """Get the latent data of the dataset."""
        return self._data["latents"][index]

    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Union[
            Shaped[Tensor, " batch_shape ..."],
    ]]:
        return {key: self._data[key][indices] for key in self._data_keys}

    def __len__(self) -> int:
        return self._data["observed"].shape[0]

    @property
    @check_initialized
    def ground_truth_data(self) -> GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.latents = self.get_latent_data(gt_batch.index)
        return GroundTruthData.from_batch(gt_batch)


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class GenericDynamicsDataset(SyntheticTimeSeriesDataset):

    dynamics_model: BaseDynamicsModel = config_field(
        default_factory=LinearDynamicsModel)
    starting_points_sampler: StartingPointsSampler = config_field(
        default_factory=HypersphereSampler)
    mixing_model: MixingModel = config_field(
        default_factory=IdentityMixingModel)
    num_steps: int = config_field(
        default=1000,
        help="Number of steps to predict forward for each starting point")
    num_trials: int = config_field(
        default=100, help="Number of trials / starting points to generate")

    def validate_config(self):
        if isinstance(self.dynamics_model,
                      GumbelSLDS) and not isinstance(self, SLDSDynamicsDataset):
            raise ValueError("Use SLDSDynamicsDataset for GumbelSLDS.")
        super().validate_config()

    @property
    def latent_dim(self) -> int:
        return self.mixing_model.input_dim

    @property
    def observed_dim(self) -> int:
        return self.mixing_model.output_dim

    @jaxtyped(typechecker=typechecked)
    def _generate_starting_points(
            self) -> Float[Tensor, "{self.num_trials} 1 {self.latent_dim}"]:
        return self.starting_points_sampler.sample(
            num_samples=self.num_trials,
            dim=self.latent_dim,
        )

    @jaxtyped(typechecker=typechecked)
    def _to_input_data(
        self, start_points: Float[Tensor,
                                  "{self.num_trials} 1 {self.latent_dim}"]
    ) -> DynamicsData:
        return self.dynamics_model.input_type.from_start_points(start_points)

    @torch.no_grad()
    def _generate_data(
            self
    ) -> Dict[str, Union[Float[Tensor, "..."], Integer[Tensor, "..."]]]:
        """Generate data."""
        start_points = self._generate_starting_points()
        input_data = self._to_input_data(start_points)

        self.dynamics_model.eval()
        for i in tqdm(range(self.num_steps - 1), desc="Generating data"):
            dynamics_pred = self.dynamics_model(input_data)
            input_data.append(dynamics_pred.to_input(), dim=1)

        data_dict = input_data.to_dict()
        data_dict["trial_id"] = torch.arange(
            self.num_trials).unsqueeze(-1).expand(-1, self.num_steps)
        data_dict["trial_time"] = data_dict["x_index"]
        del data_dict["x_index"]
        # flatten the batch (aka trial) dimension and the seq_len dimension (first two dimensions)
        for key in data_dict.keys():
            if data_dict[key] is not None:
                data_dict[key] = data_dict[key].flatten(start_dim=0, end_dim=1)

        # rename x to latents
        data_dict["latents"] = data_dict["x"]
        del data_dict["x"]

        # Apply mixing model to get observed data y
        data_dict["observed"] = self.mixing_model(data_dict["latents"])

        return {k: v for k, v in data_dict.items() if k in self._data_keys}

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        """Split the dataset into a new dataset with the given indices."""
        new_dataset = self.clone()
        new_dataset._data = self[indices]
        return new_dataset

    @property
    @check_initialized
    def ground_truth_data(self) -> GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.dynamics_model = self.dynamics_model
        return gt_batch


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class SLDSDynamicsDataset(GenericDynamicsDataset):

    dynamics_model: GumbelSLDS = config_field(default_factory=GumbelSLDS)

    @property
    def _data_keys(self) -> List[str]:
        return super()._data_keys + ["modes"]

    @jaxtyped(typechecker=typechecked)
    @property
    @check_initialized
    def ground_truth_data(self) -> GroundTruthData:
        """Get the latent data of the dataset."""
        gt_batch = super().ground_truth_data
        gt_batch.modes = self._data["modes"]
        return GroundTruthData.from_batch(gt_batch)

    def _get_input_data(
        self, start_points: Float[Tensor, "{self.num_trials} 1 {self.dim}"]
    ) -> GumbelSLDSInput:
        return self.dynamics_model.input_type.from_start_points(
            start_points,
            num_modes=self.dynamics_model.num_modes,
        )
