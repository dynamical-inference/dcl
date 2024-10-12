import warnings
from abc import abstractmethod
from dataclasses import dataclass

import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from scipy.linalg import null_space
from scipy.stats import ortho_group
from torch import nn
from torch import Tensor
from typeguard import typechecked

from dcl.models import BaseModel
from dcl.models.mixing_mlp import construct_invertible_mlp
from dcl.models.utils import ModelStorageMixin
from dcl.utils.datatypes import config_field


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class MixingModel(ModelStorageMixin, BaseModel):
    """Abstract base class for mixing transformations."""

    input_dim: int = config_field(default=2)
    output_dim: int = config_field(default=2)

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "mixing_model"

    def __new__(cls, *args, **kwargs):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    def __lazy_post_init__(self):
        ModelStorageMixin.__lazy_post_init__(self)
        BaseModel.__lazy_post_init__(self)
        self.init_parameters()

    @abstractmethod
    def init_parameters(self):
        """Initialize the transformation parameters."""

    def check_matrix_injectivity(self, matrix: torch.Tensor):
        """Check if a matrix transformation is injective."""
        if self.input_dim == self.output_dim:
            if torch.linalg.det(matrix) == 0:
                raise ValueError(
                    f"{self.__class__.__name__} has zero determinant")

        # Check null space is empty
        null_space_A = null_space(matrix.detach().cpu().numpy())
        if null_space_A.shape[-1] != 0:
            raise ValueError(
                f"{self.__class__.__name__} has non-empty null space")

        # Check singular values
        singular_values = torch.linalg.svd(matrix).S
        if torch.any(singular_values == 0):
            raise ValueError(
                f"{self.__class__.__name__} has zero singular values")

        # Warning for small singular values
        s_value_threshold = 1.0
        if torch.any(singular_values < s_value_threshold):
            warnings.warn(
                f"{self.__class__.__name__} has small singular values < {s_value_threshold}"
            )


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class LinearMixingModel(MixingModel):
    """Linear mixing transformation: x -> Ax"""

    def init_parameters(self):
        rnd_gen = torch.Generator().manual_seed(self.seed)
        self.A = nn.Parameter(
            torch.randn((self.output_dim, self.input_dim), generator=rnd_gen))
        self.check_matrix_injectivity(self.A)

    def forward(
        self,
        x: Float[Tensor, "*batch input_dim"],
    ) -> Float[Tensor, "*batch output_dim"]:
        assert x.shape[-1] == self.input_dim
        return torch.einsum('...ik,...k->...i', self.A, x)


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class AffineMixingModel(LinearMixingModel):
    """Affine mixing transformation: x -> Ax + b"""

    def init_parameters(self):
        super().init_parameters()
        rnd_gen = torch.Generator().manual_seed(self.seed)
        self.b = nn.Parameter(torch.randn(self.output_dim, generator=rnd_gen))

    def forward(
        self,
        x: Float[Tensor, "*batch input_dim"],
    ) -> Float[Tensor, "*batch output_dim"]:
        return super().forward(x) + self.b


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class IdentityMixingModel(LinearMixingModel):
    """Identity mixing transformation: x -> x"""

    def __lazy_post_init__(self):
        assert self.input_dim == self.output_dim, "Identity requires input_dim == output_dim"
        super().__lazy_post_init__()

    def init_parameters(self):
        self.A = nn.Parameter(torch.eye(self.output_dim, self.input_dim))


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class OrthogonalMixingModel(LinearMixingModel):
    """Orthogonal mixing transformation using random orthogonal matrix"""

    def __lazy_post_init__(self):
        assert self.input_dim == self.output_dim, "Orthogonal requires input_dim == output_dim"
        super().__lazy_post_init__()

    def init_parameters(self):
        self.A = nn.Parameter(
            torch.tensor(ortho_group.rvs(self.input_dim,
                                         random_state=self.seed),
                         dtype=torch.float32))


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class NonlinearMixingModel(MixingModel):
    """Nonlinear mixing transformation using invertible MLP"""
    n_layers: int = config_field(default=4)
    n_iter_cond_thresh: int = config_field(default=10000)
    cond_thresh_ratio: float = config_field(default=1e-3)

    def __lazy_post_init__(self):
        # we only check the input and output dimensions are the same
        # if this is not called from a subclass
        if self.__class__ == NonlinearMixingModel:
            assert self.input_dim == self.output_dim, "Nonlinear requires input_dim == output_dim"
        super().__lazy_post_init__()

    def _create_invertible_mlp(self, dim: int) -> nn.Module:
        """Helper to create invertible MLP with consistent parameters"""
        return construct_invertible_mlp(
            n=dim,
            n_layers=self.n_layers,
            seed=self.seed,
            n_iter_cond_thresh=self.n_iter_cond_thresh,
            cond_thresh_ratio=self.cond_thresh_ratio,
        )

    def init_parameters(self):
        self.mix = self._create_invertible_mlp(self.input_dim)
        # Freeze parameters
        for p in self.mix.parameters():
            p.requires_grad = False

    def forward(
        self,
        x: Float[Tensor, "*batch input_dim"],
    ) -> Float[Tensor, "*batch output_dim"]:
        return self.mix(x)


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class NonlinearLinearMixingModel(NonlinearMixingModel):
    """Nonlinear followed by linear mixing transformation"""

    def init_parameters(self):
        nonlinear = self._create_invertible_mlp(self.input_dim)
        linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.mix = nn.Sequential(nonlinear, linear)

        # Freeze parameters
        for p in self.mix.parameters():
            p.requires_grad = False


@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class LinearNonlinearMixingModel(NonlinearMixingModel):
    """Linear followed by nonlinear mixing transformation"""

    def init_parameters(self):
        linear = nn.Linear(self.input_dim, self.output_dim, bias=False)
        nonlinear = self._create_invertible_mlp(self.output_dim)
        self.mix = nn.Sequential(linear, nonlinear)

        # Freeze parameters
        for p in self.mix.parameters():
            p.requires_grad = False
