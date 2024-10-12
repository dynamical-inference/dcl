import math
import random
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import List, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class RotationSampler(Configurable, ABC):
    """Base class for sampling rotation matrices."""

    verbose: bool = field(default=False)

    def sample(self, dim: int) -> Float[Tensor, "dim dim"]:
        """Sample a random rotation matrix."""
        assert dim >= 2, "Dimension must be at least 2"
        rotation_matrix = self._sample(dim)
        self._verify_rotation_matrix(rotation_matrix)
        return rotation_matrix

    @abstractmethod
    def _sample(self, dim: int) -> Float[Tensor, "dim dim"]:
        """Sample a random rotation matrix."""

    def _verify_rotation_matrix(self, matrix: Tensor) -> None:
        """Verify that the matrix is indeed a rotation matrix."""
        dim = matrix.shape[-1]
        # Check orthogonality: R^T R = I
        identity = torch.eye(dim, device=matrix.device)
        is_orthogonal = torch.allclose(matrix @ matrix.T, identity, atol=1e-6)
        assert is_orthogonal, "M@M.T is not the identity matrix, matrix is not orthogonal"

        # Check determinant is 1 (proper rotation)
        det = torch.det(matrix)
        is_proper = torch.allclose(det, torch.tensor(1.0), atol=1e-6)
        assert is_proper, f"Matrix determinant is not 1 (got {det}), matrix is not a proper rotation"


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class MinMaxRotationSampler(RotationSampler):
    """
    Samples rotation matrices by applying random rotations to all pairs of dimensions.
    For each dimension pair, samples an angle between min_angle and max_angle degrees
    and creates a rotation matrix. The final matrix is the product of all indivudal 2d-rotations.
    """

    min_angle: Union[int, float] = config_field(
        default=0, help="Minimum rotation angle in degrees")
    max_angle: Union[int, float] = config_field(
        default=5, help="Maximum rotation angle in degrees")

    def __post_init__(self):
        super().__post_init__()
        assert self.max_angle >= self.min_angle, "max_angle must be >= min_angle"

    @jaxtyped(typechecker=typechecked)
    def _sample(self, dim: int) -> Float[Tensor, "dim dim"]:
        """Sample a random rotation matrix."""
        W = torch.eye(dim)
        thetas = []

        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                # Random angle between min_angle and max_angle
                theta_degree = random.uniform(self.min_angle, self.max_angle)
                # Random direction of rotation
                if random.choice([True, False]):
                    theta_degree = -theta_degree

                if self.verbose:
                    print(f"rotating {d1} and {d2} by {theta_degree} degrees")

                # Convert to radians and create rotation matrix
                theta = math.radians(theta_degree)
                Wrot = torch.eye(dim)
                Wrot[d1, d1] = math.cos(theta)
                Wrot[d1, d2] = -math.sin(theta)
                Wrot[d2, d1] = math.sin(theta)
                Wrot[d2, d2] = math.cos(theta)

                W = W @ Wrot
                thetas.append(theta_degree)

        if self.verbose:
            print(
                f"Created random matrix with mean absolute angle of {np.mean(np.abs(thetas)):.2f} degrees"
            )
        return W


@jaxtyped(typechecker=typechecked)
@dataclass(unsafe_hash=True, eq=False, kw_only=True)
class SpecifiedRotationSampler(RotationSampler):
    """
    Creates rotation matrices from specified rotations between dimension pairs.

    The rotations can be specified in two ways:

    1. As a list of (dim1, dim2, angle) tuples, where:
       - dim1, dim2: Indices of the dimensions to rotate between (must be different)
       - angle: Rotation angle in degrees (positive = counterclockwise)
       Example: [(0,1,30), (1,2,-45)] rotates dims 0-1 by 30° then dims 1-2 by -45°

    2. As a list of angles for consecutive dimensions:
       - Each angle rotates dimension i with (i+1)%dim
       - Angles are in degrees
       Example: [30, 45] with dim=3 rotates dims 0-1 by 30° then dims 1-2 by 45°
    """

    rotations: List[Union[
        Tuple[int, int, float],
        float,
    ]] = config_field(
        default_factory=list,
        help=
        "List of rotations, either as (dim1, dim2, angle) tuples or angles for consecutive dims"
    )

    @jaxtyped(typechecker=typechecked)
    def _sample(self, dim: int) -> Float[Tensor, "dim dim"]:
        """Create rotation matrix from specified rotations."""
        rotation_matrix = torch.eye(dim)

        # Convert list of angles to list of tuples if needed
        if isinstance(self.rotations[0], (float, int)):
            rotations = []
            for i, degree in enumerate(self.rotations):
                rotations.append((i, (i + 1) % self.dim, float(degree)))
        else:
            rotations = [(a, b, float(angle)) for a, b, angle in self.rotations]

        if self.verbose:
            print("Using rotations:", rotations)

        # Apply each rotation
        for a, b, angle in rotations:
            if a >= dim or b >= dim or a == b:
                raise ValueError(f"Invalid rotation plane dimensions: {a}, {b}")

            # Create rotation matrix for this plane
            theta = math.radians(angle)
            Wrot = torch.eye(self.dim)
            Wrot[a, a] = math.cos(theta)
            Wrot[a, b] = -math.sin(theta)
            Wrot[b, a] = math.sin(theta)
            Wrot[b, b] = math.cos(theta)

            rotation_matrix = rotation_matrix @ Wrot

        return rotation_matrix
