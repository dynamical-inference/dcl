import pytest
import torch
from dcl.metrics.utils import linear_assignment


def test_identical_matrices():
    """Test linear_assignment with identical matrices."""
    # Create identical matrices
    matrices = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],
                             [[9.0, 10.0], [11.0, 12.0]]])

    # Expected result: identity mapping since matrices are identical
    matrix_hat_ordered, mapping = linear_assignment(matrices, matrices)

    # Check if the ordered matrices are the same as input (identity mapping)
    assert torch.allclose(matrix_hat_ordered, matrices)

    # Check if mapping is identity (0->0, 1->1, 2->2)
    assert mapping == {0: 0, 1: 1, 2: 2}


def test_permuted_matrices():
    """Test linear_assignment with permuted matrices."""
    # Create original matrices
    matrices_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]],
                                  [[9.0, 10.0], [11.0, 12.0]]])

    # Create permuted matrices (0,1,2 -> 2,0,1)
    matrices_hat = torch.tensor([
        [[9.0, 10.0], [11.0, 12.0]],  # original index 2
        [[1.0, 2.0], [3.0, 4.0]],  # original index 0
        [[5.0, 6.0], [7.0, 8.0]]  # original index 1
    ])

    matrix_hat_ordered, mapping = linear_assignment(matrices_true, matrices_hat)

    # Check if the ordered matrices match the original
    assert torch.allclose(matrix_hat_ordered, matrices_true)

    # Check mapping: estimated index -> true index
    assert mapping == {0: 2, 1: 0, 2: 1}


def test_slightly_modified_matrices():
    """Test linear_assignment with slightly modified matrices."""
    # Create original matrices
    matrices_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]],
                                  [[9.0, 10.0], [11.0, 12.0]]])

    # Create slightly modified, permuted matrices with small noise
    matrices_hat = torch.tensor([
        [[9.1, 10.1], [11.1, 12.1]],  # original index 2 + noise
        [[1.1, 2.1], [3.1, 4.1]],  # original index 0 + noise
        [[5.1, 6.1], [7.1, 8.1]]  # original index 1 + noise
    ])

    matrix_hat_ordered, mapping = linear_assignment(matrices_true, matrices_hat)

    # The ordered matrices should be close to the original but with the noise
    expected_ordered = torch.tensor([[[1.1, 2.1], [3.1, 4.1]],
                                     [[5.1, 6.1], [7.1, 8.1]],
                                     [[9.1, 10.1], [11.1, 12.1]]])

    assert torch.allclose(matrix_hat_ordered, expected_ordered)
    assert mapping == {1: 0, 2: 1, 0: 2}


def test_single_matrix():
    """Test linear_assignment with single matrix (num_modes=1)."""
    # Create single matrix case
    matrix_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    matrix_hat = torch.tensor([[[1.1, 2.1], [3.1, 4.1]]])

    matrix_hat_ordered, mapping = linear_assignment(matrix_true, matrix_hat)

    assert matrix_hat_ordered.shape == matrix_true.shape
    assert mapping == {0: 0}


def test_large_matrices():
    """Test linear_assignment with larger matrices."""
    # Create larger matrices (more modes and dimensions)
    n_modes = 5
    dim = 4

    # Create random matrices but with clear correspondence
    matrices_true = torch.randn(n_modes, dim, dim)

    # Create a permutation
    perm = torch.randperm(n_modes)

    # Apply permutation to create matrices_hat
    matrices_hat = matrices_true[perm].clone()

    matrix_hat_ordered, mapping = linear_assignment(matrices_true, matrices_hat)

    # Check if the ordered matrices match the original
    assert torch.allclose(matrix_hat_ordered, matrices_true)

    # Verify mapping
    for hat_idx, true_idx in mapping.items():
        assert torch.allclose(matrices_hat[hat_idx], matrices_true[true_idx])


def test_nearly_identical_matrices():
    """Test with matrices that differ only by a small amount."""
    matrices_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])

    # Just slightly different
    matrices_hat = torch.tensor([
        [[5.0, 6.0], [7.0, 8.0]],
        [[1.0, 2.0001], [3.0, 4.0]]  # Just slightly different
    ])

    matrix_hat_ordered, mapping = linear_assignment(matrices_true, matrices_hat)

    # Should still find the correct mapping despite the tiny difference
    assert mapping == {0: 1, 1: 0}


def test_very_different_matrices():
    """Test with matrices that are completely different."""
    matrices_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])

    # Completely different values
    matrices_hat = torch.tensor([[[100.0, 200.0], [300.0, 400.0]],
                                 [[500.0, 600.0], [700.0, 800.0]]])

    matrix_hat_ordered, mapping = linear_assignment(matrices_true, matrices_hat)

    # Should still find a mapping based on distance
    assert len(mapping) == 2


def test_different_dtypes():
    """Test linear_assignment with different data types."""
    # Float32
    matrices_true_f32 = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=torch.float32)

    matrices_hat_f32 = torch.tensor(
        [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]],
        dtype=torch.float32)

    # Float64
    matrices_true_f64 = matrices_true_f32.to(torch.float64)
    matrices_hat_f64 = matrices_hat_f32.to(torch.float64)

    # Test with float32
    _, mapping_f32 = linear_assignment(matrices_true_f32, matrices_hat_f32)

    # Test with float64
    _, mapping_f64 = linear_assignment(matrices_true_f64, matrices_hat_f64)

    # Both should yield the same mapping
    assert mapping_f32 == mapping_f64


def test_error_different_shapes():
    """Test linear_assignment with matrices of different shapes."""
    # Different number of modes
    matrices_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])

    matrices_hat = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                 [[5.0, 6.0], [7.0, 8.0]],
                                 [[9.0, 10.0], [11.0, 12.0]]])

    # Should raise an error (different number of modes)
    with pytest.raises(expected_exception=TypeError):
        linear_assignment(matrices_true, matrices_hat)

    # Different inner dimensions
    matrices_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                  [[5.0, 6.0], [7.0, 8.0]]])

    matrices_hat = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                                  [7.0, 8.0, 9.0]],
                                 [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0],
                                  [16.0, 17.0, 18.0]]])

    # Check error is raised for matrices with different inner dimensions
    with pytest.raises(expected_exception=TypeError):
        linear_assignment(matrices_true, matrices_hat)


def test_zero_matrices():
    """Test linear_assignment with matrices of zeros."""
    # Create zero matrices
    matrices_true = torch.zeros(3, 2, 2)
    matrices_hat = torch.zeros(3, 2, 2)

    matrix_hat_ordered, mapping = linear_assignment(matrices_true, matrices_hat)

    # With all zero matrices, the assignment should be identity
    assert mapping == {0: 0, 1: 1, 2: 2}
    assert torch.allclose(matrix_hat_ordered, matrices_true)


def test_random_matrices():
    """Test linear_assignment with random matrices and verify properties."""
    # Generate random matrices
    n_modes = 4
    dim = 3
    matrices_true = torch.randn(n_modes, dim, dim)
    matrices_hat = torch.randn(n_modes, dim, dim)

    matrix_hat_ordered, mapping = linear_assignment(matrices_true, matrices_hat)

    # Check key properties
    assert matrix_hat_ordered.shape == matrices_true.shape
    assert len(mapping) == n_modes

    # Verify mapping is one-to-one
    true_indices = set(mapping.values())
    hat_indices = set(mapping.keys())
    assert len(true_indices) == n_modes
    assert len(hat_indices) == n_modes
