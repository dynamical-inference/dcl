import torch


def expand_dim(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """
    Expand a tensor along a specific dimension.
    """
    shape = list(tensor.shape)  # Get current shape
    shape[dim] = size  # Modify only the specified dimension
    return tensor.expand(*shape)
