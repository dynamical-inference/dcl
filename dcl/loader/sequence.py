from dataclasses import dataclass
from dataclasses import field
from typing import Dict, Iterator, List

import torch
from torch import Tensor

from dcl.loader import BaseDataLoader
from dcl.utils.configurable import check_initialized
from dcl.utils.datatypes import config_field


@dataclass(kw_only=True)
class SequenceDataLoader(BaseDataLoader):
    """Data loader that yields batches of sequences from a dataset.
    Used for training models that take a sequence of datapoints as input (i.e. dynamics models).
    """

    shuffle: bool = config_field(default=True)
    indices: Tensor = field(default_factory=lambda: torch.tensor([]),
                            init=False,
                            repr=False)

    sequence_length: int = config_field(default=1)

    def build_sequence_index(self) -> Tensor:
        """Build a sequence index for the dataset."""
        sequence_index = torch.arange(len(
            self.dataset)).unsqueeze(1) - torch.arange(
                self.sequence_length).unsqueeze(0)
        # filter out negative indices
        sequence_index = sequence_index[(sequence_index >= 0).any(dim=1)]
        return sequence_index

    @check_initialized
    def reset(self):
        """Reset the data loader."""
        sequence_index = self.build_sequence_index()
        if self.shuffle:
            permutation_index = torch.randperm(len(sequence_index))
            self.indices = sequence_index[permutation_index]
        else:
            self.indices = sequence_index

    @check_initialized
    def __len__(self) -> int:
        """Return number of iterations per epoch."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @check_initialized
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return iterator over batches."""
        self.reset()

        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self._get_batch_indices(start_idx)
            batch = self.load_batch(batch_indices)
            batch = self.batch_to_device(batch)
            yield batch

    def _get_batch_indices(self, start_idx: int) -> Tensor:
        """Get the indices for the next batch."""
        return self.indices[start_idx:start_idx + self.batch_size]

    def collate_fn(
            self, batch: List[Dict[str,
                                   torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Default collate function that stacks tensors."""
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}

    @check_initialized
    def load_batch(self, indices: Tensor) -> Dict[str, torch.Tensor]:
        """Load a batch of data from the dataset."""
        if hasattr(self.dataset, "__getitems__"):
            batch = self.dataset.__getitems__(indices)
        else:
            batch = [self.dataset[idx] for idx in indices]
            batch = self.collate_fn(batch)
        return batch

    def batch_to_device(
            self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move the batch to the specified device."""
        return {k: v.to(self.device) for k, v in batch.items()}
