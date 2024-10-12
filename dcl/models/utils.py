import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dcl.utils.datatypes import config_field


@dataclass(kw_only=True)
class ModelStorageMixin:

    seed: int = config_field(default=123)

    def __lazy_post_init__(self):
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    @property
    def state_dict_name(self) -> str:
        return f"{self.__class__.config_prefix()}_state_dict.pt"

    def _save_additional(
        self,
        path: Path,
    ):
        """Save model config and state_dict to disk.

        Args:
            path: Path to save directory
        """
        path.mkdir(parents=True, exist_ok=True)
        # Save weights
        state_dict_path = path / self.state_dict_name
        torch.save(self.state_dict(), state_dict_path)

    def _load_additional(
        self,
        path: Path,
    ):
        """Load model config and state_dict from disk.

        Args:
            path: Path to model directory containing state_dict.pt
        """

        # Load weights
        state_dict_path = path / self.state_dict_name
        self.load_state_dict(torch.load(state_dict_path, weights_only=True))
