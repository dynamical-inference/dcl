import random
import string
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dcl.datasets.base import BaseDataset
from dcl.datasets.splits import DatasetSplit
from dcl.loader import BaseDataLoader
from dcl.solver import BaseSolver
from dcl.utils.checkpoints import CheckpointSavingCallback
from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field
from dcl.utils.jaxtyping import disable_jaxtyping as disable_jaxtyping_context


def unique_timestamp():
    # Reset random seed to ensure unique timestamps
    random.seed()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = ''.join(
        random.choices(string.ascii_lowercase + string.digits, k=6))
    return timestamp + '_' + unique_id


@dataclass(kw_only=True)
class Experiment(Configurable):
    solver: BaseSolver = config_field(default_factory=BaseSolver)
    dataset: BaseDataset = config_field(default_factory=BaseDataset)
    train_loader: BaseDataLoader = config_field(default_factory=BaseDataLoader)
    dataset_split: Optional[DatasetSplit] = config_field(default=None)
    val_loader: Optional[BaseDataLoader] = config_field(default=None)
    test_loader: Optional[BaseDataLoader] = config_field(default=None)
    version: int = config_field(default=1)

    # Logging parameters
    base_dir: str = field(default="logs")
    name: str = field(default="experiments")
    time_stamp: str = field(default_factory=unique_timestamp)

    def __post_init__(self):
        if self.val_loader is None:
            self.val_loader = self.train_loader.clone()
        if self.test_loader is None:
            self.test_loader = self.train_loader.clone()
        super().__post_init__()

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if self.dataset_split is not None:
            (
                self.train_data,
                self.val_data,
                self.test_data,
            ) = self.dataset_split.create_split(self.dataset)
            self.train_loader.lazy_init(self.train_data)
            self.val_loader.lazy_init(self.val_data)
            self.test_loader.lazy_init(self.test_data)
        else:
            self.train_loader.lazy_init(self.dataset)
            self.val_loader.lazy_init(self.dataset)
            self.test_loader.lazy_init(self.dataset)

    @property
    def log_dir(self):
        return Path(self.base_dir) / self.name / self.time_stamp

    def run(
        self,
        *,
        save_hook: Optional[CheckpointSavingCallback] = None,
        eval_frequency: Optional[int] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
        disable_jaxtyping: bool = False,
    ):
        # override the save_hook root_dir to the log_dir
        if save_hook is not None:
            save_hook.root_dir = self.log_dir

        def fit():
            return self.solver.fit(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                save_hook=save_hook,
                eval_frequency=eval_frequency,
                eval_kwargs=eval_kwargs,
            )

        if disable_jaxtyping:
            with disable_jaxtyping_context():
                fit()
        else:
            fit()

        self.save(self.log_dir)

    def load_checkpoint(self, ckpt_path: Path):
        self.solver = self.solver.__class__.load(ckpt_path)

    def evaluate_train(self, **kwargs) -> Dict[str, Any]:
        return self.solver.evaluate(self.train_loader, **kwargs)

    def evaluate_val(self, **kwargs) -> Dict[str, Any]:
        return self.solver.evaluate(self.val_loader, **kwargs)

    def evaluate_test(self, **kwargs) -> Dict[str, Any]:
        return self.solver.evaluate(self.test_loader, **kwargs)

    def evaluate(
        self,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Evaluate the model on train, val, and test datasets."""
        train_metrics = self.evaluate_train(**kwargs)
        val_metrics = self.evaluate_val(**kwargs)
        test_metrics = self.evaluate_test(**kwargs)
        return train_metrics, val_metrics, test_metrics
