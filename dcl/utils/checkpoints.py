from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from dcl.solver import BaseSolver


@dataclass(kw_only=True)
class CheckpointSavingCallback():

    save_step: int = 1000
    root_dir: Optional[Union[str, Path]] = None
    base_dir: str = "checkpoints"
    checkpoint_dir_template: str = "checkpoint_{step:#07d}"

    def should_save(self, step: int) -> bool:
        return step % self.save_step == 0

    def build_path(
        self,
        step: Optional[int] = None,
        ckpt_dir_name: Optional[str] = None,
    ) -> Path:
        if self.root_dir is None:
            checkpoints_dir = Path(self.base_dir)
        else:
            checkpoints_dir = Path(self.root_dir) / self.base_dir

        # either step or ckpt_dir_name must be provided
        if step is None and ckpt_dir_name is None:
            raise ValueError("Either step or ckpt_dir_name must be provided")

        if ckpt_dir_name is None:
            ckpt_dir_name = self.checkpoint_dir_template.format(step=step)
        ckpt_path = checkpoints_dir / ckpt_dir_name
        return ckpt_path

    def __call__(
        self,
        *,
        solver: BaseSolver,
        step: Optional[int] = None,
        ckpt_dir_name: Optional[str] = None,
    ):
        ckpt_path = self.build_path(step=step, ckpt_dir_name=ckpt_dir_name)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        solver.save(path=ckpt_path)
        print(f"{ckpt_path} saved!")
        return ckpt_path

    def maybe_save(self,
                   *,
                   solver: "BaseSolver",
                   step: int,
                   ckpt_dir_name: Optional[str] = None):
        if self.should_save(step):
            self.__call__(solver=solver, step=step, ckpt_dir_name=ckpt_dir_name)


@dataclass(kw_only=True)
class DJCheckpointSavingCallback(CheckpointSavingCallback):
    dj_callback: Callable[[Path], None] = None

    def __post_init__(self):
        if self.dj_callback is None:
            raise ValueError("dj_callback must be provided")

    def __call__(self,
                 *,
                 solver: BaseSolver,
                 step: Optional[int] = None,
                 ckpt_dir_name: Optional[str] = None):
        ckpt_path = super().__call__(solver=solver,
                                     step=step,
                                     ckpt_dir_name=ckpt_dir_name)
        self.dj_callback(ckpt_path)
        return ckpt_path
