from comb import sweep
from tqdm import tqdm

from dcl.datajoint import schema
from dcl.datajoint.utils import get_experiment_configs_for_sweep
from dcl.experiments.experiments import Experiment


def flatten_dict(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='__'):
    result_dict = {}
    for k, v in d.items():
        parts = k.split(sep)
        d = result_dict
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = v
    return result_dict


class BaseSweep(sweep.Sweep):

    def __init__(self):
        super().__init__(format="dict")

    def get_fixed_args(self) -> dict:
        """
        The sweep.Sweep class can't handle nested dictionaries, so we need to flatten it
        """
        return flatten_dict(self.get_fixed_args_nested(), sep="__")

    def __iter__(self):
        """
        We need to convert back to nested dict, after the sweep.Sweep class has done its thing
        """
        for flat_dict in super().__iter__():
            yield unflatten_dict(flat_dict, sep="__")

    def get_fixed_args_nested(self) -> dict:
        return self.get_experiments().to_dict()

    def get_experiments(self) -> Experiment:
        raise NotImplementedError(
            "get_fixed_args_config must be implemented in subclass")

    @property
    def script(self):
        return None

    def __len__(self):
        count = 0
        for _ in iter(self):
            count += 1
        return count


class TrainConfigSweep(BaseSweep):
    sweep_name: str = "Base Train Sweep"
    save_step: int = 1000

    def __init__(self):
        super().__init__()

    def get_fixed_args(self):
        kwargs = super().get_fixed_args()
        if "sweep_name" not in kwargs.keys() and self.sweep_name is not None:
            kwargs['sweep_name'] = self.sweep_name
        return kwargs


def reset_sweeps(existing_configs, class_sweep_name):
    print(
        f"Resetting sweep '{class_sweep_name}'... Deleting {len(existing_configs)} configs"
    )
    # before deleting, let's check if some of them were already trained
    existing_checkpoints = schema.ExperimentTable.Checkpoint(
    ) & existing_configs
    if len(existing_checkpoints) > 0:
        print(
            f"Warning: There exists {len(existing_checkpoints)} checkpoints associated with this sweep. "
        )
        confirmation = input(
            "Are you sure you want to delete these configs and their results? [y/N] "
        )
        if confirmation.lower() != 'y':
            print("Aborting sweep reset")
            return

    existing_configs.delete()


def populate_exp(trainconfig_sweep: BaseSweep,
                 reset_sweep: bool = False,
                 unsafe: bool = False,
                 test_without_dj=False):
    class_sweep_name = trainconfig_sweep.sweep_name

    existing_configs = get_experiment_configs_for_sweep(
        sweep_name=class_sweep_name,
        partial=False,
    )

    if reset_sweep:
        reset_sweeps(existing_configs, class_sweep_name)

    n_experiments_start = len(schema.ExperimentConfigTable())
    num_configs = len(trainconfig_sweep)
    print(
        f"Start populating {num_configs} configurations for sweep '{class_sweep_name}'"
    )
    print(
        f"Sweep '{class_sweep_name}' alredy contains {len(existing_configs)} configs"
    )

    for trainconfig in tqdm(trainconfig_sweep, total=num_configs):
        sweep_name = trainconfig.pop("sweep_name")
        assert sweep_name == class_sweep_name

        if unsafe:
            # directly use the trainconfig dict
            # This is unsafe, but may be significantly faster as we don't need to instantiate each experiment
            the_config_dict = trainconfig
        else:
            # we instantiate the experiment to make use of the type checking to validate the config is correct
            experiment = Experiment.from_dict(trainconfig, lazy=True)
            the_config_dict = experiment.to_dict()

        if not test_without_dj:
            schema.ExperimentConfigTable().insert_from_config(
                the_config_dict,
                sweep_name=class_sweep_name,
                skip_duplicates=True,
                row_dict=dict(save_step=trainconfig_sweep.save_step,))

            n_experiments = len(
                schema.ExperimentConfigTable()) - n_experiments_start
            print(
                f"Finished populating {n_experiments} experiments for sweep '{class_sweep_name}'"
            )
        else:
            n_experiments = -99

    return n_experiments
