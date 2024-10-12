from dcl.datajoint.schema import DataLoaderConfigTable
from dcl.datajoint.schema import DatasetConfigTable
from dcl.datajoint.schema import ExperimentConfigTable
from dcl.datajoint.schema import ExperimentTable
from dcl.datajoint.schema import ModelConfigTable
from dcl.datajoint.schema import SolverConfigTable
from dcl.datajoint.schema import SweepExperimentTable
from dcl.datajoint.schema import SweepTable
from dcl.datajoint.schema import TestMetricsTable
from dcl.datajoint.schema import TrainMetricsTable
from dcl.datajoint.schema import ValMetricsTable


### Here we introduce some short hands for common queries
def get_all_solvers():
    return (SolverConfigTable() * (ModelConfigTable().proj_all(prefix="model")))


def join_experiments_configs(experiment_configs: ExperimentConfigTable):
    solvers = get_all_solvers()

    return (experiment_configs * solvers.proj_all(prefix="solver") *
            (DatasetConfigTable().proj_all(prefix="dataset")) *
            (DataLoaderConfigTable().proj_all(prefix="train_loader")) *
            (DataLoaderConfigTable().proj_all(prefix="val_loader")) *
            (DataLoaderConfigTable().proj_all(prefix="test_loader")))


def get_all_joined_experiments_configs():
    return join_experiments_configs(ExperimentConfigTable())


def get_all_checkpoints():
    return (ExperimentTable.Checkpoint() * ExperimentTable() *
            get_all_joined_experiments_configs())


def get_all_metrics():
    train_metrics = TrainMetricsTable().proj_all(prefix="train",
                                                 skip_primary_key=True)
    val_metrics = ValMetricsTable().proj_all(prefix="val",
                                             skip_primary_key=True)
    test_metrics = TestMetricsTable().proj_all(prefix="test",
                                               skip_primary_key=True)
    return train_metrics * val_metrics * test_metrics


def get_all_checkpoint_metrics(last_checkpoint: bool = True):
    checkpoint_metrics = get_all_metrics() * ExperimentTable.Checkpoint()
    if last_checkpoint:
        checkpoint_metrics = checkpoint_metrics & dict(
            checkpoint_name="last_checkpoint")
    return checkpoint_metrics


def get_experiment_results(last_checkpoint: bool = True):
    return get_all_checkpoint_metrics(last_checkpoint) * ExperimentTable()


def get_sweep_partial(partial_name: str):
    # there's no partial match in datajoint, so we need to do it with pandas
    sweeps = SweepTable()
    sweeps_df = sweeps.fetch(format="frame").reset_index()
    sweeps_df = sweeps_df[sweeps_df["name"].str.contains(partial_name)]
    sweep_restriction = [
        dict(name=sweep_name) for sweep_name in sweeps_df["name"]
    ]
    return SweepTable() & sweep_restriction


def get_experiment_configs_for_sweep(sweep_name: str, partial: bool = True):
    if partial:
        sweep_restriction = get_sweep_partial(sweep_name).proj(sweep="name")
    else:
        sweep_restriction = dict(sweep=sweep_name)

    sweeps = SweepExperimentTable().proj(
        arg_hash="experiment") & sweep_restriction
    return (ExperimentConfigTable() & sweeps)
