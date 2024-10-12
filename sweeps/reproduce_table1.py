from comb.types import zipped
from dcl.criterions.contrastive import MseInfoNCE
from dcl.datajoint.sweep import populate_exp
from dcl.datajoint.sweep import TrainConfigSweep
from dcl.datasets.synthetic import GenericDynamicsDataset
from dcl.datasets.timeseries import TensorDatasetWithLatentsFromFile
from dcl.distributions.time_distributions import OffsetTimeDistribution
from dcl.distributions.time_distributions import \
    UniformDiscreteTimeDistribution
from dcl.experiments.experiments import Experiment
from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.slds import GumbelSLDS
from dcl.models.dynamics.switching_dynamics import MSESwitchingModel
from dcl.models.dynamics.utils import IdentityLDSParameters
from dcl.models.encoder import Offset1ModelMLP
from dcl.solver.contrastive_solver import SLDSContrastiveLearningSolver
from dcl.solver.optimizer import DCLAdamOptimizer

# This file reproduces Table 1 from the paper. It defines the datasets, models, and solvers
# necessary to reproduce the results.


class BaseSweep(TrainConfigSweep):
    sweep_name = "Table 1"

    def __iter__(self):
        for args in super().__iter__():
            yield args


class DatasetConfig():

    @property
    def name(self) -> str:
        return "Dataset"

    def get_config(self) -> GenericDynamicsDataset:
        raise NotImplementedError


#####################################
########### DATASETS ################
#####################################


class DatasetConfigFromFile(DatasetConfig):

    def __init__(
        self,
        dataset_file: str,
    ):
        self.dataset_file = dataset_file

    @property
    def name(self) -> str:
        return "DatasetFromFile"

    def get_config(self):
        dataset = TensorDatasetWithLatentsFromFile(data_path=self.dataset_file)
        return dataset


###################################
########### SOLVERS ###############
###################################


def create_loader_config(
    num_iterations: int,
    batch_size: int,
    batch_size_neg: int,
):
    return DiscreteTimeContrastiveDataLoader(
        num_iterations=num_iterations,
        batch_size=batch_size,
        batch_size_neg=batch_size_neg,
        reference_distribution=UniformDiscreteTimeDistribution(),
        positive_distribution=OffsetTimeDistribution(offset=1),
        negative_distribution=UniformDiscreteTimeDistribution(),
    )


def create_encoder_config(
    input_dim: int,
    output_dim: int,
    seed: int,
):
    return Offset1ModelMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=180,
        num_layers=3,
        normalize=False,
        seed=seed,
    )


def create_slds_with_init_identity_dynamics_model_config(
    dim: int,
    num_systems: int,
):
    return GumbelSLDS(
        linear_dynamics=LinearDynamicsModel(
            dim=dim,
            num_systems=num_systems,
            initializer=IdentityLDSParameters(),
        ),
        switching_model=MSESwitchingModel(num_modes=num_systems,),
    )


class SolverSweep(BaseSweep):

    def __init__(self, dataset_config: DatasetConfig, prefix: str):
        super().__init__()
        self.dataset_config = dataset_config.get_config()
        self.sweep_name = super(
        ).sweep_name + f" {dataset_config.name} {prefix}Solver"


class DynamicsSLDSSolver(SolverSweep):

    def __init__(
            self,
            dataset_config,
            encoder_learning_rate: float,
            dynamics_learning_rate: float,
            num_iterations: int,
            batch_size: int,
            batch_size_neg: int,
            num_systems: int,
            prefix: str = 'SLDS',  # Add prefix argument with default
    ):

        super().__init__(dataset_config=dataset_config, prefix=prefix)
        self.encoder_learning_rate = encoder_learning_rate
        self.dynamics_learning_rate = dynamics_learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.batch_size_neg = batch_size_neg
        self.num_systems = num_systems

    def get_experiments(self) -> Experiment:

        dynamics_model = create_slds_with_init_identity_dynamics_model_config(
            dim=self.dataset_config.latent_dim,
            num_systems=self.num_systems,
        )

        solver = SLDSContrastiveLearningSolver(
            freeze_dynamics_model=False,
            model=create_encoder_config(
                input_dim=self.dataset_config.observed_dim,
                output_dim=self.dataset_config.latent_dim,
                seed=42,
            ),
            dynamics_model=dynamics_model,
            optimizer=DCLAdamOptimizer(
                encoder_learning_rate=self.encoder_learning_rate,
                dynamics_learning_rate=self.dynamics_learning_rate),
            criterion=MseInfoNCE(
                temperature=1.0,
                infonce_type="infonce_full_denominator",
            ),
        )
        solver.seed = zipped(42, 165, 789)
        solver.model.seed = zipped(3242, 43534, 110534)

        #NOTE: with pacakged datasets from Table 1, we don't split the dataset.
        exp = Experiment(
            dataset=self.dataset_config,
            solver=solver,
            train_loader=create_loader_config(
                num_iterations=self.num_iterations,
                batch_size=self.batch_size,
                batch_size_neg=self.batch_size_neg,
            ),
            version=0,
            name=self.sweep_name,
            lazy=False,
        )
        return exp


class IdentityDynamicsSLDSSolver(DynamicsSLDSSolver):

    def __init__(
        self,
        dataset_config: DatasetConfig,
        encoder_learning_rate: float,
        dynamics_learning_rate: float,
        num_iterations: int,
        batch_size: int,
        batch_size_neg: int,
        num_systems: int,
    ):
        super().__init__(
            dataset_config=dataset_config,
            prefix="Identity-SLDS",
            encoder_learning_rate=encoder_learning_rate,
            dynamics_learning_rate=dynamics_learning_rate,
            num_iterations=num_iterations,
            batch_size=batch_size,
            batch_size_neg=batch_size_neg,
            num_systems=num_systems,
        )

    def get_experiments(self) -> Experiment:
        # identity dynamics, freeze_dynamics_model = True
        # NOTE: DynamicsSLDSSolver already has identity dynamics for initalization,
        # so we only need to freeze the dynamics model.
        exp = super().get_experiments()
        exp.solver.freeze_dynamics_model = True
        return exp


if __name__ == "__main__":
    try:
        reset = False
        basepath = ""

        identity_paths = [
            f"{basepath}/lds_seed13328743_noise0.01_angle0.pt",
            f"{basepath}/lds_seed23123_noise0.01_angle0.pt",
            f"{basepath}/lds_seed349237_noise0.01_angle0.pt"
        ]

        identity_configs = [
            DatasetConfigFromFile(dataset_file=path) for path in identity_paths
        ]

        lds_paths = [
            f"{basepath}/lds_seed13328743_noise0.01_angle5.pt",
            f"{basepath}/lds_seed23123_noise0.01_angle5.pt",
            f"{basepath}/lds_seed349237_noise0.01_angle5.pt",
            f"{basepath}/lds_seed13328743_noise0.01_angle2.pt",
            f"{basepath}/lds_seed23123_noise0.01_angle2.pt",
            f"{basepath}/lds_seed349237_noise0.01_angle2.pt",
        ]

        lds_configs = [
            DatasetConfigFromFile(dataset_file=path) for path in lds_paths
        ]

        slds_paths = [
            f"{basepath}/slds_seed789_latentdim6_noise0.0001_maxangle10_numtrials1000_numsamples1000_nummodes5.pt",
            f"{basepath}/slds_seed849_latentdim6_noise0.0001_maxangle10_numtrials1000_numsamples1000_nummodes5.pt",
            f"{basepath}/slds_seed953_latentdim6_noise0.0001_maxangle10_numtrials1000_numsamples1000_nummodes5.pt",
        ]

        slds_configs = [
            DatasetConfigFromFile(dataset_file=path) for path in slds_paths
        ]

        lorenz_paths = [
            f"{basepath}/lorenz_dataset_julich_seed13328743_dt0.0005_noise0.1_init0.0_1.0_0.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed13328743_dt0.01_noise0.001_init0.0_1.0_0.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed23123_dt0.0005_noise0.1_init1.0_0.0_0.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed23123_dt0.01_noise0.001_init1.0_0.0_0.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed234233_dt0.0005_noise0.1_init2.0_2.0_2.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed234233_dt0.01_noise0.001_init2.0_2.0_2.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed349237_dt0.0005_noise0.1_init0.0_0.0_1.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed349237_dt0.01_noise0.001_init0.0_0.0_1.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed98234_dt0.0005_noise0.1_init3.0_3.0_3.0.pt",
            f"{basepath}/lorenz_dataset_julich_seed98234_dt0.01_noise0.001_init3.0_3.0_3.0.pt",
        ]

        lorenz_configs = [
            DatasetConfigFromFile(dataset_file=path) for path in lorenz_paths
        ]

        ### SWEEPS ###

        sweeps = []

        lds_solver_params = {
            "encoder_learning_rate": 3e-4,
            "dynamics_learning_rate": 3e-4,
            "num_iterations": 30_000,
            "batch_size": 2048,
            "batch_size_neg": 20_000,
        }

        slds_solver_params = {
            "encoder_learning_rate": 1e-3,
            "dynamics_learning_rate": 1e-2,
            "num_iterations": 50_000,
            "batch_size": 2048,
            "batch_size_neg": 2**16,
        }

        slds_lorenz_solver_params = {
            "encoder_learning_rate": 3e-4,
            "dynamics_learning_rate": 3e-4,
            "num_iterations": 30_000,
            "batch_size": 2048,
            "batch_size_neg": 20_000,
        }

        for identity_config in identity_configs:
            sweeps.extend([
                DynamicsSLDSSolver(dataset_config=identity_config,
                                   **lds_solver_params,
                                   num_systems=1),
                IdentityDynamicsSLDSSolver(dataset_config=identity_config,
                                           **lds_solver_params,
                                           num_systems=1),
            ])

        for lds_config in lds_configs:
            sweeps.extend([
                DynamicsSLDSSolver(dataset_config=lds_config,
                                   **lds_solver_params,
                                   num_systems=1),
                IdentityDynamicsSLDSSolver(dataset_config=lds_config,
                                           **lds_solver_params,
                                           num_systems=1),
            ])

        for slds_config in slds_configs:
            sweeps.extend([
                DynamicsSLDSSolver(dataset_config=slds_config,
                                   **slds_solver_params,
                                   num_systems=5),
                IdentityDynamicsSLDSSolver(dataset_config=slds_config,
                                           **slds_solver_params,
                                           num_systems=5),
            ])

        for lorenz_config in lorenz_configs:
            sweeps.extend([
                IdentityDynamicsSLDSSolver(dataset_config=lorenz_config,
                                           **slds_lorenz_solver_params,
                                           num_systems=1),
                DynamicsSLDSSolver(dataset_config=lorenz_config,
                                   **slds_lorenz_solver_params,
                                   num_systems=1),
                DynamicsSLDSSolver(dataset_config=lorenz_config,
                                   **slds_lorenz_solver_params,
                                   num_systems=200),
            ])

        sweep_n_experiments = {}
        for sweep in sweeps:
            n_experiments = populate_exp(
                sweep,
                reset_sweep=reset,
            )
            sweep_n_experiments[sweep.sweep_name] = n_experiments

        expected_n_experiments = sum([len(sweep) for sweep in sweeps])
        n_experiments = sum(sweep_n_experiments.values())
        print(
            f"Added {n_experiments} out of {expected_n_experiments} experiments in total."
        )

    except Exception:
        import traceback
        traceback.print_exc()
