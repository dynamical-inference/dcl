import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import datajoint as dj
import datajoint_pytypes.fields as db_fields
from datajoint.expression import QueryExpression

import dcl.datajoint.db_schema as db_schema
import dcl.datajoint.login as login
from dcl.experiments.experiments import Experiment
from dcl.utils.checkpoints import DJCheckpointSavingCallback
from dcl.utils.configurable import Configurable
from dcl.utils.git_check import check_repo_status


# Monkey patch proj_all into QueryExpression
def _proj_all(self,
              prefix: Optional[str] = None,
              sep: str = "__",
              skip_primary_key: bool = False,
              **primary_keys):
    if prefix is None:
        prefix = self.table_name
    named_attributes = {
        f"{prefix}{sep}{attr}": attr
        for attr in self.heading.secondary_attributes
    }

    if not skip_primary_key:
        if len(self.heading.primary_key) == 1:
            # if there's a single primary key we use the prefix as the key name
            named_attributes[f"{prefix}"] = self.heading.primary_key[0]
        else:
            primary_attributes = {}
            for key in self.heading.primary_key:
                if key in primary_keys.values():
                    # don't automatically rename primary keys that have explicitly been renamed
                    continue
                elif key == "arg_hash":
                    primary_attributes[f"{prefix}"] = "arg_hash"
                else:
                    primary_attributes[f"{prefix}{sep}{key}"] = key

            named_attributes.update(primary_attributes)
            named_attributes.update(primary_keys)

    return self.proj(**named_attributes)


QueryExpression.proj_all = _proj_all
login.connect_to_database()

# get schema_name from env variable
schema_name = os.getenv("DATAJOINT_SCHEMA_NAME")
assert schema_name is not None, "environment variable DATAJOINT_SCHEMA_NAME is not set"

schema = db_schema.Schema(schema_name,
                          locals(),
                          create_tables=True,
                          create_schema=True)


@dataclasses.dataclass
class ProjectedTableField(db_fields.Field):

    table: dj.Table = None
    # the name the foreign key attribute should have in the table
    attribute_name: str = None
    # the attribute name of the primary key in the dependent table
    key_name: str = None

    def __str__(self):
        if self.table is None:
            raise ValueError(
                "Missing argument when instantiating the TableField")
        return self.table.__name__.split(".")[-1]

    def name(self, key):
        projection = f'.proj({self.attribute_name}="{self.key_name}")'
        return super().name(key, prefix="-> {self}" + projection)


@dataclasses.dataclass
class JSONField(db_fields.Field):
    default: Optional[str] = "null"

    def __str__(self):
        return "json"


class BaseConfigTable(dj.Manual):
    arg_hash = db_fields.CharField(length=128, primary_key=True)
    config = JSONField()

    @property
    def db_fields(self):
        fields = {}
        for base_cls in self.__class__.mro():
            if base_cls is object:  # Stop at the top of the hierarchy
                continue
            for key, value in base_cls.__dict__.items():
                if isinstance(value, db_fields.Field):
                    fields[key] = value
        return fields

    def insert_from_config(self, config_dict, row_dict={}, **kwargs):
        # row is the content dict
        if not Configurable.validate_config_dict(config_dict):
            raise ValueError("Invalid config dictionary")

        # first we create the hash
        arg_hash = Configurable.hash_config_dict(config_dict)
        row = dict(arg_hash=arg_hash, config=config_dict)
        # for manually inserting additional fields
        row.update(row_dict)

        # find any fields that are TableFields and point to another ConfigTable
        for field_name, db_field in self.db_fields.items():
            if isinstance(db_field, ProjectedTableField):
                if issubclass(db_field.table, BaseConfigTable):
                    sub_config = config_dict.pop(field_name)
                    sub_config_kwargs = kwargs.copy()
                    sub_config_kwargs["skip_duplicates"] = True
                    table_ref = db_field.table()
                    row[field_name] = table_ref.insert_from_config(
                        sub_config, **sub_config_kwargs)

        self.insert1(row, **kwargs)
        return arg_hash

    def get_config_dict(self, key=None):
        config_inst = self
        if key is not None:
            config_inst = config_inst & key
        assert len(config_inst) == 1, "Expected a single row"
        row = config_inst.fetch1()
        config = row['config'].copy()

        # find any fields that are TableFields and point to another ConfigTable
        for field_name, db_field in self.db_fields.items():
            if isinstance(db_field, ProjectedTableField):
                if issubclass(db_field.table, BaseConfigTable):
                    table_ref = db_field.table()
                    sub_key = {db_field.key_name: row[field_name]}
                    config[field_name] = table_ref.get_config_dict(sub_key)

        return config


@schema
class ModelConfigTable(BaseConfigTable):
    pass


@schema
class SolverConfigTable(BaseConfigTable):
    model = ProjectedTableField(table=ModelConfigTable,
                                attribute_name="model",
                                key_name="arg_hash")


@schema
class DatasetConfigTable(BaseConfigTable):
    pass


@schema
class DataLoaderConfigTable(BaseConfigTable):
    pass


@schema
class ExperimentConfigTable(BaseConfigTable):

    dataset = ProjectedTableField(table=DatasetConfigTable,
                                  attribute_name="dataset",
                                  key_name="arg_hash")
    solver = ProjectedTableField(table=SolverConfigTable,
                                 attribute_name="solver",
                                 key_name="arg_hash")

    train_loader = ProjectedTableField(table=DataLoaderConfigTable,
                                       attribute_name="train_loader",
                                       key_name="arg_hash")
    val_loader = ProjectedTableField(table=DataLoaderConfigTable,
                                     attribute_name="val_loader",
                                     key_name="arg_hash")
    test_loader = ProjectedTableField(table=DataLoaderConfigTable,
                                      attribute_name="test_loader",
                                      key_name="arg_hash")

    save_step = db_fields.IntField(default=1000)
    eval_frequency = db_fields.IntField(default=1000)

    def insert_from_config(self,
                           config_dict,
                           row_dict: Dict[str, Any] = {},
                           sweep_name: str = "default",
                           **kwargs):

        exp_arg_hash = super().insert_from_config(
            config_dict,
            row_dict=row_dict,
            **kwargs,
        )

        SweepExperimentTable().insert1(
            dict(sweep=sweep_name, experiment=exp_arg_hash),
            **kwargs,
        )

        return exp_arg_hash


@schema
class SweepTable(dj.Manual):
    name = db_fields.CharField(length=128, primary_key=True)


@schema
class SweepExperimentTable(dj.Manual):
    sweep = ProjectedTableField(table=SweepTable,
                                attribute_name="sweep",
                                key_name="name",
                                primary_key=True)
    experiment = ProjectedTableField(table=ExperimentConfigTable,
                                     attribute_name="experiment",
                                     key_name="arg_hash",
                                     primary_key=True)

    def insert1(self, row, **kwargs):
        # automatically create the sweep if it doesn't exist
        SweepTable().insert1(
            dict(name=row["sweep"]),
            skip_duplicates=True,
        )
        super().insert1(row, **kwargs)


@schema
class ExperimentTable(dj.Computed):
    experiment_config = db_fields.TableField(table=ExperimentConfigTable,
                                             primary_key=True)

    logdir = db_fields.VarcharField(length=128)
    logs = JSONField()
    code_version = db_fields.CharField(length=40)
    state = db_fields.CharField(length=32)

    class Checkpoint(dj.Part):
        definition = """
        -> ExperimentTable
        checkpoint_base_path : varchar(128)
        checkpoint_name : varchar(128)
        ---
        """

    def make(self,
             key,
             skip_repo_hash=False,
             save_step=2_000,
             eval_frequency=None,
             **kwargs):
        print("\nMake Experiment for key\n", key, flush=True)
        print(f"With make_kwargs: {kwargs}\n", flush=True)
        try:
            # 1: get arguments necessary to initialize Experiment()
            config_dict = ExperimentConfigTable().get_config_dict(key)
            print(json.dumps(config_dict, indent=4), flush=True)

            # 2: initialize exp & get git hash
            print("Initalizing exp...", flush=True)
            exp = Experiment.from_dict(config_dict)
            print("Experiment intialized!", flush=True)
            if skip_repo_hash:
                repo_hash = "debug"
            else:
                repo_hash = check_repo_status()

            # print which sweeps this experiment belongs to
            sweeps = (SweepTable() & SweepExperimentTable() & key)
            print(f"Experiment {key} present in sweeps:", flush=True)
            print(sweeps.fetch())

            def dj_callback(ckpt_path):
                return ExperimentTable.Checkpoint.insert1(
                    dict(
                        checkpoint_base_path=ckpt_path.parent,
                        checkpoint_name=ckpt_path.name,
                        **key,
                    ))

            dj_checkpoints = DJCheckpointSavingCallback(
                dj_callback=dj_callback,
                save_step=save_step,
            )

            # run the experiment
            if eval_frequency is None:
                eval_frequency = (ExperimentConfigTable() &
                                  key).fetch1()["eval_frequency"]
            print(
                f"Running experiment with eval frequency {eval_frequency} and save frequency {save_step}",
                flush=True)

            # To be able to populate the checkpoint table, the experiment table entry must already exist
            # so we already insert it here, and update it again later
            self.insert1(
                dict(
                    **key,
                    logdir=exp.log_dir,
                    code_version=repo_hash,
                    logs=exp.solver.logs,
                    state="running",
                ))

            exp.run(
                save_hook=dj_checkpoints,
                eval_frequency=eval_frequency,
            )

            self.update1(dict(
                **key,
                logs=exp.solver.logs,
                state="finished",
            ))

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            raise e


class MetricsTable(dj.Computed):

    checkpoint = db_fields.PartTableField(table=ExperimentTable.Checkpoint,
                                          main=ExperimentTable,
                                          primary_key=True)

    metrics_code_version = db_fields.CharField(length=40)

    results = JSONField()

    def compute_metrics(self, exp: Experiment):
        pass

    def make(self, key, skip_repo_hash=False, **kwargs):
        print(f"Make Metrics for key: {key}", flush=True)
        print(f"With make_kwargs: {kwargs}", flush=True)
        try:
            if skip_repo_hash:
                repo_hash = "debug"
            else:
                repo_hash = check_repo_status()
            logdir = (ExperimentTable & key).fetch1("logdir")
            ckpt_base_path = key["checkpoint_base_path"]
            ckpt_name = key["checkpoint_name"]
            ckpt_path = Path(ckpt_base_path) / ckpt_name
            exp = Experiment.load(logdir)
            exp.load_checkpoint(ckpt_path)
            metrics = self.compute_metrics(exp, **kwargs)
            self.insert1(
                dict(
                    **key,
                    metrics_code_version=repo_hash,
                    results=metrics,
                ))

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e


@schema
class TrainMetricsTable(MetricsTable):

    def compute_metrics(self, exp: Experiment, **kwargs):
        return exp.evaluate_train(**kwargs)


@schema
class ValMetricsTable(MetricsTable):

    def compute_metrics(self, exp: Experiment, **kwargs):
        return exp.evaluate_val(**kwargs)


@schema
class TestMetricsTable(MetricsTable):

    def compute_metrics(self, exp: Experiment, **kwargs):
        return exp.evaluate_test(**kwargs)
