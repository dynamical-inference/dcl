import argparse
import os
import traceback

from dcl.datajoint import schema
from dcl.datajoint.utils import get_sweep_partial
from dcl.metrics.dynamics import AccuracyViaHungarian
from dcl.metrics.dynamics import LDSError
from dcl.metrics.dynamics import PredictiveMSE
from dcl.metrics.identifiability import CCA
from dcl.metrics.identifiability import DynamicsR2
from dcl.metrics.identifiability import MCC
from dcl.metrics.identifiability import R2

os.environ["JAXTYPING_DISABLE"] = "1"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--suppress-errors",
                        action="store_true",
                        help="Ignore exceptions during populations.")
    parser.add_argument("--order",
                        help="Job execution order",
                        choices=["original", "reverse", "random"],
                        default="random")
    parser.add_argument("--sweep",
                        help="The sweep name to run",
                        default=None,
                        required=True)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate the models, don't train any new ones.",
        default=False,
        required=False)
    parser.add_argument(
        "--limit",
        help="Max number of experiments to run, default is no limit",
        default=None,
        required=False)

    parser.add_argument(
        "--last_checkpoint_only",
        action="store_true",
        help="Only evaluate the last checkpoint for each experiment",
        default=True,
        required=False)

    args = parser.parse_args()

    print(f'CUDA_VISIBLE_DEVICES={os.getenv("CUDA_VISIBLE_DEVICES")}')

    restriction = schema.ExperimentConfigTable()
    if args.sweep is not None:
        print(f"Filter for sweep name: {args.sweep}", flush=True)
        sweep_names = get_sweep_partial(args.sweep).proj(sweep="name")
        print(sweep_names, flush=True)
        sweep_restriction = schema.SweepExperimentTable().proj(
            arg_hash="experiment") & sweep_names
        restriction = restriction & sweep_restriction

    print(f"# Experiments to choose from: {len(restriction)}", flush=True)
    assert len(restriction) > 0
    max_calls = args.limit
    max_calls = int(max_calls) if max_calls is not None else None

    # max_calls in datajoint is implemented such that it doesn't take into account reserved jobs...
    # so we need to double check this ourselves and add it as part of the restrictions

    if not args.eval_only:
        errors = schema.ExperimentTable.populate(
            restriction,
            reserve_jobs=True,
            suppress_errors=args.suppress_errors,
            order=args.order,
            max_calls=max_calls,
            display_progress=True,
            make_kwargs=dict(
                disable_jaxtyping=True,
                save_step=5_000,
                eval_frequency=1000,
                eval_kwargs=dict(metrics=[
                    AccuracyViaHungarian(),
                    PredictiveMSE(),
                    R2(
                        direction="backward",
                        bias=True,
                    ),
                ]),
                eval_batches=10,
            ))

        if errors is not None and len(errors) > 0:
            for error in errors:
                print(error, flush=True)
                traceback.print_exc()
            print("Showed errors", flush=True)
        else:
            print("Successfully populated Model", flush=True)

    # populate metrics
    print("Populating metrics", flush=True)
    metric_restriction = schema.ExperimentTable.Checkpoint() & restriction
    if args.last_checkpoint_only:
        metric_restriction = metric_restriction & dict(
            checkpoint_name="last_checkpoint")

    print(f"Checkpoints to populate metrics for: {len(metric_restriction)}",
          flush=True)

    metrics_kwargs = dict(
        metrics=[
            AccuracyViaHungarian(),
            PredictiveMSE(),
            LDSError(
                inverse_type="explicit",
                bias=True,
            ),
            LDSError(
                inverse_type="implicit",
                bias=True,
            ),
            LDSError(
                inverse_type="explicit",
                bias=False,
            ),
            LDSError(
                inverse_type="implicit",
                bias=False,
            ),
            DynamicsR2(
                direction="backward",
                bias=True,
            ),
            DynamicsR2(
                direction="forward",
                bias=True,
            ),
            DynamicsR2(
                direction="forward",
                bias=False,
            ),
            DynamicsR2(
                direction="backward",
                bias=False,
            ),
            R2(
                direction="backward",
                bias=True,
            ),
            R2(
                direction="forward",
                bias=True,
            ),
            R2(
                direction="forward",
                bias=False,
            ),
            R2(
                direction="backward",
                bias=False,
            ),
            MCC(),
            CCA(),
        ],
        eval_batches=10,
    )
    schema.TrainMetricsTable.populate(
        metric_restriction,
        reserve_jobs=True,
        suppress_errors=args.suppress_errors,
        order=args.order,
        display_progress=True,
        make_kwargs=metrics_kwargs,
    )
    schema.ValMetricsTable.populate(
        metric_restriction,
        reserve_jobs=True,
        suppress_errors=args.suppress_errors,
        order=args.order,
        display_progress=True,
        make_kwargs=metrics_kwargs,
    )
    schema.TestMetricsTable.populate(
        metric_restriction,
        reserve_jobs=True,
        suppress_errors=args.suppress_errors,
        order=args.order,
        display_progress=True,
        make_kwargs=metrics_kwargs,
    )
