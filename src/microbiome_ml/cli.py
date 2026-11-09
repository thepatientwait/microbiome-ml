"""Command line for the workflow steps in :mod:`microbiome_ml.steps`.

    python -m microbiome_ml.cli [--log-file LOG --attempt N] <command> ...

Commands: ``rank-features``, ``aggregate``, ``partition``, ``select-features``,
``fit`` and ``logo``. Each writes the files named on the command line, logs
its progress and exits non-zero on error, so it drops straight into a
workflow rule::

    shell: "python -m microbiome_ml.cli --log-file {params.logfile} \\
            --attempt {resources.attempt_no} fit --model {wildcards.model} ..."

Importing this module does not import scikit-learn or Boruta; each command
imports what it needs.
"""

import argparse
import contextlib
import logging
import sys
from typing import Any, Callable, Dict, Optional, Sequence

import yaml

logger = logging.getLogger("microbiome_ml.cli")


def _load_grid(path: Optional[str], model: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    from microbiome_ml.train import model_zoo as zoo

    key = zoo.resolve(model)
    for name, grid in cfg.items():
        try:
            if zoo.resolve(str(name)) == key:
                return grid or {}
        except ValueError:
            continue
    return {}


def _cmd_rank_features(a: argparse.Namespace) -> None:
    from microbiome_ml import steps

    steps.rank_features(
        a.profiles,
        a.rank,
        a.out,
        id_column=a.id_column,
        chunk_samples=a.chunk_samples,
        check_filled=not a.no_check_filled,
    )


def _cmd_aggregate(a: argparse.Namespace) -> None:
    from microbiome_ml import steps

    cutoff: Any = a.uncommon
    steps.aggregate_features(
        a.profiles,
        a.species_features,
        a.out,
        method=a.method,
        weighting=a.weighting,
        taxonomy_map=a.taxonomy_map,
        key_col=a.key_col,
        chunk_columns=a.chunk_columns,
        uncommon_feature_cutoff=cutoff,
        k=a.k,
        min_abundance=a.min_abundance,
    )


def _cmd_partition(a: argparse.Namespace) -> None:
    from microbiome_ml import steps

    steps.partition_samples(
        a.metadata,
        a.label,
        a.out,
        group_column=a.group,
        test_size=a.test_size,
        n_bins=a.n_bins,
        seed=a.seed,
        id_column=a.id_column,
        suffix=a.suffix,
    )


def _cmd_select(a: argparse.Namespace) -> None:
    from microbiome_ml import steps

    steps.select_features_table(
        a.features,
        a.metadata,
        a.label,
        a.partition,
        a.out_train,
        a.out_test,
        method=a.method,
        min_prevalence=a.min_prevalence,
        screen_top_k=a.screen_top_k,
        perc=a.perc,
        max_depth=a.max_depth,
        random_state=a.seed,
        n_jobs=a.threads,
        id_column=a.id_column,
        selection_table=a.selection_table,
    )


def _cmd_fit(a: argparse.Namespace) -> None:
    from microbiome_ml import steps

    steps.fit_table_model(
        a.train,
        a.model,
        _load_grid(a.params_file, a.model),
        a.out_model,
        a.out_params,
        a.out_predictions,
        test=a.test,
        out_test_predictions=a.out_test_predictions,
        out_test_metrics=a.out_test_metrics,
        cv=a.cv,
        oof_folds=a.oof_folds,
        seed=a.seed,
        n_jobs=a.threads,
        feature_label=a.feature_label,
        target_label=a.target_label,
        id_column=a.id_column,
    )


def _cmd_logo(a: argparse.Namespace) -> None:
    from microbiome_ml import steps

    steps.logo_table(
        a.train,
        a.metadata,
        a.model,
        a.params,
        a.group_factor,
        a.out_predictions,
        a.out_metrics,
        a.out_importances,
        feature_label=a.feature_label,
        target_label=a.target_label,
        n_jobs=a.threads,
        id_column=a.id_column,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m microbiome_ml.cli",
        description="Workflow steps of microbiome_ml on plain tables.",
    )
    p.add_argument("--log-file", help="send stdout/stderr to this file")
    p.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="attempt number; 1 truncates the log file, later attempts append",
    )
    p.add_argument("--log-label", default="microbiome_ml", help="log header")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], None], help: str):
        sp = sub.add_parser(name, help=help)
        sp.set_defaults(func=fn)
        return sp

    s = add("rank-features", _cmd_rank_features, "taxon abundance at one rank")
    s.add_argument("--profiles", required=True)
    s.add_argument("--rank", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--id-column", default="acc")
    s.add_argument("--chunk-samples", type=int, default=5000)
    s.add_argument("--no-check-filled", action="store_true")

    s = add("aggregate", _cmd_aggregate, "species features -> sample features")
    s.add_argument("--profiles", required=True, help="species abundance CSV")
    s.add_argument("--species-features", required=True, help="parquet/CSV")
    s.add_argument("--out", required=True)
    s.add_argument("--method", default="arithmetic_mean")
    s.add_argument("--weighting", default="none")
    s.add_argument("--taxonomy-map", help="acc,taxonomy CSV (one-to-one)")
    s.add_argument("--key-col", default="acc")
    s.add_argument("--chunk-columns", type=int)
    s.add_argument("--uncommon", type=float, default=0.0)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--min-abundance", type=float, default=0.001)

    s = add("partition", _cmd_partition, "group-aware train/test partitions")
    s.add_argument("--metadata", required=True)
    s.add_argument("--label", nargs="+", required=True, help="label columns")
    s.add_argument("--out", required=True)
    s.add_argument("--group", default="bioproject")
    s.add_argument("--test-size", type=float, default=0.2)
    s.add_argument("--n-bins", type=int, default=5)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--id-column", default="acc")
    s.add_argument("--suffix", default="_partition")

    s = add("select-features", _cmd_select, "feature selection on train rows")
    s.add_argument("--features", required=True)
    s.add_argument("--metadata", required=True)
    s.add_argument("--label", required=True, help="label column in metadata")
    s.add_argument("--partition", required=True, help="partition column")
    s.add_argument("--out-train", required=True)
    s.add_argument("--out-test", required=True)
    s.add_argument("--selection-table")
    s.add_argument(
        "--method", default="boruta", choices=["boruta", "screen", "none"]
    )
    s.add_argument("--min-prevalence", type=float, default=0.05)
    s.add_argument("--screen-top-k", type=int)
    s.add_argument("--perc", type=int, default=85)
    s.add_argument("--max-depth", type=int, default=7)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--id-column", default="acc")

    s = add("fit", _cmd_fit, "grid search + out-of-fold + test predictions")
    s.add_argument("--train", required=True)
    s.add_argument("--test")
    s.add_argument("--model", required=True)
    s.add_argument(
        "--params-file", help="YAML of per-model hyperparameter grids"
    )
    s.add_argument("--out-model", required=True)
    s.add_argument("--out-params", required=True)
    s.add_argument("--out-predictions", required=True)
    s.add_argument("--out-test-predictions")
    s.add_argument("--out-test-metrics")
    s.add_argument("--cv", type=int, default=5)
    s.add_argument("--oof-folds", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--feature-label", default="")
    s.add_argument("--target-label", default="")
    s.add_argument("--id-column", default="acc")

    s = add("logo", _cmd_logo, "leave-one-group-out with tuned parameters")
    s.add_argument("--train", required=True)
    s.add_argument("--metadata", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--params", required=True, help="best-params YAML from fit")
    s.add_argument("--group-factor", required=True)
    s.add_argument("--out-predictions", required=True)
    s.add_argument("--out-metrics", required=True)
    s.add_argument("--out-importances", required=True)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--feature-label", default="")
    s.add_argument("--target-label", default="")
    s.add_argument("--id-column", default="acc")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from microbiome_ml.utils.logging import setup_logging, tee_log

    ctx: Any = (
        tee_log(args.log_file, attempt=args.attempt, label=args.log_label)
        if args.log_file
        else contextlib.nullcontext()
    )
    with ctx:
        setup_logging(args.log_level)
        try:
            args.func(args)
        except Exception:
            logger.exception("%s failed", args.command)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
