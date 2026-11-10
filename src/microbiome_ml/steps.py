"""Workflow steps on plain tables, with explicit inputs and outputs.

Each function here is one rule of a file-based workflow (Snakemake, a shell
script, a notebook): it reads CSV/Parquet files, calls the module, and writes
files in a fixed layout. No function reads global state, so they are testable
and importable; :mod:`microbiome_ml.cli` exposes them on the command line.

Layouts (``acc`` is the sample id; ``target`` the label in model tables):

* rank features      ``acc, <taxon>...``
* sample features    ``acc, <feature>...``
* partition table    ``acc, <label>_partition`` (``train`` / ``test`` / null)
* selected features  ``<features...>, target, acc`` for train and test
* CV / test / OOF predictions  ``acc, real, pred``
* LOGO predictions   ``acc, group, real, pred``
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import polars as pl
import yaml

from microbiome_ml.train import model_zoo as zoo
from microbiome_ml.train.metrics import regression_metrics

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
ID = "acc"
TARGET = "target"


# --------------------------------------------------------------------------
# data wrangling
# --------------------------------------------------------------------------
def rank_features(
    profiles: PathLike,
    rank: str,
    out: PathLike,
    id_column: str = ID,
    chunk_samples: int = 5000,
    check_filled: bool = True,
) -> Path:
    """Relative abundance of every taxon at one rank, per sample.

    ``profiles`` is a long table ``sample, taxonomy, relabund`` (a row per
    taxon at every rank). The wide table is streamed to ``out`` a chunk of
    samples at a time, so memory is ``chunk_samples x n_taxa``. Columns are
    named by the rank token (``p__Acidobacteriota``); a sample without a taxon
    has an empty cell.
    """
    from microbiome_ml.wrangle.profiles import TaxonomicProfiles

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tp = TaxonomicProfiles(str(profiles), check_filled=check_filled)
    tp.create_features(
        rank,
        stream_path=tmp,
        chunk_samples=chunk_samples,
        id_column=id_column,
    )
    tmp.replace(out_path)
    logger.info("Wrote %s features to %s", rank, out_path)
    return out_path


def aggregate_features(
    profiles: PathLike,
    species_features: PathLike,
    out: PathLike,
    method: str = "arithmetic_mean",
    weighting: str = "none",
    taxonomy_map: Optional[PathLike] = None,
    key_col: str = ID,
    map_key_col: str = ID,
    map_taxonomy_col: str = "taxonomy",
    chunk_columns: Optional[int] = None,
    uncommon_feature_cutoff: Union[int, float] = 0,
    k: int = 10,
    min_abundance: float = 0.001,
) -> Path:
    """Aggregate a species-level feature table to samples.

    Args:
        profiles: Species-level abundance table ``sample, taxonomy, relabund``.
        species_features: Parquet file/dir or CSV of species features.
        taxonomy_map: Optional ``acc, taxonomy`` table that re-keys the species
            table from genome accession to taxonomy (must be one-to-one).
        chunk_columns: Aggregate this many feature columns at a time.
        uncommon_feature_cutoff: Drop features present in fewer samples than
            this (int) or fraction of samples (float in (0, 1)); 0 keeps all.
    """
    from microbiome_ml.core.config import AggregationConfig
    from microbiome_ml.wrangle.aggregation import FeatureAggregator
    from microbiome_ml.wrangle.species_table import LazySpeciesFeatureSet

    species = LazySpeciesFeatureSet.scan(
        species_features,
        name=Path(str(species_features)).stem,
        key_col=key_col,
        taxonomy_map=taxonomy_map,
        map_key_col=map_key_col,
        map_taxonomy_col=map_taxonomy_col,
    )
    rel = pl.scan_csv(str(profiles)).select("sample", "taxonomy", "relabund")
    cfg = AggregationConfig(
        method=method,  # type: ignore[arg-type]
        weighting=weighting,  # type: ignore[arg-type]
        uncommon_feature_cutoff=uncommon_feature_cutoff,
        chunk_columns=chunk_columns,
        k=k,
        min_abundance=min_abundance,
    )
    result = FeatureAggregator().get_sample_features(species, rel, cfg)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.collect().write_csv(out_path)
    logger.info("Wrote %s~%s features to %s", method, weighting, out_path)
    return out_path


def _write_chunks(chunks, out: Path) -> int:
    """Write an iterable of frames with identical columns as one CSV."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    n = 0
    first = True
    with tmp.open("wb") as handle:
        for frame in chunks:
            frame.write_csv(handle, include_header=first)
            first = False
            n += frame.height
    if first:
        tmp.write_text("acc\n")
    tmp.replace(out)
    return n


def aggregate_annotation_features(
    profiles: PathLike,
    annotations: PathLike,
    feature_col: str,
    out: PathLike,
    method: str = "arithmetic_mean",
    weighting: str = "none",
    taxonomy_map: Optional[PathLike] = None,
    key_col: str = ID,
    value_col: str = "count",
    min_prevalence: float = 0.05,
    chunk_samples: int = 2000,
    k: int = 10,
    min_abundance: float = 0.001,
) -> Path:
    """Aggregate a long genome annotation table (eggNOG, KEGG, ...) to samples.

    ``annotations`` has one row per genome and feature (``key_col``,
    ``feature_col``, ``value_col``). Features present in fewer than
    ``min_prevalence`` of the genomes are dropped. The result equals
    :func:`aggregate_features` on the pivoted table but is computed on the
    sparse matrix, ``chunk_samples`` samples at a time (see
    :mod:`microbiome_ml.wrangle.annotations`).
    """
    from microbiome_ml.wrangle.annotations import (
        iter_aggregate_annotations,
        scan_annotation_matrix,
    )

    ann = scan_annotation_matrix(
        annotations,
        feature_col,
        key_col=key_col,
        value_col=value_col,
        taxonomy_map=taxonomy_map,
        min_prevalence=min_prevalence,
    )
    rel = pl.scan_csv(str(profiles)).select("sample", "taxonomy", "relabund")
    n = _write_chunks(
        iter_aggregate_annotations(
            ann,
            rel,
            method=method,
            weighting=weighting,
            k=k,
            min_abundance=min_abundance,
            chunk_samples=chunk_samples,
        ),
        Path(out),
    )
    logger.info("Wrote %d samples x %d %s features to %s", n, len(ann.features), feature_col, out)
    return Path(out)


def kegg_completeness_features(
    profiles: PathLike,
    annotations: PathLike,
    taxonomy: PathLike,
    mappings: Dict[str, PathLike],
    outputs: Dict[str, PathLike],
    samples: Optional[PathLike] = None,
    keep_if_contains: Optional[Dict[str, str]] = None,
    key_col: str = "KEGG",
    chunk_samples: int = 2000,
) -> Dict[str, Path]:
    """KEGG module / pathway completeness per sample.

    For each name in ``mappings`` (e.g. ``"module"``, ``"pathway"``) writes
    the fraction of that module's (pathway's) orthologs present in at least
    one genome of the sample (see
    :func:`microbiome_ml.wrangle.annotations.iter_kegg_completeness`).

    Args:
        profiles: Species abundance table ``sample, taxonomy, relabund``.
        annotations: Long ``KEGG, acc, count`` ortholog table.
        taxonomy: GlobDB taxonomy TSV (no header): ``acc<TAB>taxonomy`` with
            spaces in species names, as shipped.
        mappings: name -> KEGG link file (``<id><TAB><ko>``).
        outputs: name -> output CSV.
        samples: Optional CSV whose ``acc`` column restricts the samples.
        keep_if_contains: name -> text a feature id must contain (``"map"``
            for reference pathways).
    """
    from microbiome_ml.wrangle.annotations import (
        iter_kegg_completeness,
        read_kegg_mapping,
        scan_annotation_matrix,
    )

    tax = pl.read_csv(
        str(taxonomy),
        separator="\t",
        has_header=False,
        new_columns=["acc", "taxonomy"],
    ).with_columns(pl.col("taxonomy").str.replace_all(" ", "_"))
    ann = scan_annotation_matrix(
        annotations,
        "KEGG",
        key_col="acc",
        taxonomy_map=tax,
        min_prevalence=0.0,
    )
    rel = pl.scan_csv(str(profiles)).select("sample", "taxonomy")
    rel = rel.filter(pl.col("taxonomy").str.contains(";s__"))
    keep = (
        pl.read_csv(str(samples))["acc"].to_list() if samples is not None else None
    )
    written = {}
    for name, mapping_path in mappings.items():
        mapping = read_kegg_mapping(
            mapping_path,
            name,
            keep_if_contains=(keep_if_contains or {}).get(name),
        )
        n = _write_chunks(
            iter_kegg_completeness(
                ann, mapping, rel, samples=keep, chunk_samples=chunk_samples
            ),
            Path(outputs[name]),
        )
        logger.info("Wrote %d samples x %s completeness to %s", n, name, outputs[name])
        written[name] = Path(outputs[name])
    return written


def partition_samples(
    metadata: PathLike,
    label_columns: Sequence[str],
    out: PathLike,
    group_column: Optional[str] = "bioproject",
    test_size: float = 0.2,
    n_bins: int = 5,
    seed: int = 42,
    id_column: str = ID,
    suffix: str = "_partition",
) -> Path:
    """Group-aware, label-stratified train/test partition for each label.

    For each ``label_columns`` entry (a numeric column of ``metadata``) the
    labelled samples are split into ``train`` / ``test`` so that whole groups
    (e.g. a bioproject) fall on one side and every label bin keeps about
    ``test_size`` of its samples in test. Samples without a value get an empty
    partition. The output has ``acc`` and one ``<label><suffix>`` column per
    label, ready to join onto the metadata.
    """
    from microbiome_ml.wrangle.splits import SplitManager

    meta = pl.read_csv(str(metadata), infer_schema_length=10000)
    if id_column not in meta.columns:
        raise ValueError(f"metadata has no '{id_column}' column")
    if group_column is not None and group_column not in meta.columns:
        raise ValueError(f"metadata has no '{group_column}' column")
    out_df = meta.select(id_column)
    for label in label_columns:
        if label not in meta.columns:
            raise ValueError(f"metadata has no label column '{label}'")
        cols = [id_column, label] + (
            [group_column] if group_column is not None else []
        )
        data = (
            meta.select(cols)
            .rename({id_column: "sample"})
            .with_columns(pl.col(label).cast(pl.Float64))
            .drop_nulls(label)
        )
        sm = SplitManager(label)
        sm.create_holdout(
            data,
            grouping=group_column,
            test_size=test_size,
            n_bins=n_bins,
            random_state=seed,
        )
        assert sm.holdout is not None
        part = sm.holdout.select(
            pl.col("sample").alias(id_column),
            pl.col("split").alias(f"{label}{suffix}"),
        )
        out_df = out_df.join(part, on=id_column, how="left")
        logger.info(
            "%s: %d train / %d test",
            label,
            int((part[f"{label}{suffix}"] == "train").sum()),
            int((part[f"{label}{suffix}"] == "test").sum()),
        )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)
    return out_path


# --------------------------------------------------------------------------
# model training
# --------------------------------------------------------------------------
def _feature_table(
    features: PathLike,
    metadata: PathLike,
    label_col: str,
    partition_col: str,
    id_column: str = ID,
) -> pl.LazyFrame:
    """Lazy feature table joined with the label and its partition."""
    path = str(features)
    names = pl.read_csv(path, n_rows=0).columns
    schema = {c: pl.Float32 for c in names if c not in (id_column, "sample")}
    lf = pl.scan_csv(path, schema_overrides=schema)
    meta = (
        pl.scan_csv(str(metadata), infer_schema_length=10000)
        .select(
            pl.col(id_column),
            pl.col(label_col).cast(pl.Float64).alias(TARGET),
            pl.col(partition_col).alias("_partition"),
        )
        .drop_nulls([TARGET, "_partition"])
    )
    return lf.join(meta, on=id_column, how="inner")


def select_features_table(
    features: PathLike,
    metadata: PathLike,
    label_col: str,
    partition_col: str,
    out_train: PathLike,
    out_test: PathLike,
    method: str = "boruta",
    min_prevalence: float = 0.05,
    screen_top_k: Optional[int] = None,
    perc: int = 85,
    max_depth: int = 7,
    random_state: int = 0,
    n_jobs: int = 1,
    id_column: str = ID,
    selection_table: Optional[PathLike] = None,
    chunk: int = 5000,
) -> Dict[str, Any]:
    """Select features on the train partition; write train and test tables.

    Reads a wide ``acc, <features>`` table and the metadata (label and its
    train/test partition). Prevalence and Boruta see **only the training
    rows**. Writes ``<selected...>, target, acc`` for train and for test.
    Absent entries (empty cells) count as absent for the prevalence filter and
    are 0 in the written tables.
    """
    from microbiome_ml.train.feature_selection import FeatureSelector

    lf = _feature_table(features, metadata, label_col, partition_col, id_column)
    train_lf = lf.filter(pl.col("_partition") == "train")
    cols = [
        c
        for c in lf.collect_schema().names()
        if c not in (id_column, TARGET, "_partition")
    ]
    n_train = train_lf.select(pl.len()).collect().item()
    if n_train == 0:
        raise ValueError("No training samples after joining features and labels")

    kept: List[str] = []
    for i in range(0, len(cols), chunk):
        part = cols[i : i + chunk]
        counts = train_lf.select(
            [pl.col(c).is_not_null().sum().alias(c) for c in part]
        ).collect()
        kept += [c for c in part if counts[c][0] / n_train >= min_prevalence]
    if not kept:
        logger.warning("No feature reaches the prevalence cutoff; keeping all")
        kept = cols
    logger.info("Prevalence >= %.3f: %d of %d features", min_prevalence, len(kept), len(cols))

    train = (
        train_lf.select([id_column, TARGET] + kept)
        .with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in kept])
        .collect()
    )
    X = train.select(kept).to_numpy()
    y = train[TARGET].to_numpy()
    selector = FeatureSelector(
        method=method,
        min_prevalence=0.0,
        screen_top_k=screen_top_k,
        perc=perc,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    result = selector.select(X, y, kept)
    selected = result.selected
    if not selected:
        raise ValueError("Feature selection kept no features")

    test = (
        lf.filter(pl.col("_partition") == "test")
        .select([id_column, TARGET] + selected)
        .with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in selected])
        .collect()
    )
    for df, path in ((train.select(selected + [TARGET, id_column]), out_train), (test.select(selected + [TARGET, id_column]), out_test)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(str(path))
    if selection_table is not None:
        result.save(selection_table)
    logger.info(
        "Selected %d features; train n=%d, test n=%d",
        len(selected),
        train.height,
        test.height,
    )
    return {
        "n_selected": len(selected),
        "n_candidates": len(kept),
        "n_train": train.height,
        "n_test": test.height,
    }


def _read_model_table(path: PathLike, id_column: str = ID):
    df = pl.read_csv(str(path))
    return df.drop([id_column, TARGET]), df[TARGET], df[id_column]


def fit_table_model(
    train: PathLike,
    model: str,
    grid: Dict[str, Any],
    out_model: PathLike,
    out_params: PathLike,
    out_predictions: PathLike,
    test: Optional[PathLike] = None,
    out_test_predictions: Optional[PathLike] = None,
    out_test_metrics: Optional[PathLike] = None,
    cv: int = 5,
    oof_folds: int = 10,
    seed: int = 0,
    n_jobs: int = 1,
    feature_label: str = "",
    target_label: str = "",
    id_column: str = ID,
    shuffle: bool = True,
) -> Dict[str, Any]:
    """Tune a model by K-fold grid search, then predict out-of-fold and test.

    1. ``GridSearchCV`` over ``grid`` (hyperparameters under their natural
       names; they are addressed for the model type) with ``cv`` folds, refit
       on all training rows.
    2. Out-of-fold predictions for the training rows with ``oof_folds``
       folds and the best hyperparameters.
    Folds are shuffled (seed ``seed``) unless ``shuffle=False``, which uses
    consecutive folds like ``cv=5`` in scikit-learn (for tables whose rows
    are already in random order).
    3. If ``test`` is given, predictions and metrics for the held-out rows.

    Files: the fitted best estimator (pickle), the best parameters (YAML,
    addressed as in the estimator), ``acc,real,pred`` for the out-of-fold and
    test predictions, and a one-row metrics table.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

    model = zoo.resolve(model)
    X, y, acc = _read_model_table(train, id_column)
    estimator = zoo.build(model)
    param_grid = zoo.prefix_grid(model, grid or {})
    logger.info(
        "Grid search for %s on %d samples x %d features",
        model,
        X.height,
        X.width,
    )
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=KFold(cv, shuffle=shuffle, random_state=seed if shuffle else None),
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)
    best_params = search.best_params_
    best_model = search.best_estimator_
    logger.info("Best hyperparameters: %s (score %.4f)", best_params, search.best_score_)

    oof = cross_val_predict(
        clone(estimator).set_params(**best_params),
        X,
        y,
        cv=KFold(
            oof_folds,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        ),
        n_jobs=n_jobs,
    )

    for path in (out_model, out_params, out_predictions):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"acc": acc, "real": y, "pred": oof}).write_csv(
        str(out_predictions)
    )
    with open(out_params, "w") as fh:
        yaml.dump(best_params, fh)
    with open(out_model, "wb") as fh:
        pickle.dump(best_model, fh)

    summary: Dict[str, Any] = {
        "best_params": best_params,
        "cv_score": float(search.best_score_),
        "oof": regression_metrics(y, oof),
    }
    if test is not None:
        Xt, yt, acct = _read_model_table(test, id_column)
        pred = best_model.predict(Xt)
        m = regression_metrics(yt, pred, alternative="greater")
        summary["test"] = m
        if out_test_predictions is not None:
            Path(out_test_predictions).parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"acc": acct, "real": yt, "pred": pred}).write_csv(
                str(out_test_predictions)
            )
        if out_test_metrics is not None:
            Path(out_test_metrics).parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "model": model,
                    "feature": feature_label,
                    "target": target_label,
                    "mse": m["mse"],
                    "r2": m["r2"],
                    "q2": m["q2"],
                    "pcc": m["pcc"],
                    "pval": m["pval"],
                    "n_samples": len(yt),
                    "comment": "held-out validation",
                }
            ).write_csv(str(out_test_metrics))
    return summary


def logo_table(
    train: PathLike,
    metadata: PathLike,
    model: str,
    params: PathLike,
    group_factor: str,
    out_predictions: PathLike,
    out_metrics: PathLike,
    out_importances: PathLike,
    feature_label: str = "",
    target_label: str = "",
    n_jobs: int = 1,
    id_column: str = ID,
    out_by_group: Optional[PathLike] = None,
) -> Optional[Dict[str, Any]]:
    """Leave-one-group-out on a training table, with tuned hyperparameters.

    ``params`` is the YAML written by :func:`fit_table_model`. ``metadata``
    supplies ``group_factor`` per sample (rows without a group are dropped).
    Returns ``None`` (and writes nothing) if there is only one group.
    """
    from microbiome_ml.train.validation import leave_one_group_out

    model = zoo.resolve(model)
    df = pl.read_csv(str(train)).unique(id_column, keep="first")
    meta = pl.read_csv(str(metadata), infer_schema_length=10000).select(
        pl.col(id_column), pl.col(group_factor).cast(pl.Utf8)
    )
    df = df.join(meta, on=id_column, how="inner").drop_nulls(group_factor)
    if df[group_factor].n_unique() < 2:
        logger.error("Only one group in %s; nothing to do", group_factor)
        return None
    with open(params) as fh:
        hyper = yaml.safe_load(fh) or {}
    estimator = zoo.build(model, **hyper)
    features = [c for c in df.columns if c not in (id_column, TARGET, group_factor)]
    res = leave_one_group_out(
        estimator,
        df.select(features).to_numpy(),
        df[TARGET].to_numpy(),
        df[group_factor].to_list(),
        samples=df[id_column].to_list(),
        feature_names=features,
        n_jobs=n_jobs,
        alternative="greater",
    )
    for path in (out_predictions, out_metrics, out_importances):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    res.predictions.rename({"sample": "acc"}).write_csv(str(out_predictions))
    m = res.metrics
    pl.DataFrame(
        {
            "feature": feature_label,
            "target": target_label,
            "model": model,
            "factor": group_factor,
            "mse": m["mse"],
            "r2": m["r2"],
            "q2": m["q2"],
            "pcc": m["pcc"],
            "pval": m["pval"],
            "n_samples": int(df.height),
            "comment": "",
        },
        schema_overrides={"n_samples": pl.Int32},
    ).write_csv(str(out_metrics))
    imp = res.importances
    if imp is None:
        imp = pl.DataFrame({"feature": [], "importance": []})
    imp.with_columns(
        pl.lit(model).alias("model"),
        pl.lit(target_label).alias("target"),
        pl.lit(group_factor).alias("factor"),
    ).select("model", "feature", "target", "factor", "importance").write_csv(
        str(out_importances)
    )
    if out_by_group is not None:
        Path(out_by_group).parent.mkdir(parents=True, exist_ok=True)
        res.by_group.write_csv(str(out_by_group))
    return {"n_groups": res.n_groups, **res.metrics}


def logi_table(
    train: PathLike,
    metadata: PathLike,
    model: str,
    params: PathLike,
    group_factor: str,
    out_predictions: PathLike,
    out_metrics: PathLike,
    out_importances: PathLike,
    out_holdout_predictions: PathLike,
    out_holdout_metrics: PathLike,
    subgroup_column: str = "bioproject",
    feature_label: str = "",
    target_label: str = "",
    n_jobs: int = 1,
    id_column: str = ID,
    shuffle: bool = False,
    seed: int = 42,
    holdout_fn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Leave-one-group-in on a training table, with tuned hyperparameters.

    For every ``group_factor`` group the model is cross-validated on the
    group's own training rows and scored on a holdout of the same group (whole
    ``subgroup_column`` values when there are enough, see
    :func:`microbiome_ml.train.validation.group_holdout_split`). The split of
    a group is decided from the training rows only.

    Files: ``acc,group,real,pred`` out-of-fold predictions; a metrics table
    (``feature,target,model,factor,group,mse,r2,q2,pcc,pval,n_samples,
    comment``); importances (``model,feature,target,factor,group,
    importance``); and the same predictions/metrics for the holdout rows
    (predictions with a ``split_method`` column). A skipped group has null
    metrics and its reason in ``comment``.
    """
    from microbiome_ml.train.validation import leave_one_group_in

    model = zoo.resolve(model)
    df = pl.read_csv(str(train)).unique(id_column, keep="first")
    meta = pl.read_csv(str(metadata), infer_schema_length=10000).select(
        pl.col(id_column),
        pl.col(group_factor).cast(pl.Utf8),
        pl.col(subgroup_column).cast(pl.Utf8).alias("_subgroup")
        if subgroup_column != group_factor
        else pl.col(group_factor).cast(pl.Utf8).alias("_subgroup"),
    )
    df = df.join(meta, on=id_column, how="inner").drop_nulls(group_factor)
    if df[group_factor].n_unique() < 2:
        logger.error("Only one group in %s; nothing to do", group_factor)
        return None
    with open(params) as fh:
        hyper = yaml.safe_load(fh) or {}
    estimator = zoo.build(model, **hyper)
    features = [
        c
        for c in df.columns
        if c not in (id_column, TARGET, group_factor, "_subgroup")
    ]
    res = leave_one_group_in(
        estimator,
        df.select(features).to_numpy(),
        df[TARGET].to_numpy(),
        df[group_factor].to_list(),
        df["_subgroup"].to_list(),
        samples=df[id_column].to_list(),
        feature_names=features,
        shuffle=shuffle,
        seed=seed,
        n_jobs=n_jobs,
        alternative="greater",
        subgroup_name=subgroup_column,
        holdout_fn=holdout_fn,
    )

    def table(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(
            pl.lit(feature_label).alias("feature"),
            pl.lit(target_label).alias("target"),
            pl.lit(model).alias("model"),
            pl.lit(group_factor).alias("factor"),
            "group",
            "mse",
            "r2",
            "q2",
            "pcc",
            "pval",
            pl.col("n_samples").cast(pl.Int32),
            "comment",
        )

    for path in (
        out_predictions,
        out_metrics,
        out_importances,
        out_holdout_predictions,
        out_holdout_metrics,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    res.predictions.rename({"sample": "acc"}).write_csv(str(out_predictions))
    table(res.metrics).write_csv(str(out_metrics))
    res.importances.select(
        pl.lit(model).alias("model"),
        "feature",
        pl.lit(target_label).alias("target"),
        pl.lit(group_factor).alias("factor"),
        "group",
        "importance",
    ).write_csv(str(out_importances))
    res.holdout_predictions.rename({"sample": "acc"}).write_csv(
        str(out_holdout_predictions)
    )
    table(res.holdout_metrics).write_csv(str(out_holdout_metrics))
    fitted = res.metrics.filter(pl.col("r2").is_not_null()).height
    logger.info(
        "LOGI over %d %s groups: %d fitted, %d with a holdout score",
        res.n_groups,
        group_factor,
        fitted,
        res.holdout_metrics.filter(pl.col("r2").is_not_null()).height,
    )
    return {"n_groups": res.n_groups, "n_fitted": fitted}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def plot_predictions(
    predictions: PathLike,
    out: PathLike,
    title: str,
    units: str = "",
    density_cmap: str = "inferno_r",
    validation: Optional[PathLike] = None,
    style: Optional[PathLike] = None,
    left_label: str = "CV",
    right_label: str = "Validation",
    dpi: int = 500,
) -> Path:
    """Actual vs predicted density from an ``acc,real,pred`` table.

    With ``validation`` (another ``acc,real,pred`` table, e.g. the held-out
    test predictions) the two are drawn side by side.
    """
    from microbiome_ml.visualise import diagnostics as dg

    cv = pl.read_csv(str(predictions)).drop_nulls()
    pred, real = cv["pred"].to_numpy(), cv["real"].to_numpy()
    if validation is None:
        dg.plot_prediction_density(
            pred, real, title=title, units=units, density_cmap=density_cmap,
            file_name=out, style=style, dpi=dpi,
        )
    else:
        val = pl.read_csv(str(validation)).drop_nulls()
        dg.plot_two_panels(
            (pred, real), (val["pred"].to_numpy(), val["real"].to_numpy()),
            title=title, units=units, density_cmap=density_cmap,
            left_label=left_label, right_label=right_label,
            file_name=out, style=style, dpi=dpi,
        )
    return Path(out)


def plot_logo_predictions(
    predictions: PathLike,
    out: PathLike,
    title: str,
    units: str = "",
    group_label: Optional[str] = None,
    hist_cmap: str = "viridis",
    density_cmap: str = "inferno_r",
    style: Optional[PathLike] = None,
    dpi: int = 500,
) -> Path:
    """LOGO predictions (``acc,group,real,pred``): density plus the error
    histogram coloured by the held-out group."""
    from microbiome_ml.visualise import diagnostics as dg

    df = pl.read_csv(str(predictions)).drop_nulls(subset=["group", "real", "pred"])
    if "group" not in df.columns:
        raise ValueError(f"Expected a 'group' column in {predictions}")
    dg.plot_predictions_by_group(
        df["pred"].to_numpy(), df["real"].to_numpy(), df["group"].cast(pl.Utf8),
        title=title, units=units, hist_cmap=hist_cmap, density_cmap=density_cmap,
        cbar_label=group_label, file_name=out, style=style, dpi=dpi,
    )
    return Path(out)


def plot_logi_predictions(
    cv_predictions: PathLike,
    holdout_predictions: PathLike,
    out_dir: PathLike,
    title: str,
    units: str = "",
    density_cmap: str = "inferno_r",
    style: Optional[PathLike] = None,
    dpi: int = 500,
) -> List[Path]:
    """One CV-vs-holdout figure per group from the LOGI prediction tables.
    ``title`` may contain ``{group}``."""
    from microbiome_ml.visualise import diagnostics as dg

    return dg.plot_logi_groups(
        pl.read_csv(str(cv_predictions)),
        pl.read_csv(str(holdout_predictions)),
        out_dir, title, units, density_cmap, style=style, dpi=dpi,
    )


def plot_pca(
    train: PathLike,
    test: PathLike,
    metadata: PathLike,
    group_factors: Sequence[str],
    out: PathLike,
    title: str,
    target_name: str = "label",
    n_components: int = 2,
    scale: bool = True,
    top_n: int = 10,
    legend_max_levels: int = 12,
    other_colour: str = "lightgrey",
    clip_percentile: Optional[float] = 0.5,
    style: Optional[PathLike] = None,
    id_column: str = ID,
    dpi: int = 300,
) -> Path:
    """PCA of the selected features (scaler and PCA fitted on the training
    rows, test rows projected) coloured by each group factor and by the label.

    ``train`` / ``test`` are the ``<features>, target, acc`` tables;
    ``metadata`` supplies the group factors. ``title`` gets
    ``\n(<n> features, <n> train / <n> test samples)`` appended.
    """
    from microbiome_ml.visualise import diagnostics as dg

    tr = pl.read_csv(str(train)).with_columns(pl.lit("train").alias("split"))
    te = pl.read_csv(str(test)).with_columns(pl.lit("test").alias("split"))
    features = [c for c in tr.columns if c not in (TARGET, id_column, "split")]
    data = pl.concat([tr, te.select(tr.columns)], how="vertical_relaxed")
    meta = pl.read_csv(str(metadata), infer_schema_length=10000).select(
        [id_column] + [pl.col(g).cast(pl.Utf8) for g in group_factors]
    )
    data = data.join(meta, on=id_column, how="left")
    X = (
        data.select(features).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float64)
    )
    is_train = (data["split"] == "train").to_numpy()
    coords, evr = dg.pca_embedding(X, is_train, n_components, scale)
    logger.info("Explained variance: %s", np.round(evr * 100, 2))
    dg.plot_pca_overview(
        coords, evr, is_train,
        {g: data[g].to_list() for g in group_factors},
        data[TARGET].to_numpy().astype(float), target_name,
        title=(
            f"{title}\n({len(features)} features, {int(is_train.sum())} train / "
            f"{int((~is_train).sum())} test samples)"
        ),
        top_n=top_n, legend_max_levels=legend_max_levels, other_colour=other_colour,
        clip_percentile=clip_percentile, file_name=out, style=style, dpi=dpi,
    )
    return Path(out)


def plot_feature_importances(
    model: PathLike,
    train: PathLike,
    out_image: PathLike,
    out_table: PathLike,
    model_name: str = "",
    target: str = "",
    feature_set: str = "",
    shorten_taxa: bool = False,
    top: int = 10,
    style: Optional[PathLike] = None,
    id_column: str = ID,
) -> pl.DataFrame:
    """Importances of a pickled model on the columns of a training table.

    Writes the full table (``feature, importance, model, target,
    feature_set``, largest first) and a bar chart of the top ``top``;
    ``shorten_taxa`` reduces lineage-named features to their last rank.
    """
    from microbiome_ml.visualise import diagnostics as dg

    features = pl.read_csv(str(train), n_rows=0).drop([TARGET, id_column]).columns
    with open(model, "rb") as fh:
        fitted = pickle.load(fh)
    imp = zoo.feature_importance(fitted)
    table = (
        pl.DataFrame({"feature": features, "importance": imp})
        .sort(by="importance", descending=True)
        .with_columns(
            model=pl.lit(model_name), target=pl.lit(target), feature_set=pl.lit(feature_set)
        )
    )
    names = table["feature"].head(top).to_list()
    if shorten_taxa:
        names = dg.short_taxon_names(names)
    dg.plot_top_importances(
        names, table["importance"].to_list(), out_image, top=top, style=style
    )
    Path(out_table).parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(str(out_table))
    return table


__all__ = [
    "rank_features",
    "aggregate_features",
    "aggregate_annotation_features",
    "kegg_completeness_features",
    "partition_samples",
    "select_features_table",
    "fit_table_model",
    "logo_table",
    "logi_table",
    "plot_predictions",
    "plot_logo_predictions",
    "plot_logi_predictions",
    "plot_pca",
    "plot_feature_importances",
]
