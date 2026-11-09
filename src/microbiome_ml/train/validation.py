"""Validation schemes that return per-sample predictions.

* :func:`out_of_fold_predictions` - every sample predicted by a model that did
  not see it, from predefined folds (a CV scheme) or K random folds.
* :func:`leave_one_group_out` - LOGO: each group (e.g. bioproject) is held out
  in turn, so every prediction is for a group the model never saw. This is the
  robustness check across categories.

Both take plain arrays, so they work on anything (a ``Dataset`` frame, a
notebook, a test) and return polars frames in the ``acc,real,pred`` layout the
downstream analyses read.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    PredefinedSplit,
    cross_val_predict,
)

from microbiome_ml.train import model_zoo as zoo
from microbiome_ml.train.metrics import regression_metrics

if TYPE_CHECKING:
    from microbiome_ml.wrangle.dataset import Dataset

logger = logging.getLogger(__name__)


def out_of_fold_predictions(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    samples: Optional[Sequence[str]] = None,
    folds: Optional[Sequence[int]] = None,
    n_splits: int = 10,
    seed: int = 0,
    n_jobs: Optional[int] = None,
) -> pl.DataFrame:
    """Predict every sample with a model fitted on the other folds.

    Args:
        estimator: Unfitted scikit-learn estimator (cloned per fold).
        X, y: Features and target.
        samples: Sample ids, in row order (defaults to row numbers).
        folds: Fold id per row (a CV scheme). If omitted, ``n_splits`` random
            folds are drawn with ``seed``.
        n_splits: Number of random folds when ``folds`` is not given.

    Returns:
        Frame with ``sample``, ``fold``, ``real``, ``pred``.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    ids = (
        [str(i) for i in range(len(y))]
        if samples is None
        else [str(s) for s in samples]
    )
    if folds is None:
        splitter: Any = KFold(n_splits, shuffle=True, random_state=seed)
        fold_ids = np.empty(len(y), dtype=int)
        for k, (_, test) in enumerate(splitter.split(X)):
            fold_ids[test] = k
        cv: Any = splitter
    else:
        fold_ids = np.asarray(folds).astype(int)
        cv = PredefinedSplit(test_fold=fold_ids)
    pred = cross_val_predict(clone(estimator), X, y, cv=cv, n_jobs=n_jobs)
    return pl.DataFrame(
        {
            "sample": ids,
            "fold": fold_ids,
            "real": y,
            "pred": np.asarray(pred, dtype=float),
        }
    )


@dataclass
class LogoResult:
    """Outcome of :func:`leave_one_group_out`.

    Attributes:
        predictions: ``sample, group, real, pred`` for every sample.
        metrics: Pooled metrics over all predictions (see
            :func:`~microbiome_ml.train.metrics.regression_metrics`).
        by_group: One row per held-out group: ``group, n, mae, rmse, bias,
            r2`` (``r2`` is null for groups with fewer than 2 samples or a
            constant label). ``bias`` is mean(pred - real).
        importances: ``feature, importance`` (sum to 1) from a model fitted on
            all the data, or ``None``.
        n_groups: Number of groups held out.
    """

    predictions: pl.DataFrame
    metrics: Dict[str, Optional[float]]
    by_group: pl.DataFrame
    importances: Optional[pl.DataFrame] = None
    n_groups: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, directory: Union[str, Path], prefix: str = "logo") -> Path:
        """Write the result as CSV/JSON files under ``directory``.

        ``<prefix>_predictions.csv`` (sample, group, real, pred),
        ``<prefix>_by_group.csv``, ``<prefix>_metrics.json`` and, when
        available, ``<prefix>_importances.csv``.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.predictions.write_csv(out / f"{prefix}_predictions.csv")
        self.by_group.write_csv(out / f"{prefix}_by_group.csv")
        record = {**self.metrics, "n_groups": self.n_groups, **self.extra}
        (out / f"{prefix}_metrics.json").write_text(
            json.dumps(record, indent=2, default=str)
        )
        if self.importances is not None:
            self.importances.write_csv(out / f"{prefix}_importances.csv")
        return out


def leave_one_group_out(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any],
    samples: Optional[Sequence[str]] = None,
    feature_names: Optional[List[str]] = None,
    importances: bool = True,
    n_jobs: Optional[int] = None,
    alternative: str = "two-sided",
) -> LogoResult:
    """Leave-one-group-out cross-validation.

    Each distinct group is predicted by a model fitted on all the others.
    Null group values are treated as one group per sample (an unknown group
    cannot be left out as a whole).

    Raises:
        ValueError: if there are fewer than 2 groups.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    ids = (
        [str(i) for i in range(len(y))]
        if samples is None
        else [str(s) for s in samples]
    )
    grp = np.array(
        [f"__null_{i}" if g is None else str(g) for i, g in enumerate(groups)]
    )
    uniq = np.unique(grp)
    if uniq.size < 2:
        raise ValueError(
            f"Leave-one-group-out needs at least 2 groups, got {uniq.size}"
        )
    pred = cross_val_predict(
        clone(estimator),
        X,
        y,
        cv=LeaveOneGroupOut(),
        groups=grp,
        n_jobs=n_jobs,
    )
    predictions = pl.DataFrame(
        {
            "sample": ids,
            "group": grp,
            "real": y,
            "pred": np.asarray(pred, dtype=float),
        }
    )
    metrics = regression_metrics(y, pred, alternative=alternative)

    rows = []
    for g in uniq:
        m = grp == g
        mg = regression_metrics(y[m], np.asarray(pred)[m])
        rows.append(
            {
                "group": str(g),
                "n": int(m.sum()),
                "mae": mg["mae"],
                "rmse": mg["rmse"],
                "bias": float(np.mean(np.asarray(pred)[m] - y[m])),
                "r2": mg["r2"],
            }
        )
    by_group = pl.DataFrame(
        rows,
        schema={
            "group": pl.Utf8,
            "n": pl.Int64,
            "mae": pl.Float64,
            "rmse": pl.Float64,
            "bias": pl.Float64,
            "r2": pl.Float64,
        },
    ).sort("n", descending=True)

    imp_df = None
    if importances:
        fitted = clone(estimator).fit(X, y)
        try:
            imp = zoo.feature_importance(fitted)
            names = feature_names or [f"f{i}" for i in range(X.shape[1])]
            imp_df = pl.DataFrame(
                {"feature": names, "importance": imp}
            ).sort("importance", descending=True)
        except ValueError:
            logger.info("Model has no feature importances; skipped")

    return LogoResult(
        predictions=predictions,
        metrics=metrics,
        by_group=by_group,
        importances=imp_df,
        n_groups=int(uniq.size),
    )


def logo_for_dataset(
    dataset: "Dataset",
    feature_set: str,
    label: str,
    grouping: str,
    estimator: Any,
    split: str = "train",
    importances: bool = True,
    n_jobs: Optional[int] = None,
    alternative: str = "two-sided",
) -> LogoResult:
    """Leave-one-group-out on a dataset's feature set and label.

    Args:
        dataset: Dataset with the feature set, label and grouping.
        feature_set, label: What to model.
        grouping: Grouping column to leave out (e.g. ``"bioproject"``).
        estimator: Unfitted estimator, e.g. ``clone(result.model)`` of a CV
            winner (it keeps the tuned hyperparameters).
        split: ``"train"`` (the holdout-train samples, the default: the
            holdout test set stays untouched), ``"test"`` or ``"all"``.
    """
    xy = dataset.get_xy(feature_set, label, split=split, groupings=[grouping])
    result = leave_one_group_out(
        estimator,
        xy.X,
        xy.y,
        xy.groups[grouping],
        samples=xy.samples,
        feature_names=xy.feature_names,
        importances=importances,
        n_jobs=n_jobs,
        alternative=alternative,
    )
    result.extra.update(
        {
            "feature_set": feature_set,
            "label": label,
            "grouping": grouping,
            "split": split,
            "n_samples": len(xy.y),
        }
    )
    return result
