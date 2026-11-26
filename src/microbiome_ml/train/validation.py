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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    PredefinedSplit,
    cross_val_predict,
    train_test_split,
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


# --------------------------------------------------------------------------
# leave-one-group-in
# --------------------------------------------------------------------------
HoldoutFn = Callable[..., Tuple[np.ndarray, np.ndarray, str, float]]


def group_holdout_split(
    y: Any,  # array-like of float
    subgroups: Sequence[Any],
    test_size: float = 0.2,
    seed: int = 42,
    min_labelled: int = 25,
    min_subgroups: int = 5,
    n_bins: int = 5,
    subgroup_name: str = "bioproject",
    samples: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """Split one group's samples into train and holdout rows.

    ``samples`` (ids, same order as ``y``) is accepted for symmetry with
    custom split functions and is not used here.

    Whole sub-groups (e.g. bioprojects inside an ecoregion) are held out
    together when there are at least ``min_subgroups`` of them and the split
    reaches ``test_size`` within a tolerance of ``max(0.05, 1 / n)``;
    otherwise the samples are split stratified by label bin, and at random if
    some bin is too small to stratify.

    Returns:
        ``(train_idx, test_idx, method, achieved_fraction)`` where the indices
        are positions in ``y``. ``method`` records which route was taken.

    Raises:
        ValueError: with fewer than ``min_labelled`` samples.
    """
    from microbiome_ml.wrangle.splits import SplitManager

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < min_labelled:
        raise ValueError(
            f"only {n} labelled samples; need at least {min_labelled}"
        )
    sub = np.array([str(g) for g in subgroups])
    n_sub = len(np.unique(sub))
    tolerance = max(0.05, 1.0 / n)
    method = f"{subgroup_name}_grouped"
    frac = 0.0
    constant = bool(y.min() == y.max())
    use_grouped = n_sub >= min_subgroups and not constant
    train_idx = test_idx = np.zeros(0, dtype=int)
    if use_grouped:
        frame = pl.DataFrame(
            {"sample": [str(i) for i in range(n)], "y": y, "g": sub}
        )
        sm = SplitManager("y")
        sm.create_holdout(
            frame,
            grouping="g",
            test_size=test_size,
            n_bins=n_bins,
            random_state=seed,
        )
        assert sm.holdout is not None
        side = dict(
            zip(sm.holdout["sample"].to_list(), sm.holdout["split"].to_list())
        )
        is_test = np.array([side[str(i)] == "test" for i in range(n)])
        train_idx, test_idx = np.where(~is_test)[0], np.where(is_test)[0]
        frac = len(test_idx) / n
        use_grouped = abs(frac - test_size) <= tolerance
        if not use_grouped:
            method = "sample_stratified_fallback: grouped fraction out of range"
    elif constant:
        method = "sample_stratified_fallback: constant label"
    else:
        method = f"sample_stratified_fallback: only {n_sub} {subgroup_name}s"

    if not use_grouped:
        strata = None
        if y.min() != y.max():
            edges = np.linspace(y.min(), y.max(), n_bins + 1)
            strata = np.digitize(y, edges[1:-1], right=False)
        positions = np.arange(n)
        try:
            train_idx, test_idx = train_test_split(
                positions,
                test_size=test_size,
                random_state=seed,
                stratify=strata,
            )
        except ValueError:
            method = "sample_random_fallback: insufficient target-bin counts"
            train_idx, test_idx = train_test_split(
                positions, test_size=test_size, random_state=seed
            )
        frac = len(test_idx) / n
    return np.asarray(train_idx), np.asarray(test_idx), method, frac


@dataclass
class LogiResult:
    """Outcome of :func:`leave_one_group_in`.

    Attributes:
        metrics: One row per group: ``group, n_samples, mae, mse, rmse, r2,
            q2, pcc, pval, comment``. ``n_samples`` is the number of training
            rows of the group; metrics are null (and ``comment`` says why) for
            a group that was skipped.
        predictions: ``sample, group, real, pred`` out-of-fold predictions
            inside each group's training rows.
        importances: ``group, feature, importance`` from a model fitted on the
            group's training rows.
        holdout_metrics: One row per group with the same columns, scored on
            the group's holdout rows (``comment`` = split method + fraction).
        holdout_predictions: ``sample, group, real, pred, split_method``.
    """

    metrics: pl.DataFrame
    predictions: pl.DataFrame
    importances: pl.DataFrame
    holdout_metrics: pl.DataFrame
    holdout_predictions: pl.DataFrame
    n_groups: int = 0


_METRIC_COLS = ["mae", "mse", "rmse", "r2", "q2", "pcc", "pval"]


def _metric_row(
    group: str,
    n: int,
    metrics: Optional[Dict[str, Optional[float]]],
    comment: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"group": group, "n_samples": n}
    for c in _METRIC_COLS:
        row[c] = None if metrics is None else metrics.get(c)
    row["comment"] = comment
    return row


def leave_one_group_in(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any],
    subgroups: Sequence[Any],
    samples: Optional[Sequence[str]] = None,
    feature_names: Optional[List[str]] = None,
    cv_folds: int = 10,
    min_train: int = 20,
    test_size: float = 0.2,
    min_labelled: int = 25,
    min_subgroups: int = 5,
    n_bins: int = 5,
    seed: int = 42,
    shuffle: bool = True,
    n_jobs: Optional[int] = None,
    alternative: str = "two-sided",
    subgroup_name: str = "bioproject",
    holdout_fn: Optional[HoldoutFn] = None,
) -> LogiResult:
    """Leave-one-group-in: how much signal is there *within* each group?

    The complement of LOGO. For every group (e.g. an ecoregion) the model is
    cross-validated on that group's samples alone, and scored on a holdout of
    the same group that it never saw:

    1. the group's samples are split into training and holdout rows
       (:func:`group_holdout_split`: whole ``subgroups`` such as bioprojects
       when there are enough of them);
    2. out-of-fold predictions inside the training rows (``cv_folds`` folds);
    3. a model fitted on the training rows gives the importances and the
       holdout predictions.

    A group is skipped (metrics null, reason in ``comment``) if it has fewer
    than ``min_labelled`` samples ("skipped: ..."), fewer than ``min_train``
    training rows ("Not enough samples") or a constant label ("Only one y
    value"). Rows with a null group or subgroup are ignored.

    Args:
        estimator: Unfitted estimator (cloned per use).
        X, y: Features and target of all samples.
        groups: Group per sample (what is left in).
        subgroups: Sub-group per sample used to hold out whole sub-groups.
        shuffle: Shuffle the inner CV folds (seeded); ``False`` uses
            consecutive folds like scikit-learn's integer ``cv``.
        holdout_fn: Replacement for :func:`group_holdout_split`, called as
            ``holdout_fn(y, subgroups, test_size=, seed=, min_labelled=,
            min_subgroups=, n_bins=, subgroup_name=, samples=)`` for one
            group and returning ``(train_idx, test_idx, method, fraction)``
            (indices into ``y``); raise ``ValueError`` to skip the group.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    ids = (
        [str(i) for i in range(len(y))]
        if samples is None
        else [str(s) for s in samples]
    )
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    grp = [None if g is None else str(g) for g in groups]
    sub = [None if g is None else str(g) for g in subgroups]
    split_fn = holdout_fn or group_holdout_split

    metric_rows: List[Dict[str, Any]] = []
    hold_rows: List[Dict[str, Any]] = []
    pred_frames: List[pl.DataFrame] = []
    imp_frames: List[pl.DataFrame] = []
    hold_pred_frames: List[pl.DataFrame] = []
    pred_schema = {
        "sample": pl.Utf8,
        "group": pl.Utf8,
        "real": pl.Float64,
        "pred": pl.Float64,
    }

    order = sorted({g for g in grp if g is not None})
    for g in order:
        rows = np.array(
            [
                i
                for i in range(len(y))
                if grp[i] == g and sub[i] is not None and not np.isnan(y[i])
            ],
            dtype=int,
        )
        try:
            tr, te, method, frac = split_fn(
                y[rows],
                [sub[i] for i in rows],
                test_size=test_size,
                seed=seed,
                min_labelled=min_labelled,
                min_subgroups=min_subgroups,
                n_bins=n_bins,
                subgroup_name=subgroup_name,
                samples=[ids[i] for i in rows],
            )
        except ValueError as exc:
            logger.warning("Could not create a holdout for %s: %s", g, exc)
            tr = te = np.zeros(0, dtype=int)
            method, frac = f"skipped: {exc}", 0.0
        train_rows, hold_rows_idx = rows[np.asarray(tr, int)], rows[np.asarray(te, int)]
        Xt, yt = X[train_rows], y[train_rows]

        pred = None
        fitted = None
        comment = ""
        if len(train_rows) < min_train:
            comment = "Not enough samples"
        elif len(np.unique(yt)) == 1:
            comment = "Only one y value"
        else:
            k = min(cv_folds, len(yt))
            splitter = KFold(
                k, shuffle=shuffle, random_state=seed if shuffle else None
            )
            pred = cross_val_predict(
                clone(estimator), Xt, yt, cv=splitter, n_jobs=n_jobs
            )
            fitted = clone(estimator).fit(Xt, yt)
            try:
                imp = zoo.feature_importance(fitted)
                imp_frames.append(
                    pl.DataFrame(
                        {"group": g, "feature": names, "importance": imp}
                    ).sort("importance", descending=True)
                )
            except ValueError:
                pass
        m = None if pred is None else regression_metrics(yt, pred, alternative)
        metric_rows.append(_metric_row(g, int(len(yt)), m, comment))
        pred_frames.append(
            pl.DataFrame(
                {
                    "sample": [ids[i] for i in train_rows],
                    "group": [g] * len(train_rows),
                    "real": yt,
                    "pred": [None] * len(yt) if pred is None else pred,
                },
                schema=pred_schema,
            )
        )

        h_comment = method
        hm = None
        hn = 0
        h_pred = None
        if len(hold_rows_idx) and fitted is not None:
            Xh, yh = X[hold_rows_idx], y[hold_rows_idx]
            h_pred = fitted.predict(Xh)
            hm = regression_metrics(yh, h_pred, alternative)
            hn = len(yh)
            h_comment = f"{method}; fraction={frac:.3f}"
            hold_pred_frames.append(
                pl.DataFrame(
                    {
                        "sample": [ids[i] for i in hold_rows_idx],
                        "group": [g] * hn,
                        "real": yh,
                        "pred": h_pred,
                        "split_method": [method] * hn,
                    },
                    schema={**pred_schema, "split_method": pl.Utf8},
                )
            )
        hold_rows.append(_metric_row(g, hn, hm, h_comment))

    metric_schema = {
        "group": pl.Utf8,
        "n_samples": pl.Int64,
        **{c: pl.Float64 for c in _METRIC_COLS},
        "comment": pl.Utf8,
    }
    empty_pred = pl.DataFrame(schema=pred_schema)
    return LogiResult(
        metrics=pl.DataFrame(metric_rows, schema=metric_schema),
        predictions=pl.concat(pred_frames) if pred_frames else empty_pred,
        importances=pl.concat(imp_frames)
        if imp_frames
        else pl.DataFrame(
            schema={
                "group": pl.Utf8,
                "feature": pl.Utf8,
                "importance": pl.Float64,
            }
        ),
        holdout_metrics=pl.DataFrame(hold_rows, schema=metric_schema),
        holdout_predictions=pl.concat(hold_pred_frames)
        if hold_pred_frames
        else pl.DataFrame(schema={**pred_schema, "split_method": pl.Utf8}),
        n_groups=len(order),
    )
