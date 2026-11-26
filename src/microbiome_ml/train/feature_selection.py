"""Feature selection.

:class:`FeatureSelector` reduces a wide feature matrix before modelling. The
default method is Boruta (all-relevant selection against shuffled shadow
features) wrapped around a shallow random forest, preceded by two optional,
cheap screens:

1. a *prevalence* filter: features present (non-null) in fewer than
   ``min_prevalence`` of the samples are dropped. Absence must therefore still
   be NaN in the matrix passed in (``Dataset.get_xy(..., fillna=None)``);
2. a *screen*: keep only the ``screen_top_k`` features with the highest
   importance in a quick forest, which keeps Boruta tractable for matrices
   with thousands of columns.

Selection is a supervised step: run it on the training samples only (the
holdout test set must not influence which features are kept).

Boruta is an optional dependency (extra ``selection``). It pins old numpy and
scikit-learn versions, so it usually lives in an isolated environment; the
aliases for removed numpy types that it needs are installed on first use.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from microbiome_ml.wrangle.dataset import Dataset

logger = logging.getLogger(__name__)

METHODS = ("boruta", "screen", "none")


def _patch_numpy_aliases() -> None:
    """Boruta uses ``np.int`` / ``np.float`` / ``np.bool``, removed from numpy."""
    for name, value in (("int", np.int32), ("float", np.float64), ("bool", np.bool_)):
        if name not in np.__dict__:
            setattr(np, name, value)


@dataclass
class SelectionResult:
    """Outcome of :meth:`FeatureSelector.select`.

    Attributes:
        selected: Names of the kept features, in input order.
        table: One row per input feature: ``feature, prevalence, kept_prevalence,
            kept_screen, selected, ranking`` (Boruta rank, 1 = confirmed;
            null when Boruta did not see the feature).
        method: Method used.
        params: Settings of the run.
    """

    selected: List[str]
    table: pl.DataFrame
    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.selected)

    def save(self, path: Union[str, Path]) -> Path:
        """Write ``<path>`` (selection table CSV) and ``<path>.json``
        (selected names + settings)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.table.write_csv(out)
        out.with_suffix(out.suffix + ".json").write_text(
            json.dumps(
                {
                    "method": self.method,
                    "params": self.params,
                    "selected": self.selected,
                },
                indent=2,
                default=str,
            )
        )
        return out


class FeatureSelector:
    """Select features for one feature set and label.

    Args:
        method: ``"boruta"`` (default), ``"screen"`` (importance screen only)
            or ``"none"`` (prevalence filter only).
        min_prevalence: Keep features non-null in at least this fraction of
            the samples (``0`` disables).
        screen_top_k: Before Boruta, keep this many features by quick-forest
            importance (``None`` disables).
        perc: Boruta's percentile of shadow importance a feature must beat
            (lower is more permissive; 100 is the original, stricter test).
        max_depth: Depth of the forest wrapped by Boruta / the screen.
        n_estimators: ``"auto"`` lets Boruta size the forest.
        max_iter: Boruta iterations.
        random_state: Seed. Also fixes the column/row shuffle.
        n_jobs: Threads for the forests.
    """

    def __init__(
        self,
        method: str = "boruta",
        min_prevalence: float = 0.05,
        screen_top_k: Optional[int] = None,
        perc: int = 85,
        max_depth: int = 7,
        n_estimators: Union[int, str] = "auto",
        max_iter: int = 100,
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if not 0 <= min_prevalence <= 1:
            raise ValueError("min_prevalence must be in [0, 1]")
        self.method = method
        self.min_prevalence = min_prevalence
        self.screen_top_k = screen_top_k
        self.perc = perc
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def params(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "min_prevalence": self.min_prevalence,
            "screen_top_k": self.screen_top_k,
            "perc": self.perc,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

    def _forest(self, n_estimators: int = 100) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> SelectionResult:
        """Select features.

        Args:
            X: (n_samples x n_features) matrix. Absent entries should be NaN
                if the prevalence filter is to mean anything; they are set to
                0 before any model is fitted.
            y: Target.
            feature_names: Column names of ``X``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        names = list(feature_names)
        if X.shape[1] != len(names):
            raise ValueError("feature_names does not match the columns of X")
        n_features = len(names)

        prevalence = (~np.isnan(X)).mean(axis=0) if len(X) else np.zeros(n_features)
        keep = prevalence >= self.min_prevalence
        if not keep.any():
            logger.warning(
                "No feature reaches prevalence %.3f; keeping all features",
                self.min_prevalence,
            )
            keep = np.ones(n_features, dtype=bool)
        kept_prevalence = keep.copy()
        idx = np.where(keep)[0]
        Xk = np.nan_to_num(X[:, idx], nan=0.0)

        screen_keep = np.ones(len(idx), dtype=bool)
        if (
            self.method != "none"
            and self.screen_top_k is not None
            and len(idx) > self.screen_top_k
        ):
            forest = self._forest().fit(Xk, y)
            order = np.argsort(-forest.feature_importances_, kind="stable")
            screen_keep = np.zeros(len(idx), dtype=bool)
            screen_keep[order[: self.screen_top_k]] = True
            logger.info(
                "Screened %d -> %d features by forest importance",
                len(idx),
                int(screen_keep.sum()),
            )
        kept_screen = np.zeros(n_features, dtype=bool)
        kept_screen[idx[screen_keep]] = True

        ranking = np.full(n_features, np.nan)
        selected_mask = np.zeros(n_features, dtype=bool)
        cand = idx[screen_keep]

        if self.method == "none":
            selected_mask[cand] = True
        elif self.method == "screen":
            selected_mask[cand] = True
            ranking[cand] = 1
        else:
            selected_mask, ranking = self._boruta(
                np.nan_to_num(X[:, cand], nan=0.0), y, cand, n_features
            )

        selected = [n for n, m in zip(names, selected_mask) if m]
        table = pl.DataFrame(
            {
                "feature": names,
                "prevalence": prevalence,
                "kept_prevalence": kept_prevalence,
                "kept_screen": kept_screen,
                "selected": selected_mask,
                "ranking": [None if np.isnan(r) else int(r) for r in ranking],
            },
            schema_overrides={"ranking": pl.Int64},
        )
        logger.info(
            "Feature selection (%s): %d of %d features selected",
            self.method,
            len(selected),
            n_features,
        )
        return SelectionResult(
            selected=selected,
            table=table,
            method=self.method,
            params=self.params(),
        )

    def _boruta(
        self,
        Xc: np.ndarray,
        y: np.ndarray,
        cand: np.ndarray,
        n_features: int,
    ) -> Any:
        _patch_numpy_aliases()
        try:
            from boruta import BorutaPy
        except ImportError as exc:
            raise ImportError(
                "Boruta selection needs the optional dependency 'boruta_py' "
                "(extra 'selection'); it pins old numpy / scikit-learn, so "
                "use an isolated environment. Use method='screen' or 'none' "
                "without it."
            ) from exc
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(Xc.shape[1])  # shuffle columns, as the workflow does
        boruta = BorutaPy(
            self._forest(),
            n_estimators=self.n_estimators,
            perc=self.perc,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,
        )
        boruta.fit(Xc[:, perm], y)
        support = np.zeros(Xc.shape[1], dtype=bool)
        support[perm] = np.asarray(boruta.support_, dtype=bool)
        rank = np.zeros(Xc.shape[1])
        rank[perm] = np.asarray(boruta.ranking_, dtype=float)
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[cand[support]] = True
        ranking = np.full(n_features, np.nan)
        ranking[cand] = rank
        return selected_mask, ranking


def select_features_for_dataset(
    dataset: "Dataset",
    feature_set: str,
    label: str,
    selector: Optional[FeatureSelector] = None,
    split: str = "train",
) -> SelectionResult:
    """Run a selector on a dataset's feature set and label.

    Uses the holdout-train samples by default and keeps absent entries as NaN
    for the prevalence filter.
    """
    selector = selector or FeatureSelector()
    xy = dataset.get_xy(feature_set, label, split=split, fillna=None)
    return selector.select(xy.X, xy.y, xy.feature_names)
