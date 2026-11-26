"""Regression metrics shared by CV, holdout evaluation and LOGO."""

import math
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    alternative: str = "two-sided",
) -> Dict[str, Optional[float]]:
    """Return n, mae, mse, rmse, r2, q2, pcc and pval for a set of
    predictions.

    ``q2`` is ``1 - SS_res / SS_tot`` about the mean of ``y_true``: for a
    single evaluation set it equals ``r2``. It is reported separately because
    out-of-fold predictions are conventionally quoted as Q2.

    ``pcc`` / ``pval`` are the Pearson correlation and its p-value
    (``alternative`` as in :func:`scipy.stats.pearsonr`, e.g. ``"greater"``
    for a one-sided test of a positive correlation). They are ``None`` when
    undefined (fewer than 2 points or a constant input). ``r2`` and ``q2`` are
    ``None`` when ``y_true`` is constant.

    Empty input returns ``n = 0`` and ``None`` for everything else.
    """
    from scipy.stats import pearsonr

    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    out: Dict[str, Optional[float]] = {
        "n": float(y.size),
        "mae": None,
        "mse": None,
        "rmse": None,
        "r2": None,
        "q2": None,
        "pcc": None,
        "pval": None,
    }
    if y.size == 0 or y.size != p.size:
        return out
    err = y - p
    mse = float(np.mean(err**2))
    out["mae"] = float(np.mean(np.abs(err)))
    out["mse"] = mse
    out["rmse"] = math.sqrt(mse)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot > 0:
        r2 = 1.0 - float((err**2).sum()) / ss_tot
        out["r2"] = r2
        out["q2"] = r2
    if y.size >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0:
        try:
            res: Any = pearsonr(y, p, alternative=alternative)
            out["pcc"], out["pval"] = float(res[0]), float(res[1])
        except Exception:  # pragma: no cover - scipy corner cases
            pass
    return out
