"""Model types for regression: how each is built, how its hyperparameters are
named and how its feature importances are read.

  rf     RandomForestRegressor
  gb     GradientBoostingRegressor
  xgb    XGBRegressor                (optional dependency)
  ridge  Ridge regression            (median imputation + standardisation + Ridge)
  en     ElasticNet                  (median imputation + standardisation + ElasticNet)
  mlp    MLPRegressor                (median imputation + standardisation + MLP)

Linear models and the MLP are sensitive to feature scale (microbiome features
easily span 1e-9 ... 1e7), so they are scikit-learn Pipelines with the steps
``impute``, ``scale`` and ``model``. Their hyperparameters are written in a
parameter file under the natural name (``alpha``) and addressed as
``model__alpha`` by :func:`prefix_params`. Pipelines pickle like any other
estimator (no custom classes), so another environment needs only scikit-learn.

:func:`feature_importance` returns importances that sum to 1 for every model
type, so an importance cut-off means the same thing for all of them:

  forests / boosting  impurity importances
  ridge / en          |coefficient| on standardised features (comparable
                      across features)
  mlp                 summed |weight| of the first layer per feature (a crude
                      proxy; use with care)
"""

from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

# name -> (module, class, scaled, fixed parameters)
MODEL_SPECS: Dict[str, Tuple[str, str, bool, Dict[str, Any]]] = {
    "rf": ("sklearn.ensemble", "RandomForestRegressor", False, {}),
    "gb": ("sklearn.ensemble", "GradientBoostingRegressor", False, {}),
    "xgb": ("xgboost", "XGBRegressor", False, {}),
    "ridge": ("sklearn.linear_model", "Ridge", True, {}),
    "en": (
        "sklearn.linear_model",
        "ElasticNet",
        True,
        {"max_iter": 20000, "tol": 1e-3},
    ),
    "mlp": (
        "sklearn.neural_network",
        "MLPRegressor",
        True,
        {"max_iter": 500, "early_stopping": True, "random_state": 0},
    ),
}
STEP = "model"

# Human-friendly spellings accepted wherever a model is named.
ALIASES: Dict[str, List[str]] = {
    "rf": [
        "rf",
        "r f",
        "rforest",
        "r forest",
        "randomforest",
        "random forest",
        "random forest regressor",
        "random-forest",
        "random_forest",
    ],
    "gb": [
        "gb",
        "gradient boosting",
        "gradient_boosting",
        "gradient-boosting",
        "gradientboosting",
    ],
    "xgb": ["xgb", "xgboost", "x gboost", "x g b"],
    "ridge": ["ridge", "ridge regression", "ridge_regression"],
    "en": [
        "en",
        "elasticnet",
        "elastic net",
        "elastic-net",
        "elastic_net",
    ],
    "mlp": ["mlp", "neural network", "neural_network", "mlpregressor"],
}


def known(name: str) -> None:
    """Raise ``ValueError`` if ``name`` is not a model type."""
    if name not in MODEL_SPECS:
        raise ValueError(
            f"unknown model type {name!r}; known: {sorted(MODEL_SPECS)}"
        )


def is_scaled(name: str) -> bool:
    """True for model types wrapped in an impute + scale pipeline."""
    known(name)
    return MODEL_SPECS[name][2]


def build(name: str, **params: Any) -> Any:
    """A new estimator of model type ``name``.

    ``params`` are already addressed for the built estimator (see
    :func:`prefix_params`).
    """
    known(name)
    module, cls, scaled, fixed = MODEL_SPECS[name]
    try:
        klass = getattr(import_module(module), cls)
    except ImportError as exc:
        raise ImportError(
            f"Model {name!r} needs the optional dependency "
            f"{module.split('.')[0]!r}; add it to the environment."
        ) from exc
    estimator = klass(**fixed)
    if not scaled:
        return estimator.set_params(**params) if params else estimator
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (STEP, estimator),
        ]
    )
    return pipe.set_params(**params) if params else pipe


def prefix_params(name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Hyperparameters as written in the parameter file -> names the built
    estimator understands (``alpha`` -> ``model__alpha`` for pipelines)."""
    known(name)
    if not is_scaled(name):
        return dict(params)
    return {
        (k if "__" in k else f"{STEP}__{k}"): v for k, v in params.items()
    }


def prefix_grid(name: str, grid: Any) -> Any:
    """:func:`prefix_params` for a parameter grid (a dict of lists, or a list
    of such dicts)."""
    if isinstance(grid, Mapping):
        return prefix_params(name, grid)
    if isinstance(grid, (list, tuple)):
        return [prefix_params(name, g) for g in grid]
    return grid


def resolve(name: str) -> str:
    """Canonical model type for a name or alias (``"random forest"`` ->
    ``"rf"``)."""
    if not isinstance(name, str):
        raise ValueError(f"Model alias must be a string, got: {type(name)}")
    norm = _normalise(name)
    for canon, aliases in ALIASES.items():
        if any(_normalise(a) == norm for a in aliases):
            return canon
    if norm in MODEL_SPECS:
        return norm
    raise ValueError(f"Unknown model alias: {name}")


def _normalise(name: str) -> str:
    s = (name or "").lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def final_estimator(model: Any) -> Any:
    """The regressor inside a Pipeline (or the estimator itself)."""
    # last step, not STEP: cv.ScaledSVR names its regressor `svr`
    return model.steps[-1][1] if hasattr(model, "steps") else model


def feature_importance(
    model: Any, n_features: Optional[int] = None
) -> np.ndarray:
    """Normalised (sum to 1) feature importances of a fitted model."""
    est = final_estimator(model)
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.ravel(est.coef_)).astype(float)
    elif hasattr(est, "coefs_"):
        imp = np.abs(est.coefs_[0]).sum(axis=1).astype(float)
    else:
        raise ValueError(f"no feature importances for {type(est).__name__}")
    total = imp.sum()
    return imp / total if total > 0 else imp


def has_trees(model: Any) -> bool:
    """True for random forests (leaf indices and per-tree structure are
    available); boosting has trees but no per-tree leaf support."""
    return type(final_estimator(model)).__module__.startswith(
        "sklearn.ensemble._forest"
    )


def fit_weighted(
    model: Any,
    X: Any,
    y: Any,
    sample_weight: Union[None, np.ndarray] = None,
) -> Any:
    """Fit with sample weights, routing them to the regressor inside a
    Pipeline."""
    if sample_weight is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        step = model.steps[-1][0]
        return model.fit(X, y, **{f"{step}__sample_weight": sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def predict_with_spread(model: Any, X: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Mean prediction and the standard deviation across trees.

    Only for random forests (see :func:`has_trees`); the spread is the basis
    of locally weighted conformal intervals.
    """
    if not has_trees(model):
        raise ValueError("predict_with_spread needs a random forest")
    est = final_estimator(model)
    Xt = X
    if hasattr(model, "named_steps"):
        for name, step in model.steps[:-1]:
            Xt = step.transform(Xt)
    per_tree = np.stack([t.predict(Xt) for t in est.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)
