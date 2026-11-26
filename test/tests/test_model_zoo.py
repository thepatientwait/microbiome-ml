"""Tests for the model zoo and its use by the CrossValidator."""

import json

import numpy as np
import polars as pl
import pytest

from microbiome_ml.train import model_zoo as zoo
from microbiome_ml.train.cv import CrossValidator


def _data(n=60, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X @ np.array([2.0, -1.0, 0.0, 0.5][:p]) + rng.normal(0, 0.1, n)
    return X, y


@pytest.mark.parametrize("name", ["rf", "gb", "ridge", "en", "mlp"])
def test_build_fit_predict_and_importance(name):
    X, y = _data()
    est = zoo.build(name)
    est.fit(X, y)
    pred = est.predict(X)
    assert pred.shape == y.shape
    imp = zoo.feature_importance(est)
    assert imp.shape == (X.shape[1],)
    assert imp.sum() == pytest.approx(1.0)
    assert np.all(imp >= 0)


def test_scaled_models_are_pipelines_with_named_steps():
    for name in ("ridge", "en", "mlp"):
        est = zoo.build(name)
        assert [s for s, _ in est.steps] == ["impute", "scale", "clip", "model"]
    assert not hasattr(zoo.build("rf"), "named_steps")


def test_scaled_models_impute_missing_values():
    X, y = _data()
    X[3, 1] = np.nan
    zoo.build("ridge").fit(X, y)  # would raise without the imputer


@pytest.mark.parametrize("name", ["ridge", "en", "mlp"])
def test_scaled_models_survive_an_almost_constant_feature(name):
    # training SD ~1e-8 for feature 0, test value 1.0: without the clip step that is z ~ 1e7 and a wild prediction
    rng = np.random.default_rng(0)
    X, y = _data(n=200, p=4)
    X[:, 0] = 1e-8 + rng.uniform(0, 1e-8, len(X))
    est = zoo.build(name).fit(X, y)
    X_test = X[:20].copy()
    X_test[0, 0] = 1.0
    pred = est.predict(X_test)
    assert np.all(np.isfinite(pred))
    assert np.abs(pred).max() < 100 * np.abs(y).max()


def test_pickled_pipeline_needs_only_sklearn_and_numpy():
    import pickle

    est = zoo.build("ridge").fit(*_data())
    assert b"microbiome_ml" not in pickle.dumps(est)


def test_prefix_params_only_for_scaled_models():
    assert zoo.prefix_params("ridge", {"alpha": 1.0}) == {"model__alpha": 1.0}
    assert zoo.prefix_params("ridge", {"model__alpha": 1.0}) == {
        "model__alpha": 1.0
    }
    assert zoo.prefix_params("rf", {"n_estimators": 5}) == {"n_estimators": 5}
    assert zoo.prefix_grid("en", {"alpha": [1, 2]}) == {"model__alpha": [1, 2]}
    assert zoo.prefix_grid("en", [{"alpha": [1]}, {"l1_ratio": [0.1]}]) == [
        {"model__alpha": [1]},
        {"model__l1_ratio": [0.1]},
    ]


def test_build_with_params_sets_them():
    est = zoo.build("ridge", **zoo.prefix_params("ridge", {"alpha": 7.0}))
    assert est.get_params()["model__alpha"] == 7.0
    assert zoo.build("rf", n_estimators=7).n_estimators == 7


def test_resolve_aliases():
    assert zoo.resolve("Random Forest") == "rf"
    assert zoo.resolve("xgboost") == "xgb"
    assert zoo.resolve("elastic-net") == "en"
    assert zoo.resolve("ridge") == "ridge"
    with pytest.raises(ValueError):
        zoo.resolve("nonsense")


def test_unknown_model_type():
    with pytest.raises(ValueError, match="unknown model type"):
        zoo.build("nope")


def test_xgb_without_the_package_has_a_clear_error():
    pytest.importorskip("sklearn")
    try:
        import xgboost  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="optional dependency"):
            zoo.build("xgb")


def test_has_trees_only_for_forests():
    X, y = _data()
    assert zoo.has_trees(zoo.build("rf", n_estimators=5).fit(X, y))
    assert not zoo.has_trees(zoo.build("gb", n_estimators=5).fit(X, y))
    assert not zoo.has_trees(zoo.build("ridge").fit(X, y))


def test_fit_weighted_routes_weights_through_pipelines():
    X, y = _data()
    w = np.ones(len(y))
    w[:10] = 100.0
    zoo.fit_weighted(zoo.build("ridge"), X, y, w)
    zoo.fit_weighted(zoo.build("rf", n_estimators=5), X, y, w)
    zoo.fit_weighted(zoo.build("rf", n_estimators=5), X, y, None)


def test_importances_and_weights_reach_the_svm_inside_its_pipeline():
    from microbiome_ml.train.cv import ScaledSVR

    X, y = _data()
    est = zoo.fit_weighted(ScaledSVR(), X, y, np.ones(len(y)))
    assert zoo.feature_importance(est).sum() == pytest.approx(1.0)


def test_predict_with_spread_for_forests():
    X, y = _data()
    est = zoo.build("rf", n_estimators=20, random_state=0).fit(X, y)
    mean, sd = zoo.predict_with_spread(est, X)
    np.testing.assert_allclose(mean, est.predict(X), atol=1e-9)
    assert (sd >= 0).all() and sd.max() > 0
    with pytest.raises(ValueError):
        zoo.predict_with_spread(zoo.build("ridge").fit(X, y), X)


class _FS:
    def __init__(self, samples, X):
        self._df = pl.DataFrame(
            {"sample": samples, **{f"f{i}": X[:, i] for i in range(X.shape[1])}}
        )

    def to_df(self):
        return self._df


class _DS:
    def __init__(self, n=40):
        X, y = _data(n)
        samples = [f"s{i}" for i in range(n)]
        self.feature_sets = {"feat": _FS(samples, X)}
        self.labels = pl.DataFrame({"sample": samples, "target": y})
        self.splits = {"target": {"k": {s: i % 4 for i, s in enumerate(samples)}}}

    def iter_labels(self):
        yield ("target", self.labels)

    def iter_cv_folds(self, label=None):
        yield ("target", "k", None)


@pytest.mark.parametrize("mode", ["run", "run_grid"])
@pytest.mark.parametrize(
    "model,grid",
    [
        ("ridge", {"alpha": [0.1, 10.0]}),
        ("en", {"alpha": [0.01], "l1_ratio": [0.5]}),
        ("rf", {"n_estimators": [5]}),
    ],
)
def test_cross_validator_runs_zoo_models(tmp_path, mode, model, grid):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({model: grid}))
    cv = CrossValidator(_DS(), models=model, label="target", scheme="k")
    res = getattr(cv, mode)(param_path=str(p))
    assert res
    best = max(res.values(), key=lambda r: r.avg_validation_r2)
    assert best.avg_validation_r2 > 0.5
    # the stored parameters must be accepted by a clone of the stored model
    from sklearn.base import clone

    clone(best.model).set_params(**best.best_params)
    if model in ("ridge", "en"):
        assert all(k.startswith("model__") for k in best.best_params)
