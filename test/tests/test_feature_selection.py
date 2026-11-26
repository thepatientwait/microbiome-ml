"""Tests for feature selection and its use by the CrossValidator/trainer."""

import json

import numpy as np
import polars as pl
import pytest

from microbiome_ml.train.feature_selection import (
    FeatureSelector,
    select_features_for_dataset,
)


def _signal_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(0, 0.2, n)
    names = [f"f{i}" for i in range(12)]
    return X, y, names


def test_none_method_applies_only_the_prevalence_filter():
    X, y, names = _signal_data()
    X[:, 5] = np.nan
    X[:20, 6] = 1.0
    X[20:, 6] = np.nan  # present in 20 / 300 = 6.7 %
    res = FeatureSelector(method="none", min_prevalence=0.1).select(X, y, names)
    assert "f5" not in res.selected and "f6" not in res.selected
    assert len(res.selected) == 10
    assert res.table.filter(pl.col("feature") == "f6")["prevalence"][0] == pytest.approx(
        20 / 300
    )
    low = FeatureSelector(method="none", min_prevalence=0.05).select(X, y, names)
    assert "f6" in low.selected


def test_screen_keeps_the_most_important_features():
    X, y, names = _signal_data()
    res = FeatureSelector(method="screen", screen_top_k=2, random_state=0).select(
        X, y, names
    )
    assert set(res.selected) == {"f0", "f1"}


def test_boruta_selects_the_signal_features():
    pytest.importorskip("boruta")
    X, y, names = _signal_data(n=250)
    res = FeatureSelector(
        method="boruta", min_prevalence=0, max_iter=40, random_state=0, n_jobs=2
    ).select(X, y, names)
    assert {"f0", "f1"} <= set(res.selected)
    assert len(res.selected) < 8
    assert res.table["selected"].sum() == len(res.selected)


def test_selection_is_reproducible():
    X, y, names = _signal_data()
    sel = FeatureSelector(method="screen", screen_top_k=4, random_state=3)
    assert sel.select(X, y, names).selected == sel.select(X, y, names).selected


def test_boruta_missing_gives_a_clear_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "boruta":
            raise ImportError("no boruta")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    X, y, names = _signal_data(n=60)
    with pytest.raises(ImportError, match="boruta_py"):
        FeatureSelector(method="boruta", min_prevalence=0).select(X, y, names)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        FeatureSelector(method="nope")
    with pytest.raises(ValueError):
        FeatureSelector(min_prevalence=2)
    with pytest.raises(ValueError, match="does not match"):
        FeatureSelector().select(np.zeros((3, 2)), np.zeros(3), ["a"])


def test_no_feature_reaching_prevalence_keeps_all():
    X, y, names = _signal_data(n=40)
    X[:] = np.nan
    X[0, 0] = 1.0
    res = FeatureSelector(method="none", min_prevalence=0.9).select(X, y, names)
    assert len(res.selected) == len(names)


def test_save_writes_table_and_json(tmp_path):
    X, y, names = _signal_data()
    res = FeatureSelector(method="screen", screen_top_k=3).select(X, y, names)
    out = res.save(tmp_path / "sel" / "fs~y.csv")
    assert pl.read_csv(out).height == 12
    meta = json.loads((tmp_path / "sel" / "fs~y.csv.json").read_text())
    assert meta["selected"] == res.selected and meta["method"] == "screen"


def _dataset(n=120, seed=0):
    from microbiome_ml.wrangle.dataset import Dataset
    from microbiome_ml.wrangle.features import SampleFeatureSet
    from microbiome_ml.wrangle.splits import SplitManager

    rng = np.random.default_rng(seed)
    samples = [f"s{i}" for i in range(n)]
    cols = {f"f{i}": rng.normal(size=n) for i in range(6)}
    y = 2 * cols["f0"] - cols["f1"] + rng.normal(0, 0.1, n)
    ds = Dataset()
    ds.feature_sets["fs"] = SampleFeatureSet.from_df(
        pl.DataFrame({"sample": samples, **cols}), name="fs", acc_column="sample"
    )
    ds.labels = pl.DataFrame({"sample": samples, "y": y})
    ds.groupings = pl.DataFrame(
        {"sample": samples, "bioproject": [f"bp{i // 8}" for i in range(n)]}
    )
    sm = SplitManager("y")
    sm.create_holdout(
        ds.labels.join(ds.groupings, on="sample"), grouping="bioproject", n_bins=5
    )
    ds.splits["y"] = sm
    return ds


def test_select_for_dataset_uses_only_the_training_split():
    ds = _dataset()
    res = select_features_for_dataset(
        ds, "fs", "y", FeatureSelector(method="screen", screen_top_k=2)
    )
    assert set(res.selected) == {"f0", "f1"}


def test_cv_and_trainer_use_only_the_selected_columns(tmp_path):
    from microbiome_ml.train.cv import CrossValidator
    from microbiome_ml.train.trainer import ModelTrainer

    ds = _dataset()
    ds.create_cv_folds(
        label="y", n_folds=4, grouping="bioproject", use_holdout=True
    )
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"ridge": {"alpha": [1.0]}}))

    cv = CrossValidator(
        ds,
        models="ridge",
        label="y",
        scheme="bioproject",
        feature_subset={"fs": ["f0", "f1"]},
    )
    res = cv.run(param_path=str(params), n_jobs=1)
    best = next(iter(res.values()))
    assert best.feature_names == ["f0", "f1"]
    assert best.model.named_steps["scale"].n_features_in_ == 2

    trainer = ModelTrainer(
        ds, cv.best_result_by_label_scheme, tmp_path / "holdout"
    )
    ev = trainer.train_and_evaluate()
    one = next(iter(ev.values())) if isinstance(ev, dict) else ev
    assert one.feature_names == ["f0", "f1"]
    assert one.metrics["r2"] > 0.9


def test_feature_subset_per_label_wins_and_validates(tmp_path):
    from microbiome_ml.train.cv import CrossValidator

    ds = _dataset()
    cv = CrossValidator(
        ds,
        models="ridge",
        feature_subset={"fs": ["f0"], "fs::y": ["f1", "f2"]},
    )
    assert cv._columns_for("fs", "y", ["f0", "f1", "f2", "f3"]) == ["f1", "f2"]
    assert cv._columns_for("fs", "other", ["f0", "f1"]) == ["f0"]
    assert cv._columns_for("zz", "y", ["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError, match="not in the feature set"):
        cv._columns_for("fs", "y", ["f1"])
