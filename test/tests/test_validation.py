"""Tests for metrics, out-of-fold predictions, LOGO and Dataset.get_xy."""

import numpy as np
import polars as pl
import pytest

from microbiome_ml.train import model_zoo as zoo
from microbiome_ml.train.metrics import regression_metrics
from microbiome_ml.train.validation import (
    leave_one_group_out,
    out_of_fold_predictions,
)


def _grouped_data(n_groups=6, per=15, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for k in range(n_groups):
        x = rng.normal(size=(per, 3))
        X.append(x)
        y.append(x[:, 0] * 2.0 + shift * k + rng.normal(0, 0.1, per))
        g += [f"g{k}"] * per
    return np.vstack(X), np.concatenate(y), np.array(g)


def test_metrics_match_hand_computation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.2, 3.8])
    m = regression_metrics(y, p)
    assert m["n"] == 4
    assert m["mae"] == pytest.approx(0.15)
    assert m["mse"] == pytest.approx(0.025)
    assert m["rmse"] == pytest.approx(np.sqrt(0.025))
    assert m["r2"] == pytest.approx(1 - 0.1 / 5.0)
    assert m["q2"] == m["r2"]
    assert m["pcc"] == pytest.approx(np.corrcoef(y, p)[0, 1])
    assert 0 <= m["pval"] <= 1


def test_metrics_one_sided_pvalue_is_half_for_positive_correlation():
    y = np.arange(10.0)
    p = y + np.random.default_rng(0).normal(0, 1, 10)
    two = regression_metrics(y, p)["pval"]
    one = regression_metrics(y, p, alternative="greater")["pval"]
    assert one == pytest.approx(two / 2)


def test_metrics_degenerate_inputs():
    empty = regression_metrics([], [])
    assert empty["n"] == 0 and empty["r2"] is None
    const = regression_metrics([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert const["r2"] is None and const["pcc"] is None
    assert const["mae"] == pytest.approx(1.0)
    single = regression_metrics([1.0], [2.0])
    assert single["pcc"] is None and single["mae"] == 1.0


def test_out_of_fold_predictions_use_predefined_folds_and_cover_everyone():
    X, y, _ = _grouped_data()
    folds = np.arange(len(y)) % 5
    ids = [f"s{i}" for i in range(len(y))]
    out = out_of_fold_predictions(
        zoo.build("ridge"), X, y, samples=ids, folds=folds
    )
    assert out["sample"].to_list() == ids
    assert out["fold"].to_list() == folds.tolist()
    assert regression_metrics(out["real"], out["pred"])["r2"] > 0.9


def test_out_of_fold_random_folds_are_reproducible():
    X, y, _ = _grouped_data()
    a = out_of_fold_predictions(zoo.build("ridge"), X, y, n_splits=10, seed=1)
    b = out_of_fold_predictions(zoo.build("ridge"), X, y, n_splits=10, seed=1)
    c = out_of_fold_predictions(zoo.build("ridge"), X, y, n_splits=10, seed=2)
    assert a.equals(b)
    assert a["fold"].n_unique() == 10
    assert not a["fold"].equals(c["fold"])


def test_out_of_fold_prediction_never_sees_its_own_sample():
    # memorising model: out-of-fold error must not be ~0
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = rng.normal(size=80)  # no signal
    from sklearn.ensemble import RandomForestRegressor

    oof = out_of_fold_predictions(
        RandomForestRegressor(50, random_state=0), X, y, n_splits=5
    )
    assert regression_metrics(oof["real"], oof["pred"])["r2"] < 0.2


def test_logo_holds_out_whole_groups_and_returns_pooled_and_group_metrics():
    X, y, g = _grouped_data()
    res = leave_one_group_out(
        zoo.build("ridge"),
        X,
        y,
        g,
        samples=[f"s{i}" for i in range(len(y))],
        feature_names=["a", "b", "c"],
    )
    assert res.n_groups == 6
    assert res.predictions.columns == ["sample", "group", "real", "pred"]
    assert res.predictions.height == len(y)
    assert res.metrics["r2"] > 0.9
    assert set(res.by_group["group"].to_list()) == set(g)
    assert (res.by_group["n"] == 15).all()
    assert res.importances is not None
    assert res.importances["importance"].sum() == pytest.approx(1.0)
    assert res.importances["feature"][0] == "a"


def test_logo_is_not_fooled_by_group_specific_offsets():
    # each group has its own offset: a model that cannot know the group's
    # offset must score worse under LOGO than under random folds
    X, y, g = _grouped_data(shift=5.0)
    est = zoo.build("rf", n_estimators=40, random_state=0)
    logo = leave_one_group_out(est, X, y, g, importances=False)
    rnd = out_of_fold_predictions(est, X, y, n_splits=5)
    r2_logo = logo.metrics["r2"]
    r2_rnd = regression_metrics(rnd["real"], rnd["pred"])["r2"]
    assert r2_logo < r2_rnd


def test_logo_needs_two_groups():
    X, y, _ = _grouped_data()
    with pytest.raises(ValueError, match="at least 2 groups"):
        leave_one_group_out(zoo.build("ridge"), X, y, ["a"] * len(y))


def test_logo_null_groups_are_singletons():
    X, y, g = _grouped_data(n_groups=3)
    groups = [None if i % 10 == 0 else gi for i, gi in enumerate(g)]
    res = leave_one_group_out(zoo.build("ridge"), X, y, groups, importances=False)
    assert res.predictions.height == len(y)
    assert res.n_groups > 3


def test_logo_importances_skipped_for_models_without_them():
    X, y, g = _grouped_data()

    class Const:
        def get_params(self, deep=True):
            return {}

        def set_params(self, **p):
            return self

        def fit(self, X, y):
            self.m = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.m)

    res = leave_one_group_out(Const(), X, y, g)
    assert res.importances is None


def _dataset_with_split():
    from microbiome_ml.wrangle.dataset import Dataset
    from microbiome_ml.wrangle.features import SampleFeatureSet
    from microbiome_ml.wrangle.splits import SplitManager

    rng = np.random.default_rng(0)
    n = 60
    samples = [f"s{i}" for i in range(n)]
    bp = [f"bp{i // 6}" for i in range(n)]
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f2[3] = np.nan
    y = f1 * 2 + rng.normal(0, 0.1, n)
    y_missing = y.copy().tolist()
    y_missing[5] = None

    ds = Dataset()
    ds.feature_sets["fs"] = SampleFeatureSet.from_df(
        pl.DataFrame({"sample": samples, "f1": f1, "f2": f2}),
        name="fs",
        acc_column="sample",
    )
    ds.labels = pl.DataFrame({"sample": samples, "y": y_missing})
    ds.groupings = pl.DataFrame({"sample": samples, "bioproject": bp})
    sm = SplitManager("y")
    data = ds.labels.drop_nulls("y").join(ds.groupings, on="sample")
    sm.create_holdout(data, grouping="bioproject", test_size=0.2, n_bins=5)
    ds.splits["y"] = sm
    return ds


def test_get_xy_train_test_all_and_groups():
    ds = _dataset_with_split()
    tr = ds.get_xy("fs", "y", split="train", groupings=["bioproject"])
    te = ds.get_xy("fs", "y", split="test", groupings=["bioproject"])
    al = ds.get_xy("fs", "y", split="all")
    assert tr.feature_names == ["f1", "f2"]
    assert tr.X.shape == (len(tr.samples), 2)
    assert len(tr.samples) + len(te.samples) == len(al.samples) == 59
    assert "s5" not in al.samples  # null label dropped
    assert not np.isnan(al.X).any()  # f2[3] filled
    assert set(tr.groups["bioproject"]).isdisjoint(te.groups["bioproject"])
    assert len(tr.groups["bioproject"]) == len(tr.samples)


def test_get_xy_errors():
    ds = _dataset_with_split()
    with pytest.raises(ValueError, match="not found"):
        ds.get_xy("nope", "y")
    with pytest.raises(ValueError, match="not found"):
        ds.get_xy("fs", "nope")
    with pytest.raises(ValueError, match="split must be"):
        ds.get_xy("fs", "y", split="other")
    with pytest.raises(ValueError, match="Groupings not found"):
        ds.get_xy("fs", "y", groupings=["zzz"])


def test_logo_on_dataset_arrays_end_to_end():
    ds = _dataset_with_split()
    d = ds.get_xy("fs", "y", split="train", groupings=["bioproject"])
    res = leave_one_group_out(
        zoo.build("ridge"),
        d.X,
        d.y,
        d.groups["bioproject"],
        samples=d.samples,
        feature_names=d.feature_names,
    )
    assert res.n_groups == len(set(d.groups["bioproject"]))
    assert res.metrics["r2"] > 0.9


def test_cross_validator_out_of_fold_predictions_and_export(tmp_path):
    import json

    from microbiome_ml.train.cv import CrossValidator
    from microbiome_ml.train.results import CV_Result

    ds = _dataset_with_split()
    ds.create_cv_folds(
        label="y", n_folds=4, grouping="bioproject", use_holdout=True
    )
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"ridge": {"alpha": [0.1, 10.0]}}))
    cv = CrossValidator(ds, models="ridge", label="y", scheme="bioproject")
    cv.run(param_path=str(params), n_jobs=1)
    out = cv.add_oof_predictions()
    assert list(out) == [("y", "bioproject")]
    oof = out[("y", "bioproject")]

    train = set(ds.get_train_samples("y", metadata=False)["sample"].to_list())
    assert set(oof["sample"].to_list()) == train
    assert oof.columns == ["sample", "fold", "real", "pred"]
    assert regression_metrics(oof["real"], oof["pred"])["r2"] > 0.9

    # random K-fold predictions on the same samples
    ten = cv.add_oof_predictions(oof_folds=10, seed=3)[("y", "bioproject")]
    assert ten["fold"].n_unique() == 10
    assert set(ten["sample"].to_list()) == train

    CV_Result.export_best_results_by_scheme(
        cv.best_result_by_label_scheme, tmp_path / "best"
    )
    csv = tmp_path / "best" / "y" / "bioproject" / "oof_predictions.csv"
    assert csv.exists()
    assert pl.read_csv(csv).height == len(train)


def test_logo_for_dataset_and_save(tmp_path):
    import json

    from microbiome_ml.train.validation import logo_for_dataset

    ds = _dataset_with_split()
    res = logo_for_dataset(ds, "fs", "y", "bioproject", zoo.build("ridge"))
    assert res.extra["split"] == "train"
    assert res.extra["n_samples"] == len(res.predictions)
    # the holdout test samples were not touched
    test_ids = set(ds.get_test_samples("y", metadata=False)["sample"].to_list())
    assert test_ids.isdisjoint(res.predictions["sample"].to_list())

    out = res.save(tmp_path / "logo", prefix="y~fs")
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "y~fs_by_group.csv",
        "y~fs_importances.csv",
        "y~fs_metrics.json",
        "y~fs_predictions.csv",
    ]
    rec = json.loads((out / "y~fs_metrics.json").read_text())
    assert rec["grouping"] == "bioproject" and rec["r2"] > 0.9
    assert pl.read_csv(out / "y~fs_predictions.csv").height == len(res.predictions)
