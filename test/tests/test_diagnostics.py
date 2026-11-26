"""Tests for the diagnostic figures (drawn with the Agg backend)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from microbiome_ml import steps  # noqa: E402
from microbiome_ml.visualise import diagnostics as dg  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    real = rng.normal(15, 5, n)
    return real, 0.8 * real + rng.normal(2, 2, n)


def _texts(fig):
    out = [t.get_text() for t in fig.texts]
    out += [ax.get_title() for ax in fig.axes]
    out.append(fig._suptitle.get_text() if fig._suptitle else "")
    return "\n".join(out)


def test_prediction_stats_match_numpy():
    real, pred = _data()
    mse, r2, mae, pcc = dg.prediction_stats(pred, real)
    assert mse == pytest.approx(np.mean((real - pred) ** 2))
    assert mae == pytest.approx(np.mean(np.abs(real - pred)))
    assert pcc == pytest.approx(np.corrcoef(real, pred)[0, 1])
    assert r2 == pytest.approx(1 - ((real - pred) ** 2).sum() / ((real - real.mean()) ** 2).sum())


def test_density_plot_title_carries_the_statistics(tmp_path):
    real, pred = _data()
    fig = dg.plot_prediction_density(pred, real, title="RF vs Temp", units="°C")
    txt = _texts(fig)
    mse, r2, mae, pcc = dg.prediction_stats(pred, real)
    assert "RF vs Temp" in txt and f"R2: {r2:.3f}" in txt and f"n: {len(real)}" in txt
    # square axes with the same limits, 1:1 line and regression line drawn
    ax = fig.axes[0]
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_xlabel() == "Actual Values (°C)" and ax.get_ylabel() == "Predicted Values (°C)"
    out = tmp_path / "sub" / "d.png"
    assert dg.plot_prediction_density(pred, real, file_name=out, dpi=60) is None
    assert out.exists() and out.stat().st_size > 1000
    assert plt.get_fignums() == [fig.number]  # the saved figure was closed


def test_two_panels_have_their_own_statistics():
    real, pred = _data()
    real2, pred2 = _data(80, seed=1)
    fig = dg.plot_two_panels((pred, real), (pred2, real2), "Title", "pH", "inferno_r", "CV", "Holdout\nx")
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == 2
    assert titles[0].startswith("CV\n") and f"n: {len(real)}" in titles[0]
    assert titles[1].startswith("Holdout\nx\n") and "n: 80" in titles[1]
    assert fig._suptitle.get_text() == "Title"


def test_group_plot_text_groups_get_a_legend_and_numeric_a_colour_bar():
    real, pred = _data()
    rng = np.random.default_rng(0)
    groups = pl.Series([f"g{i}" for i in rng.integers(0, 5, len(real))])
    fig = dg.plot_predictions_by_group(pred, real, groups, "T", "u", cbar_label="eco")
    hist_ax = [a for a in fig.axes if a.get_xlabel().startswith("Error")][0]
    assert hist_ax.get_legend() is not None
    assert len(hist_ax.get_legend().get_texts()) == 5
    plt.close(fig)

    many = pl.Series([f"g{i:02d}" for i in rng.integers(0, 25, len(real))])
    fig = dg.plot_predictions_by_group(pred, real, many, "T", "u")
    hist_ax = [a for a in fig.axes if a.get_xlabel().startswith("Error")][0]
    assert len(hist_ax.get_legend().get_texts()) == 10  # legend capped at ten
    plt.close(fig)

    fig = dg.plot_predictions_by_group(pred, real, pl.Series(rng.integers(0, 12, len(real))), "T", "u", cbar_label="year")
    assert any(a.get_xlabel() == "year" for a in fig.axes)  # colour bar for a numeric category


def test_group_plot_without_groups_or_with_one_value():
    real, pred = _data()
    for groups in (None, pl.Series(["a"] * len(real))):
        fig = dg.plot_predictions_by_group(pred, real, groups, "T", "u")
        assert [a for a in fig.axes if a.get_visible()]
        plt.close(fig)


def test_logi_groups_writes_one_file_per_group_with_both_parts(tmp_path):
    real, pred = _data(120)
    cv = pl.DataFrame({"group": ["a b"] * 60 + ["b"] * 40 + ["c"] * 20, "real": real, "pred": pred})
    ho = pl.DataFrame({"group": ["a b"] * 10 + ["b"] * 10, "real": real[:20], "pred": pred[:20],
                       "split_method": ["bioproject_grouped"] * 10 + ["sample_stratified_fallback: only 3 bioprojects"] * 10})
    files = dg.plot_logi_groups(cv, ho, tmp_path / "logi", "m [{group}]", "u", dpi=40)
    assert sorted(f.name for f in files) == ["a_b.png", "b.png"]   # "c" has no holdout; spaces become underscores
    assert all(f.exists() for f in files)


def test_style_is_applied_only_while_drawing(tmp_path):
    style = tmp_path / "x.mplstyle"
    style.write_text("axes.labelsize: 21\n")
    real, pred = _data(100)
    before = plt.rcParams["axes.labelsize"]
    fig = dg.plot_prediction_density(pred, real, style=style)
    assert fig.axes[0].xaxis.label.get_size() == 21
    assert plt.rcParams["axes.labelsize"] == before


def test_pca_fits_on_training_rows_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6))
    is_train = np.arange(100) < 70
    c1, evr1 = dg.pca_embedding(X, is_train)
    X2 = X.copy()
    X2[~is_train] += 50  # change only the test rows
    c2, evr2 = dg.pca_embedding(X2, is_train)
    np.testing.assert_allclose(evr1, evr2)
    np.testing.assert_allclose(c1[is_train], c2[is_train])
    with pytest.raises(ValueError, match="Cannot plot a 2D PCA"):
        dg.pca_embedding(X[:, :1], is_train)


def test_collapse_levels_top_n_other_and_na():
    values = ["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"] + [None] * 2
    labels, top, n_levels, has_other, has_na = dg.collapse_levels(values, top_n=2)
    assert top == ["a", "b"] and n_levels == 4 and has_other and has_na
    assert list(labels[-3:]) == ["other", "NA", "NA"]
    assert dg.collapse_levels(["x", "y"], top_n=5)[3:] == (False, False)


def test_view_limits_clip_outliers():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(1000, 2))
    coords[0] = [500, 500]
    xlim, ylim, outside = dg.pca_view_limits(coords, 0.5)
    assert xlim[1] < 20 and outside >= 1
    assert dg.pca_view_limits(coords, None) == (None, None, 0)


def test_short_taxon_names():
    assert dg.short_taxon_names(["d__Bacteria;p__Acid", "g__Foo", "plain"]) == ["Acid", "Foo", "plain"]


def test_top_importances_chart(tmp_path):
    out = tmp_path / "i.png"
    dg.plot_top_importances([f"f{i}" for i in range(15)], list(np.linspace(1, 0.1, 15)), out, top=10, dpi=50)
    assert out.exists()


def _csvs(tmp_path):
    real, pred = _data(150)
    acc = [f"S{i}" for i in range(150)]
    pl.DataFrame({"acc": acc, "real": real, "pred": pred}).write_csv(tmp_path / "cv.csv")
    pl.DataFrame({"acc": acc[:40], "real": real[:40], "pred": pred[:40]}).write_csv(tmp_path / "test.csv")
    rng = np.random.default_rng(1)
    pl.DataFrame({"acc": acc, "group": [f"e{i}" for i in rng.integers(0, 4, 150)], "real": real, "pred": pred}).write_csv(tmp_path / "logo.csv")
    return acc, real, pred


def test_plot_steps_write_images(tmp_path):
    _csvs(tmp_path)
    steps.plot_predictions(tmp_path / "cv.csv", tmp_path / "a.png", "T", "u", dpi=40)
    steps.plot_predictions(tmp_path / "cv.csv", tmp_path / "b.png", "T", "u", validation=tmp_path / "test.csv", dpi=40)
    steps.plot_logo_predictions(tmp_path / "logo.csv", tmp_path / "c.png", "T", "u", group_label="eco", dpi=40)
    assert all((tmp_path / f).exists() for f in ("a.png", "b.png", "c.png"))
    bad = pl.read_csv(tmp_path / "cv.csv")
    bad.write_csv(tmp_path / "nogroup.csv")
    with pytest.raises(Exception):
        steps.plot_logo_predictions(tmp_path / "nogroup.csv", tmp_path / "d.png", "T")


def test_plot_pca_and_importances_steps(tmp_path):
    import pickle

    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    n, p = 120, 5
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2 + rng.normal(size=n)
    acc = [f"S{i}" for i in range(n)]
    names = [f"d__B;p__P{i}" for i in range(p)]
    tr = pl.DataFrame({**{k: X[:90, i] for i, k in enumerate(names)}, "target": y[:90], "acc": acc[:90]})
    te = pl.DataFrame({**{k: X[90:, i] for i, k in enumerate(names)}, "target": y[90:], "acc": acc[90:]})
    tr.write_csv(tmp_path / "train.csv")
    te.write_csv(tmp_path / "test.csv")
    pl.DataFrame({"acc": acc, "domain": rng.choice(["m", "t", None], n), "bioproject": [f"b{i // 6}" for i in range(n)]}).write_csv(tmp_path / "meta.csv")
    steps.plot_pca(tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "meta.csv", ["domain", "bioproject"], tmp_path / "pca.png", "PCA", "Temperature", top_n=3, dpi=40)
    assert (tmp_path / "pca.png").exists()

    model = RandomForestRegressor(10, random_state=0).fit(tr.drop("target", "acc").to_numpy(), tr["target"].to_numpy())
    pickle.dump(model, open(tmp_path / "m.pkl", "wb"))
    table = steps.plot_feature_importances(tmp_path / "m.pkl", tmp_path / "train.csv", tmp_path / "imp.png", tmp_path / "imp.csv", "rf", "temp", "rank_relabund", shorten_taxa=True)
    assert table.columns == ["feature", "importance", "model", "target", "feature_set"]
    assert table["importance"].is_sorted(descending=True) and table["importance"].sum() == pytest.approx(1.0)
    assert (tmp_path / "imp.png").exists() and pl.read_csv(tmp_path / "imp.csv").height == p
