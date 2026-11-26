"""End-to-end tests of the file-based workflow steps and the CLI."""

import pickle
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
import yaml

from microbiome_ml import steps
from microbiome_ml.cli import main


def _write_profiles(path, samples=("S1", "S2", "S3")):
    prefixes = ["d", "p", "c", "o", "f", "g", "s"]
    rows = []
    leaves = {
        "S1": {"X/X1/o/f/g/s1": 0.4, "X/X2/o/f/g/s2": 0.2, "Y/Y1/o/f/g/s3": 0.4},
        "S2": {"X/X1/o/f/g/s1": 0.9, "Y/Y1/o/f/g/s3": 0.1},
        "S3": {"Z/Z1/o/f/g/s9": 1.0},
    }
    for s in samples:
        totals = {}
        for leaf, v in leaves[s].items():
            parts = ["A"] + leaf.split("/")
            for d in range(1, 8):
                key = ";".join(f"{prefixes[i]}__{parts[i]}" for i in range(d))
                totals[key] = totals.get(key, 0.0) + v
        rows += [(s, t, round(v, 6)) for t, v in totals.items()]
    pl.DataFrame(rows, schema=["sample", "taxonomy", "relabund"], orient="row").write_csv(path)


def test_rank_features_command(tmp_path):
    prof = tmp_path / "p.csv"
    _write_profiles(prof)
    out = tmp_path / "out" / "phylum.csv"
    rc = main(["rank-features", "--profiles", str(prof), "--rank", "phylum", "--out", str(out), "--chunk-samples", "2"])
    assert rc == 0
    df = pl.read_csv(out).sort("acc")
    assert df.columns[0] == "acc"
    assert sorted(df.columns[1:]) == ["p__X", "p__Y", "p__Z"]
    # absent taxa are written as 0.0 (create_features fills them)
    assert df["p__X"].to_list() == [pytest.approx(0.6), pytest.approx(0.9), 0.0]
    assert df["p__Z"].to_list() == [0.0, 0.0, 1.0]


def test_aggregate_command_matches_numpy(tmp_path):
    taxa = [f"d__A;p__P;c__C;o__O;f__F;g__G;s__sp{i}" for i in range(4)]
    prof = pl.DataFrame(
        {
            "sample": ["A", "A", "A", "B", "B"],
            "taxonomy": [taxa[0], taxa[1], taxa[2], taxa[0], taxa[3]],
            "relabund": [0.5, 0.3, 0.2, 0.7, 0.3],
        }
    )
    prof.write_csv(tmp_path / "rel.csv")
    vals = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0], [2.0, 8.0]])
    sp = pl.DataFrame({"acc": [f"G{i}" for i in range(4)], "f1": vals[:, 0], "f2": vals[:, 1]})
    sp.write_parquet(tmp_path / "sp.parquet")
    pl.DataFrame({"acc": sp["acc"], "taxonomy": taxa}).write_csv(tmp_path / "map.csv")
    out = tmp_path / "agg.csv"
    rc = main(
        [
            "aggregate", "--profiles", str(tmp_path / "rel.csv"),
            "--species-features", str(tmp_path / "sp.parquet"),
            "--taxonomy-map", str(tmp_path / "map.csv"),
            "--method", "arithmetic_mean", "--weighting", "abundance",
            "--out", str(out), "--chunk-columns", "1",
        ]
    )
    assert rc == 0
    df = pl.read_csv(out).sort("acc")
    a = (vals[[0, 1, 2]] * np.array([0.5, 0.3, 0.2])[:, None]).sum(0) / 1.0
    b = (vals[[0, 3]] * np.array([0.7, 0.3])[:, None]).sum(0) / 1.0
    np.testing.assert_allclose(df.drop("acc").to_numpy(), np.vstack([a, b]))


def _metadata(n=160, seed=0):
    rng = np.random.default_rng(seed)
    acc = [f"S{i}" for i in range(n)]
    x = rng.normal(size=(n, 8))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(0, 0.2, n)
    return acc, x, y, [f"bp{i // 8}" for i in range(n)]


def test_partition_command_group_pure_and_columns(tmp_path):
    acc, x, y, bp = _metadata()
    meta = pl.DataFrame({"acc": acc, "bioproject": bp, "label~temp": y, "label~ph": [None] * 10 + list(y[10:])})
    meta.write_csv(tmp_path / "meta.csv")
    out = tmp_path / "part.csv"
    rc = main(["partition", "--metadata", str(tmp_path / "meta.csv"), "--label", "label~temp", "label~ph", "--out", str(out)])
    assert rc == 0
    part = pl.read_csv(out)
    assert part.columns == ["acc", "label~temp_partition", "label~ph_partition"]
    j = meta.join(part, on="acc")
    sides = j.group_by("bioproject").agg(pl.col("label~temp_partition").n_unique())
    assert sides["label~temp_partition"].max() == 1
    assert j.filter(pl.col("label~ph").is_null())["label~ph_partition"].null_count() == 10
    assert set(part["label~temp_partition"].unique().to_list()) == {"train", "test"}


def test_partition_unknown_label(tmp_path):
    pl.DataFrame({"acc": ["a"], "bioproject": ["p"]}).write_csv(tmp_path / "m.csv")
    with pytest.raises(ValueError, match="no label column"):
        steps.partition_samples(tmp_path / "m.csv", ["zzz"], tmp_path / "o.csv")
    assert main(["partition", "--metadata", str(tmp_path / "m.csv"), "--label", "zzz", "--out", str(tmp_path / "o.csv")]) == 1


@pytest.fixture
def model_files(tmp_path):
    acc, x, y, bp = _metadata()
    feats = pl.DataFrame({"acc": acc, **{f"f{i}": x[:, i] for i in range(8)}})
    feats.write_csv(tmp_path / "features.csv")
    meta = pl.DataFrame({"acc": acc, "bioproject": bp, "label~y": y})
    meta.write_csv(tmp_path / "meta.csv")
    main(["partition", "--metadata", str(tmp_path / "meta.csv"), "--label", "label~y", "--out", str(tmp_path / "part.csv")])
    full = meta.join(pl.read_csv(tmp_path / "part.csv"), on="acc")
    full.write_csv(tmp_path / "meta_full.csv")
    return tmp_path


def test_select_fit_logo_pipeline_schemas(model_files):
    d = model_files
    rc = main(
        [
            "select-features", "--features", str(d / "features.csv"),
            "--metadata", str(d / "meta_full.csv"), "--label", "label~y",
            "--partition", "label~y_partition", "--method", "screen",
            "--screen-top-k", "3", "--out-train", str(d / "train.csv"),
            "--out-test", str(d / "test.csv"), "--selection-table", str(d / "sel.csv"),
        ]
    )
    assert rc == 0
    train = pl.read_csv(d / "train.csv")
    test = pl.read_csv(d / "test.csv")
    assert train.columns[-2:] == ["target", "acc"]
    assert train.columns[:-2] == test.columns[:-2]
    assert set(train.columns[:-2]) == {"f0", "f1", train.columns[-3]}.union(set(train.columns[:-2]))
    assert {"f0", "f1"} <= set(train.columns)
    part = pl.read_csv(d / "part.csv")
    assert set(train["acc"]) == set(part.filter(pl.col("label~y_partition") == "train")["acc"])
    assert set(test["acc"]) == set(part.filter(pl.col("label~y_partition") == "test")["acc"])

    grid = d / "grids.yaml"
    grid.write_text(yaml.safe_dump({"ridge": {"alpha": [0.1, 10.0]}, "rf": {"n_estimators": [10]}}))
    rc = main(
        [
            "fit", "--train", str(d / "train.csv"), "--test", str(d / "test.csv"),
            "--model", "ridge", "--params-file", str(grid),
            "--out-model", str(d / "m.pkl"), "--out-params", str(d / "best.yaml"),
            "--out-predictions", str(d / "cv_pred.csv"),
            "--out-test-predictions", str(d / "test_pred.csv"),
            "--out-test-metrics", str(d / "test_metrics.csv"),
            "--feature-label", "tax/genus", "--target-label", "y",
        ]
    )
    assert rc == 0
    best = yaml.safe_load((d / "best.yaml").read_text())
    assert list(best) == ["model__alpha"]
    cv_pred = pl.read_csv(d / "cv_pred.csv")
    assert cv_pred.columns == ["acc", "real", "pred"] and cv_pred.height == train.height
    metrics = pl.read_csv(d / "test_metrics.csv")
    assert metrics.columns == ["model", "feature", "target", "mse", "r2", "q2", "pcc", "pval", "n_samples", "comment"]
    assert metrics["feature"][0] == "tax/genus" and metrics["r2"][0] > 0.8
    assert pl.read_csv(d / "test_pred.csv").height == test.height
    with open(d / "m.pkl", "rb") as fh:
        fitted = pickle.load(fh)
    # the pickle loads and predicts on the test table's feature columns
    Xt = test.drop(["acc", "target"])
    assert fitted.predict(Xt).shape == (test.height,)

    rc = main(
        [
            "logo", "--train", str(d / "train.csv"), "--metadata", str(d / "meta_full.csv"),
            "--model", "ridge", "--params", str(d / "best.yaml"),
            "--group-factor", "bioproject",
            "--out-predictions", str(d / "logo_pred.csv"),
            "--out-metrics", str(d / "logo_metrics.csv"),
            "--out-importances", str(d / "logo_imp.csv"),
            "--out-by-group", str(d / "logo_by_group.csv"),
            "--feature-label", "tax/genus", "--target-label", "y",
        ]
    )
    assert rc == 0
    lp = pl.read_csv(d / "logo_pred.csv")
    assert lp.columns == ["acc", "group", "real", "pred"] and lp.height == train.height
    lm = pl.read_csv(d / "logo_metrics.csv")
    assert lm.columns == ["feature", "target", "model", "factor", "mse", "r2", "q2", "pcc", "pval", "n_samples", "comment"]
    assert lm["factor"][0] == "bioproject" and lm["r2"][0] > 0.8
    li = pl.read_csv(d / "logo_imp.csv")
    assert li.columns == ["model", "feature", "target", "factor", "importance"]
    assert li["importance"].sum() == pytest.approx(1.0)
    assert pl.read_csv(d / "logo_by_group.csv").height == lp["group"].n_unique()


def test_cli_log_file_and_failure_exit_code(model_files):
    d = model_files
    log = d / "logs" / "fit.log"
    run = lambda attempt: subprocess.run(  # noqa: E731
        [
            sys.executable, "-m", "microbiome_ml.cli",
            "--log-file", str(log), "--attempt", str(attempt), "--log-label", "fit",
            "fit", "--train", str(d / "nope.csv"), "--model", "rf",
            "--out-model", str(d / "a"), "--out-params", str(d / "b"),
            "--out-predictions", str(d / "c"),
        ],
        capture_output=True, text=True,
    )
    first = run(1)
    assert first.returncode == 1
    text = log.read_text()
    assert "=== fit, attempt 1" in text and "fit failed" in text
    assert "Traceback" in text
    second = run(2)
    assert second.returncode == 1
    text2 = log.read_text()
    assert "attempt 1" in text2 and "attempt 2" in text2  # retry appends


def test_fit_with_rf_and_no_test(model_files):
    d = model_files
    main(["select-features", "--features", str(d / "features.csv"), "--metadata", str(d / "meta_full.csv"), "--label", "label~y", "--partition", "label~y_partition", "--method", "none", "--out-train", str(d / "train.csv"), "--out-test", str(d / "test.csv")])
    grid = d / "g.yaml"
    grid.write_text(yaml.safe_dump({"rf": {"n_estimators": [10], "max_features": [None]}}))
    summary = steps.fit_table_model(d / "train.csv", "rf", yaml.safe_load(grid.read_text())["rf"], d / "m.pkl", d / "b.yaml", d / "p.csv", n_jobs=2)
    assert "test" not in summary
    assert summary["oof"]["r2"] > 0.6
    assert yaml.safe_load((d / "b.yaml").read_text())["n_estimators"] == 10


def test_module_entry_point_runs_as_a_subprocess(tmp_path):
    out = subprocess.run([sys.executable, "-m", "microbiome_ml.cli", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "rank-features" in out.stdout and "select-features" in out.stdout
    bad = subprocess.run([sys.executable, "-m", "microbiome_ml.cli", "fit"], capture_output=True, text=True)
    assert bad.returncode != 0


def test_logi_command_schemas(model_files):
    d = model_files
    # metadata with a coarser grouping (ecoregion) containing several bioprojects
    meta = pl.read_csv(d / "meta_full.csv").with_columns(
        (pl.col("bioproject").str.extract(r"(\d+)").cast(pl.Int64) // 6).cast(pl.Utf8).alias("eco")
    )
    meta.write_csv(d / "meta_eco.csv")
    main(["select-features", "--features", str(d / "features.csv"), "--metadata", str(d / "meta_full.csv"), "--label", "label~y", "--partition", "label~y_partition", "--method", "none", "--out-train", str(d / "train.csv"), "--out-test", str(d / "test.csv")])
    (d / "best.yaml").write_text(yaml.safe_dump({"model__alpha": 1.0}))
    rc = main(
        [
            "logi", "--train", str(d / "train.csv"), "--metadata", str(d / "meta_eco.csv"),
            "--model", "ridge", "--params", str(d / "best.yaml"), "--group-factor", "eco",
            "--out-predictions", str(d / "p.csv"), "--out-metrics", str(d / "m.csv"),
            "--out-importances", str(d / "i.csv"), "--out-holdout-predictions", str(d / "hp.csv"),
            "--out-holdout-metrics", str(d / "hm.csv"), "--feature-label", "tax/genus", "--target-label", "y",
        ]
    )
    assert rc == 0
    cols = ["feature", "target", "model", "factor", "group", "mse", "r2", "q2", "pcc", "pval", "n_samples", "comment"]
    m, hm = pl.read_csv(d / "m.csv"), pl.read_csv(d / "hm.csv")
    assert m.columns == cols and hm.columns == cols
    assert pl.read_csv(d / "p.csv").columns == ["acc", "group", "real", "pred"]
    assert pl.read_csv(d / "hp.csv").columns == ["acc", "group", "real", "pred", "split_method"]
    assert pl.read_csv(d / "i.csv").columns == ["model", "feature", "target", "factor", "group", "importance"]
    assert m["factor"][0] == "eco" and set(m["group"].cast(pl.Utf8)) == set(meta["eco"].unique())


def _annotation_world(tmp_path):
    taxa = [f"d__A;p__P;c__C;o__O;f__F;g__G;s__sp{i}" for i in range(6)]
    rows = [
        ("G0", "K1", 2), ("G0", "K2", 1), ("G1", "K1", 1), ("G2", "K3", 4),
        ("G3", "K2", 2), ("G3", "K3", 1), ("G4", "K1", 3), ("G5", "K9", 1),
    ]
    pl.DataFrame(rows, schema=["acc", "KEGG", "count"], orient="row").write_csv(tmp_path / "ann.csv")
    pl.DataFrame({"acc": [f"G{i}" for i in range(6)], "taxonomy": taxa}).write_csv(tmp_path / "map.csv")
    prof = [
        ("S1", taxa[0], 0.5), ("S1", taxa[1], 0.3), ("S1", taxa[2], 0.2),
        ("S2", taxa[3], 0.6), ("S2", taxa[4], 0.4),
        ("S3", "d__A;p__P;c__C;o__O;f__F;g__G;s__unknown", 1.0),
    ]
    pl.DataFrame(prof, schema=["sample", "taxonomy", "relabund"], orient="row").write_csv(tmp_path / "rel.csv")
    return taxa


def test_aggregate_annotations_command(tmp_path):
    _annotation_world(tmp_path)
    out = tmp_path / "agg.csv"
    rc = main(
        [
            "aggregate-annotations", "--profiles", str(tmp_path / "rel.csv"),
            "--annotations", str(tmp_path / "ann.csv"), "--feature-col", "KEGG",
            "--taxonomy-map", str(tmp_path / "map.csv"), "--min-prevalence", "0.2",
            "--method", "arithmetic_mean", "--weighting", "abundance", "--out", str(out),
            "--chunk-samples", "1",
        ]
    )
    assert rc == 0
    df = pl.read_csv(out).sort("acc")
    # K1 in 3 of 6 genomes, K2 in 2, K3 in 2 (>= int(6*0.2) = 1); K9 in 1 -> kept too (1 >= 1)
    assert df.columns == ["acc", "K1", "K2", "K3", "K9"]
    # S1: genomes G0 (K1=2,K2=1), G1 (K1=1), G2 (K3=4) weighted 0.5/0.3/0.2
    assert df.filter(pl.col("acc") == "S1").row(0) == pytest.approx(("S1", 0.5 * 2 + 0.3 * 1, 0.5 * 1, 0.2 * 4, 0.0))
    assert df["acc"].to_list() == ["S1", "S2"]  # S3's species has no genome


def test_kegg_completeness_command(tmp_path):
    _annotation_world(tmp_path)
    tax = pl.read_csv(tmp_path / "map.csv")
    tsv = tmp_path / "tax.tsv"
    # as shipped: tab separated, no header, spaces in the species name
    tsv.write_text("".join(f"{a}\t{t.replace('_', ' ') if False else t}\n" for a, t in tax.iter_rows()))
    (tmp_path / "mod").write_text("md:M1\tko:K1\nmd:M1\tko:K2\nmd:M2\tko:K3\nmd:M2\tko:K8\n")
    (tmp_path / "path").write_text("path:map1\tko:K1\npath:map1\tko:K3\npath:ko9\tko:K1\n")
    pl.DataFrame({"acc": ["S1", "S2"]}).write_csv(tmp_path / "samples.csv")
    rc = main(
        [
            "kegg-completeness", "--profiles", str(tmp_path / "rel.csv"),
            "--annotations", str(tmp_path / "ann.csv"), "--taxonomy", str(tsv),
            "--modules", str(tmp_path / "mod"), "--pathways", str(tmp_path / "path"),
            "--out-modules", str(tmp_path / "m.csv"), "--out-pathways", str(tmp_path / "p.csv"),
            "--samples", str(tmp_path / "samples.csv"),
        ]
    )
    assert rc == 0
    m = pl.read_csv(tmp_path / "m.csv").sort("acc")
    p = pl.read_csv(tmp_path / "p.csv").sort("acc")
    # S1 has K1,K2,K3 in its genomes: M1 = 2/2, M2 = 1/2 ; S2 has K2,K3,K1 (G3: K2,K3; G4: K1)
    assert set(m.columns) == {"acc", "M1", "M2"}
    assert m.filter(pl.col("acc") == "S1").select("M1", "M2").row(0) == pytest.approx((1.0, 0.5))
    assert m.filter(pl.col("acc") == "S2").select("M1", "M2").row(0) == pytest.approx((1.0, 0.5))
    assert p.columns == ["acc", "map1"] and p["map1"].to_list() == pytest.approx([1.0, 1.0])
