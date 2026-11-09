"""Numerical checks of species -> sample aggregation against numpy references,
plus the lazy species table (Parquet/CSV scans, accession -> taxonomy map)."""

import numpy as np
import polars as pl
import pytest

from microbiome_ml.core.config import (
    AggregationConfig,
    AggregationMethod,
    WeightingStrategy,
)
from microbiome_ml.wrangle.aggregation import FeatureAggregator
from microbiome_ml.wrangle.features import SpeciesFeatureSet
from microbiome_ml.wrangle.species_table import LazySpeciesFeatureSet

TAX = [f"d__B;p__P;c__C;o__O;f__F;g__G;s__sp{i}" for i in range(6)]
FEATS = ["f1", "f2", "f3"]
VALUES = np.array(
    [
        [1.0, 0.0, 2.0],
        [3.0, 2.0, 0.0],
        [5.0, 4.0, 1.0],
        [2.0, 8.0, 3.0],
        [0.0, 1.0, 6.0],
        [7.0, 3.0, 9.0],
    ]
)
# sample -> {species index: relabund}
PROFILES = {
    "A": {0: 0.5, 1: 0.3, 2: 0.2},
    "B": {0: 0.6, 2: 0.3, 3: 0.1},
    "C": {1: 0.4, 3: 0.4, 4: 0.15, 5: 0.05},
}
NOT_IN_TABLE = "d__B;p__P;c__C;o__O;f__F;g__G;s__absent"


def _profiles() -> pl.LazyFrame:
    rows = [
        (s, TAX[i], r) for s, d in PROFILES.items() for i, r in d.items()
    ]
    # a species without features and a higher-rank row must be ignored
    rows.append(("A", NOT_IN_TABLE, 0.05))
    rows.append(("A", "d__B;p__P", 0.9))
    return pl.DataFrame(
        rows, schema=["sample", "taxonomy", "relabund"], orient="row"
    ).lazy()


def _species_set() -> SpeciesFeatureSet:
    return SpeciesFeatureSet(TAX, FEATS, VALUES.copy(), "traits")


def _weights(r: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "none":
        return np.ones_like(r)
    if weighting == "abundance":
        return r
    return np.sqrt(r)


def _expected(method: str, weighting: str, **kw) -> dict:
    out = {}
    for s, d in PROFILES.items():
        idx = np.array(list(d))
        r = np.array([d[i] for i in idx])
        x = VALUES[idx]
        w = _weights(r, weighting)
        if method == "arithmetic_mean":
            v = (x * w[:, None]).sum(0) / w.sum()
        elif method == "geometric_mean":
            eps = kw.get("pseudocount", 1e-8)
            v = np.exp((np.log(x + eps) * w[:, None]).sum(0) / w.sum())
        elif method == "harmonic_mean":
            eps = kw.get("pseudocount", 1e-8)
            v = w.sum() / (w[:, None] / (x + eps)).sum(0)
        elif method == "median":
            v = np.median(x, axis=0)
        elif method == "max_abundance":
            v = x[np.argmax(r)]
        elif method == "min_abundance":
            v = x[np.argmin(r)]
        elif method == "presence_absence":
            keep = r >= kw.get("min_abundance", 0.001)
            v = ((x > 0) & keep[:, None]).any(axis=0).astype(float)
        elif method == "top_k_abundant":
            k = kw["k"]
            top = np.argsort(-r, kind="stable")[:k]
            ww = _weights(r[top], weighting)
            v = (x[top] * ww[:, None]).sum(0) / ww.sum()
        else:
            raise AssertionError(method)
        out[s] = v
    return out


def _run(species, method, weighting="none", **kw) -> dict:
    cfg = AggregationConfig(
        method=AggregationMethod(method),
        weighting=WeightingStrategy(weighting),
        uncommon_feature_cutoff=1,
        **kw,
    )
    res = FeatureAggregator().get_sample_features(
        species, _profiles(), cfg
    )
    df = res.collect()
    return {
        row["acc"]: np.array([row[f] for f in FEATS])
        for row in df.iter_rows(named=True)
    }


@pytest.mark.parametrize("weighting", ["none", "abundance", "sqrt_abundance"])
@pytest.mark.parametrize(
    "method", ["arithmetic_mean", "geometric_mean", "harmonic_mean"]
)
def test_weighted_means_match_numpy(method, weighting):
    got = _run(_species_set(), method, weighting)
    exp = _expected(method, weighting)
    for s in PROFILES:
        np.testing.assert_allclose(got[s], exp[s], rtol=1e-9)


@pytest.mark.parametrize("method", ["median", "max_abundance", "min_abundance"])
def test_unweighted_methods_match_numpy(method):
    got = _run(_species_set(), method, "none")
    exp = _expected(method, "none")
    for s in PROFILES:
        np.testing.assert_allclose(got[s], exp[s])


def test_presence_absence_is_per_feature_and_respects_min_abundance():
    got = _run(_species_set(), "presence_absence", "none", min_abundance=0.1)
    exp = _expected("presence_absence", "none", min_abundance=0.1)
    for s in PROFILES:
        np.testing.assert_array_equal(got[s], exp[s])
    # C: species 4 (0.15) and 5 (0.05): species 5 is below the threshold
    assert set(np.unique(got["C"])) <= {0.0, 1.0}


@pytest.mark.parametrize("weighting", ["none", "abundance"])
def test_top_k_abundant_matches_numpy(weighting):
    got = _run(_species_set(), "top_k_abundant", weighting, k=2)
    exp = _expected("top_k_abundant", weighting, k=2)
    for s in PROFILES:
        np.testing.assert_allclose(got[s], exp[s], rtol=1e-9)


def test_richness_counts_species_with_features():
    cfg = AggregationConfig(
        method=AggregationMethod.RICHNESS,
        weighting=WeightingStrategy.NONE,
        min_abundance=0.1,
        uncommon_feature_cutoff=1,
    )
    df = (
        FeatureAggregator()
        .get_sample_features(_species_set(), _profiles(), cfg)
        .collect()
        .sort("acc")
    )
    assert df["acc"].to_list() == ["A", "B", "C"]
    assert df["richness"].to_list() == [3, 3, 3]  # absent species ignored
    np.testing.assert_allclose(
        df["total_abundance"].to_numpy(), [1.0, 1.0, 0.95]
    )


def test_uncommon_filter_by_count_and_fraction():
    df = pl.DataFrame(
        {
            "acc": [f"S{i}" for i in range(10)],
            "common": [1.0] * 10,
            "half": [1.0] * 5 + [None] * 5,
            "rare": [1.0] + [None] * 9,
        }
    )
    agg = FeatureAggregator()
    assert agg.remove_uncommon_features(df, 2).columns == [
        "acc",
        "common",
        "half",
    ]
    assert agg.remove_uncommon_features(df, 0.6).columns == ["acc", "common"]
    assert agg.remove_uncommon_features(df, 0.05).columns == df.columns
    # fewer samples than an integer cutoff: nothing can be judged rare
    assert agg.remove_uncommon_features(df.head(3), 10).columns == df.columns


def test_uncommon_filter_runs_once_on_a_lazy_frame():
    lf = pl.DataFrame({"acc": ["a", "b"], "x": [1.0, None]}).lazy()
    out = FeatureAggregator().remove_uncommon_features(lf, 1)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["acc", "x"]


def test_column_chunking_gives_identical_result():
    cfg_kwargs = dict(uncommon_feature_cutoff=1)
    for method in ("arithmetic_mean", "geometric_mean", "top_k_abundant"):
        base = FeatureAggregator().get_sample_features(
            _species_set(),
            _profiles(),
            AggregationConfig(
                method=AggregationMethod(method),
                weighting=WeightingStrategy.ABUNDANCE,
                k=2,
                **cfg_kwargs,
            ),
        )
        chunked = FeatureAggregator().get_sample_features(
            _species_set(),
            _profiles(),
            AggregationConfig(
                method=AggregationMethod(method),
                weighting=WeightingStrategy.ABUNDANCE,
                k=2,
                chunk_columns=1,
                **cfg_kwargs,
            ),
        )
        a = base.collect().sort("acc")
        b = chunked.collect().sort("acc")
        assert a.columns == b.columns
        np.testing.assert_allclose(
            a.drop("acc").to_numpy(), b.drop("acc").to_numpy()
        )


def _write_species_files(tmp_path):
    df = pl.DataFrame({"acc": [f"G{i}" for i in range(6)]}).with_columns(
        [pl.Series(f, VALUES[:, j]) for j, f in enumerate(FEATS)]
    )
    csv = tmp_path / "species.csv"
    df.write_csv(csv)
    pq_dir = tmp_path / "pq"
    pq_dir.mkdir()
    df.head(3).write_parquet(pq_dir / "b0.parquet")
    df.tail(3).write_parquet(pq_dir / "b1.parquet")
    mapping = pl.DataFrame(
        {"acc": [f"G{i}" for i in range(6)], "taxonomy": TAX}
    )
    return csv, pq_dir, mapping


@pytest.mark.parametrize("source", ["csv", "parquet_dir"])
def test_lazy_species_set_matches_eager(tmp_path, source):
    csv, pq_dir, mapping = _write_species_files(tmp_path)
    path = csv if source == "csv" else pq_dir
    lazy = LazySpeciesFeatureSet.scan(
        path, "traits", key_col="acc", taxonomy_map=mapping
    )
    assert lazy.feature_names == FEATS
    got = _run(lazy, "geometric_mean", "abundance")
    exp = _expected("geometric_mean", "abundance")
    for s in PROFILES:
        np.testing.assert_allclose(got[s], exp[s], rtol=1e-9)


def test_lazy_species_csv_features_are_read_as_float(tmp_path):
    # integer-looking first rows must not type a column as int
    df = pl.DataFrame({"acc": ["a", "b", "c"], "k": [0, 0, 1]}).with_columns(
        pl.col("k").cast(pl.Float64)
    )
    path = tmp_path / "s.csv"
    path.write_text("acc,k\na,0\nb,0\nc,0.5\n")
    lazy = LazySpeciesFeatureSet.scan(path, "t", key_col="acc")
    assert lazy.collect()["k"].dtype == pl.Float64
    assert df.height == 3


def test_map_rejects_repeated_accessions(tmp_path):
    csv, _, mapping = _write_species_files(tmp_path)
    bad = pl.concat([mapping, mapping.head(1)])
    with pytest.raises(ValueError, match="repeated"):
        LazySpeciesFeatureSet.scan(csv, "t", key_col="acc", taxonomy_map=bad)


def test_map_rejects_shared_taxonomy_unless_averaged(tmp_path):
    csv, _, mapping = _write_species_files(tmp_path)
    shared = mapping.with_columns(
        pl.when(pl.col("acc") == "G1")
        .then(pl.lit(TAX[0]))
        .otherwise(pl.col("taxonomy"))
        .alias("taxonomy")
    )
    with pytest.raises(ValueError, match="share a taxonomy"):
        LazySpeciesFeatureSet.scan(
            csv, "t", key_col="acc", taxonomy_map=shared
        )
    lazy = LazySpeciesFeatureSet.scan(
        csv,
        "t",
        key_col="acc",
        taxonomy_map=shared,
        duplicate_taxa="mean",
    )
    df = lazy.collect()
    assert df.height == 5
    row = df.filter(pl.col("taxonomy") == TAX[0]).row(0, named=True)
    np.testing.assert_allclose(
        [row[f] for f in FEATS], (VALUES[0] + VALUES[1]) / 2
    )


def test_map_drops_unmapped_rows(tmp_path):
    csv, _, mapping = _write_species_files(tmp_path)
    lazy = LazySpeciesFeatureSet.scan(
        csv, "t", key_col="acc", taxonomy_map=mapping.head(4)
    )
    assert lazy.collect().height == 4


def test_lazy_set_normalises_semicolon_spaces(tmp_path):
    df = pl.DataFrame(
        {"taxonomy": ["d__B; p__P; s__x"], "f": [1.0]}
    )
    path = tmp_path / "t.parquet"
    df.write_parquet(path)
    lazy = LazySpeciesFeatureSet.scan(path, "t", key_col="taxonomy")
    assert lazy.collect()["taxonomy"].to_list() == ["d__B;p__P;s__x"]


def test_dataset_add_species_features_lazy(tmp_path):
    from microbiome_ml import Dataset

    csv, _, mapping = _write_species_files(tmp_path)
    ds = Dataset().add_species_features(
        "traits",
        data=csv,
        accession_col="acc",
        lazy=True,
        taxonomy_map=mapping,
    )
    fs = ds.species_feature_sets["traits"]
    assert isinstance(fs, LazySpeciesFeatureSet)
    assert fs.feature_names == FEATS
