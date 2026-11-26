"""Sparse aggregation of long-format genome annotations against the dense
computation, and KEGG module completeness against a set-based reference."""

import numpy as np
import polars as pl
import pytest

from microbiome_ml.core.config import AggregationConfig
from microbiome_ml.wrangle.aggregation import FeatureAggregator
from microbiome_ml.wrangle.annotations import (
    aggregate_annotations,
    iter_kegg_completeness,
    read_kegg_mapping,
    scan_annotation_matrix,
)
from microbiome_ml.wrangle.species_table import LazySpeciesFeatureSet

N_GENOMES = 40
N_FEATURES = 12
TAXA = [f"d__A;p__P;c__C;o__O;f__F;g__G;s__sp{i}" for i in range(N_GENOMES)]


def _world(seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.integers(0, 6, size=(N_GENOMES, N_FEATURES)) * (
        rng.random((N_GENOMES, N_FEATURES)) < 0.35
    )
    dense[7] = 0  # a genome with no annotation at all
    feats = [f"K{j:05d}" for j in range(N_FEATURES)]
    rows = [
        (f"G{g}", feats[j], int(dense[g, j]))
        for g in range(N_GENOMES)
        for j in range(N_FEATURES)
        if dense[g, j] > 0
    ]
    long = pl.DataFrame(rows, schema=["acc", "KEGG", "count"], orient="row")
    amap = pl.DataFrame({"acc": [f"G{g}" for g in range(N_GENOMES)], "taxonomy": TAXA})
    prof = []
    for s in range(9):
        n = int(rng.integers(3, 14))
        picks = rng.choice(N_GENOMES + 4, size=n, replace=False)  # +4: species with no genome
        rel = rng.random(n) + 0.01
        rel = rel / rel.sum()
        for p, r in zip(picks, rel):
            tax = TAXA[p] if p < N_GENOMES else f"d__A;p__P;c__C;o__O;f__F;g__G;s__none{p}"
            prof.append((f"S{s}", tax, float(r)))
    # a sample whose species have no annotated genome at all
    prof.append(("S_empty", TAXA[7], 1.0))
    profiles = pl.DataFrame(prof, schema=["sample", "taxonomy", "relabund"], orient="row")
    return dense, feats, long, amap, profiles


def _dense_reference(tmp_path, dense, feats, amap, profiles, min_prevalence, method, weighting, **kw):
    """The dense pipeline: prevalence filter, pivot (genomes with a kept
    feature only), FeatureAggregator."""
    occurs = (dense > 0).sum(axis=0)
    min_count = int(N_GENOMES * min_prevalence)
    keep = [j for j in range(N_FEATURES) if occurs[j] >= min_count]
    sub = dense[:, keep]
    has = sub.sum(axis=1) > 0
    table = pl.DataFrame(
        {"acc": [f"G{g}" for g in range(N_GENOMES) if has[g]]}
        | {feats[j]: sub[has, i].astype(float) for i, j in enumerate(keep)}
    )
    path = tmp_path / "dense.parquet"
    table.write_parquet(path)
    species = LazySpeciesFeatureSet.scan(path, "t", key_col="acc", taxonomy_map=amap)
    cfg = AggregationConfig(
        method=method, weighting=weighting, uncommon_feature_cutoff=0, **kw
    )
    return (
        FeatureAggregator()
        .get_sample_features(species, profiles.lazy(), cfg)
        .collect()
        .sort("acc")
    )


def _sparse(tmp_path, long, amap, profiles, min_prevalence, method, weighting, **kw):
    path = tmp_path / "long.csv"
    long.write_csv(path)
    ann = scan_annotation_matrix(
        path, "KEGG", taxonomy_map=amap, min_prevalence=min_prevalence
    )
    return aggregate_annotations(
        ann, profiles.lazy(), method=method, weighting=weighting, chunk_samples=4, **kw
    ).sort("acc")


@pytest.mark.parametrize("min_prevalence", [0.0, 0.3])
@pytest.mark.parametrize("weighting", ["none", "abundance", "sqrt_abundance"])
@pytest.mark.parametrize(
    "method", ["arithmetic_mean", "geometric_mean", "harmonic_mean"]
)
def test_weighted_means_equal_the_dense_computation(tmp_path, method, weighting, min_prevalence):
    dense, feats, long, amap, profiles = _world()
    ref = _dense_reference(tmp_path, dense, feats, amap, profiles, min_prevalence, method, weighting)
    got = _sparse(tmp_path, long, amap, profiles, min_prevalence, method, weighting)
    assert got["acc"].to_list() == ref["acc"].to_list()
    assert "S_empty" not in got["acc"].to_list()
    cols = [c for c in ref.columns if c != "acc"]
    assert sorted(got.columns) == sorted(ref.columns)
    np.testing.assert_allclose(got.select(cols).to_numpy(), ref.select(cols).to_numpy(), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    "method,weighting",
    [
        ("median", "none"),
        ("max_abundance", "none"),
        ("min_abundance", "none"),
        ("presence_absence", "none"),
        ("top_k_abundant", "none"),
        ("top_k_abundant", "abundance"),
    ],
)
def test_per_sample_methods_equal_the_dense_computation(tmp_path, method, weighting):
    dense, feats, long, amap, profiles = _world()
    ref = _dense_reference(tmp_path, dense, feats, amap, profiles, 0.0, method, weighting, k=3)
    got = _sparse(tmp_path, long, amap, profiles, 0.0, method, weighting, k=3)
    assert got["acc"].to_list() == ref["acc"].to_list()
    cols = [c for c in ref.columns if c != "acc"]
    np.testing.assert_allclose(got.select(cols).to_numpy(), ref.select(cols).to_numpy(), rtol=1e-9, atol=1e-12)


def test_richness_equals_the_dense_computation(tmp_path):
    dense, feats, long, amap, profiles = _world()
    ref = _dense_reference(tmp_path, dense, feats, amap, profiles, 0.0, "richness", "none", min_abundance=0.05)
    got = _sparse(tmp_path, long, amap, profiles, 0.0, "richness", "none", min_abundance=0.05)
    assert got["acc"].to_list() == ref["acc"].to_list()
    assert got["richness"].to_list() == ref["richness"].to_list()
    np.testing.assert_allclose(got["total_abundance"].to_numpy(), ref["total_abundance"].to_numpy())


def test_weighted_median_falls_back_to_weighted_mean(tmp_path):
    dense, feats, long, amap, profiles = _world()
    med = _sparse(tmp_path, long, amap, profiles, 0.0, "median", "abundance")
    mean = _sparse(tmp_path, long, amap, profiles, 0.0, "arithmetic_mean", "abundance")
    assert med.equals(mean)


def test_prevalence_filter_and_matrix_shape(tmp_path):
    dense, feats, long, amap, profiles = _world()
    path = tmp_path / "long.csv"
    long.write_csv(path)
    ann = scan_annotation_matrix(path, "KEGG", taxonomy_map=amap, min_prevalence=0.3)
    occurs = (dense > 0).sum(axis=0)
    assert ann.features == [f for f, o in zip(feats, occurs) if o >= int(N_GENOMES * 0.3)]
    assert ann.n_genomes_in_table == long["acc"].n_unique()
    assert ann.matrix.shape[1] == len(ann.features)
    assert ann.matrix.shape[0] == len(ann.taxa) <= N_GENOMES
    with pytest.raises(ValueError, match="No feature occurs"):
        scan_annotation_matrix(path, "KEGG", taxonomy_map=amap, min_prevalence=1.01)


def test_map_must_be_one_to_one(tmp_path):
    dense, feats, long, amap, profiles = _world()
    path = tmp_path / "long.csv"
    long.write_csv(path)
    bad = amap.with_columns(
        pl.when(pl.col("acc") == "G1").then(pl.lit(TAXA[0])).otherwise(pl.col("taxonomy")).alias("taxonomy")
    )
    with pytest.raises(ValueError, match="one-to-one"):
        scan_annotation_matrix(path, "KEGG", taxonomy_map=bad, min_prevalence=0)


def test_without_a_map_the_key_is_the_taxonomy(tmp_path):
    long = pl.DataFrame({"acc": ["tA", "tA", "tB"], "KEGG": ["K1", "K2", "K1"], "count": [1, 2, 3]})
    path = tmp_path / "l.csv"
    long.write_csv(path)
    ann = scan_annotation_matrix(path, "KEGG", min_prevalence=0)
    assert ann.taxa == ["tA", "tB"] and ann.features == ["K1", "K2"]
    assert ann.matrix.toarray().tolist() == [[1.0, 2.0], [3.0, 0.0]]


def _kegg_reference(dense, feats, mapping, profiles, amap):
    ko_in_genome = [{feats[j] for j in range(N_FEATURES) if dense[g, j] > 0} for g in range(N_GENOMES)]
    tax_to_g = {t: g for g, t in enumerate(TAXA)}
    sizes = mapping.group_by("feature").len()
    size = dict(zip(sizes["feature"], sizes["len"]))
    out = {}
    for s, grp in profiles.group_by("sample"):
        s = s[0]
        kos = set()
        for t in grp["taxonomy"]:
            if t in tax_to_g:
                kos |= ko_in_genome[tax_to_g[t]]
        counts = {}
        for f, ko in mapping.iter_rows():
            if ko in kos:
                counts[f] = counts.get(f, 0) + 1
        if counts:
            out[s] = {f: c / size[f] for f, c in counts.items()}
    return out, size


def test_kegg_completeness_matches_a_set_reference(tmp_path):
    dense, feats, long, amap, profiles = _world()
    path = tmp_path / "long.csv"
    long.write_csv(path)
    # a mapping file as KEGG ships it: prefixed ids, tab separated, no header
    rng = np.random.default_rng(3)
    lines = []
    for m in range(5):
        for ko in rng.choice(feats + ["K99999"], size=int(rng.integers(2, 7)), replace=False):
            lines.append(f"md:M{m:05d}\tko:{ko}")
    lines += ["md:M00000\tko:" + feats[0]]  # a duplicated link
    lines += ["path:ko01100\tko:" + feats[1], "path:map00010\tko:" + feats[2], "path:map00010\tko:" + feats[3]]
    mp = tmp_path / "mapping"
    mp.write_text("\n".join(lines) + "\n")

    modules = read_kegg_mapping(mp, "module", keep_if_contains="M0")
    pathways = read_kegg_mapping(mp, "pathway", keep_if_contains="map")
    assert modules.columns == ["feature", "KEGG"] and modules["feature"].str.starts_with("M").all()
    assert set(pathways["feature"]) == {"map00010"}

    ann = scan_annotation_matrix(path, "KEGG", taxonomy_map=amap, min_prevalence=0)
    for mapping in (modules, pathways):
        ref, size = _kegg_reference(dense, feats, mapping, profiles, amap)
        got = pl.concat(list(iter_kegg_completeness(ann, mapping.select("feature", "KEGG"), profiles.lazy(), chunk_samples=3))).sort("acc")
        assert got["acc"].to_list() == sorted(ref)
        order = list(got.columns[1:])
        assert order == [f for f, _ in sorted(size.items(), key=lambda kv: (kv[1], kv[0]))]
        for row in got.iter_rows(named=True):
            for f in order:
                assert row[f] == pytest.approx(ref[row["acc"]].get(f, 0.0))


def test_kegg_completeness_sample_filter(tmp_path):
    dense, feats, long, amap, profiles = _world()
    path = tmp_path / "long.csv"
    long.write_csv(path)
    ann = scan_annotation_matrix(path, "KEGG", taxonomy_map=amap, min_prevalence=0)
    mapping = pl.DataFrame({"feature": ["M1"] * 3, "KEGG": feats[:3]})
    got = pl.concat(list(iter_kegg_completeness(ann, mapping, profiles.lazy(), samples=["S0", "S1"])))
    assert set(got["acc"]) <= {"S0", "S1"}
