"""Sample-level features from long-format genome annotation tables.

Annotation tables such as eggNOG orthologous groups or KEGG orthologs are
stored long: one row per (genome, feature) with a count. Pivoting them to a
dense genome x feature table before aggregating (what a wide table needs)
spends most of its memory on zeros. Here the table is a *sparse* matrix
``F`` (genomes x features) and a sample is a weighted combination of genome
rows, so the means become sparse products:

* arithmetic mean   ``(S F) / W``
* geometric mean    ``exp(log e + (S L) / W)``  with ``L = log((f+e)/e)`` on
  the non-zeros (a genome without the feature contributes ``log e``)
* harmonic mean     ``W / (W0/e + S I)``         with ``I = 1/(f+e)`` on the
  non-zeros and ``W0 = W - S B`` the weight of the genomes lacking the feature
  (``B`` marks the non-zeros)

where ``S`` is the (samples x genomes) weight matrix and ``W`` its row sums.
Median, max/min abundance, top-k and presence/absence use the genome rows of
one sample at a time. All of them equal the dense computation exactly; they
just never build the zeros.

Which genomes count
-------------------
Features are first filtered by *prevalence* (kept if annotated in at least
``min_prevalence`` of the genomes in the table), and only genomes that carry at
least one kept feature enter a sample's denominator (a genome with no kept
annotation is, as in a pivoted table, absent rather than all zeros).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np
import polars as pl

from microbiome_ml.core.config import AggregationMethod, WeightingStrategy
from microbiome_ml.wrangle.species_table import (
    MapInput,
    _as_lazy,
    _scan_table,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
_SEMICOLON_SPACE = r";\s+"


def _sp():
    import scipy.sparse as sp  # imported lazily: scipy is optional elsewhere

    return sp


@dataclass
class AnnotationMatrix:
    """Sparse genome x feature count matrix.

    Attributes:
        taxa: Taxonomy key of each row (genome).
        features: Name of each column, sorted.
        matrix: ``scipy.sparse.csr_matrix`` of counts (float32).
        n_genomes_in_table: Distinct genomes in the source table, before any
            filter (the denominator of the prevalence cut-off).
        min_count: Smallest number of genomes a kept feature occurs in.
    """

    taxa: List[str]
    features: List[str]
    matrix: object
    n_genomes_in_table: int
    min_count: int

    @property
    def shape(self):
        return self.matrix.shape


def scan_annotation_matrix(
    path: PathLike,
    feature_col: str,
    key_col: str = "acc",
    value_col: str = "count",
    taxonomy_map: Optional[MapInput] = None,
    map_key_col: str = "acc",
    map_taxonomy_col: str = "taxonomy",
    min_prevalence: float = 0.05,
    features: Optional[Sequence[str]] = None,
) -> AnnotationMatrix:
    """Read a long annotation table into an :class:`AnnotationMatrix`.

    Args:
        path: CSV or Parquet with ``key_col``, ``feature_col`` and
            ``value_col`` (a row per genome and feature).
        feature_col: Column holding the feature (e.g. ``KEGG``,
            ``eggNOG_OGs``).
        taxonomy_map: ``acc, taxonomy`` table re-keying genomes to taxonomy
            strings (must be one-to-one). Rows without a mapping are dropped.
            Without a map the key column is used as the taxonomy.
        min_prevalence: Keep a feature if it occurs in at least
            ``int(n_genomes * min_prevalence)`` genomes. ``features`` overrides
            the filter with an explicit list.

    The table is read lazily (streamed); only the kept (genome, feature,
    count) triples are materialised, as integer indices and counts.
    """
    p = Path(path)
    if p.is_dir() or p.suffix == ".parquet":
        lf = _scan_table(p, key_col)
    else:
        lf = pl.scan_csv(
            p,
            schema_overrides={
                key_col: pl.Utf8,
                feature_col: pl.Utf8,
                value_col: pl.Float64,
            },
        )
    lf = lf.select(
        pl.col(key_col).alias("_key"),
        pl.col(feature_col).alias("_feature"),
        pl.col(value_col).cast(pl.Float64).alias("_value"),
    )

    n_total = lf.select(pl.col("_key").n_unique()).collect(engine="streaming").item()
    if features is None:
        min_count = int(n_total * min_prevalence)
        counts = (
            lf.group_by("_feature")
            .agg(pl.len().alias("n"))
            .collect(engine="streaming")
        )
        keep = sorted(
            counts.filter(pl.col("n") >= min_count)["_feature"].to_list()
        )
    else:
        keep = sorted(set(features))
        min_count = 0
    if not keep:
        raise ValueError(
            f"No feature occurs in at least {min_count} of {n_total} genomes"
        )
    logger.info(
        "%d of %d features kept (present in >= %d of %d genomes)",
        len(keep),
        len(counts) if features is None else len(keep),
        min_count,
        n_total,
    )

    if taxonomy_map is not None:
        mapping = (
            _as_lazy(taxonomy_map)
            .select(
                pl.col(map_key_col).alias("_key"),
                pl.col(map_taxonomy_col)
                .str.replace_all(_SEMICOLON_SPACE, ";")
                .alias("_taxon"),
            )
            .collect()
        )
        if mapping.height != mapping["_key"].n_unique():
            raise ValueError("Map has repeated accessions")
        if mapping.height != mapping["_taxon"].n_unique():
            raise ValueError(
                "Several genomes share a taxonomy; the map must be one-to-one"
            )
    else:
        keys = lf.select("_key").unique().collect(engine="streaming")
        mapping = keys.with_columns(pl.col("_key").alias("_taxon"))
    mapping = mapping.sort("_taxon").with_row_index("_gi")
    feature_index = pl.DataFrame(
        {"_feature": keep, "_fi": np.arange(len(keep), dtype=np.uint32)}
    )

    triples = (
        lf.join(feature_index.lazy(), on="_feature", how="inner")
        .join(
            mapping.select("_key", pl.col("_gi").cast(pl.UInt32)).lazy(),
            on="_key",
            how="inner",
        )
        .select("_gi", "_fi", "_value")
        .collect(engine="streaming")
    )
    sp = _sp()
    # float32 holds the integer counts exactly (< 2**24) and halves the memory
    mat = sp.coo_matrix(
        (
            triples["_value"].to_numpy().astype(np.float32),
            (
                triples["_gi"].to_numpy().astype(np.int32),
                triples["_fi"].to_numpy().astype(np.int32),
            ),
        ),
        shape=(mapping.height, len(keep)),
    ).tocsr()
    mat.sum_duplicates()
    has_row = np.diff(mat.indptr) > 0
    taxa_all = mapping["_taxon"].to_list()
    taxa = [t for t, h in zip(taxa_all, has_row) if h]
    mat = mat[np.where(has_row)[0]]
    logger.info(
        "Annotation matrix: %d genomes x %d features, %d non-zeros",
        mat.shape[0],
        mat.shape[1],
        mat.nnz,
    )
    return AnnotationMatrix(
        taxa=taxa,
        features=keep,
        matrix=mat,
        n_genomes_in_table=n_total,
        min_count=min_count,
    )


def _weights(r: np.ndarray, weighting: WeightingStrategy) -> np.ndarray:
    if weighting == WeightingStrategy.NONE:
        return np.ones_like(r)
    if weighting == WeightingStrategy.ABUNDANCE:
        return r
    if weighting == WeightingStrategy.SQRT_ABUNDANCE:
        return np.sqrt(r)
    raise ValueError(f"Unknown weighting strategy: {weighting}")


def _column_median(sub, n: int) -> np.ndarray:
    """Median of every column of a sparse (n x K) matrix, zeros included."""
    csc = sub.tocsc()
    csc.sort_indices()
    K = csc.shape[1]
    counts = np.diff(csc.indptr)
    out = np.zeros(K)
    nz = np.where(counts > 0)[0]
    if nz.size == 0:
        return out
    col_of = np.repeat(np.arange(K), counts)
    order = np.lexsort((csc.data, col_of))
    srt = csc.data[order]
    zeros = n - counts[nz]
    lo, hi = (n - 1) // 2, n // 2

    def at(pos):
        idx = pos - zeros
        vals = np.zeros(nz.size)
        have = idx >= 0
        vals[have] = srt[csc.indptr[nz][have] + idx[have]]
        return vals

    out[nz] = (at(lo) + at(hi)) / 2.0
    return out


def iter_aggregate_annotations(
    annotations: AnnotationMatrix,
    relabund: pl.LazyFrame,
    method: Union[str, AggregationMethod] = "arithmetic_mean",
    weighting: Union[str, WeightingStrategy] = "none",
    k: int = 10,
    min_abundance: float = 0.001,
    pseudocount: float = 1e-8,
    chunk_samples: int = 2000,
) -> Iterator[pl.DataFrame]:
    """Aggregate genome annotations to samples, ``chunk_samples`` at a time.

    ``relabund`` has ``sample, taxonomy, relabund`` (one row per species).
    Species without an annotated genome are ignored; a sample with none is
    absent from the output. Yields frames with ``acc`` plus one column per
    feature (``richness`` and ``total_abundance`` for the richness method).
    Method and weighting semantics are those of
    :class:`~microbiome_ml.wrangle.aggregation.FeatureAggregator`.
    """
    sp = _sp()
    method = AggregationMethod(method)
    weighting = WeightingStrategy(weighting)
    F = annotations.matrix
    n_taxa, K = F.shape
    taxa_index = pl.DataFrame(
        {
            "taxonomy": annotations.taxa,
            "_gi": np.arange(n_taxa, dtype=np.int64),
        }
    ).lazy()
    rel = (
        relabund.select(
            pl.col("sample"),
            pl.col("taxonomy").str.replace_all(_SEMICOLON_SPACE, ";"),
            pl.col("relabund").cast(pl.Float64),
        )
        .join(taxa_index, on="taxonomy", how="inner")
        .select("sample", "_gi", "relabund")
        .collect()
        .sort("sample", maintain_order=True)
    )
    if rel.height == 0:
        return
    samples = rel["sample"].to_numpy()
    gi = rel["_gi"].to_numpy()
    r_all = rel["relabund"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, samples[1:] != samples[:-1]])
    starts = bounds
    ends = np.r_[bounds[1:], len(samples)]
    names = samples[starts]
    names_cols = annotations.features

    if method == AggregationMethod.MEDIAN and weighting != WeightingStrategy.NONE:
        logger.warning(
            "Weighted median not implemented, falling back to weighted arithmetic mean"
        )
        method = AggregationMethod.ARITHMETIC_MEAN

    log_F = None
    inv_F = None
    ind_F = None
    if method == AggregationMethod.GEOMETRIC_MEAN:
        log_F = F.astype(np.float64)
        log_F.data = np.log((log_F.data + pseudocount) / pseudocount)
    elif method == AggregationMethod.HARMONIC_MEAN:
        inv_F = F.astype(np.float64)
        inv_F.data = 1.0 / (inv_F.data + pseudocount)
        ind_F = F.astype(np.float64)
        ind_F.data = np.ones_like(ind_F.data)

    for a in range(0, len(names), chunk_samples):
        b = min(a + chunk_samples, len(names))
        lo, hi = starts[a], ends[b - 1]
        nb = b - a
        sample_in_chunk = np.repeat(np.arange(nb), ends[a:b] - starts[a:b])
        g, r = gi[lo:hi], r_all[lo:hi]
        out_names = names[a:b]

        if method == AggregationMethod.RICHNESS:
            keep = r >= min_abundance
            rich = np.bincount(sample_in_chunk[keep], minlength=nb)
            tot = np.bincount(sample_in_chunk[keep], weights=r[keep], minlength=nb)
            present = rich > 0
            yield pl.DataFrame(
                {
                    "acc": out_names[present],
                    "richness": rich[present],
                    "total_abundance": tot[present],
                }
            )
            continue

        if method in (
            AggregationMethod.ARITHMETIC_MEAN,
            AggregationMethod.GEOMETRIC_MEAN,
            AggregationMethod.HARMONIC_MEAN,
        ):
            w = _weights(r, weighting)
            S = sp.csr_matrix((w, (sample_in_chunk, g)), shape=(nb, n_taxa))
            W = np.asarray(S.sum(axis=1)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                if method == AggregationMethod.ARITHMETIC_MEAN:
                    block = (S @ F).toarray() / W[:, None]
                elif method == AggregationMethod.GEOMETRIC_MEAN:
                    block = np.exp(
                        np.log(pseudocount) + (S @ log_F).toarray() / W[:, None]
                    )
                else:
                    # weight of the genomes WITHOUT the feature (each counts
                    # 1/pseudocount); exactly zero where there are none, so
                    # the subtraction cannot leave rounding noise that the
                    # tiny pseudocount would amplify
                    w0 = W[:, None] - (S @ ind_F).toarray()
                    w0[w0 < 1e-9 * W[:, None]] = 0.0
                    block = W[:, None] / (
                        w0 / pseudocount + (S @ inv_F).toarray()
                    )
        else:
            block = np.zeros((nb, K))
            for j in range(nb):
                s0, s1 = starts[a + j] - lo, ends[a + j] - lo
                gg, rr = g[s0:s1], r[s0:s1]
                if method == AggregationMethod.MEDIAN:
                    block[j] = _column_median(F[gg], len(gg))
                elif method in (
                    AggregationMethod.MAX_ABUNDANCE,
                    AggregationMethod.MIN_ABUNDANCE,
                ):
                    ext = rr.max() if method == AggregationMethod.MAX_ABUNDANCE else rr.min()
                    tied = gg[rr == ext]
                    block[j] = np.asarray(F[tied].sum(axis=0)).ravel() / len(tied)
                elif method == AggregationMethod.PRESENCE_ABSENCE:
                    sel = gg[rr >= min_abundance]
                    if len(sel):
                        block[j] = (np.asarray(F[sel].sum(axis=0)).ravel() > 0)
                elif method == AggregationMethod.TOP_K_ABUNDANT:
                    top = np.argsort(-rr, kind="stable")[:k]
                    wt = _weights(rr[top], weighting)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        block[j] = (
                            np.asarray(
                                F[gg[top]].multiply(wt[:, None]).sum(axis=0)
                            ).ravel()
                            / wt.sum()
                        )
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")

        frame = pl.DataFrame(block, schema=names_cols, orient="row")
        yield frame.insert_column(0, pl.Series("acc", out_names))


def aggregate_annotations(
    annotations: AnnotationMatrix,
    relabund: pl.LazyFrame,
    **kwargs,
) -> pl.DataFrame:
    """All chunks of :func:`iter_aggregate_annotations` as one frame."""
    parts = list(iter_aggregate_annotations(annotations, relabund, **kwargs))
    if not parts:
        return pl.DataFrame({"acc": []})
    return pl.concat(parts)


def read_kegg_mapping(
    path: PathLike,
    feature_name: str,
    keep_if_contains: Optional[str] = None,
) -> pl.DataFrame:
    """Read a KEGG ``module``/``pathway`` -> ortholog link table.

    The file is tab separated without header, ``<feature>\t<KO>`` with
    prefixed ids (``md:M00001``, ``ko:K00844``); the prefixes are removed.
    ``keep_if_contains`` keeps rows whose feature id contains the text (use
    ``"map"`` for the reference pathways). Returns ``feature, KEGG``.
    """
    df = pl.read_csv(
        str(path),
        separator="\t",
        has_header=False,
        new_columns=[feature_name, "KEGG"],
    ).with_columns(
        pl.col(feature_name).str.split(":").list.get(1),
        pl.col("KEGG").str.split(":").list.get(1),
    )
    if keep_if_contains is not None:
        df = df.filter(pl.col(feature_name).str.contains(keep_if_contains))
    return df.rename({feature_name: "feature"})


def iter_kegg_completeness(
    annotations: AnnotationMatrix,
    mapping: pl.DataFrame,
    relabund: pl.LazyFrame,
    samples: Optional[Sequence[str]] = None,
    chunk_samples: int = 2000,
) -> Iterator[pl.DataFrame]:
    """Fraction of each module/pathway's orthologs found in each sample.

    For a sample, the orthologs present in *any* of its genomes (a set, no
    abundance weighting) are counted per module/pathway and divided by the
    number of orthologs in it (rows of ``mapping``, as in the retired
    script). ``mapping`` has ``feature, KEGG``; ``annotations`` must have
    been read with every ortholog (``min_prevalence=0``). Samples with no
    module hit are omitted. Features are ordered by size then name.
    """
    sp = _sp()
    ko_pos = {k: i for i, k in enumerate(annotations.features)}
    sizes = mapping.group_by("feature").len().sort("len", "feature")
    feats = sizes["feature"].to_list()
    f_pos = {f: i for i, f in enumerate(feats)}
    hit = mapping.filter(pl.col("KEGG").is_in(list(ko_pos)))
    M = sp.coo_matrix(
        (
            np.ones(hit.height),
            (
                np.array([ko_pos[k] for k in hit["KEGG"].to_list()], dtype=np.int64),
                np.array([f_pos[f] for f in hit["feature"].to_list()], dtype=np.int64),
            ),
        ),
        shape=(len(ko_pos), len(feats)),
    ).tocsr()
    size = sizes["len"].to_numpy().astype(float)

    present = annotations.matrix.copy()
    present.data = np.ones_like(present.data)
    rel = relabund.select(
        pl.col("sample"),
        pl.col("taxonomy").str.replace_all(_SEMICOLON_SPACE, ";"),
    )
    if samples is not None:
        rel = rel.filter(pl.col("sample").is_in(list(samples)))
    rel = (
        rel.join(
            pl.DataFrame(
                {
                    "taxonomy": annotations.taxa,
                    "_gi": np.arange(len(annotations.taxa), dtype=np.int64),
                }
            ).lazy(),
            on="taxonomy",
            how="inner",
        )
        .select("sample", "_gi")
        .unique()
        .collect()
        .sort("sample", maintain_order=True)
    )
    if rel.height == 0:
        return
    names_all = rel["sample"].to_numpy()
    g_all = rel["_gi"].to_numpy()
    starts = np.flatnonzero(np.r_[True, names_all[1:] != names_all[:-1]])
    ends = np.r_[starts[1:], len(names_all)]
    names = names_all[starts]
    for a in range(0, len(names), chunk_samples):
        b = min(a + chunk_samples, len(names))
        lo, hi = starts[a], ends[b - 1]
        rows = np.repeat(np.arange(b - a), ends[a:b] - starts[a:b])
        S = sp.csr_matrix(
            (np.ones(hi - lo), (rows, g_all[lo:hi])),
            shape=(b - a, present.shape[0]),
        )
        P = (S @ present).tocsr()
        P.data = np.ones_like(P.data)          # which orthologs a sample has
        counts = (P @ M).toarray()
        block = counts / size[None, :]
        keep = block.sum(axis=1) > 0
        if not keep.any():
            continue
        frame = pl.DataFrame(block[keep], schema=feats, orient="row")
        yield frame.insert_column(0, pl.Series("acc", names[a:b][keep]))


__all__ = [
    "read_kegg_mapping",
    "iter_kegg_completeness",
    "AnnotationMatrix",
    "scan_annotation_matrix",
    "iter_aggregate_annotations",
    "aggregate_annotations",
]
