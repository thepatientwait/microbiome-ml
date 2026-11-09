"""Lazy, memory-safe species-level feature tables.

``SpeciesFeatureSet`` keeps every value in one dense numpy array, which is
unusable for genome-scale tables (hundreds of thousands of rows by thousands
of columns). ``LazySpeciesFeatureSet`` keeps a polars ``LazyFrame`` instead:
a Parquet table (a file or a directory of ``*.parquet`` files, read by column
and row group) or a CSV whose feature columns are forced to ``Float64``.

Tables are often keyed by a genome accession while the taxonomic profiles are
keyed by a taxonomy string. A mapping (accession -> taxonomy) re-keys the table
on scan; it is checked to be one-to-one, because a taxonomy with several rows
would otherwise be counted several times when it is joined to the profiles.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)

TAXONOMY = "taxonomy"
_SEMICOLON_SPACE = r";\s+"

MapInput = Union[str, Path, pl.DataFrame, pl.LazyFrame]


def _scan_table(path: Union[str, Path], key_col: str) -> pl.LazyFrame:
    """Scan a Parquet file/directory or CSV; CSV features are read as float."""
    p = Path(path)
    if p.is_dir():
        return pl.scan_parquet(str(p / "*.parquet"))
    if p.suffix == ".parquet":
        return pl.scan_parquet(str(p))
    names = pl.read_csv(p, n_rows=0).columns
    return pl.scan_csv(
        p,
        schema_overrides={c: pl.Float64 for c in names if c != key_col},
    )


def _as_lazy(table: MapInput) -> pl.LazyFrame:
    """A mapping table as a LazyFrame (plain scan: columns are text)."""
    if isinstance(table, pl.LazyFrame):
        return table
    if isinstance(table, pl.DataFrame):
        return table.lazy()
    p = Path(table)
    if p.suffix == ".parquet" or p.is_dir():
        return _scan_table(p, "")
    return pl.scan_csv(p)


class LazySpeciesFeatureSet:
    """Species-level feature table backed by a polars ``LazyFrame``.

    Rows are taxa, columns are features. ``lazy()`` always exposes the key
    column as ``taxonomy`` (whitespace after ``;`` removed, matching the
    taxonomic profiles) so it can be joined directly to a profile.
    """

    def __init__(
        self,
        lazy_frame: pl.LazyFrame,
        name: str,
        key_col: str = TAXONOMY,
    ) -> None:
        if name is None:
            raise ValueError("FeatureSet name must be defined")
        columns = lazy_frame.collect_schema().names()
        if key_col not in columns:
            raise ValueError(
                f"Key column '{key_col}' not found; columns start with "
                f"{columns[:5]}"
            )
        self.name = name
        self.key_col = key_col
        self._lf = lazy_frame
        self.feature_names: List[str] = [c for c in columns if c != key_col]

    @classmethod
    def scan(
        cls,
        path: Union[str, Path],
        name: str,
        key_col: str = "acc",
        taxonomy_map: Optional[MapInput] = None,
        map_key_col: str = "acc",
        map_taxonomy_col: str = TAXONOMY,
        duplicate_taxa: str = "error",
    ) -> "LazySpeciesFeatureSet":
        """Scan a table, optionally re-keying it from accession to taxonomy.

        Args:
            path: Parquet file, Parquet directory, or CSV.
            name: Feature set name.
            key_col: Key column of the table (accession or taxonomy).
            taxonomy_map: Table with ``map_key_col`` and ``map_taxonomy_col``;
                when given, the table is re-keyed to taxonomy and rows without
                a mapping are dropped.
            map_key_col: Key column of the map (matches ``key_col``).
            map_taxonomy_col: Taxonomy column of the map.
            duplicate_taxa: What to do when several rows map to one taxonomy:
                ``"error"`` (default) raises, ``"mean"`` averages them.

        Raises:
            ValueError: if the mapping is not one-to-one and
                ``duplicate_taxa="error"``.
        """
        if duplicate_taxa not in ("error", "mean"):
            raise ValueError("duplicate_taxa must be 'error' or 'mean'")
        lf = _scan_table(path, key_col)
        if taxonomy_map is None:
            return cls(lf.rename({key_col: TAXONOMY}), name, key_col=TAXONOMY)

        mapping = (
            _as_lazy(taxonomy_map)
            .select(
                pl.col(map_key_col).alias(key_col),
                pl.col(map_taxonomy_col).alias(TAXONOMY),
            )
            .collect()
        )
        n_dup_keys = mapping.height - mapping[key_col].n_unique()
        if n_dup_keys:
            raise ValueError(
                f"Map has {n_dup_keys} repeated '{map_key_col}' values; "
                "each accession must map to exactly one taxonomy"
            )
        n_dup_tax = mapping.height - mapping[TAXONOMY].n_unique()
        if n_dup_tax and duplicate_taxa == "error":
            raise ValueError(
                f"{n_dup_tax} rows share a taxonomy with another row; a "
                "taxonomy would be counted several times. Pass "
                "duplicate_taxa='mean' to average them, or deduplicate the map."
            )
        lf = lf.join(mapping.lazy(), on=key_col, how="inner").drop(key_col)
        if n_dup_tax:
            logger.warning(
                "%d accessions share a taxonomy; averaging them", n_dup_tax
            )
            lf = lf.group_by(TAXONOMY).agg(pl.exclude(TAXONOMY).mean())
        return cls(lf, name, key_col=TAXONOMY)

    @property
    def features(self) -> pl.LazyFrame:
        """The lazy table (key column untouched)."""
        return self._lf

    def lazy(self) -> pl.LazyFrame:
        """Lazy frame with a normalised ``taxonomy`` key as first column."""
        lf = self._lf
        if self.key_col != TAXONOMY:
            lf = lf.rename({self.key_col: TAXONOMY})
        return lf.with_columns(
            pl.col(TAXONOMY).str.replace_all(_SEMICOLON_SPACE, ";")
        ).select([TAXONOMY] + self.feature_names)

    def collect(self) -> pl.DataFrame:
        """Materialise the whole table (large for genome-scale data)."""
        return self.lazy().collect()

    def select_features(self, names: List[str]) -> "LazySpeciesFeatureSet":
        """Keep only some feature columns (still lazy)."""
        missing = set(names) - set(self.feature_names)
        if missing:
            raise ValueError(f"Feature names {sorted(missing)} not found")
        return LazySpeciesFeatureSet(
            self.lazy().select([TAXONOMY] + list(names)), self.name, TAXONOMY
        )

    def filter_samples(self, taxonomy_ids: List[str]) -> "LazySpeciesFeatureSet":
        """Keep only the given taxa (still lazy)."""
        return LazySpeciesFeatureSet(
            self.lazy().filter(pl.col(TAXONOMY).is_in(taxonomy_ids)),
            self.name,
            TAXONOMY,
        )

    def __repr__(self) -> str:
        return (
            f"LazySpeciesFeatureSet(name={self.name!r}, "
            f"n_features={len(self.feature_names)})"
        )


def is_lazy_species(obj: object) -> bool:
    """True for lazy species feature sets."""
    return isinstance(obj, LazySpeciesFeatureSet)


__all__ = ["LazySpeciesFeatureSet", "is_lazy_species"]
