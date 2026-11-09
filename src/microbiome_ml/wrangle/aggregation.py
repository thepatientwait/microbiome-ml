"""Feature aggregation functionality for converting species-wise to sample-wise
features."""

import logging
from typing import Any, List, Union

import polars as pl

from microbiome_ml.core.config import (
    AggregationConfig,
    AggregationMethod,
    WeightingStrategy,
)
from microbiome_ml.wrangle.features import SampleFeatureSet, SpeciesFeatureSet
from microbiome_ml.wrangle.species_table import LazySpeciesFeatureSet

logger = logging.getLogger(__name__)


class FeatureAggregator:
    """Memory-efficient feature aggregation using polars LazyFrame.

    Note:
        All aggregation methods require taxonomic profiles with relative abundance
        data (relabund column). If you have coverage data, use TaxonomicProfiles
        to automatically convert coverage to relative abundance before aggregation.
    """

    def __init__(self) -> None:
        """Initialize aggregator."""

    def _get_feature_columns(
        self, species_features_lf: pl.LazyFrame
    ) -> List[str]:
        """Get feature column names (excluding 'taxonomy')."""
        return [
            col
            for col in species_features_lf.collect_schema().names()
            if col not in ("acc", "sample", "taxonomy")
        ]

    def _apply_weighting(
        self, joined_data: pl.LazyFrame, weighting: WeightingStrategy
    ) -> pl.LazyFrame:
        """Apply weighting strategy to joined data."""

        if weighting == WeightingStrategy.NONE:
            return joined_data.with_columns(pl.lit(1.0).alias("weight"))

        elif weighting == WeightingStrategy.ABUNDANCE:
            return joined_data.with_columns(pl.col("relabund").alias("weight"))

        elif weighting == WeightingStrategy.SQRT_ABUNDANCE:
            return joined_data.with_columns(
                pl.col("relabund").sqrt().alias("weight")
            )

        else:
            raise ValueError(f"Unknown weighting strategy: {weighting}")

    def aggregate_arithmetic_mean(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Arithmetic mean aggregation with weighting."""

        weighted_data = self._apply_weighting(joined_data, weighting)

        agg_exprs = [
            (pl.col(col) * pl.col("weight")).sum() / pl.col("weight").sum()
            for col in feature_cols
        ]

        return (
            weighted_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_geometric_mean(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Geometric mean aggregation with weighting."""

        pseudocount = kwargs.get("pseudocount", 1e-8)
        weighted_data = self._apply_weighting(joined_data, weighting)

        agg_exprs = [
            (
                ((pl.col(col) + pseudocount).log() * pl.col("weight")).sum()
                / pl.col("weight").sum()
            )
            .exp()
            .alias(col)
            for col in feature_cols
        ]

        return (
            weighted_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_harmonic_mean(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Harmonic mean aggregation with weighting."""

        pseudocount = kwargs.get("pseudocount", 1e-8)
        weighted_data = self._apply_weighting(joined_data, weighting)

        agg_exprs = [
            (
                pl.col("weight").sum()
                / (pl.col("weight") / (pl.col(col) + pseudocount)).sum()
            ).alias(col)
            for col in feature_cols
        ]

        return (
            weighted_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_median(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Median aggregation (weighted median is approximated)."""

        if weighting != WeightingStrategy.NONE:
            logger.warning(
                "Weighted median not implemented, falling back to weighted arithmetic mean"
            )
            return self.aggregate_arithmetic_mean(
                joined_data, feature_cols, weighting, **kwargs
            )

        # For unweighted median, we don't need the weight column
        agg_exprs = [pl.col(col).median() for col in feature_cols]

        return (
            joined_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_presence_absence(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Presence/absence of every feature in a sample (weighting not
        applicable).

        A feature is 1.0 in a sample if at least one species with relative
        abundance >= ``min_abundance`` has a value > 0 for it, else 0.0.
        """

        min_abundance = kwargs.get("min_abundance", 0.001)
        present = pl.col("relabund") >= min_abundance

        agg_exprs = [
            ((pl.col(col) > 0) & present).any().cast(pl.Float64).alias(col)
            for col in feature_cols
        ]

        return (
            joined_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_richness(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Species richness and total abundance per sample.

        Independent of the species features: counts species with relative
        abundance >= ``min_abundance`` that have an entry in the feature set.
        """

        min_abundance = kwargs.get("min_abundance", 0.001)

        return (
            joined_data.filter(pl.col("relabund") >= min_abundance)
            .group_by("sample")
            .agg(
                [
                    pl.col("taxonomy").count().alias("richness"),
                    pl.col("relabund").sum().alias("total_abundance"),
                ]
            )
            .rename({"sample": "acc"})
        )

    def aggregate_max_abundance(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Max abundance aggregation (weighting not applicable)."""

        agg_exprs = [
            pl.col(col)
            .filter(pl.col("relabund") == pl.col("relabund").max())
            .first()
            for col in feature_cols
        ]

        return (
            joined_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_min_abundance(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Min abundance aggregation (weighting not applicable)."""

        agg_exprs = [
            pl.col(col)
            .filter(pl.col("relabund") == pl.col("relabund").min())
            .first()
            for col in feature_cols
        ]

        return (
            joined_data.group_by("sample")
            .agg(agg_exprs)
            .rename({"sample": "acc"})
        )

    def aggregate_top_k_abundant(
        self,
        joined_data: pl.LazyFrame,
        feature_cols: List[str],
        weighting: WeightingStrategy,
        **kwargs: Any,
    ) -> pl.LazyFrame:
        """Top-k abundant taxa aggregation."""

        k = kwargs.get("k", 10)

        # Keep the k most abundant species per sample (among the species that
        # have features), then average their features.
        filtered_data = (
            joined_data.sort("relabund", descending=True)
            .group_by("sample", maintain_order=True)
            .head(k)
        )

        # Use arithmetic mean on the filtered data
        return self.aggregate_arithmetic_mean(
            filtered_data, feature_cols, weighting, **kwargs
        )

    def remove_uncommon_features(
        self,
        feature_set: Union[pl.LazyFrame, pl.DataFrame],
        uncommon_feature_cutoff: Union[int, float] = 10,
    ) -> pl.DataFrame:
        """Drop feature columns that have a value in too few samples.

        A feature is kept when it is non-null in at least
        ``uncommon_feature_cutoff`` samples (an int) or in at least that
        fraction of samples (a float in (0, 1)). The sample id column ``acc``
        is always kept. Counts are taken in a single pass. With fewer samples
        than an integer cutoff, rarity cannot be judged and nothing is dropped.
        """

        df = (
            feature_set.collect()
            if isinstance(feature_set, pl.LazyFrame)
            else feature_set
        )
        feature_cols = [c for c in df.columns if c != "acc"]
        if not feature_cols or df.height == 0:
            return df
        if isinstance(uncommon_feature_cutoff, float) and (
            0 < uncommon_feature_cutoff < 1
        ):
            min_samples = uncommon_feature_cutoff * df.height
        else:
            min_samples = float(uncommon_feature_cutoff)
            if df.height < min_samples:
                logger.info(
                    "Only %d samples (< cutoff %s): rare-feature filter skipped",
                    df.height,
                    min_samples,
                )
                return df
        non_null = df.select(
            [pl.col(c).count().alias(c) for c in feature_cols]
        ).row(0)
        keep = [c for c, n in zip(feature_cols, non_null) if n >= min_samples]
        dropped = len(feature_cols) - len(keep)
        if dropped:
            logger.info(
                "Dropped %d of %d features present in < %s samples",
                dropped,
                len(feature_cols),
                min_samples,
            )
        return df.select(["acc"] + keep)

    def get_sample_features(
        self,
        species_features: Union[SpeciesFeatureSet, LazySpeciesFeatureSet],
        relabund_lf: pl.LazyFrame,
        config: AggregationConfig,
    ) -> SampleFeatureSet:
        """Convert species-wise features into sample-wise features using
        aggregation.

        Args:
            species_features: SpeciesFeatureSet with species-level features
            relabund_lf: LazyFrame with relative abundance data (sample, taxonomy, relabund)
                        Must contain 'relabund' column with relative abundance values [0-1]
            config: Aggregation configuration

        Returns:
            SampleFeatureSet with aggregated features

        Note:
            The relabund_lf must contain relative abundance data, not raw coverage.
            Use TaxonomicProfiles to convert coverage to relative abundance if needed.
            relabund_lf: LazyFrame with relative abundance data (sample, taxonomy, relabund)
            config: AggregationConfig with method and parameters

        Returns:
            SampleFeatureSet with aggregated sample-level features
        """
        # Species features as a LazyFrame keyed by ``taxonomy``. Lazy sets
        # (Parquet / CSV scans) are never materialised as a whole.
        if isinstance(species_features, LazySpeciesFeatureSet):
            species_lf = species_features.lazy()
        else:
            species_lf = species_features.collect().lazy()

        feature_cols = self._get_feature_columns(species_lf)

        # Strategy mapping
        aggregation_strategies = {
            AggregationMethod.ARITHMETIC_MEAN: self.aggregate_arithmetic_mean,
            AggregationMethod.GEOMETRIC_MEAN: self.aggregate_geometric_mean,
            AggregationMethod.HARMONIC_MEAN: self.aggregate_harmonic_mean,
            AggregationMethod.MEDIAN: self.aggregate_median,
            AggregationMethod.PRESENCE_ABSENCE: self.aggregate_presence_absence,
            AggregationMethod.RICHNESS: self.aggregate_richness,
            AggregationMethod.MAX_ABUNDANCE: self.aggregate_max_abundance,
            AggregationMethod.MIN_ABUNDANCE: self.aggregate_min_abundance,
            AggregationMethod.TOP_K_ABUNDANT: self.aggregate_top_k_abundant,
        }

        # Get the appropriate aggregation function
        if config.method not in aggregation_strategies:
            raise ValueError(f"Unknown aggregation method: {config.method}")

        aggregation_func = aggregation_strategies[config.method]

        # Aggregate in column chunks (one chunk = everything by default), so
        # the join with the profiles only ever carries one chunk's columns.
        chunk = config.chunk_columns
        if config.method == AggregationMethod.RICHNESS or not chunk:
            chunks = [feature_cols]
        else:
            chunks = [
                feature_cols[i : i + chunk]
                for i in range(0, len(feature_cols), chunk)
            ]

        parts: List[pl.DataFrame] = []
        for cols in chunks:
            joined_data = relabund_lf.join(
                species_lf.select(["taxonomy"] + cols),
                on="taxonomy",
                how="inner",
                coalesce=True,
            )
            part = aggregation_func(
                joined_data,
                cols,
                config.weighting,
                k=config.k,
                pseudocount=config.pseudocount,
                min_abundance=config.min_abundance,
            ).collect()
            parts.append(part.sort("acc"))

        if len(parts) == 1:
            sample_features_df = parts[0]
        else:
            ref = parts[0]["acc"]
            for part in parts[1:]:
                if not part["acc"].equals(ref):
                    raise RuntimeError(
                        "Column chunks produced different sample sets"
                    )
            sample_features_df = pl.concat(
                [parts[0]] + [p.drop("acc") for p in parts[1:]],
                how="horizontal",
            )

        sample_features_df = self.remove_uncommon_features(
            sample_features_df, config.uncommon_feature_cutoff
        )

        return SampleFeatureSet.from_df(
            df=sample_features_df,
            name=f"{species_features.name}_{config.method.value}",
            acc_column="acc",
        )


def aggregate_species_to_samples(
    species_features: Union[SpeciesFeatureSet, LazySpeciesFeatureSet],
    relabund_lf: pl.LazyFrame,
    config: AggregationConfig,
) -> SampleFeatureSet:
    """Convenience function to aggregate species features to sample features.

    Args:
        species_features: SpeciesFeatureSet with species-level features
        relabund_lf: LazyFrame with relative abundance data
        config: AggregationConfig with method and parameters

    Returns:
        SampleFeatureSet with aggregated sample-level features
    """
    aggregator = FeatureAggregator()
    return aggregator.get_sample_features(
        species_features, relabund_lf, config
    )
