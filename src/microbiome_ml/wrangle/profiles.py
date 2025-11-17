"""Core data models for microbiome data structures."""

import numpy as np
import polars as pl
from typing import List, Any, Optional, Union
from pathlib import Path
from enum import Enum
import warnings

import sys
sys.path.append("/home/n9985158/git/microbiomeML/src")

from microbiome_ml.utils._cache import load_data
from microbiome_ml.utils.taxonomy import TaxonomicRanks
from microbiome_ml.wrangle.features import FeatureSet


class Fields(Enum):
    "Base class for field definitions with validation properties."

    def __init__(self, column_name: str, dtype, required: bool, description: str, alternatives: List[str] = None):
        self.column_name = column_name
        self.dtype = dtype
        self.required = required
        self.description = description
        self.alternatives = alternatives or []
        self.all_names = [column_name] + self.alternatives

    def find_column_name(self, lf: pl.LazyFrame) -> Optional[str]:
        """Find the actual column name in the LazyFrame from possible alternatives."""
        schema_names = lf.collect_schema().names()
        for name in self.all_names:
            if name in schema_names:
                return name
        return None


class ProfilesFields(Fields):
    """Enumeration of profiles fields with validation properties."""
    
    SAMPLE = ("sample", pl.Utf8, True, "Sample identifier", ["acc", "sample_id"])
    TAXONOMY = ("taxonomy", pl.Utf8, True, "Taxonomic classification string", ["tax"])
    RELABUND = ("relabund", pl.Float64, False, "Relative abundance value", ["relative_abundance", "abundance", "proportion"])
    COVERAGE = ("coverage", pl.Float64, False, "Coverage value", ["cov"])


class RootFields(Fields):
    """Enumeration of root fields with validation properties."""
    
    SAMPLE = ("sample", pl.Utf8, True, "Sample identifier", ["acc", "sample_id"])
    COVERAGE = ("root_coverage", pl.Float64, True, "Coverage value", ["coverage", "cov"])



class TaxonomicProfiles:
    """
    Long-form taxonomic abundance data container.
    
    Stores taxonomic profiles in long format with exactly 3 columns:
    - sample: Sample identifier
    - taxonomy: Taxonomic classification string  
    - relabund or coverage: Relative abundance or coverage value
    
    Attributes:
        profiles: Polars LazyFrame containing the taxonomic profiles
        root: Optional Polars LazyFrame containing root coverage data
        is_filled: Boolean indicating if profiles are in filled format
    """
    
    def __init__(
        self, 
        profiles: Union[Path, str, pl.LazyFrame], 
        root: Optional[Union[Path, str, pl.LazyFrame]] = None,
        check_filled: bool = True,
        sample_size: int = 1000
    ):
        """
        Initialize TaxonomicProfiles with validation and standardization.
        Optionally checks if profiles are filled and sets is_filled attribute.
        
        Args:
            profiles: Path to profiles file or LazyFrame with profiles data
            root: Optional path to root file or LazyFrame with root coverage data  
            check_filled: Whether to check if profiles are in filled format
            sample_size: Sample size for filled format checking
        """
        self.root = None

        # Load and standardize profiles
        self.profiles = self._load_and_standardize(profiles, ProfilesFields)
        self.profiles = self._standardize_tax_strings(self.profiles)

        # Check if profiles are filled and store the result
        self.is_filled = None
        if check_filled:
            self.is_filled = self._is_filled(self.profiles, sample_size=sample_size)
            
            if not self.is_filled:
                # warnings.warn(
                #     "Taxonomic profiles are not in filled format. Parent taxa abundances "
                #     "may not equal the sum of their children. This could affect downstream "
                #     "analyses. Consider using fill_taxonomy() to correct this.",
                #     UserWarning,
                #     stacklevel=2
                # )

                self.profiles = self._fill_profiles_lf()
                self.is_filled = True  # Now filled after correction

        
        # Load and standardize root if provided
        if root is not None:
            self.root = self._load_and_standardize(root, RootFields)
        else:
            try:
                self.root = self._create_root_lf()
            except ValueError:
                self.root = None

        if 'coverage' in self.profiles.collect_schema().names():
            if self.root is None:
                raise ValueError("Root coverage data is required when profiles contain coverage values.")
            else:
                self.profiles = self._to_relabund_lf()



    def _load_data(self, data_source: Union[Path, str, pl.LazyFrame]) -> pl.LazyFrame:
        """
        Load data from various sources.
        
        Args:
            data_source: Path to file or existing LazyFrame
            
        Returns:
            LazyFrame
        """
        if isinstance(data_source, pl.LazyFrame):
            return data_source
        elif isinstance(data_source, (str, Path)):
            return load_data(data_source)
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

    def _validate_and_standardize_fields(self, lf: pl.LazyFrame, field_enum: Enum) -> pl.LazyFrame:
        """
        Validate required fields and standardize column names.
        
        Args:
            lf: LazyFrame to validate and standardize
            field_enum: Enum class with field definitions
            
        Returns:
            LazyFrame with standardized column names
        """
        missing_required = []
        rename_mapping = {}
        
        # Find actual columns and build rename mapping
        for field in field_enum:
            actual_column = field.find_column_name(lf)
            if actual_column:
                # Only rename if the actual column name is different from standard
                if actual_column != field.column_name:
                    rename_mapping[actual_column] = field.column_name
            elif field.required:
                missing_required.append(f"{field.column_name} (tried: {field.all_names})")
        
        if missing_required:
            raise ValueError(f"Missing required fields: {missing_required}")
        
        # Rename columns to standard names
        if rename_mapping:
            lf = lf.rename(rename_mapping)
        
        return lf

    def _load_and_standardize(self, data_source: Union[Path, str, pl.LazyFrame], field_enum: Enum) -> pl.LazyFrame:
        """
        Load data and standardize column names.
        
        Args:
            data_source: Data source to load
            field_enum: Field enum for validation
            
        Returns:
            Standardized LazyFrame
        """
        lf = self._load_data(data_source)
        return self._validate_and_standardize_fields(lf, field_enum)

    def _standardize_tax_strings(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Standardize taxonomy strings to use ';{prefix}' (no space after semicolon).
        
        Args:
            lf: LazyFrame with taxonomy column
            
        Returns:
            LazyFrame with standardized taxonomy strings
        """
        return lf.with_columns([
            pl.col('taxonomy').str.replace_all(r";\s+", ";").alias('taxonomy')
        ])



    def _is_filled_for_rank(self, cached_profiles: pl.LazyFrame, parent_rank: TaxonomicRanks, 
                           child_rank: TaxonomicRanks, sample_size: int = 1000) -> bool:
        """
        Check if profiles are filled for a specific parent/child rank pair.
        
        Args:
            cached_profiles: Cached LazyFrame with profiles data
            parent_rank: Parent taxonomic rank
            child_rank: Child taxonomic rank  
            sample_size: Maximum number of samples to check
            
        Returns:
            True if profiles are filled for this rank pair
        """
        schema_names = cached_profiles.collect_schema().names()
        abundance_col = 'relabund' if 'relabund' in schema_names else 'coverage'
        
        # Subset for efficiency
        unique_samples = cached_profiles.select('sample').unique()
        if unique_samples.collect().height > sample_size:
            subset_samples = unique_samples.sample(n=sample_size, seed=42)
            df = cached_profiles.join(subset_samples, on='sample', how='semi')
        else:
            df = cached_profiles

        # Use get_rank to get child and parent entries
        children = self._get_rank(df, child_rank)
        parents = self._get_rank(df, parent_rank)

        if children.collect().height == 0 or parents.collect().height == 0:
            return False

        # Extract parent taxonomy from child taxonomy
        children = (
            children.with_columns([
                pl.col('taxonomy')
                .str.extract(f"^(.*?{parent_rank.prefix}[^;]+)")
                .alias('parent_taxonomy')
            ])
            .filter(pl.col('parent_taxonomy').is_not_null())
        )

        # Sum child abundances by parent/sample
        child_sums = (
            children
            .group_by(['sample', 'parent_taxonomy'])
            .agg(pl.col(abundance_col).sum().alias('child_sum'))
        )

        # Prepare parent abundances
        parents = (
            parents
            .select(['sample', 'taxonomy', abundance_col])
            .rename({'taxonomy': 'parent_taxonomy', abundance_col: 'parent_abundance'})
        )

        # Join and compare
        comparison = parents.join(child_sums, on=['sample', 'parent_taxonomy'], how='inner')
        invalid = comparison.filter(pl.col('parent_abundance') < (pl.col('child_sum') - 1e-10))
        return invalid.collect().height == 0

    def _is_filled(self, profiles: pl.LazyFrame, sample_size: int = 1000) -> bool:
        """
        Check if profiles are in filled format.
        
        Args:
            profiles: LazyFrame with profiles data
            sample_size: Maximum number of samples to check
            
        Returns:
            True if profiles are filled
        """
        for rank in TaxonomicRanks.PHYLUM.iter_down():
            parent = rank.parent
            if not self._is_filled_for_rank(profiles, parent, rank, sample_size=sample_size):
                return False
        return True


    def _filter_by_sample(self, samples: pl.LazyFrame) -> "TaxonomicProfiles":
        """
        Create new TaxonomicProfiles filtered by sample IDs.
        
        Args:
            samples: LazyFrame containing sample IDs to keep
            
        Returns:
            New TaxonomicProfiles instance with filtered data
        """
        filtered_profiles = self._filter_profiles_by_sample(samples)
        
        filtered_root = None
        if self.root is not None:
            filtered_root = self._filter_root_by_sample(samples)
        
        # Create new instance without running __init__ validation
        new_instance = TaxonomicProfiles.__new__(TaxonomicProfiles)
        new_instance.profiles = filtered_profiles
        new_instance.root = filtered_root  
        new_instance.is_filled = self.is_filled  # Inherit filled status
        return new_instance

    def _get_rank(self, lf: pl.LazyFrame, rank: Union[str, TaxonomicRanks]) -> pl.LazyFrame:
        """Internal method to get entries for a specific taxonomic rank."""
        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)

        rank_regex = f"{rank.prefix}[^;]+$"
        return lf.filter(pl.col('taxonomy').str.contains(rank_regex))
    
    def get_rank(self, rank: Union[str, TaxonomicRanks]) -> pl.LazyFrame:
        """Get entries for a specific taxonomic rank.
        
        Args:
            rank: Taxonomic rank to filter by
            
        Returns:
            LazyFrame with entries for the specified rank
        """
        return self._get_rank(self.profiles, rank)


    def filter_by_coverage(self, cov_cutoff: float = 50.0) -> "TaxonomicProfiles":
        """
        Filter samples based on coverage metrics.
        
        Args:
            cov_cutoff: Minimum coverage cutoff (default: 50.0)
            
        Returns:
            Filtered TaxonomicProfiles instance
        """
        if self.root is None:
            raise ValueError("Root coverage data not available")

        passing_samples = self.root.filter(pl.col('root_coverage') > cov_cutoff).select('sample')

        return self._filter_by_sample(passing_samples)

    def get_samples(self) -> List[str]:
        """
        Get list of unique sample identifiers.
        
        Returns:
            List of unique sample IDs
        """
        return self.profiles.select('sample').unique().collect().to_series().to_list()

    
    def filter_dominated_samples(self, dominated_cutoff: float = 0.99, rank: Union[TaxonomicRanks, str] = TaxonomicRanks.ORDER) -> "TaxonomicProfiles":
        """
        Filter out samples dominated by a single taxonomic group.
        
        Args:
            dominated_cutoff: Maximum relative abundance for single group (default: 0.99)
            rank: Taxonomic rank to check for domination
            
        Returns:
            New TaxonomicProfiles instance with filtered data
        """
        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)

        # Determine abundance column
        if 'coverage' in self.profiles.collect_schema().names():
            raise ValueError("Profiles must be in relative abundance format to filter dominated samples.")
        
        # Find samples that are NOT dominated
        non_dominated_samples = (
            self.profiles
            .filter(
                pl.col('taxonomy').str.contains(rank.prefix) & 
                ~pl.col('taxonomy').str.contains(rank.child.prefix if rank.child else "NONE")
            )
            .group_by('sample')
            .agg(pl.col('relabund').max().alias('max_abund'))
            .filter(pl.col('max_abund') < dominated_cutoff)
            .select('sample')
        )
        
        # Use central filtering method
        return self._filter_by_sample(non_dominated_samples)

    
    def default_qc(self, cov_cutoff: float = 50.0, dominated_cutoff: float = 0.99, rank: Union[TaxonomicRanks, str] = TaxonomicRanks.ORDER) -> "TaxonomicProfiles":
        """
        Apply default quality control: coverage and domination filters.
        
        Args:
            cov_cutoff: Minimum coverage cutoff (default: 50.0)
            dominated_cutoff: Maximum relative abundance for single group (default: 0.99)
            rank: Taxonomic rank to check for domination
        """
        return (
            self
            .filter_by_coverage(cov_cutoff)
            .filter_dominated_samples(dominated_cutoff, rank)
            )

    def _fill_profiles_lf(self) -> pl.LazyFrame:
        """Internal method: fill profiles and return LazyFrame."""
        schema_names = self.profiles.collect_schema().names()
        abundance_col = 'relabund' if 'relabund' in schema_names else 'coverage'

        filled_profiles = self.profiles
        species = self.get_rank(TaxonomicRanks.SPECIES)

        filled_ranks = [species]

        # Traverse from most specific to most general (lowest to highest rank)
        for rank in TaxonomicRanks.GENUS.iter_up():
            # Get entries at this rank (using get_rank helper)
            filled_profiles = (
                filled_profiles
                .with_columns(
                    pl.col('taxonomy')
                      .str.extract(f"^(.*?{rank.prefix}[^;]+)"))
                .with_columns(
                    pl.col(abundance_col)
                      .sum()
                      .over(['sample', 'taxonomy']))
                .unique()
                )
            
            filled_ranks.append(
                filled_profiles
                .filter(pl.col('taxonomy').str.contains(rank.prefix))
            )

        # Combine all filled ranks
        return pl.concat(filled_ranks).unique()

    def _create_root_lf(self) -> pl.LazyFrame:
        """Internal method: create root and return LazyFrame."""
        schema_names = self.profiles.collect_schema().names()
        if 'relabund' in schema_names:
            raise ValueError("Cannot create root from relative abundance profiles.")

        # If filled, sum domain-level coverages per sample
        if self.is_filled:
            domain_entries = self.get_rank(TaxonomicRanks.DOMAIN)
            return (
                domain_entries
                .group_by('sample')
                .agg(pl.col('coverage').sum().alias('root_coverage'))
                .select(['sample', 'root_coverage'])
            )
        else:
            # If not filled, sum all coverages per sample
            return (
                self.profiles
                .group_by('sample')
                .agg(pl.col('coverage').sum().alias('root_coverage'))
                .select(['sample', 'root_coverage'])
            )

    def _to_relabund_lf(self) -> pl.LazyFrame:
        """
        Convert coverage profiles to relative abundance.
        Returns a new TaxonomicProfiles instance with relabund column.
        """
        if self.root is None:
            raise ValueError("Root coverage is required to convert to relative abundance. Use create_root() first.")

        schema_names = self.profiles.collect_schema().names()
        if 'relabund' in schema_names:
            return self  # Already relative abundance

        # Join root coverage to profiles
        relabund = (
            self.profiles
            .join(self.root.select(['sample', 'root_coverage']), on='sample')
            .with_columns([
                (pl.col('coverage') / pl.col('root_coverage')).alias('relabund')
            ])
            .select(['sample', 'taxonomy', 'relabund'])
        )

        return relabund

    def _filter_profiles_by_sample(self, samples: pl.LazyFrame) -> pl.LazyFrame:
        """Internal method: filter profiles by samples and return LazyFrame."""
        return self.profiles.join(samples, on='sample', how='semi')

    def _filter_root_by_sample(self, samples: pl.LazyFrame) -> pl.LazyFrame:
        """Internal method: filter root by samples and return LazyFrame."""
        if self.root is None:
            raise ValueError("No root data to filter")
        return self.root.join(samples, on='sample', how='semi')

    def fill_profiles(self) -> "TaxonomicProfiles":
        """
        Fill the taxonomic profiles so that each parent's abundance is the sum of its children.
        Returns a new TaxonomicProfiles instance with filled profiles.
        """
        if self.is_filled:
            return self  # Already filled

        filled_lf = self._fill_profiles_lf()
        
        # Create new instance
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.profiles = filled_lf
        new_instance.root = self.root
        new_instance.is_filled = True
        return new_instance


    def create_root(self) -> "TaxonomicProfiles":
        """
        Create the root dataframe if not provided.
        Only possible if profiles are in coverage (not relabund) format.
        Returns a new TaxonomicProfiles instance with root set.
        """
        if hasattr(self, 'root') and self.root is not None:
            return self

        root_lf = self._create_root_lf()
        
        # Create new instance
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.profiles = self.profiles
        new_instance.root = root_lf
        new_instance.is_filled = self.is_filled
        return new_instance


    def create_features(self, rank: [Union[str, TaxonomicRanks]]) -> "FeatureSet":
        """
        Create a FeatureSet from the taxonomic profiles at a specified rank.
        
        Args:
            rank: Taxonomic rank to extract features from
        Returns:
            FeatureSet instance with features at the specified rank
        """
        if self.profiles is None:
            raise ValueError("No profiles available to create features.")
        if self.is_filled is False:
            raise ValueError("Profiles must be in filled format to create features. Use fill_profiles() first.")
        if 'relabund' not in self.profiles.collect_schema().names():
            raise ValueError("Profiles must be in relative abundance format to create features. Use _to_relabund_lf() first.")

        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)

        # Extract features at the specified rank
        features = (
            self.profiles
            .filter(pl.col("taxonomy").str.contains(rank.prefix))
            .select(["sample", "taxonomy", "relabund"])
            .collect()
            .pivot(values="relabund", index="sample", columns="taxonomy", aggregate_function="first")
        )

        return FeatureSet.from_df(
            df=features,
            label_column=None)