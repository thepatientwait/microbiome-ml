"""Core data models for microbiome data structures."""

import numpy as np
import polars as pl
from typing import List, Any, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from microbiome_ml.wrangle.profiles import TaxonomicProfiles
    from microbiome_ml.utils.taxonomy import TaxonomicRanks


class FeatureSet:
    """
    ML-ready feature set with numpy arrays and metadata.
    
    Stores features as numpy arrays with associated metadata lists
    for efficient ML operations while maintaining data provenance.
    """
    
    def __init__(
        self, 
        accessions: List[str], 
        feature_names: List[str], 
        features: np.ndarray,
        name: Optional[str] = None):
        """
        Initialize FeatureSet with validation.
        
        Args:
            accessions: Ordered sample/species IDs
            feature_names: Ordered feature names
            features: Two-dimensional numpy array (accessions by features)
            name: Optional name for the FeatureSet
            
        Raises:
            ValueError: If array dimensions don't match list lengths
        """
        # Internal validation: numpy array dimensions match list lengths
        if features.shape[0] != len(accessions):
            raise ValueError(f"Features array rows ({features.shape[0]}) must match accessions length ({len(accessions)})")
        if features.shape[1] != len(feature_names):
            raise ValueError(f"Features array cols ({features.shape[1]}) must match feature_names length ({len(feature_names)})")
            
        self.accessions = accessions
        self.feature_names = feature_names
        self.features = features
        self.name = name
        
    @classmethod
    def from_df(
        cls, 
        df: pl.DataFrame, 
        acc_column: Optional[str] = None,
        name: Optional[str] = None
        ) -> "FeatureSet":
        """
        Create FeatureSet from wide-form DataFrame (features only, no labels).
        
        Args:
            df: Wide-form DataFrame with features as columns
            acc_column: Column name for accessions (defaults to auto-detect)
            name: Optional name for the FeatureSet
            
        Returns:
            FeatureSet instance
        """
        # If acc_column=None, test for 'acc' first, then 'sample'
        if acc_column is None:
            if 'acc' in df.columns:
                acc_column = 'acc'
            elif 'sample' in df.columns:
                acc_column = 'sample'
            else:
                raise ValueError("No accession column found. Expected 'acc' or 'sample' column, or specify acc_column")
        
        # Extract components
        accessions = df.select(acc_column).to_series().to_list()
        
        # Feature columns are everything except acc_column
        feature_columns = [col for col in df.columns if col != acc_column]
        feature_names = feature_columns
        
        # Convert features to numpy array
        features = df.select(feature_columns).to_numpy()
        
        # All validation happens in __init__
        return cls(accessions=accessions, feature_names=feature_names,
                  features=features, name=name)
    
    @classmethod
    def from_profiles(
        cls, 
        profiles: "TaxonomicProfiles", 
        sample_ids: List[str], 
        rank: "TaxonomicRanks",
        name: Optional[str] = None
        ) -> "FeatureSet":
        """
        Create FeatureSet from TaxonomicProfiles at specified taxonomic rank.
        
        Args:
            profiles: TaxonomicProfiles instance
            sample_ids: List of sample IDs to include
            rank: Taxonomic rank to extract features from
            name: Optional name for the FeatureSet
            
        Returns:
            FeatureSet instance with features at the specified rank
        """
        # Convert rank to enum if string
        from microbiome_ml.utils.taxonomy import TaxonomicRanks
        if isinstance(rank, str):
            rank = TaxonomicRanks.from_name(rank)
        
        # Filter profiles to specified samples
        import polars as pl
        sample_filter = pl.DataFrame({"sample": sample_ids}).lazy()
        filtered_profiles = profiles._filter_by_sample(sample_filter)
        
        # Use the TaxonomicProfiles create_features method
        feature_set = filtered_profiles.create_features(rank)
        
        # Set name if provided
        if name is not None:
            feature_set.name = name
        
        return feature_set
    
    def get_samples(self) -> List[str]:
        """Get list of sample/accession IDs."""
        return self.accessions.copy()
    
    def get_features(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()
    
    def filter_samples(self, sample_ids: List[str]) -> "FeatureSet":
        """
        Filter FeatureSet to specific samples.
        
        Args:
            sample_ids: List of sample IDs to keep
            
        Returns:
            New FeatureSet with filtered samples
        """
        # Find indices of samples to keep
        keep_indices = [i for i, acc in enumerate(self.accessions) if acc in sample_ids]
        
        if not keep_indices:
            raise ValueError("No matching samples found")
        
        # Filter arrays/lists by indices
        filtered_accessions = [self.accessions[i] for i in keep_indices]
        filtered_features = self.features[keep_indices, :]
        
        # Create new FeatureSet with filtered data
        return FeatureSet(
            accessions=filtered_accessions,
            feature_names=self.feature_names.copy(),
            features=filtered_features,
            name=self.name
        )