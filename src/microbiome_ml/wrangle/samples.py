"""High-level sample coordination class for ML workflows."""

import numpy as np
import polars as pl
from typing import List, Union, Optional, Dict, Any, Tuple
from pathlib import Path

from microbiome_ml.wrangle.metadata import SampleMetadata
from microbiome_ml.wrangle.profiles import TaxonomicProfiles
from microbiome_ml.wrangle.features import FeatureSet
from microbiome_ml.utils.taxonomy import TaxonomicRanks


class Dataset:
    """
    Global entry point for microbiome ML workflows with flexible builder pattern.
    
    Can be initialized empty and components added iteratively, or with all data at once.
    Serves as the central data structure for ML workflows.
    
    Examples:
        # Empty initialization with builder pattern
        samples = (Dataset()
                  .add_metadata(metadata='path/to/metadata.csv', attributes='path/to/attributes.csv')
                  .add_profiles('path/to/profiles.csv')
                  .add_features('genus', rank=TaxonomicRanks.GENUS))
        
        # Full initialization
        samples = Dataset(
            metadata={'metadata': 'path/to/metadata.csv', 'attributes': 'path/to/attributes.csv'},
            profiles='path/to/profiles.csv'
        )
        
        # Mixed approach
        samples = Dataset(metadata=existing_metadata)
        samples.add_profiles('path/to/profiles.csv').add_features('genus', rank=TaxonomicRanks.GENUS)
    """
    
    def __init__(
        self,
        metadata: Optional[Union[SampleMetadata, Dict[str, Union[str, Path, pl.LazyFrame]]]] = None,
        profiles: Optional[Union[TaxonomicProfiles, str, Path, pl.LazyFrame]] = None,
        features: Optional[Dict[str, FeatureSet]] = None
    ):
        """
        Initialize Dataset with optional components.
        
        Args:
            metadata: SampleMetadata instance or dict with metadata sources
            profiles: TaxonomicProfiles instance or data source
            features: Dict of FeatureSets by name
        """
        # Initialize empty structure
        self.metadata: Optional[SampleMetadata] = None
        self.profiles: Optional[TaxonomicProfiles] = None
        self.feature_sets: Dict[str, FeatureSet] = {}
        self.labels: Dict[str, Any] = {}
        self._sample_ids: Optional[List[str]] = None
        
        # Process provided components if given
        if metadata is not None:
            self.add_metadata(metadata)
        if profiles is not None:
            self.add_profiles(profiles)
        if features is not None:
            self.feature_sets.update(features)
            self._update_sample_ids()
    
    def add_metadata(
        self, 
        data: Optional[Union[SampleMetadata, Dict[str, Union[str, Path, pl.LazyFrame]]]] = None,
        metadata: Optional[Union[str, Path, pl.LazyFrame]] = None,
        attributes: Optional[Union[str, Path, pl.LazyFrame]] = None,
        study_titles: Optional[Union[str, Path, pl.LazyFrame]] = None,
        **kwargs
    ) -> "Dataset":
        """
        Add metadata component with flexible input handling.
        
        Args:
            data: SampleMetadata instance or dict with metadata sources
            metadata: Direct metadata source
            attributes: Direct attributes source  
            study_titles: Direct study_titles source
            **kwargs: Additional metadata sources
            
        Returns:
            Self for chaining
            
        Examples:
            .add_metadata(metadata='path.csv', attributes='attr.csv')
            .add_metadata({'metadata': df, 'attributes': 'path.csv'})
            .add_metadata(existing_metadata_instance)
        """
        # Handle SampleMetadata instance
        if isinstance(data, SampleMetadata):
            self.metadata = data
        # Handle dict with named metadata sources
        elif isinstance(data, dict):
            self.metadata = SampleMetadata(
                metadata=data.get('metadata'),
                attributes=data.get('attributes'),
                study_titles=data.get('study_titles', None)
            )
        # Handle direct keyword arguments
        else:
            # Use provided kwargs or direct parameters
            meta_source = metadata or kwargs.get('metadata')
            attr_source = attributes or kwargs.get('attributes')
            study_source = study_titles or kwargs.get('study_titles')
            
            if meta_source and attr_source:
                self.metadata = SampleMetadata(
                    metadata=meta_source,
                    attributes=attr_source,
                    study_titles=study_source
                )
        
        self._update_sample_ids()
        return self
    
    def add_profiles(
        self,
        profiles: Union[TaxonomicProfiles, str, Path, pl.LazyFrame],
        root: Optional[Union[str, Path, pl.LazyFrame]] = None,
        check_filled: bool = True,
        sample_size: int = 1000,
        **kwargs
    ) -> "Dataset":
        """
        Add profiles component with flexible input handling.
        
        Args:
            profiles: TaxonomicProfiles instance or data source
            root: Optional root coverage data
            check_filled: Whether to check if profiles are filled
            sample_size: Sample size for filled format checking
            **kwargs: Additional TaxonomicProfiles parameters
            
        Returns:
            Self for chaining
        """
        # Handle TaxonomicProfiles instance
        if isinstance(profiles, TaxonomicProfiles):
            self.profiles = profiles
        # Handle file paths, DataFrames, LazyFrames
        else:
            self.profiles = TaxonomicProfiles(
                profiles=profiles,
                root=root,
                check_filled=check_filled,
                sample_size=sample_size,
                **kwargs
            )
        
        self._update_sample_ids()
        return self
    
    def add_features(
        self,
        name: str,
        features: Optional[Union[FeatureSet, pl.DataFrame]] = None,
        rank: Optional[Union[str, TaxonomicRanks]] = None,
        accessions: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        feature_array: Optional[np.ndarray] = None,
        **kwargs
    ) -> "Dataset":
        """
        Add feature set with flexible input handling.
        
        Args:
            name: Name for the feature set
            features: FeatureSet instance or DataFrame
            rank: Taxonomic rank to create features from profiles
            accessions: Direct accessions list
            feature_names: Direct feature names list
            feature_array: Direct feature numpy array
            **kwargs: Additional parameters for FeatureSet creation
            
        Returns:
            Self for chaining
            
        Examples:
            .add_features('genus', rank=TaxonomicRanks.GENUS)  # From profiles
            .add_features('custom', features=dataframe)
            .add_features('existing', features=feature_set_instance)
            .add_features('direct', accessions=[...], feature_names=[...], feature_array=array)
        """
        # Handle FeatureSet instance
        if isinstance(features, FeatureSet):
            self.feature_sets[name] = features
        # Handle DataFrame conversion
        elif isinstance(features, pl.DataFrame):
            self.feature_sets[name] = FeatureSet.from_df(features, name=name, **kwargs)
        # Create from profiles using rank
        elif rank is not None:
            if self.profiles is None:
                raise ValueError("Profiles must be added before creating taxonomic features")
            feature_set = self.profiles.create_features(rank)
            feature_set.name = name
            self.feature_sets[name] = feature_set
        # Create directly from components
        elif accessions and feature_names and feature_array is not None:
            self.feature_sets[name] = FeatureSet(
                accessions=accessions,
                feature_names=feature_names,
                features=feature_array,
                name=name
            )
        else:
            raise ValueError("Must provide FeatureSet instance, DataFrame, rank parameter, or direct components")
        
        return self
    
    def iter_feature_sets(self, names: Optional[List[str]] = None):
        """
        Iterate over feature sets.
        
        Args:
            names: List of feature set names to iterate over (all if None)
            
        Yields:
            Tuple of (name, FeatureSet) for each feature set
        """
        if names is None:
            names = list(self.feature_sets.keys())
        for name in names:
            yield name, self.feature_sets[name]

    def iter_labels(self, names: Optional[List[str]] = None):
        """
        Iterate over label sets.
        
        Args:
            names: List of label set names to iterate over (all if None)
            
        Yields:
            Tuple of (name, label_data) for each label set
        """
        if names is None:
            names = list(self.labels.keys())
        for name in names:
            yield name, self.labels[name]
    
    def _update_sample_ids(self):
        """Recalculate canonical sample list when components change."""
        if self.metadata is None or self.profiles is None:
            self._sample_ids = None
            return
        
        # Get samples from both metadata and profiles
        metadata_samples = set(self.metadata.metadata.select("sample").collect().to_series().to_list())
        profiles_samples = set(self.profiles.get_samples())
        
        # Find intersection and sort for canonical ordering
        common_samples = sorted(list(metadata_samples.intersection(profiles_samples)))
        self._sample_ids = common_samples
