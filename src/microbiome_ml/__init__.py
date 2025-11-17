"""
microbiomeML

A Python module for standardizing machine learning applications within microbiome research groups.
Provides tools for feature manipulation, data annotation, and label processing.
"""

# Core data structures
from .utils.taxonomy import TaxonomicRanks

# Key utilities
from .utils.logging import setup_logging, get_logger
from .wrangle import TaxonomicProfiles, SampleMetadata, FeatureSet

__version__ = "0.1.0"
__author__ = "Your Research Group"

__all__ = [
    "TaxonomicProfiles",
    "SampleMetadata",  
    "FeatureSet",
    "TaxonomicRanks",
    "setup_logging",
    "get_logger",
]

# Configure default logging
setup_logging()
