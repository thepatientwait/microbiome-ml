"""microbiomeML.

A Python module for standardizing machine learning applications within
microbiome research groups. Provides tools for feature manipulation, data
annotation, and label processing.
"""

# Key utilities
from .utils.logging import get_logger, setup_logging

# Core data structures
from .utils.taxonomy import TaxonomicRanks
from .wrangle import FeatureSet, SampleMetadata, TaxonomicProfiles

__version__ = "0.1.0"
__author__ = "Your Research Group"

# Heavier classes are imported on first access (PEP 562) so that
# ``import microbiome_ml`` stays cheap and does not pull in sklearn/matplotlib.
_LAZY = {
    "Dataset": "microbiome_ml.wrangle.dataset",
    "LazySpeciesFeatureSet": "microbiome_ml.wrangle.species_table",
    "CrossValidator": "microbiome_ml.train.cv",
    "ModelTrainer": "microbiome_ml.train.trainer",
    "Visualiser": "microbiome_ml.visualise.visualisations",
}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'microbiome_ml' has no attribute {name!r}")


__all__ = [
    "Dataset",
    "LazySpeciesFeatureSet",
    "CrossValidator",
    "ModelTrainer",
    "Visualiser",
    "TaxonomicProfiles",
    "SampleMetadata",
    "FeatureSet",
    "TaxonomicRanks",
    "setup_logging",
    "get_logger",
]

# Importing the package does not configure logging: that is the application's
# job (a Snakemake job, a script or a notebook). Call ``setup_logging()`` or
# use ``tee_log(path)`` from a command-line entry point.
