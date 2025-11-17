"""Utility functions for microbiome ML data processing.

This module provides data loading and validation utilities for Polars LazyFrames.
"""

from pathlib import Path
import polars as pl
from typing import Union


def load_data(path: Union[str, Path]) -> pl.LazyFrame:
    """Smart file loading with automatic format detection.
    
    Automatically detects file format based on extension and returns
    a LazyFrame for efficient processing.
    
    Args:
        path: Path to data file
        
    Returns:
        LazyFrame for the loaded data
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
        
    Supported formats:
        - CSV (.csv)
        - TSV (.tsv) 
        - IPC (.ipc)
        - Parquet (.parquet, .pq)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    extension = path.suffix.lower()
    
    if extension == '.csv':
        return pl.scan_csv(path)
    elif extension == '.tsv':
        return pl.scan_csv(path, separator='\t')
    elif extension == '.ipc':
        return pl.scan_ipc(path)
    elif extension in ['.parquet', '.pq']:
        return pl.scan_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: .csv, .tsv, .ipc, .parquet, .pq"
        )


def validate_columns(lf: pl.LazyFrame, required_columns: list[str]) -> None:
    """Validate that required columns exist in LazyFrame.
    
    Args:
        lf: LazyFrame to validate
        required_columns: List of column names that must exist
        
    Raises:
        ValueError: If any required columns are missing
    """
    schema = lf.collect_schema()
    missing_columns = [col for col in required_columns if col not in schema.names()]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


