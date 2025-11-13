"""Utilities for loading raw tabular datasets and preparing them for the
synthetic data evaluation pipeline.

The functions in this module are intentionally lightweight so that the
project can be executed end-to-end without manual data preparation steps.
They take care of reading the configured raw dataset, performing a small set
of cleaning operations, persisting the processed dataset, and generating the
SDV metadata artefacts required by the downstream components.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata

from src.config import DatasetConfig

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for *path* if it does not already exist."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_and_clean_data(raw_path: str, processed_path: str) -> Optional[pd.DataFrame]:
    """Load the configured raw dataset and apply basic cleaning steps.

    The cleaning intentionally stays minimal so it works for the Adult income
    dataset shipped with the repository while still being easy to adapt for
    future datasets.  The current steps are:

    1. Read the CSV file from ``raw_path``.
    2. Drop columns flagged in :class:`DatasetConfig`.
    3. Trim whitespace from string columns and normalise their dtype to
       ``str`` so downstream encoders behave deterministically.
    4. Impute missing values (median for numeric, mode for categorical).
    5. Persist the cleaned data to ``processed_path`` for reuse.
    """

    if not os.path.exists(raw_path):
        logger.error("Raw data file not found at %s", raw_path)
        return None

    logger.info("Loading raw dataset from %s", raw_path)
    try:
        data = pd.read_csv(raw_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load raw dataset: %s", exc, exc_info=True)
        return None

    # Drop user-specified columns if they are present.
    if DatasetConfig.COLS_TO_DROP:
        data = data.drop(columns=[col for col in DatasetConfig.COLS_TO_DROP if col in data.columns])

    # Trim whitespace from string columns and ensure a consistent dtype.
    for column in data.select_dtypes(include=["object", "string", "category"]).columns:
        data[column] = data[column].astype(str).str.strip()

    # Basic imputation so downstream models do not fail on NaN values.
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns

    data[numeric_cols] = data[numeric_cols].apply(lambda col: col.fillna(col.median()))

    def _fill_categorical(col: pd.Series) -> pd.Series:
        if col.mode().empty:
            return col.fillna("Unknown")
        return col.fillna(col.mode().iloc[0])

    data[categorical_cols] = data[categorical_cols].apply(_fill_categorical)

    _ensure_parent_dir(processed_path)
    data.to_csv(processed_path, index=False)
    logger.info("Processed dataset saved to %s", processed_path)

    return data


def generate_and_save_metadata(data: pd.DataFrame, metadata_path: str) -> SingleTableMetadata:
    """Create SDV metadata for *data* and persist it to ``metadata_path``."""

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    _ensure_parent_dir(metadata_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    metadata.save_to_json(metadata_path)
    logger.info("Metadata saved to %s", metadata_path)

    return metadata
