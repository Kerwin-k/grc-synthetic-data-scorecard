"""
加载原始表格数据并为合成数据评估流程做好准备的工具。
Utilities for loading raw tabular datasets and preparing them for the synthetic data evaluation pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import DatasetConfig, PathConfig

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    """如果路径不存在，则创建父目录。 / Create the parent directory for *path* if it does not already exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_and_clean_data() -> pd.DataFrame:
    """
    统一的数据加载入口 / Unified Data Loading Entry Point:

    1. 从 RAW_DIR 读取原始 CSV / Read raw CSV from RAW_DIR
    2. 中心化采样 / Centralized Sampling:
       - 根据 DatasetConfig.SAMPLING_MODE 处理（full 或 fixed）
       - 如果行数超过 SAMPLE_SIZE，执行分层或随机采样
    3. 删除指定列 / Drop specified columns (COLS_TO_DROP)
    4. 基础清洗 / Basic Cleaning:
       - 去除空白字符 / Strip whitespace
       - 统一缺失值标记 / Unify missing value markers
    5. 降精度 / Downcasting (float64 -> float32)
    6. 保存到 PROCESSED_DIR / Save to PROCESSED_DIR
    """
    raw_path = DatasetConfig.RAW_PATH
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}")

    logger.info(f"[DataLoader] Loading raw data from {raw_path}...")

    # 读取原始数据
    df = pd.read_csv(raw_path)

    # ----------------------------
    # 2) 中心化采样 / Centralized Sampling
    # ----------------------------
    # 如果配置为固定大小采样，且数据量超过阈值
    if DatasetConfig.SAMPLING_MODE == "fixed" and len(df) > DatasetConfig.SAMPLE_SIZE:
        target_col = getattr(DatasetConfig, "TARGET_COLUMN", None)

        if target_col and target_col in df.columns:
            # 分层采样
            logger.info(f"[DataLoader] Stratified sampling to {DatasetConfig.SAMPLE_SIZE} rows...")
            split = StratifiedShuffleSplit(n_splits=1, train_size=DatasetConfig.SAMPLE_SIZE, random_state=42)
            for train_index, _ in split.split(df, df[target_col]):
                df = df.iloc[train_index]
        else:
            # 随机采样
            logger.info(f"[DataLoader] Random sampling to {DatasetConfig.SAMPLE_SIZE} rows...")
            df = df.sample(n=DatasetConfig.SAMPLE_SIZE, random_state=42)

    # ----------------------------
    # 3) 删除列与清洗 / Dropping & Cleaning
    # ----------------------------

    # 删除配置中指定的列 (COLS_TO_DROP)
    if hasattr(DatasetConfig, 'COLS_TO_DROP') and DatasetConfig.COLS_TO_DROP:
        cols_to_drop = [c for c in DatasetConfig.COLS_TO_DROP if c in df.columns]
        if cols_to_drop:
            logger.info(f"[DataLoader] Dropping ignored columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

    # 1. Basic Cleaning
    # Remove identifiers if configured
    if hasattr(DatasetConfig, 'ID_COLUMN') and DatasetConfig.ID_COLUMN in df.columns:
        logger.info(f"[DataLoader] Removing ID column: {DatasetConfig.ID_COLUMN}")
        df = df.drop(columns=[DatasetConfig.ID_COLUMN])

    # 统一缺失值
    # 将常见的缺失值标记替换为 np.nan
    df = (
        df.replace(r"^\s*$", np.nan, regex=True)
        .replace(["", "NA", "NaN", "?", "null"], np.nan)
    )

    # 尝试将对象列转换为数值 / Try converting object columns to numeric
    for col in df.columns:
        if df[col].dtype == "object":
            try_num = pd.to_numeric(df[col], errors="ignore")
            # 只有当转换后确实变成了数值类型才覆盖
            if pd.api.types.is_numeric_dtype(try_num):
                df[col] = try_num

    # ----------------------------
    # 4) 降精度 / Downcasting
    # ----------------------------
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype("float32")

    # ----------------------------
    # 5) 持久化 / Persistence
    # ----------------------------
    processed_path = DatasetConfig.PROCESSED_PATH
    _ensure_parent_dir(processed_path)
    df.to_csv(processed_path, index=False)
    logging.info(f"[DataLoader] Processed dataset saved to {processed_path}")
    logging.info(f"[DataLoader] Final dataset shape: {df.shape}")

    return df


def generate_and_save_metadata(data: pd.DataFrame, metadata_path: str) -> SingleTableMetadata:
    """
    为数据生成 SDV 元数据并保存。
    Create SDV metadata for data and persist it.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    _ensure_parent_dir(metadata_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    metadata.save_to_json(metadata_path)
    logger.info("Metadata saved to %s", metadata_path)

    return metadata