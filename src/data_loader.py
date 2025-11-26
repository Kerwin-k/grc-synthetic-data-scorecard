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
       - 尝试转换为数值 / Attempt numeric conversion
    5. 降精度 (float64 -> float32) 以节省内存 / Downcast float64 to float32
    6. 保存处理后的数据 / Save processed data
    """
    raw_path = os.path.join(PathConfig.RAW_DIR, DatasetConfig.RAW_DATA_FILE)
    logging.info(f"[DataLoader] Loading raw dataset from {raw_path}")

    df = pd.read_csv(raw_path)

    # ----------------------------
    # 1) 中心化采样 / Centralized Sampling
    # ----------------------------
    mode = getattr(DatasetConfig, "SAMPLING_MODE", "fixed")
    sample_size: Optional[int] = None

    if mode == "full":
        logging.info("...")  # 保留原逻辑
        sample_size = None
    elif mode == "auto":
        # 新增: 自动模式调用动态检测
        sample_size = DatasetConfig.get_dynamic_sample_size()
        logging.info(
            "[DataLoader] SAMPLING_MODE='auto' -> Resource-Aware System detected safe sample size: %d",
            sample_size
        )
    else:
        # 默认: Fixed fixed mode (论文使用模式)
        sample_size = DatasetConfig.SAMPLE_SIZE
        logging.info(
            "[DataLoader] SAMPLING_MODE='fixed' -> Using strict cap: %d rows for reproducibility.",
            sample_size
        )

    if sample_size is not None and len(df) > sample_size:
        logging.info(
            f"[DataLoader] Dataset has {len(df)} rows; "
            f"downsampling to {sample_size} rows "
            f"(stratify={getattr(DatasetConfig, 'STRATIFY_BY_TARGET', False)})."
        )
        if getattr(DatasetConfig, "STRATIFY_BY_TARGET", False) and DatasetConfig.TARGET_COLUMN in df.columns:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=sample_size,
                random_state=getattr(DatasetConfig, "RANDOM_STATE", 42),
            )
            y = df[DatasetConfig.TARGET_COLUMN]
            _, sample_idx = next(splitter.split(df, y))
            df = df.iloc[sample_idx].reset_index(drop=True)
        else:
            df = df.sample(
                n=sample_size,
                random_state=getattr(DatasetConfig, "RANDOM_STATE", 42),
            ).reset_index(drop=True)

    logging.info("[DataLoader] Shape after central sampling: %s", df.shape)

    # ----------------------------
    # 2) 删除指定列 / Drop Columns
    # ----------------------------
    if getattr(DatasetConfig, "COLS_TO_DROP", None):
        drop_cols = [c for c in DatasetConfig.COLS_TO_DROP if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logging.info("[DataLoader] Dropped columns: %s", drop_cols)

    # ----------------------------
    # 3) 基础清洗 / Basic Cleaning
    # ----------------------------
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "NA": np.nan, "NaN": np.nan})
        )

    # 尝试将对象列转换为数值 / Try converting object columns to numeric
    for col in df.columns:
        if df[col].dtype == "object":
            try_num = pd.to_numeric(df[col], errors="ignore")
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