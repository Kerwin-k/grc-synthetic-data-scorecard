#
# 文件名: src/config.py
# File: src/config.py
#
# 架构师说明 (Architect's Note):
# 这是整个框架的唯一控制面板。
# This is the SINGLE CONTROL PANEL for the entire framework.
#
# 当您需要评估一个新数据集时，这是您唯一需要修改的文件。
# When you need to evaluate a new dataset, this is the ONLY file you need to modify.
#

import os
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer
)

# --- 路径配置 (Path Configuration) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PathConfig:
    """定义关键目录和文件路径 / Defines all key directory and file paths"""
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    SYNTH_DIR = os.path.join(RESULTS_DIR, "synthetic_data")
    EMISSIONS_DIR = os.path.join(RESULTS_DIR, "emissions")

    METRICS_REPORT_PATH = os.path.join(RESULTS_DIR, "metrics_report.json")
    GRC_SCORECARD_CSV_PATH = os.path.join(RESULTS_DIR, "grc_scorecard.csv")
    GRC_SCORECARD_IMG_PATH = os.path.join(RESULTS_DIR, "grc_scorecard.png")

# --- 碳排放配置 (CodeCarbon Configuration) ---
class SustainabilityConfig:
    """
    CodeCarbon 设置 / CodeCarbon settings.
    FIXED_COUNTRY_ISO: 设为 ISO 代码 (如 'MYS', 'CHN') 可固定排放因子，方便复现。
                       Set to ISO code (e.g., 'MYS') for reproducible emissions factors.
    """
    FIXED_COUNTRY_ISO: str | None = None     # 默认 None 使用自动 IP 定位 / Default None uses auto IP lookup
    FALLBACK_COUNTRY_LABEL: str = "Canada"    # 日志显示的后备国家 / Fallback country label for logs

# --- 数据集配置 (Dataset Configuration) ---
class DatasetConfig:
    # 原始文件名 (位于 data/raw/) / Raw filename (in data/raw/)
    RAW_DATA_FILE = "adult.csv"
    # Home_Credit Dataset
    # RAW_DATA_FILE = "application_train.csv"

    # 目标列 (用于 ML 效用和公平性) / Target column (for ML Utility & Fairness)
    TARGET_COLUMN = "income"
    # Home_Credit Dataset
    # TARGET_COLUMN = "TARGET"

    # 正类标签 (用于 F1 分数) / Positive label (for F1 score)
    POSITIVE_LABEL = ">50K"
    # Home_Credit Dataset
    # POSITIVE_LABEL = "1"

    # 敏感属性 (用于公平性评估) / Sensitive attributes (for Fairness evaluation)
    SENSITIVE_FEATURES = ["sex", "race"]
    # Home_Credit Dataset
    # SENSITIVE_FEATURES = ["CODE_GENDER", "DAYS_BIRTH"]

    # 需要丢弃的列 (ID, 权重等) / Columns to drop (IDs, weights, etc.)
    COLS_TO_DROP = ["fnlwgt", "education"]
    # Home_Credit Dataset
    # COLS_TO_DROP = ["SK_ID_CURR"]

    # 采样大小 (防止内存溢出) / Sample size (to prevent OOM)
    # None = 使用全部数据 / None = use full data
    SAMPLE_SIZE: int | None = 50_000

    # 是否分层抽样 / Stratify sampling by target
    STRATIFY_BY_TARGET: bool = True

    # 随机种子 / Random state
    RANDOM_STATE: int = 42

    # 采样模式 / Sampling Mode: 'fixed' (下采样), 'full' (全量), 'auto'
    SAMPLING_MODE: str = "fixed"

    @staticmethod
    def get_dynamic_sample_size():
        """
        Implements Resource-Aware Heuristic Assessment.
        Returns a safe sample size based on available RAM.
        """
        if psutil is None:
            return 50_000  # Fallback if psutil not installed

        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)

            if available_gb > 16:
                return 100_000
            elif available_gb > 8:
                return 50_000
            else:
                return 20_000
        except Exception:
            return 50_000

    # 自动路径 (勿改) / Auto-generated paths (Do not edit)
    RAW_PATH = os.path.join(PathConfig.RAW_DIR, RAW_DATA_FILE)
    CLEAN_FILE = RAW_DATA_FILE.replace(".csv", "_clean.csv")
    PROCESSED_PATH = os.path.join(PathConfig.PROCESSED_DIR, CLEAN_FILE)
    META_FILE = RAW_DATA_FILE.replace(".csv", "_metadata.json")
    METADATA_PATH = os.path.join(PathConfig.METADATA_DIR, META_FILE)

# --- 资源配置 (Resource Configuration) ---
class ResourceConfig:
    """防止内存溢出 (OOM) 的限制配置 / Constraints to prevent Out-Of-Memory (OOM)"""
    # 训练数据行数上限 / Max training rows per model
    MAX_TRAIN_ROWS_PER_MODEL: int | None = 50_000

    # TSTR/MIA 评估行数上限 / Max rows for TSTR/MIA evaluation
    MAX_ROWS_TSTR: int | None = 50_000
    MAX_ROWS_MIA: int | None = 50_000

    # 启用浮点数降精度 / Enable float64 -> float32 downcasting
    ENABLE_DTYPE_DOWNCAST: bool = True

    # 资源密集型模型 / Heavy models
    HEAVY_MODELS = ("CTGAN", "TVAE")

# --- RAG 阈值配置 (RAG Threshold Configuration) ---
class RAGThresholdConfig:
    """红黄绿 (Red/Amber/Green) 评分阈值 / RAG Scoring Thresholds"""

    # 1) 质量 (越高越好) / Quality (Higher is better)
    QUALITY_JSD = {"green": 0.90, "amber": 0.80}
    QUALITY_NMI = {"green": 0.80, "amber": 0.60}

    # 2) 效用 (越高越好)(需根据实际 XGBoost 基线调整)  / Utility (Higher is better)
    UTILITY_TSTR_F1 = {"green": 0.76, "amber": 0.70}
    #UTILITY_TSTR_F1 = {"green": 0.40, "amber": 0.20}  # Home_Credit Dataset

    # 3) 风险 (越低越好) / Risk (Lower is better)
    PRIVACY_MIA = {"green": 0.55, "amber": 0.65} # MIA AUC
    FAIRNESS = {"green": 0.10, "amber": 0.20}   # Avg Difference
    # Home_Credit Dataset
    # PRIVACY_MIA = {"green": 0.55, "amber": 0.65}
    # FAIRNESS = {"green": 0.10, "amber": 0.20}

    # 4) 可持续性 (越低越好) / Sustainability (Lower is better)
    SUSTAIN_CO2 = {"green": 0.005, "amber": 0.05} # kg CO2
    SUSTAIN_CO2_NEAR_ZERO = 1e-3  # 忽略极小排放的阈值 / Threshold to ignore negligible emissions

    # SUSTAIN_TIME = {"green": 60.0, "amber": 300.0} # Seconds
    SUSTAIN_TIME = {"green": 60.0, "amber": 600.0}  # Seconds

# --- 模型配置 (Model Configuration) ---
MODELS_CONFIG = {
    "GaussianCopula": {
        "class": GaussianCopulaSynthesizer,
        "params": {}
    },
    "CTGAN": {
        "class": CTGANSynthesizer,
        "params": {"epochs": 5} # 测试用 / For testing
    },
    "TVAE": {
        "class": TVAESynthesizer,
        "params": {"epochs": 5} # 测试用 / For testing
    }
}

# --- GRC 可视化配置 (GRC Visualization Config) ---
class GRCConfig:
    RAG_COLORS = {
        'Green': '#90EE90',
        'Amber': '#FFBF00',
        'Red': '#F08080',
        'N/A': '#D3D3D3'
    }