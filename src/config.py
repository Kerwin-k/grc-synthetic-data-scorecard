#
# 文件名: src/config.py
# File: src/config.py
#
# 架构师说明 (Architect's Note):
#
# 这-是-整-个-框-架-的-唯-一-控-制-面-板。
# This is the SINGLE CONTROL PANEL for the entire framework.
#
# 当您需要评估一个新数据集时 (例如, 'company_x_data.csv')，
# 这-是-您-唯-一-需-要-修-改-的-文-件。
# When you need to evaluate a new dataset (e.g., 'company_x_data.csv'),
# this is the ONLY file you need to modify.
#
# 核心逻辑模块 (data_loader, model_trainer, evaluation_engine, grc_translator)
# 将从此文件导入所有配置。
# The core logic modules (data_loader, model_trainer, evaluation_engine, grc_translator)
# will import all configurations from this file.
#

import os
import numpy as np

from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer
)

# --- 1. 路径配置 (Path Configuration) ---
# 自动检测项目根目录 (thesis_project)
# Automatically detects the project root directory (thesis_project)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 定义所有关键目录和文件路径
# Defines all key directory and file paths
class PathConfig:
    # 核心目录 / Core Directories
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # 子目录 / Sub-directories
    SYNTH_DIR = os.path.join(RESULTS_DIR, "synthetic_data")
    EMISSIONS_DIR = os.path.join(RESULTS_DIR, "emissions")

    # 结果文件 / Result Files
    METRICS_REPORT_PATH = os.path.join(RESULTS_DIR, "metrics_report.json")
    GRC_SCORECARD_CSV_PATH = os.path.join(RESULTS_DIR, "grc_scorecard.csv")
    GRC_SCORECARD_IMG_PATH = os.path.join(RESULTS_DIR, "grc_scorecard.png")  # [1]


# --- 2. 数据集配置 (Dataset Configuration) ---
# [!!] 关键区域: 当您更换数据集时，请在此处更新 [1]
# [!!] KEY SECTION: Update this section when you change datasets [1]
class DatasetConfig:
    # 2.1. 文件定义 / File Definition
    RAW_DATA_FILE = "adult.csv"  # 位于 /data/raw/ 中的原始文件名 [1]
    # The raw data filename in /data/raw/ [1]

    # 2.2. 模式定义 / Schema Definition
    # 定义用于机器学习效用 (TSTR) 和公平性评估的目标列
    # Define the target column for Machine Learning Utility (TSTR) and Fairness
    TARGET_COLUMN = "income"

    # 定义目标列中的“阳性”标签 (用于计算 F1 分数)
    # Define the "positive" label in the target column (for F1 score)
    POSITIVE_LABEL = ">50K"

    # 定义用于公平性评估的受保护属性
    # Define the protected attributes for fairness evaluation
    SENSITIVE_FEATURES = [
    "sex", "race"
    ]

    # 定义在预处理期间应*丢弃*的列
    # (例如, 标识符, 采样权重, 或冗余列)
    # Define columns to *drop* during preprocessing
    # (e.g., identifiers, sampling weights, or redundant columns)
    COLS_TO_DROP = [
    "fnlwgt", "education"
    ]

    # 2.3. 自动生成的路径 (请勿编辑)
    # 2.3. Auto-generated paths (DO NOT EDIT)
    RAW_PATH = os.path.join(PathConfig.RAW_DIR, RAW_DATA_FILE)

    CLEAN_FILE = RAW_DATA_FILE.replace(".csv", "_clean.csv")
    PROCESSED_PATH = os.path.join(PathConfig.PROCESSED_DIR, CLEAN_FILE)

    META_FILE = RAW_DATA_FILE.replace(".csv", "_metadata.json")
    METADATA_PATH = os.path.join(PathConfig.METADATA_DIR, META_FILE)


# --- 3. 模型训练配置 (Model Training Configuration) ---
# 定义要比较的模型 / Define the models to compare

# 填充模型配置
# Populate model configuration
MODELS_CONFIG = {
    "GaussianCopula": {
        "class": GaussianCopulaSynthesizer,
        "params": {}
    },
    "CTGAN": {
        "class": CTGANSynthesizer,
        "params": {"epochs": 5}  # 低轮数用于本地测试 / Low epochs for local testing
    },
    "TVAE": {
        "class": TVAESynthesizer,
        "params": {"epochs": 5}  # 低轮数用于本地测试 / Low epochs for local testing
    }
}


# --- 复制到此结束 ---


# --- 4. GRC 记分卡配置 (GRC Scorecard Configuration) ---
# 定义记分卡的结构、指标和 RAG (红/黄/绿) 阈值 [1]
# Defines the scorecard structure, metrics, and RAG thresholds [1]
class GRCConfig:
    # 定义记分卡中值的顺序 (英文)
    # Defines the order of values in the scorecard (English)
    METRIC_ORDER = [
        'Score', 'RAG'
    ]

    # RAG 阈值定义 / RAG Threshold Definitions
    THRESHOLDS = {
        'JSD': {'green': 0.9, 'amber': 0.8},  # 越高越好 / Higher-is-better
        'NMI': {'green': 0.8, 'amber': 0.6},  # 越高越好 / Higher-is-better
        'TSTR': {'green': 0.76, 'amber': 0.70},  # 越高越好 / Higher-is-better
        'MIA': {'green': 0.55, 'amber': 0.65},  # 越低越好 / Lower-is-better
        'FAIR': {'green': 0.1, 'amber': 0.2}  # 越低越好 / Lower-is-better
    }

    #
    # 映射: (类别, 指标) -> (JSON 键, 阈值, 逻辑)
    # Mapping: (Category, Metric) -> (JSON key, Threshold, Logic)
    METRIC_MAP = {
        'Quality': [
            # (显示名称 / Display Name, JSON 键 / JSON Key, 阈值键 / Threshold Key)
            ("Distribution (JSD Score)", "fidelity_jsd_avg", THRESHOLDS),
            ("Correlation (NMI Score)", "fidelity_nmi_avg", THRESHOLDS['NMI'])
        ],

        'Utility': [
            ("ML Utility (TSTR F1)", "utility_tstr_f1", THRESHOLDS)
        ],

        'Risk': [
            ("Privacy (MIA AUC)", "privacy_mia_auc", THRESHOLDS['MIA']),
            ("Fairness (Avg Diff)", "avg_fairness", THRESHOLDS)
        ],

        'Sustainability': [
            # (阈值为 None 表示相对排名)
            # (Threshold=None means relative ranking)
            ("CO2 Emissions (kg)", "co2_eq_kg", None),
            ("Training Time (s)", "training_time_sec", None)
        ]
    }

    # 用于可视化的 RAG 颜色 / New: RAG colors for visualization
    RAG_COLORS = {
        'Green': '#90EE90',  # 浅绿色 / Light Green
        'Amber': '#FFBF00',  # 琥珀色 / Amber
        'Red': '#F08080',  # 浅红色 / Light Red
        'N/A': '#D3D3D3'  # 灰色 / Gray
    }
