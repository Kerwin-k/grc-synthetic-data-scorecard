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

# --- 路径配置 (Path Configuration) ---
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
    GRC_SCORECARD_IMG_PATH = os.path.join(RESULTS_DIR, "grc_scorecard.png")  #


# --- 碳排放 / CodeCarbon 配置 ---
class SustainabilityConfig:
    """
    配置 CodeCarbon 的国家设定。

    FIXED_COUNTRY_ISO:
        - 为 None 时 (默认): 使用 CodeCarbon 自带的 IP 自动定位。
          如果定位失败，CodeCarbon 内部会回退到默认国家（目前是 Canada），
          日志里会出现 “Using 'Canada' as the default value”。
        - 设置为 "MYS" / "CHN" / "HKG" / "CAN" 等时：强制所有实验使用该国家
          的电网排放因子，完全不再调用地理定位 API，更方便论文复现。

    FALLBACK_COUNTRY_LABEL 只用于日志说明，帮助你在 log 里看到写的是什么。
    """
    FIXED_COUNTRY_ISO: str | None = None     # 例如 "MYS" 或 "CHN"，不想固定就保持 None
    FALLBACK_COUNTRY_LABEL: str = "Canada"    # 仅用于说明文字

# --- 数据集配置 (Dataset Configuration) ---
# [!!] 关键区域: 当您更换数据集时，请在此处更新
# [!!] KEY SECTION: Update this section when you change datasets
class DatasetConfig:
    # 文件定义 / File Definition
    RAW_DATA_FILE = "adult.csv"  # 位于 /data/raw/ 中的原始文件名
    #     RAW_DATA_FILE = "application_train.csv"

    # 模式定义 / Schema Definition
    # 定义用于机器学习效用 (TSTR) 和公平性评估的目标列
    # Define the target column for Machine Learning Utility (TSTR) and Fairness
    TARGET_COLUMN = "income"
    #     TARGET_COLUMN = "TARGET"

    # 定义目标列中的“阳性”标签 (用于计算 F1 分数)
    # Define the "positive" label in the target column (for F1 score)
    POSITIVE_LABEL = ">50K"
    #     POSITIVE_LABEL = 1

    # 定义用于公平性评估的受保护属性
    # Define the protected attributes for fairness evaluation
    SENSITIVE_FEATURES =["sex", "race"]
    #     SENSITIVE_FEATURES = [
    #         "CODE_GENDER",  # gender
    #         "NAME_EDUCATION_TYPE",
    #         "NAME_FAMILY_STATUS",
    #         "DAYS_BIRTH"  # age proxy
    #     ]

    # 定义在预处理期间应*丢弃*的列
    # (例如, 标识符, 采样权重, 或冗余列)
    # Define columns to *drop* during preprocessing
    # (e.g., identifiers, sampling weights, or redundant columns)
    COLS_TO_DROP = ["fnlwgt", "education"]
    #     COLS_TO_DROP = [
    #         "SK_ID_CURR",  # useless ID
    #     ]
    #



    # 自动生成的路径 (请勿编辑)
    # Auto-generated paths (DO NOT EDIT)
    RAW_PATH = os.path.join(PathConfig.RAW_DIR, RAW_DATA_FILE)
    CLEAN_FILE = RAW_DATA_FILE.replace(".csv", "_clean.csv")
    PROCESSED_PATH = os.path.join(PathConfig.PROCESSED_DIR, CLEAN_FILE)
    META_FILE = RAW_DATA_FILE.replace(".csv", "_metadata.json")
    METADATA_PATH = os.path.join(PathConfig.METADATA_DIR, META_FILE)

class RAGThresholdConfig:
    """
    所有记分卡 RAG 阈值在这里集中配置。
    后面 grc_translator.py 会从这里读取，不要在那边写死常数。
    """

    # 1) Quality / Fidelity（越高越好）
    # JSD：接近 1 表示分布很像
    QUALITY_JSD = {"green": 0.90, "amber": 0.80}

    # NMI：相关性越高越好
    QUALITY_NMI = {"green": 0.80, "amber": 0.60}

    # 2) Utility（越高越好）
    # TSTR F1，可以按任务难度调整
    UTILITY_TSTR_F1 = {"green": 0.76, "amber": 0.70}

    # 3) Risk（越低越好）
    # MIA AUC：越高隐私风险越大
    PRIVACY_MIA = {"green": 0.55, "amber": 0.65}

    # Fairness 平均差：越接近 0 越公平
    FAIRNESS = {"green": 0.10, "amber": 0.20}

    # 4) Sustainability（越低越好）
    # 单次实验的 CO2（kg）
    # < 0.005 kg 视为 Green；0.005–0.05 kg Amber；>0.05 kg Red
    SUSTAIN_CO2 = {"green": 0.005, "amber": 0.05}

    # 如果所有模型的 CO2 都低于这个阈值，认为整体排放“接近 0”，
    # 不再在它们之间分红黄绿（后面统一标 Green）。
    SUSTAIN_CO2_NEAR_ZERO = 1e-3  # 0.001 kg = 1 g

    # 训练时间（秒）阈值示例：<60s Green，60–300s Amber，>300s Red
    SUSTAIN_TIME = {"green": 60.0, "amber": 300.0}



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




# --- 4. GRC 记分卡配置 (GRC Scorecard Configuration) ---
# 定义记分卡的结构、指标和 RAG (红/黄/绿) 阈值
# Defines the scorecard structure, metrics, and RAG thresholds
class GRCConfig:
    # 定义记分卡中值的顺序 (英文)
    # Defines the order of values in the scorecard (English)
    METRIC_ORDER = ['Score', 'RAG']

    # RAG 阈值定义 / RAG Threshold Definitions
    # JSD 分数 (1-JSD) 越高越好
    # JSD Score (1-JSD) is "higher is better"
    THRESHOLDS = {
        'JSD': {'green': 0.9, 'amber': 0.8},  # 越高越好 / Higher-is-better
        'NMI': {'green': 0.8, 'amber': 0.6},  # 越高越好 / Higher-is-better
        'TSTR': {'green': 0.76, 'amber': 0.70},  # 越高越好 / Higher-is-better
        'MIA': {'green': 0.55, 'amber': 0.65},  # 越低越好 / Lower-is-better
        'FAIR': {'green': 0.1, 'amber': 0.2}  # 越低越好 / Lower-is-better
    }

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

    # [!!] 新增: 用于可视化的 RAG 颜色 / New: RAG colors for visualization
    RAG_COLORS = {
        'Green': '#90EE90',  # 浅绿色 / Light Green
        'Amber': '#FFBF00',  # 琥珀色 / Amber
        'Red': '#F08080',  # 浅红色 / Light Red
        'N/A': '#D3D3D3'  # 灰色 / Gray
    }

    # [!!] 新增: 用于图例和标题的英文文本 / New: English text for legend and title
    LEGEND_LABELS = {
        'Green': 'Good / Low-Risk / Best-in-Class',
        'Amber': 'Warning / Requires Review',
        'Red': 'Bad / High-Risk / Worst-in-Class',
        'N/A': 'N/A (e.g., Metric Failed)'
    }

    IMG_TITLE = 'GRC Quality & Risk Scorecard: Synthetic Data Model Benchmark'
    IMG_SUBTITLE = 'Comparative Analysis of Generative Models\n(Cells show "Score", Color shows "RAG" Risk Rating)'