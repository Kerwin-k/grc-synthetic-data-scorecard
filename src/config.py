#
# 文件名: src/config.py
# File: src/config.py
#
# 论文题目: Auditing Synthetic Data in Resource-Constrained Environments
# Thesis Title: Auditing Synthetic Data in Resource-Constrained Environments
# 框架名称: SynTab-GRC (Synthetic Tabular Data Governance, Risk and Compliance)
# Framework: SynTab-GRC
#
# 架构师说明 (Architect's Note):
# 这是整个框架的唯一控制面板，用于编排论文 Phase I 的计算实验。
# This is the SINGLE CONTROL PANEL orchestrating the Phase I computational experiments.
#

import os
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
    对应论文 Section 3.3.1: 为了确保实验在不同机器上的可比性，固定 ISO 代码。
    Ref Thesis Sec 3.3.1: Fixed ISO code ensures comparability across machines.
    """
    # 使用 'CAN' (加拿大) 作为基准，因为其能源结构相对稳定
    # Using 'CAN' (Canada) as baseline for stable energy mix
    FIXED_COUNTRY_ISO: str | None = 'CAN'
    FALLBACK_COUNTRY_LABEL: str = "Canada"

# --- 数据集配置 (Dataset Configuration) ---
class DatasetConfig:
    """
    数据集参数配置。
    支持论文 Chapter 4 中的两个核心实验：Adult 和 Home Credit。
    Supports the two core experiments in Thesis Chapter 4: Adult and Home Credit.
    """

    # =========================================================
    # 实验 A: Adult Census Dataset (基线实验 / Baseline)
    # 对应论文 Section 4.2.2 / Ref Thesis Sec 4.2.2
    # 状态: 已注释 (默认不激活) / Status: Commented out (Default inactive)
    # =========================================================
    # RAW_DATA_FILE = "adult.csv"
    # ID_COLUMN = None
    # TARGET_COLUMN = "income"           # 目标列 / Target column
    # POSITIVE_LABEL = ">50K"            # 正类标签 / Positive label
    # SENSITIVE_FEATURES = ["sex", "race"] # 公平性分析的敏感属性 / Sensitive attributes
    # COLS_TO_DROP = ["fnlwgt", "education"] # 删除 fnlwgt 以符合数据最小化原则 / Drop identifiers

    # =========================================================
    # 实验 B: Home_Credit Default Risk (复杂场景 / Complex Scenario)
    # 对应论文 Section 4.2.3 / Ref Thesis Sec 4.2.3
    # 状态: 当前激活 / Status: Currently Active
    # =========================================================
    RAW_DATA_FILE = "application_train.csv"
    ID_COLUMN = "SK_ID_CURR"
    TARGET_COLUMN = "TARGET"
    POSITIVE_LABEL = "1"
    SENSITIVE_FEATURES = ["CODE_GENDER", "DAYS_BIRTH"]

    # 复杂的列删除逻辑 (针对 Home Credit 数据集的特定预处理)
    # 对应论文 Section 3.3.1 & 4.2.1: 仅保留关键特征以模拟资源受限环境。
    # Ref Thesis Sec 3.3.1: Resource-aware pre-processing.
    COLS_TO_DROP = [
        "ORGANIZATION_TYPE",  # 类别过多 (58类)，One-Hot 后内存消耗巨大 / High cardinality (58 classes), heavy memory usage
        "OCCUPATION_TYPE",    # 类别多 / High cardinality
        "FONDKAPREMONT_MODE", # 缺失值多且复杂 / High missing values & complexity
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
        "WEEKDAY_APPR_PROCESS_START", # 对信贷预测贡献低 / Low predictive power
        "HOUR_APPR_PROCESS_START",
        "OWN_CAR_AGE"         # 缺失值极多 / Extremely high missing rate
    ]

    # --- 通用采样配置 (General Sampling Config) ---

    # 采样大小: 50,000 行
    # 对应论文 Section 3.3.1: "Fixed low-resource budget (50,000 rows)"
    # 旨在模拟资源受限的企业环境
    # Simulating resource-constrained enterprise environments
    SAMPLE_SIZE: int | None = 50_000

    # 是否分层抽样: True
    # 对应论文 Section 3.3.1: 保持原始类分布
    # Ref Thesis Sec 3.3.1: Preserving marginal class distribution
    STRATIFY_BY_TARGET: bool = True

    # 随机种子，确保可复现性 / Random state for reproducibility
    RANDOM_STATE: int = 42

    # 采样模式: 'fixed' (固定预算), 'full' (全量), 'auto' (动态)
    # Sampling Mode: 'fixed' (budget), 'full', 'auto'
    SAMPLING_MODE: str = "fixed"

    @staticmethod
    def get_dynamic_sample_size():
        """
        实施资源感知型启发式评估。
        Implements Resource-Aware Heuristic Assessment.
        """
        if psutil is None:
            return 50_000

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

    # 自动路径生成 (请勿修改) / Auto-generated paths (Do not edit)
    RAW_PATH = os.path.join(PathConfig.RAW_DIR, RAW_DATA_FILE)
    CLEAN_FILE = RAW_DATA_FILE.replace(".csv", "_clean.csv")
    PROCESSED_PATH = os.path.join(PathConfig.PROCESSED_DIR, CLEAN_FILE)
    META_FILE = RAW_DATA_FILE.replace(".csv", "_metadata.json")
    METADATA_PATH = os.path.join(PathConfig.METADATA_DIR, META_FILE)

# --- 资源配置 (Resource Configuration) ---
class ResourceConfig:
    """
    防止内存溢出 (OOM) 的限制配置。
    Constraints to prevent Out-Of-Memory (OOM).
    Ref Thesis Sec 3.3.1: Resource guardrails.
    """
    # 训练数据行数上限 / Max training rows per model
    MAX_TRAIN_ROWS_PER_MODEL: int | None = 50_000

    # 评估阶段行数上限 / Max rows for evaluation
    MAX_ROWS_TSTR: int | None = 50_000
    MAX_ROWS_MIA: int | None = 50_000

    # 启用浮点数降精度 (float64 -> float32)
    # Ref Thesis Sec 3.3.1: "Automatic downcasting... to 32-bit precision"
    ENABLE_DTYPE_DOWNCAST: bool = True

    # 资源密集型模型 / Heavy models
    HEAVY_MODELS = ("CTGAN", "TVAE")

# --- RAG 阈值配置 (RAG Threshold Configuration) ---
class RAGThresholdConfig:
    """
    红黄绿 (Red/Amber/Green) 评分阈值配置。
    用于生成 GRC Scorecard。
    Ref Thesis Section 3.3.6 & 4.2.3.
    """

    # 1) 质量 (越高越好) / Quality (Higher is better)
    QUALITY_JSD = {"green": 0.90, "amber": 0.80}
    QUALITY_NMI = {"green": 0.80, "amber": 0.60}

    # 2) 效用 (越高越好) / Utility (Higher is better)
    # 注意：此阈值针对 Home Credit 数据集进行了校准。
    # Note: These thresholds are calibrated for Home Credit as per Thesis Section 3.3.6.
    # UTILITY_TSTR_F1 = {"green": 0.76, "amber": 0.70} # (Adult Benchmark)
    UTILITY_TSTR_F1 = {"green": 0.40, "amber": 0.20}  # (Home Credit Benchmark)

    # 3) 风险 (越低越好) / Risk (Lower is better)
    # PRIVACY_MIA = {"green": 0.55, "amber": 0.65} # (Adult Benchmark)
    # FAIRNESS = {"green": 0.10, "amber": 0.20}   # (Adult Benchmark)

    # Home Credit Dataset Settings
    # 对应论文 Figure 4.2 中的风险评级逻辑
    # Corresponds to risk rating logic in Thesis Figure 4.2
    PRIVACY_MIA = {"green": 0.55, "amber": 0.65}
    FAIRNESS = {"green": 0.10, "amber": 0.20}

    # 4) 可持续性 (越低越好) / Sustainability (Lower is better)
    # 单位: kg CO2
    SUSTAIN_CO2 = {"green": 0.005, "amber": 0.05}
    SUSTAIN_CO2_NEAR_ZERO = 1e-3  # 忽略极小排放的阈值 / Threshold to ignore negligible emissions

    # 训练时间阈值 / Training time thresholds (Seconds)
    # 对应论文 Table 4.2: CTGAN (4436s) 为红色, Gaussian (190s) 为琥珀色/绿色区间
    SUSTAIN_TIME = {"green": 60.0, "amber": 600.0}

# --- 模型配置 (Model Configuration) ---
MODELS_CONFIG = {
    # 统计基线模型 / Statistical Baseline
    "GaussianCopula": {
        "class": GaussianCopulaSynthesizer,
        "params": {}
    },
    # 深度学习模型 (GAN) / Deep Learning (GAN)
    "CTGAN": {
        "class": CTGANSynthesizer,
        "params": {"epochs": 100, "verbose": True}
    },
    # 深度学习模型 (VAE) / Deep Learning (VAE)
    "TVAE": {
        "class": TVAESynthesizer,
        "params": {"epochs": 100, "verbose": True}
    }
}

# --- GRC 可视化配置 (GRC Visualization Config) ---
class GRCConfig:
    RAG_COLORS = {
        'Green': '#90EE90', # 低风险 / Low Risk / Good
        'Amber': '#FFBF00', # 中等风险 / Medium Risk / Review Required
        'Red': '#F08080',   # 高风险 / High Risk / Critical
        'N/A': '#D3D3D3'    # 不适用 / Not Applicable
    }