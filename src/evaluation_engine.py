import logging
import warnings

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sdmetrics.reports.single_table import QualityReport
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DatasetConfig, ResourceConfig

# 忽略常见 ML 库警告 / Suppress common ML library warnings
warnings.filterwarnings("ignore")


def enforce_schema(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    """
    全局类型强制与模式对齐。
    Global Type Enforcement + Schema Alignment.

    确保 synth_df 与 real_df 在列名、顺序和数据类型上一致：
    1. 重新索引以对齐列顺序 / Reindex to align columns
    2. 强制数值列转换 / Coerce numeric conversion
    3. 强制分类列为字符串 / Force categorical columns to string
    """
    synth = synth_df.copy()
    synth = synth.reindex(columns=real_df.columns)

    for col in real_df.columns:
        real_dtype = real_df[col].dtype

        if pd.api.types.is_numeric_dtype(real_dtype):
            synth[col] = pd.to_numeric(synth[col], errors="coerce")
        else:
            synth[col] = synth[col].astype(str)

    return synth


def _aligned_sample(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int | None,
    *,
    random_state: int = 42,
):
    """
    监督学习对齐采样：X 和 y 使用相同的索引进行采样。
    Aligned sampling for supervised learning: X and y sampled with same indices.
    """
    if max_rows is None or len(X) <= max_rows:
        return X, y

    idx = X.sample(n=max_rows, random_state=random_state).index
    return X.loc[idx], y.loc[idx]


def _create_ml_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    """
    创建 sklearn 预处理管道。
    Create sklearn preprocessing pipeline.

    - 数值: 中位数填充 + 标准化 / Numeric: Median Imputation + Scaling
    - 分类: 众数填充 + OneHot 编码 / Categorical: Mode Imputation + OneHot
    """
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    categorical_cols = categorical_cols.drop(DatasetConfig.TARGET_COLUMN, errors="ignore")

    numerical_cols = data.select_dtypes(include=np.number).columns
    numerical_cols = numerical_cols.drop(DatasetConfig.SENSITIVE_FEATURES, errors="ignore")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_quality_fup(real_data, synth_data, metadata_dict):
    """
    评估维度 1: 保真度、效用与隐私。
    Evaluates Dimension 1: Fidelity, Utility, and Privacy (FUP).
    """
    logging.info("... Evaluating Dimension 1: FUP...")
    metrics = {}
    rs = getattr(DatasetConfig, "RANDOM_STATE", 42)

    # --- 1.1 保真度 (JSD, NMI) / Fidelity ---
    logging.info("... Calculating Fidelity (JSD, NMI)...")
    try:
        quality_report = QualityReport()
        quality_report.generate(real_data, synth_data, metadata_dict)

        shape_details = quality_report.get_details(property_name="Column Shapes")
        metrics["fidelity_jsd_avg"] = shape_details["Score"].mean()

        pair_details = quality_report.get_details(property_name="Column Pair Trends")
        metrics["fidelity_nmi_avg"] = pair_details["Score"].mean()

    except Exception as e:
        logging.warning(f"SDMetrics report failed: {e}. Setting fidelity scores to NaN.")
        metrics["fidelity_jsd_avg"] = np.nan
        metrics["fidelity_nmi_avg"] = np.nan

    # --- 1.2 效用 (TSTR) / Utility ---
    logging.info("... Calculating Utility (TSTR)...")
    real_train = None

    try:
        # 1) 拆分真实数据 / Split real data
        real_train, real_test = train_test_split(
            real_data,
            test_size=0.3,
            random_state=rs,
        )

        # 2) 准备目标变量和特征 / Prepare Target & Features
        y_real_test = real_test[DatasetConfig.TARGET_COLUMN].apply(
            lambda value: 1 if value == DatasetConfig.POSITIVE_LABEL else 0
        )
        X_real_test = real_test.drop(columns=[DatasetConfig.TARGET_COLUMN])

        y_synth_train = synth_data[DatasetConfig.TARGET_COLUMN].apply(
            lambda value: 1 if value == DatasetConfig.POSITIVE_LABEL else 0
        )
        X_synth_train = synth_data.drop(columns=[DatasetConfig.TARGET_COLUMN])

        # 3) 下采样以提高效率 / Down-sample for efficiency
        max_rows_tstr = ResourceConfig.MAX_ROWS_TSTR
        X_real_test, y_real_test = _aligned_sample(X_real_test, y_real_test, max_rows_tstr, random_state=rs)
        X_synth_train, y_synth_train = _aligned_sample(X_synth_train, y_synth_train, max_rows_tstr, random_state=rs)

        # 4) 拟合预处理器 (在合成数据上) / Fit Preprocessor (on Synthetic)
        preprocessor = _create_ml_preprocessor(X_synth_train)
        X_synth_train_processed = preprocessor.fit_transform(X_synth_train)
        X_real_test_processed = preprocessor.transform(X_real_test)

        # 5) 训练和评估模型 / Train & Evaluate Model
        model = LogisticRegression(max_iter=1000, random_state=rs)
        model.fit(X_synth_train_processed, y_synth_train)
        y_pred_on_real = model.predict(X_real_test_processed)
        metrics["utility_tstr_f1"] = f1_score(y_real_test, y_pred_on_real)

        tstr_results = {
            "y_true": y_real_test,
            "y_pred": y_pred_on_real,
            "sensitive_features_df": X_real_test.reset_index(drop=True),
        }
    except Exception as e:
        logging.warning(f"TSTR utility calculation failed: {e}. Setting TSTR F1 to NaN.")
        metrics["utility_tstr_f1"] = np.nan
        tstr_results = {}

    # --- 1.3 隐私 (MIA) / Privacy ---
    logging.info("... Calculating Privacy (MIA)...")
    try:
        reference_real = real_train if real_train is not None else real_data
        n_synth = len(synth_data)
        max_rows_mia = ResourceConfig.MAX_ROWS_MIA
        if max_rows_mia is not None and n_synth > max_rows_mia:
            n_synth = max_rows_mia

        # 创建 MIA 数据集 / Create MIA dataset
        real_subset = reference_real.sample(n=n_synth, replace=True, random_state=rs)
        real_subset = real_subset.copy()
        real_subset["is_real"] = 1
        synth_data_copy = synth_data.copy()
        synth_data_copy["is_real"] = 0

        mia_data = pd.concat([real_subset, synth_data_copy], ignore_index=True)
        y_mia = mia_data["is_real"]
        X_mia = mia_data.drop(columns=["is_real"])

        X_mia_train, X_mia_test, y_mia_train, y_mia_test = train_test_split(
            X_mia, y_mia, test_size=0.3, random_state=rs, stratify=y_mia,
        )

        mia_preprocessor = _create_ml_preprocessor(X_mia_train)
        X_mia_train_processed = mia_preprocessor.fit_transform(X_mia_train)
        X_mia_test_processed = mia_preprocessor.transform(X_mia_test)

        mia_model = LogisticRegression(max_iter=1000, random_state=rs)
        mia_model.fit(X_mia_train_processed, y_mia_train)

        y_mia_pred_proba = mia_model.predict_proba(X_mia_test_processed)[:, 1]
        metrics["privacy_mia_auc"] = roc_auc_score(y_mia_test, y_mia_pred_proba)

    except Exception as e:
        logging.warning(f"MIA privacy calculation failed: {e}. Setting MIA AUC to NaN.")
        metrics["privacy_mia_auc"] = np.nan

    return metrics, tstr_results


def evaluate_fairness(tstr_results):
    """
    评估维度 4: 算法公平性。
    Evaluates Dimension 4: Algorithmic Fairness.
    """
    logging.info("... Evaluating Dimension 4: Fairness...")
    metrics = {}

    if not tstr_results:
        logging.warning("Skipping fairness evaluation as TSTR results are missing.")
        for feature in DatasetConfig.SENSITIVE_FEATURES:
            metrics[f"fairness_dp_diff_{feature}"] = np.nan
            metrics[f"fairness_eo_diff_{feature}"] = np.nan
        return metrics

    try:
        y_true = tstr_results["y_true"]
        y_pred = tstr_results["y_pred"]

        for feature in DatasetConfig.SENSITIVE_FEATURES:
            sensitive_df = tstr_results["sensitive_features_df"]
            if feature not in sensitive_df.columns:
                logging.warning("Sensitive feature '%s' not found in evaluation data.", feature)
                metrics[f"fairness_dp_diff_{feature}"] = np.nan
                metrics[f"fairness_eo_diff_{feature}"] = np.nan
                continue

            sf_vector = sensitive_df[feature]

            dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sf_vector)
            eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sf_vector)

            metrics[f"fairness_dp_diff_{feature}"] = dpd
            metrics[f"fairness_eo_diff_{feature}"] = eod

    except Exception as e:
        logging.error(f"Error evaluating fairness: {e}")
        for feature in DatasetConfig.SENSITIVE_FEATURES:
            metrics[f"fairness_dp_diff_{feature}"] = np.nan
            metrics[f"fairness_eo_diff_{feature}"] = np.nan

    return metrics


def run_evaluation_pipeline(real_data, synthetic_data_map, metadata_dict, sustainability_report):
    """
    完整的定量评估编排器。
    Complete quantitative evaluation orchestrator.
    """
    all_metrics = {}

    # 基础类型标准化 / Basic type normalization
    real_base = real_data.copy()
    for col in real_base.select_dtypes(include=["object", "category"]).columns:
        real_base[col] = real_base[col].astype(str).str.strip()

    for model_name, synth_data in synthetic_data_map.items():
        logging.info(f"--- Quantitative Evaluation: {model_name} ---")

        # 1) 模式强制对齐 / Schema Enforcement
        aligned_synth = enforce_schema(real_base, synth_data)
        model_metrics = {}

        # 2) 运行 FUP (质量、效用、隐私) / Run FUP (Quality, Utility, Privacy)
        fup_metrics, tstr_results = evaluate_quality_fup(real_base, aligned_synth, metadata_dict)
        model_metrics.update(fup_metrics)

        # 3) 运行公平性 / Run Fairness
        fairness_metrics = evaluate_fairness(tstr_results)
        model_metrics.update(fairness_metrics)

        # 4) 合并可持续性 / Merge Sustainability
        if model_name in sustainability_report:
            model_metrics.update(sustainability_report[model_name])
        else:
            logging.warning(f"Sustainability report not found for {model_name}.")

        all_metrics[model_name] = model_metrics

    logging.info("Quantitative evaluation pipeline complete.")
    return all_metrics