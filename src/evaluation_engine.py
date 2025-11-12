import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sdmetrics.reports.single_table import QualityReport
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
import warnings
import logging
import numpy as np

# 抑制来自 ML 库的常见警告 / Suppress common warnings from ML libraries
warnings.filterwarnings("ignore")

# 基于 Adult 数据集定义常量 / Define constants based on Adult dataset
TARGET_COLUMN = "income"
SENSITIVE_FEATURES = ['sex', 'race']  # 用于公平性评估 / For fairness evaluation


def _create_ml_preprocessor(data):
    """
    创建一个 scikit-learn ColumnTransformer 流程以确保一致的预处理。
    Creates a scikit-learn ColumnTransformer pipeline for consistent preprocessing.
    """
    # 从输入 dataframe 识别列类型 / Identify column types from input dataframe
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    # 从特征集中排除目标列 / Exclude target column from features
    categorical_cols = categorical_cols.drop(TARGET_COLUMN, errors='ignore')
    # [!!] 'race' 和 'sex' 是分类特征，应在此处保留以进行独热编码
    # [!!] 'race' and 'sex' are categorical and should be kept here for OHE

    numerical_cols = data.select_dtypes(include=np.number).columns
    # 确保敏感特征（如果它们是数字的话）不被缩放
    # Ensure sensitive features are not scaled if they are numeric
    numerical_cols = numerical_cols.drop(SENSITIVE_FEATURES, errors='ignore')

    # 创建预处理流程 / Create preprocessing pipelines
    numeric_transformer = Pipeline(steps= [('std_scaler',StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # 创建列转换器 / Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # 丢弃未指定的列 / Drop columns not specified
    )
    return preprocessor



def evaluate_quality_fup(real_data, synth_data, metadata_dict):
    """
    评估维度 1: 保真度、效用和隐私。
    Evaluates Dimension 1: Fidelity, Utility, and Privacy.
    """
    logging.info("... Evaluating Dimension 1: FUP...")
    metrics = {}

    # --- 1.1 保真度 (JSD, NMI) / Fidelity (JSD, NMI) ---
    logging.info("... Calculating Fidelity (JSD, NMI)...")
    try:
        quality_report = QualityReport()
        quality_report.generate(real_data, synth_data, metadata_dict)

        shape_details = quality_report.get_details(property_name='Column Shapes')
        metrics['fidelity_jsd_avg'] = shape_details['Score'].mean()

        pair_details = quality_report.get_details(property_name='Column Pair Trends')
        metrics['fidelity_nmi_avg'] = pair_details['Score'].mean()

    except Exception as e:
        logging.warning(f"SDMetrics report failed: {e}. Setting fidelity scores to NaN.")
        metrics['fidelity_jsd_avg'] = np.nan
        metrics['fidelity_nmi_avg'] = np.nan

    # --- 1.2 效用 (TSTR) / Utility (TSTR) ---
    logging.info("... Calculating Utility (TSTR)...")
    try:
        # 1. 将真实数据拆分为一个保留的测试集 / Split real data into a holdout test set
        real_train, real_test = train_test_split(real_data, test_size=0.3, random_state=42)

        # 2. 准备 ML 数据 / Prepare ML data
        y_real_test = real_test['income'].apply(lambda x: 1 if x == '>50K' else 0)
        X_real_test = real_test.drop(columns=['income'])

        y_synth_train = synth_data['income'].apply(lambda x: 1 if x == '>50K' else 0)
        X_synth_train = synth_data.drop(columns=['income'])

        # 3. 在 *合成* 训练数据上创建并拟合预处理器 / Create and fit preprocessor on *synthetic* train data
        preprocessor = _create_ml_preprocessor(X_synth_train)
        X_synth_train_processed = preprocessor.fit_transform(X_synth_train)

        # 4. 使用 *相同* 的预处理器转换 *真实* 测试数据 / Transform *real* test data with *same* preprocessor
        X_real_test_processed = preprocessor.transform(X_real_test)

        # 5. 在合成数据上训练模型 / Train model on synthetic data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_synth_train_processed, y_synth_train)

        # 6. 在真实数据上测试模型 / Test model on real data
        y_pred_on_real = model.predict(X_real_test_processed)
        metrics['utility_tstr_f1'] = f1_score(y_real_test, y_pred_on_real)

        # 7. 存储结果用于公平性评估 / Store results for fairness evaluation
        tstr_results = {
            'y_true': y_real_test,
            'y_pred': y_pred_on_real,
            'sensitive_features_df': X_real_test.reset_index(drop=True)
        }
    except Exception as e:
        logging.warning(f"TSTR utility calculation failed: {e}. Setting TSTR F1 to NaN.")
        metrics['utility_tstr_f1'] = np.nan
        tstr_results = {}  # 返回空字典 / Return empty dict

    # --- 1.3 隐私 (MIA) / Privacy (MIA) ---
    logging.info("... Calculating Privacy (MIA)...")
    try:
        # 我们使用 'real_train' 分割作为“真实”数据 / We use 'real_train' split as "real" data
        n_synth = len(synth_data)
        real_subset = real_train.sample(n=n_synth, replace=True, random_state=42)

        # 创建标签: 1 = 真实, 0 = 合成 / Create labels: 1 = Real, 0 = Synthetic
        real_subset['is_real'] = 1
        synth_data_copy = synth_data.copy()
        synth_data_copy['is_real'] = 0

        # 合并数据集 / Concatenate datasets
        mia_data = pd.concat([real_subset, synth_data_copy], ignore_index=True)

        # 准备 ML / Prepare for ML
        y_mia = mia_data['is_real']
        X_mia = mia_data.drop(columns=['is_real'])

        # 将 MIA 数据拆分为其自己的训练/测试集 / Split MIA data into its own train/test sets
        X_mia_train, X_mia_test, y_mia_train, y_mia_test = train_test_split(
            X_mia, y_mia, test_size=0.3, random_state=42, stratify=y_mia
        )

        # 创建并拟合预处理器 / Create and fit preprocessor
        mia_preprocessor = _create_ml_preprocessor(X_mia_train)
        X_mia_train_processed = mia_preprocessor.fit_transform(X_mia_train)
        X_mia_test_processed = mia_preprocessor.transform(X_mia_test)

        # 训练 MIA 分类器 / Train MIA classifier
        mia_model = LogisticRegression(max_iter=1000, random_state=42)
        mia_model.fit(X_mia_train_processed, y_mia_train)

        # 预测概率 / Predict probabilities
        y_mia_pred_proba = mia_model.predict_proba(X_mia_test_processed)[:, 1]

        # 0.5 = 完美隐私 (随机); 1.0 = 零隐私 (完美检测)
        # 0.5 = Perfect Privacy (random); 1.0 = Zero Privacy (perfect detection)
        metrics['privacy_mia_auc'] = roc_auc_score(y_mia_test, y_mia_pred_proba)

    except Exception as e:
        logging.warning(f"MIA privacy calculation failed: {e}. Setting MIA AUC to NaN.")
        metrics['privacy_mia_auc'] = np.nan

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
        for feature in SENSITIVE_FEATURES:
            metrics[f'fairness_dp_diff_{feature}'] = np.nan
            metrics[f'fairness_eo_diff_{feature}'] = np.nan
        return metrics

    try:
        y_true = tstr_results['y_true']
        y_pred = tstr_results['y_pred']

        for feature in SENSITIVE_FEATURES:
            sf_vector = tstr_results['sensitive_features_df'][feature]

            # 计算人口统计均等差异 (0 是完美的)
            # Calculate Demographic Parity Difference (0 is perfect)
            dpd = demographic_parity_difference(
                y_true,
                y_pred,
                sensitive_features=sf_vector
            )

            # 计算均等化赔率差异 (0 是完美的)
            # Calculate Equalized Odds Difference (0 is perfect)
            eod = equalized_odds_difference(
                y_true,
                y_pred,
                sensitive_features=sf_vector
            )

            metrics[f'fairness_dp_diff_{feature}'] = dpd
            metrics[f'fairness_eo_diff_{feature}'] = eod

    except Exception as e:
        logging.error(f"Error evaluating fairness: {e}")
        for feature in SENSITIVE_FEATURES:
            metrics[f'fairness_dp_diff_{feature}'] = np.nan
            metrics[f'fairness_eo_diff_{feature}'] = np.nan

    return metrics


def run_evaluation_pipeline(real_data, synthetic_data_map, metadata_dict, sustainability_report):
    """
    完整的定量评估编排器。
    Complete quantitative evaluation orchestrator.
    """
    all_metrics = {}
    for model_name, synth_data in synthetic_data_map.items():
        logging.info(f"--- Quantitative Evaluation: {model_name} ---")

        # 处理 dtypes / Handle dtypes
        for col in real_data.select_dtypes(include=['object', 'category']).columns:
            if col in synth_data.columns:
                synth_data[col] = synth_data[col].astype(str)
                real_data[col] = real_data[col].astype(str)

        model_metrics = {}

        # --- 运行维度 1 (FUP) / Run Dimension 1 (FUP) ---
        fup_metrics, tstr_results = evaluate_quality_fup(real_data, synth_data, metadata_dict)
        model_metrics.update(fup_metrics)

        # --- 运行维度 4 (公平性) / Run Dimension 4 (Fairness) ---
        fairness_metrics = evaluate_fairness(tstr_results)
        model_metrics.update(fairness_metrics)

        # --- 添加维度 5 (可持续性) / Add Dimension 5 (Sustainability) ---
        if model_name in sustainability_report:
            model_metrics.update(sustainability_report[model_name])
        else:
            logging.warning(f"Sustainability report not found for {model_name}.")

        all_metrics[model_name] = model_metrics

    logging.info("Quantitative evaluation pipeline complete.")
    return all_metrics