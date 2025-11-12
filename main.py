import pandas as pd
from sdv.metadata import SingleTableMetadata
from src.data_loader import load_and_clean_data, generate_and_save_metadata
from src.model_trainer import train_and_generate, MODELS_CONFIG
from src.evaluation_engine import run_evaluation_pipeline
from src.grc_translator import create_grc_scorecard, save_scorecard_as_image
import json
import logging
import os
import numpy as np
from src.config import PathConfig, DatasetConfig # 导入配置 / Import configs

# 配置日志 (英文) / Configure logging (English)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - INFO - %(message)s')

# [!!] 新增: 自定义 JSON 编码器以处理 np.nan，防止 JSON 保存失败
# [!!] New: Custom JSON encoder to handle np.nan, preventing JSON save failure
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    """
    编排完整的 5 步流程 / Orchestrate the complete 5-step process:
    1. 加载/清理 / Load/Clean
    2. 训练/生成 (维度 5) / Train/Generate (Dimension 5)
    3. 评估 (维度 1, 4) / Evaluate (Dimensions 1, 4)
    4. 转化 (GRC 记分卡) / Transform (GRC Scorecard)
    5. 报告 / Report
    """

    logging.info("==============================================")
    logging.info("  Starting: Multi-Dimensional Evaluation Pipeline ")
    logging.info("==============================================")

    # --- 步骤 1: 摄入与准备 / Step 1: Ingest & Prepare ---
    logging.info("  Step 1: Ingesting and processing data...")
    # [!!] 修正: 使用 Config 文件中的路径 / FIX: Use paths from Config file
    real_data = load_and_clean_data(DatasetConfig.RAW_PATH, DatasetConfig.PROCESSED_PATH)
    if real_data is None:
        logging.error("Data loading failed. Exiting.")
        return

    metadata = generate_and_save_metadata(real_data, DatasetConfig.METADATA_PATH)
    num_rows_to_generate = len(real_data)
    logging.info(f"  Data preparation complete. Loaded {num_rows_to_generate} rows.")

    # --- 步骤 2: 生成与可持续性追踪 / Step 2: Generate & Track Sustainability ---
    logging.info("  Step 2: Training models and tracking sustainability (Dimension 5)...")
    sustainability_report = train_and_generate(real_data, metadata, num_rows_to_generate, MODELS_CONFIG)
    logging.info("  Model training and data generation complete.")

    # --- 步骤 3: 评估 (定量引擎) / Step 3: Evaluate (Quantitative Engine) ---
    logging.info("  Step 3: Starting 5-Dimension metrics engine (Dimensions 1, 4)...")

    # 加载所有生成的数据集进行评估 / Load all generated datasets for evaluation
    synthetic_data_map = {}
    for name in MODELS_CONFIG.keys():
        # [!!] 修正: 使用 Config 文件中的路径 / FIX: Use paths from Config file
        synth_path = os.path.join(PathConfig.SYNTH_DIR, f"synth_{name.lower()}.csv")
        try:
            synthetic_data_map[name] = pd.read_csv(synth_path)
        except FileNotFoundError:
            logging.warning(f"Synthetic data for {name} not found. Skipping.")
        except pd.errors.EmptyDataError:
            logging.warning(f"Synthetic data for {name} is empty. Skipping.")

    if not synthetic_data_map:
        logging.error("No synthetic datasets found or loaded. Evaluation cannot continue. Exiting.")
        return

    # 运行完整的评估流程 / Run the full evaluation pipeline
    all_metrics = run_evaluation_pipeline(
        real_data,
        synthetic_data_map,
        metadata.to_dict(), # 传递元数据字典 / Pass the metadata dictionary
        sustainability_report
    )

    # 保存原始定量指标 / Save raw quantitative metrics
    # [!!] 修正: 使用 Config 文件中的路径 / FIX: Use paths from Config file
    os.makedirs(os.path.dirname(PathConfig.METRICS_REPORT_PATH), exist_ok=True)
    with open(PathConfig.METRICS_REPORT_PATH, 'w') as f:
        # [!!] 修正: 使用自定义编码器处理 NaN / Fix: Use custom encoder for NaN
        json.dump(all_metrics, f, indent=4, cls=NpEncoder)
    logging.info(f"  Quantitative metrics report saved to {PathConfig.METRICS_REPORT_PATH}")

    # --- 步骤 4: 转化 (定性引擎) / Step 4: Transform (Qualitative Engine) ---
    logging.info("  Step 4: Transforming metrics into GRC 'Quality & Risk Scorecard'...")
    scorecard_df = create_grc_scorecard(all_metrics, MODELS_CONFIG)

    # 保存最终的 GRC 记分卡 / Save the final GRC Scorecard
    # [!!] 修正: 使用 Config 文件中的路径 / FIX: Use paths from Config file
    scorecard_df.to_csv(PathConfig.GRC_SCORECARD_CSV_PATH)
    logging.info(f"  GRC Scorecard (.csv) saved to {PathConfig.GRC_SCORECARD_CSV_PATH}")

    # --- 步骤 5: 报告 (可视化) / Step 5: Report (Visualize) ---
    # [!!] 新增: 保存 GRC 记分卡图像 (.png)
    # [!!] New: Save GRC Scorecard Image (.png)
    try:
        # [!!] 修正: 使用 Config 文件中的路径 / FIX: Use paths from Config file
        save_scorecard_as_image(scorecard_df, PathConfig.GRC_SCORECARD_IMG_PATH)
        logging.info(f"  GRC Scorecard Image (.png) saved to {PathConfig.GRC_SCORECARD_IMG_PATH}")
    except Exception as e:
        logging.error(f"Failed to save GRC Scorecard image: {e}", exc_info=True)

    # --- 完成 / Complete ---
    logging.info("  Pipeline successfully completed.")
    logging.info("==============================================")
    # [!!] 修正: 打印英文预览 / Fix: Print English preview
    print("\n--- GRC Scorecard (Preview) ---")
    print(scorecard_df)
    print("---------------------------------")


if __name__ == "__main__":
    main()