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
from src.config import PathConfig, DatasetConfig

# 配置日志 / Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - INFO - %(message)s')

# 自定义 JSON 编码器以处理 NaN / Custom JSON encoder to handle NaN values
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
    编排完整的 5 步评估流程。
    Orchestrate the complete 5-step evaluation process.

    步骤 / Steps:
    1. 数据摄入与清洗 (Ingestion & Cleaning): 加载数据，预处理，生成元数据。
    2. 模型训练与生成 (Training & Generation): 训练生成模型并追踪碳排放 (Dimension 5)。
    3. 定量评估 (Quantitative Eval): 计算质量、效用、隐私和公平性指标 (Dimensions 1-4)。
    4. 定性转化 (Qualitative Transformation): 将原始指标转换为 GRC RAG 记分卡。
    5. 报告可视化 (Visualization): 生成最终的记分卡图像报告。
    """

    logging.info("==============================================")
    logging.info("  Starting: Multi-Dimensional Evaluation Pipeline ")
    logging.info("==============================================")

    # --- 步骤 1: 摄入与准备 / Step 1: Ingest & Prepare ---
    logging.info("  Step 1: Ingesting and processing data...")
    real_data = load_and_clean_data()
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

    # --- 步骤 3: 定量评估 / Step 3: Quantitative Evaluation ---
    logging.info("  Step 3: Starting 5-Dimension metrics engine (Dimensions 1, 4)...")

    # 加载所有生成的合成数据集 / Load all generated synthetic datasets
    synthetic_data_map = {}
    for name in MODELS_CONFIG.keys():
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
        metadata.to_dict(),
        sustainability_report
    )

    # 保存原始定量指标 / Save raw quantitative metrics
    os.makedirs(os.path.dirname(PathConfig.METRICS_REPORT_PATH), exist_ok=True)
    with open(PathConfig.METRICS_REPORT_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=4, cls=NpEncoder)
    logging.info(f"  Quantitative metrics report saved to {PathConfig.METRICS_REPORT_PATH}")

    # --- 步骤 4: 定性转化 (GRC) / Step 4: Qualitative Transformation (GRC) ---
    logging.info("  Step 4: Transforming metrics into GRC 'Quality & Risk Scorecard'...")
    scorecard_df = create_grc_scorecard(all_metrics, MODELS_CONFIG)

    # 保存 CSV 格式记分卡 / Save Scorecard as CSV
    scorecard_df.to_csv(PathConfig.GRC_SCORECARD_CSV_PATH)
    logging.info(f"  GRC Scorecard (.csv) saved to {PathConfig.GRC_SCORECARD_CSV_PATH}")

    # --- 步骤 5: 报告可视化 / Step 5: Visualization & Reporting ---
    try:
        save_scorecard_as_image(scorecard_df, PathConfig.GRC_SCORECARD_IMG_PATH)
        logging.info(f"  GRC Scorecard Image (.png) saved to {PathConfig.GRC_SCORECARD_IMG_PATH}")
    except Exception as e:
        logging.error(f"Failed to save GRC Scorecard image: {e}", exc_info=True)

    # --- 完成 / Complete ---
    logging.info("  Pipeline successfully completed.")
    logging.info("==============================================")
    print("\n--- GRC Scorecard (Preview) ---")
    print(scorecard_df)
    print("---------------------------------")


if __name__ == "__main__":
    main()