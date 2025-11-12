import pandas as pd
from sdv.metadata import SingleTableMetadata
from codecarbon import EmissionsTracker
import os
import logging
import time
# [!!] 新增: 从 config.py 导入配置
# [!!] New: Import configurations from config.py
from src.config import DatasetConfig, PathConfig, MODELS_CONFIG

# [!!] 修正: 导入正确的 SDV 1.0+ 合成器类名
# [!!] Fix: Import the correct SDV 1.0+ synthesizer class names
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer


def train_and_generate(real_data, metadata):
    """
    循环遍历模型，在追踪排放的同时训练它们，并生成合成数据。
    Loops through models, trains them while tracking emissions, and generates synthetic data.
    """
    logging.info("Starting model training and generation process...")

    # 确保所有输出目录都存在 / Ensure all output directories exist
    os.makedirs(PathConfig.MODELS_DIR, exist_ok=True)
    os.makedirs(PathConfig.SYNTH_DIR, exist_ok=True)
    os.makedirs(PathConfig.EMISSIONS_DIR, exist_ok=True)

    sustainability_report = {}
    num_rows = len(real_data)  # 获取要生成的行数 / Get number of rows to generate

    for name, config in MODELS_CONFIG.items():
        logging.info(f"--- Processing Model: {name} ---")

        # 1. 初始化模型 / Initialize model
        model_class = config['class']
        model_params = config['params']
        model = model_class(metadata, **model_params)

        # 2. 配置可持续性追踪器 (维度 5) / Configure sustainability tracker (Dimension 5)
        tracker = EmissionsTracker(
            project_name=f"thesis-sdg-{name}",
            output_dir=PathConfig.EMISSIONS_DIR,
            output_file=f"{name}_emissions.csv"
        )

        # 3. 训练模型并追踪排放 / Train model and track emissions
        logging.info(f"Starting training for {name}...")
        tracker.start()
        start_time = time.time()

        try:
            model.fit(real_data)
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
            tracker.stop()
            continue

        end_time = time.time()
        # 停止追踪器并检索排放数据 / Stop tracker and retrieve emissions data
        emissions_kg = tracker.stop()

        # 从 tracker 实例中获取详细的能耗数据 / Get detailed energy data from tracker
        energy_kwh = tracker.final_emissions_data.energy_consumed if tracker.final_emissions_data else 0.0

        training_duration_sec = end_time - start_time

        logging.info(f"Emissions for {name}: {emissions_kg:.6f} kgCO2eq")
        logging.info(f"{name} training complete. Duration: {training_duration_sec:.2f}s")

        # 4. 存储可持续性报告 / Store sustainability report
        sustainability_report[name] = {
            "training_time_sec": training_duration_sec,
            "energy_kwh": energy_kwh,
            "co2_eq_kg": emissions_kg
        }

        # 5. 保存训练好的模型 / Save trained model
        model_path = os.path.join(PathConfig.MODELS_DIR, f"{name.lower()}.pkl")
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")

        # 6. 生成并保存合成数据 / Generate and save synthetic data
        logging.info(f"Generating {num_rows} synthetic samples for {name}...")
        synthetic_data = model.sample(num_rows=num_rows)
        synth_path = os.path.join(PathConfig.SYNTH_DIR, f"synth_{name.lower()}.csv")
        synthetic_data.to_csv(synth_path, index=False)
        logging.info(f"Synthetic data saved to {synth_path}")

    return sustainability_report


if __name__ == "__main__":
    # 允许此脚本直接运行以进行设置 / Allow this script to be run directly for setup
    logging.info("Loading data for model training (Standalone)...")
    try:
        data = pd.read_csv(DatasetConfig.PROCESSED_PATH)
        metadata = SingleTableMetadata.load_from_json(DatasetConfig.METADATA_PATH)
        num_to_generate = len(data)  # 生成 1:1 匹配 / Generate 1:1 match

        report = train_and_generate(data, metadata, num_to_generate)

        logging.info("--- Sustainability Report Summary ---")
        for model, metrics in report.items():
            logging.info(f"{model}: Time={metrics['training_time_sec']:.2f}s, CO2={metrics['co2_eq_kg']:.8f}kg")

    except FileNotFoundError:
        logging.error("Processed data or metadata not found. Please run data_loader.py first.")
    except Exception as e:
        logging.error(f"Standalone model training failed: {e}")