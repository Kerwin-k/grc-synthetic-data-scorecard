import os
import time
import logging

import pandas as pd
from sdv.metadata import SingleTableMetadata
from codecarbon import EmissionsTracker

# 从 config.py 导入配置 / Import configurations from config.py
from src.config import DatasetConfig, PathConfig, MODELS_CONFIG, SustainabilityConfig

# 正确的 SDV 1.x 合成器类 / Correct SDV 1.x synthesizer classes
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)


def _create_emissions_tracker_for_model(model_name: str):
    """
    尽量探测 GPU 总能耗 API 是否可用，用来标记 coverage，
    但不再往 EmissionsTracker 里传 measure_gpu（你这版 codecarbon 已经不支持这个参数）。
    """
    measure_gpu = False
    gpu_note = "GPU energy not included (CPU + RAM only)."

    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                # 如果这个 API 可用，说明 GPU 总能耗是可以测到的
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                measure_gpu = True
                gpu_note = "GPU energy included via NVML total energy API."
            except pynvml.NVMLError:
                measure_gpu = False
                gpu_note = (
                    "GPU detected but NVML total energy API not supported; "
                    "GPU energy will NOT be included (CPU + RAM only)."
                )

        pynvml.nvmlShutdown()
    except Exception as e:
        measure_gpu = False
        gpu_note = (
            "No usable GPU NVML interface; GPU energy will NOT be included "
            f"(CPU + RAM only). Detail: {type(e).__name__}"
        )

    logging.info(f"[Sustainability/{model_name}] {gpu_note}")

    # --- 组装 EmissionsTracker 参数，加入国家配置 ---
    tracker_kwargs = {
        "project_name": f"thesis-sdg-{model_name}",
        "output_dir": PathConfig.EMISSIONS_DIR,
        "output_file": f"{model_name}_emissions.csv",
    }

    if SustainabilityConfig.FIXED_COUNTRY_ISO:
        # 显式指定国家：不再调用地理定位 API，完全可复现
        tracker_kwargs["country_iso_code"] = SustainabilityConfig.FIXED_COUNTRY_ISO
        logging.info(
            f"[Sustainability/{model_name}] Using fixed country_iso_code = "
            f"{SustainabilityConfig.FIXED_COUNTRY_ISO} from config.py."
        )
    else:
        logging.info(
            f"[Sustainability/{model_name}] Using CodeCarbon automatic "
            f"geolocation; if lookup fails, CodeCarbon falls back to "
            f"'{SustainabilityConfig.FALLBACK_COUNTRY_LABEL}'."
        )

    tracker = EmissionsTracker(**tracker_kwargs)
    return tracker, measure_gpu, gpu_note



def train_and_generate(
    real_data: pd.DataFrame,
    metadata: SingleTableMetadata,
    num_rows_to_generate: int | None = None,
    models_config: dict | None = None,
):
    """
    循环遍历模型，在追踪排放的同时训练它们，并生成合成数据。
    Loops through models, trains them while tracking emissions, and generates synthetic data.
    """
    logging.info("Starting model training and generation process...")

    # 确保所有输出目录都存在 / Ensure all output directories exist
    os.makedirs(PathConfig.MODELS_DIR, exist_ok=True)
    os.makedirs(PathConfig.SYNTH_DIR, exist_ok=True)
    os.makedirs(PathConfig.EMISSIONS_DIR, exist_ok=True)

    sustainability_report: dict[str, dict] = {}
    num_rows = num_rows_to_generate or len(real_data)
    models_to_use = models_config or MODELS_CONFIG

    for name, config in models_to_use.items():
        logging.info("=" * 80)
        logging.info(f"--- Processing Model: {name} ---")

        # 1. 初始化模型 / Initialize model
        model_class = config["class"]
        model_params = config["params"]
        model = model_class(metadata, **model_params)

        # 2. 配置可持续性追踪器 (维度 5) / Configure sustainability tracker (Dimension 5)
        tracker, gpu_included, gpu_note = _create_emissions_tracker_for_model(name)

        # 3. 训练模型并追踪排放 / Train model and track emissions
        logging.info(f"[{name}] Starting training with emissions tracking...")
        tracker.start()
        start_time = time.time()

        try:
            model.fit(real_data)
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
            # 尽量优雅停止 tracker / try to stop tracker gracefully
            try:
                tracker.stop()
            except Exception:
                pass
            continue

        end_time = time.time()

        # 3.1 停止追踪器并检索排放数据 / Stop tracker and retrieve emissions data
        try:
            emissions_kg = tracker.stop()  # 可能返回 float，也可能抛错
        except Exception as e:
            logging.error(f"Error stopping EmissionsTracker for {name}: {e}")
            emissions_kg = None

        # 3.2 从 tracker 实例中获取详细的能耗数据（可能为 None） / Get detailed energy data
        energy_kwh = None
        try:
            if getattr(tracker, "final_emissions_data", None) is not None:
                energy_kwh = tracker.final_emissions_data.energy_consumed
        except Exception as e:
            logging.warning(
                f"[{name}] Failed to retrieve total energy from tracker: {e}. "
                "Energy will be recorded as N/A."
            )
            energy_kwh = None

        training_duration_sec = end_time - start_time

        # --- 覆盖率分类：full / partial / none ---
        # 逻辑：
        #   - emissions_kg 和 energy_kwh 都有值：
        #       · 如果 gpu_included=True → 'full'
        #       · 否则 → 'partial'（只 CPU+RAM）
        #   - 任何一个为 None → 'none'
        if (emissions_kg is None) or (energy_kwh is None):
            coverage = "none"
        elif gpu_included:
            coverage = "full"
        else:
            coverage = "partial"

        logging.info(
            f"[{name}] Training done. "
            f"Time={training_duration_sec:.2f}s, "
            f"CO2={emissions_kg if emissions_kg is not None else 'N/A'} kgCO2eq, "
            f"Energy={energy_kwh if energy_kwh is not None else 'N/A'} kWh, "
            f"GPU_included={gpu_included}, "
            f"coverage={coverage}"
        )

        # 4. 存储可持续性报告 / Store sustainability report
        sustainability_report[name] = {
            "training_time_sec": training_duration_sec,
            "energy_kwh": energy_kwh,
            "co2_eq_kg": emissions_kg,
            "gpu_energy_included": gpu_included,
            "gpu_note": gpu_note,
            "sustainability_coverage": coverage,  # ⭐ 关键字段：full/partial/none
        }

        # 5. 保存训练好的模型 / Save trained model
        model_path = os.path.join(PathConfig.MODELS_DIR, f"{name.lower()}.pkl")
        try:
            model.save(model_path)
            logging.info(f"[{name}] Model saved to {model_path}")
        except Exception as e:
            logging.warning(f"[{name}] Failed to save model: {e}")

        # 6. 生成并保存合成数据 / Generate and save synthetic data
        logging.info(f"[{name}] Generating {num_rows} synthetic samples...")
        try:
            synthetic_data = model.sample(num_rows=num_rows)
            synth_path = os.path.join(PathConfig.SYNTH_DIR, f"synth_{name.lower()}.csv")
            synthetic_data.to_csv(synth_path, index=False)
            logging.info(f"[{name}] Synthetic data saved to {synth_path}")
        except Exception as e:
            logging.error(f"[{name}] Failed to generate or save synthetic data: {e}")

    logging.info("All models processed. Sustainability report ready.")
    return sustainability_report


if __name__ == "__main__":
    # 允许此脚本直接运行以进行设置 / Allow this script to be run directly for setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Loading data for model training (Standalone)...")
    try:
        data = pd.read_csv(DatasetConfig.PROCESSED_PATH)
        metadata = SingleTableMetadata.load_from_json(DatasetConfig.METADATA_PATH)
        num_to_generate = len(data)  # 生成 1:1 匹配 / Generate 1:1 match

        report = train_and_generate(data, metadata, num_rows_to_generate=num_to_generate)

        logging.info("--- Sustainability Report Summary ---")
        for model, metrics in report.items():
            logging.info(
                f"{model}: "
                f"Time={metrics['training_time_sec']:.2f}s, "
                f"CO2={metrics['co2_eq_kg'] if metrics['co2_eq_kg'] is not None else 'N/A'} kg, "
                f"Energy={metrics['energy_kwh'] if metrics['energy_kwh'] is not None else 'N/A'} kWh, "
                f"Coverage={metrics['sustainability_coverage']}"
            )

    except FileNotFoundError:
        logging.error("Processed data or metadata not found. Please run data_loader.py first.")
    except Exception as e:
        logging.error(f"Standalone model training failed: {e}")
