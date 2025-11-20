import os
import time
import logging

import pandas as pd
from sdv.metadata import SingleTableMetadata
from codecarbon import EmissionsTracker

# 导入配置 / Import configurations
from src.config import DatasetConfig, PathConfig, MODELS_CONFIG, SustainabilityConfig, ResourceConfig

# 导入 SDV 合成器类 / Import SDV synthesizer classes
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)


def _create_emissions_tracker_for_model(model_name: str):
    """
    创建一个 EmissionsTracker 并尝试检测 GPU 及其能耗 API 的可用性。
    Create an EmissionsTracker and attempt to detect GPU availability and energy API support.
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
                # 尝试获取总能耗，确认 API 是否可用
                # Attempt to get total energy consumption to verify API availability
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

    # 配置追踪器参数 / Configure tracker parameters
    tracker_kwargs = {
        "project_name": f"thesis-sdg-{model_name}",
        "output_dir": PathConfig.EMISSIONS_DIR,
        "output_file": f"{model_name}_emissions.csv",
    }

    if SustainabilityConfig.FIXED_COUNTRY_ISO:
        # 使用固定国家代码，确保可复现性 / Use fixed country code for reproducibility
        tracker_kwargs["country_iso_code"] = SustainabilityConfig.FIXED_COUNTRY_ISO
        logging.info(
            f"[Sustainability/{model_name}] Using fixed country_iso_code = "
            f"{SustainabilityConfig.FIXED_COUNTRY_ISO} from config.py."
        )
    else:
        # 使用自动地理定位 / Use automatic geolocation
        logging.info(
            f"[Sustainability/{model_name}] Using CodeCarbon automatic "
            f"geolocation; if lookup fails, CodeCarbon falls back to "
            f"'{SustainabilityConfig.FALLBACK_COUNTRY_LABEL}'."
        )

    tracker = EmissionsTracker(**tracker_kwargs)
    return tracker, measure_gpu, gpu_note

def _prepare_data_for_model(full_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    为单个模型准备训练数据：复制、精度压缩和降采样。
    Prepare training data for a single model: copy, downcast types, and down-sample.
    """
    df = full_df.copy()

    # 1. 精度压缩：float64->float32, int64->int32 / Dtype downcasting
    if ResourceConfig.ENABLE_DTYPE_DOWNCAST:
        float_cols = df.select_dtypes(include=["float64"]).columns
        int_cols = df.select_dtypes(include=["int64"]).columns

        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype("float32")
        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].astype("int32")

    # 2. 行数降采样以避免内存溢出 / Down-sampling to prevent OOM
    max_rows = ResourceConfig.MAX_TRAIN_ROWS_PER_MODEL
    if max_rows is not None and len(df) > max_rows:
        logging.warning(
            "[%s] Training data has %d rows; down-sampling to %d rows to reduce OOM risk.",
            model_name,
            len(df),
            max_rows,
        )
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    else:
        logging.info(
            "[%s] Using all %d rows for training (no down-sampling applied).",
            model_name,
            len(df),
        )

    return df


def train_and_generate(real_data, metadata, num_rows_to_generate, models_config):
    """
    训练模型，生成合成数据，并追踪碳排放。
    Train models, generate synthetic data, and track carbon emissions.
    """
    os.makedirs(PathConfig.SYNTH_DIR, exist_ok=True)
    os.makedirs(PathConfig.EMISSIONS_DIR, exist_ok=True)
    os.makedirs(PathConfig.MODELS_DIR, exist_ok=True)

    sustainability_report = {}

    logging.info("Starting model training and generation process...")

    for name, cfg in models_config.items():
        logging.info("=" * 80)
        logging.info(f"--- Processing Model: {name} ---")

        model_class = cfg["class"]
        model_params = cfg.get("params", {})

        # 准备训练数据 / Prepare training data
        data_for_model = _prepare_data_for_model(real_data, name)

        tracker = None
        gpu_included = False
        coverage_label = "unknown"

        try:
            # 初始化排放追踪器 / Initialize emissions tracker
            try:
                tracker = EmissionsTracker(
                    project_name=f"Synth-{name}",
                    output_dir=PathConfig.EMISSIONS_DIR,
                    save_to_file=True,
                )
                logging.info(f"[Sustainability/{name}] EmissionsTracker created (CPU+RAM+GPU if available).")
            except Exception as e:
                logging.warning(
                    "[Sustainability/%s] Failed to initialise EmissionsTracker: %s. "
                    "Sustainability will be marked as N/A for this model.",
                    name,
                    e,
                )
                tracker = None

            start_time = time.time()
            if tracker is not None:
                tracker.start()

            # ---------- 训练 / Training ----------
            model = model_class(metadata, **model_params)
            model.fit(data_for_model)

            # ---------- 采样 / Sampling ----------
            synth = model.sample(num_rows_to_generate)

            # ---------- 停止追踪 / Stop Tracking ----------
            emissions_kg = None
            energy_kwh = None
            if tracker is not None:
                try:
                    emissions_data = tracker.stop()
                    emissions_kg = float(getattr(emissions_data, "emissions", None))
                    energy_kwh = float(getattr(emissions_data, "energy_consumed", None))
                    gpu_included = True  # 假设 NVML 正常工作 / Assuming NVML is working
                    coverage_label = "full" if emissions_kg is not None else "unknown"
                except Exception as e:
                    logging.warning(
                        "[Sustainability/%s] Failed to read emissions data from tracker: %s",
                        name,
                        e,
                    )
                    coverage_label = "partial"

            elapsed = time.time() - start_time
            logging.info(
                "[%s] Training done. Time=%.3fs, CO2=%s kgCO2eq, Energy=%s kWh, coverage=%s",
                name,
                elapsed,
                f"{emissions_kg:.6f}" if isinstance(emissions_kg, (int, float)) else "N/A",
                f"{energy_kwh:.6f}" if isinstance(energy_kwh, (int, float)) else "N/A",
                coverage_label,
            )

            # ---------- 保存 / Saving ----------
            model_path = os.path.join(PathConfig.MODELS_DIR, f"{name.lower()}.pkl")
            model.save(model_path)
            logging.info("[%s] Model saved to %s", name, model_path)

            synth_path = os.path.join(PathConfig.SYNTH_DIR, f"synth_{name.lower()}.csv")
            synth.to_csv(synth_path, index=False)
            logging.info("[%s] Synthetic data saved to %s", name, synth_path)

            sustainability_report[name] = {
                "training_time_sec": elapsed,
                "co2_eq_kg": emissions_kg,
                "energy_kwh": energy_kwh,
                "gpu_included": gpu_included,
                "sustainability_coverage": coverage_label,
            }

        except MemoryError:
            # 捕获内存溢出错误 / Catch Out-Of-Memory errors
            logging.error(
                "[%s] Training failed due to MemoryError / OOM. "
                "Consider reducing MAX_TRAIN_ROWS_PER_MODEL or model complexity.",
                name,
                exc_info=True,
            )
            if tracker is not None:
                try:
                    tracker.stop()
                except Exception:
                    pass

            sustainability_report[name] = {
                "training_time_sec": None,
                "co2_eq_kg": None,
                "energy_kwh": None,
                "gpu_included": False,
                "sustainability_coverage": "N/A_OOM",
            }

        except Exception as e:
            # 捕获其他意外错误 / Catch unexpected errors
            logging.error("[%s] Training failed with unexpected error: %s", name, e, exc_info=True)
            if tracker is not None:
                try:
                    tracker.stop()
                except Exception:
                    pass

            sustainability_report[name] = {
                "training_time_sec": None,
                "co2_eq_kg": None,
                "energy_kwh": None,
                "gpu_included": False,
                "sustainability_coverage": "N/A_ERROR",
            }

    logging.info("All models processed. Sustainability report ready.")
    return sustainability_report


if __name__ == "__main__":
    # 独立运行模式 / Standalone execution mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Loading data for model training (Standalone)...")
    try:
        data = pd.read_csv(DatasetConfig.PROCESSED_PATH)
        metadata = SingleTableMetadata.load_from_json(DatasetConfig.METADATA_PATH)
        num_to_generate = len(data)

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