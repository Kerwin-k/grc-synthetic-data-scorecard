import os
import time
import logging
import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from sklearn.utils import resample

# 导入配置
from src.config import DatasetConfig, PathConfig, MODELS_CONFIG, SustainabilityConfig, ResourceConfig

# 导入 SDV 合成器类
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)


def _create_emissions_tracker_for_model(model_name: str):
    # 1. 定义所有追踪器通用的参数
    common_kwargs = {
        "project_name": f"thesis-sdg-{model_name}",
        "output_dir": PathConfig.EMISSIONS_DIR,
        "output_file": f"{model_name}_emissions.csv",
        "save_to_file": True,
        "log_level": "error"  # 减少日志噪音
    }

    try:
        # 2. 根据是否有 ISO 代码决定模式
        if SustainabilityConfig.FIXED_COUNTRY_ISO:
            # --- 离线模式 (Offline) ---
            # 关键修复：country_iso_code 必须作为显式参数传递，不要混在 common_kwargs 里
            logging.info(
                f"[Sustainability] Init Offline Tracker for {model_name} (ISO: {SustainabilityConfig.FIXED_COUNTRY_ISO})")

            tracker = OfflineEmissionsTracker(
                country_iso_code=SustainabilityConfig.FIXED_COUNTRY_ISO,
                **common_kwargs
            )
        else:
            # --- 在线模式 (Online) ---
            logging.info(f"[Sustainability] Init Online Tracker for {model_name}")
            tracker = EmissionsTracker(**common_kwargs)

        return tracker

    except Exception as e:
        logging.warning(f"[Sustainability] Failed to init tracker: {e}")
        return None


def _prepare_data_for_model(full_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    安全地准备训练数据：
    1. 先降采样到 MAX_TRAIN_ROWS (防止 OOM)
    2. 再进行类平衡 (防止 Mode Collapse)
    """
    df = full_df.copy()
    target_col = getattr(DatasetConfig, "TARGET_COLUMN", "TARGET")
    max_rows = ResourceConfig.MAX_TRAIN_ROWS_PER_MODEL

    # --- 步骤 1: 预先降采样 (Pre-emptive Downsampling) ---
    # 在做任何处理前，先限制总行数，防止后续处理撑爆内存
    if max_rows is not None and len(df) > max_rows:
        logging.info(f"[{model_name}] Downsampling from {len(df)} to {max_rows} before balancing.")
        if target_col in df.columns:
            # 分层抽样以保持原始比例
            df = df.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(frac=max_rows / len(full_df), random_state=42)
            )
        else:
            df = df.sample(n=max_rows, random_state=42)

    # --- 步骤 2: 处理类别不平衡 (Class Balancing) ---
    # 只有当目标列存在且确实不平衡时才执行
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        if len(counts) == 2:  # 仅处理二分类
            minority_class = counts.idxmin()
            majority_class = counts.idxmax()

            min_count = counts[minority_class]
            maj_count = counts[majority_class]

            ratio = min_count / maj_count

            # 阈值设为 0.3
            if ratio < 0.3:
                logging.info(f"[{model_name}] Triggering Hybrid Resampling (Ratio: {ratio:.2%})...")

                df_minority = df[df[target_col] == minority_class]
                df_majority = df[df[target_col] == majority_class]

                # 混合策略 (Hybrid Strategy)
                # 1. 保持多数类数据尽可能多 (Minimal Invasive)，只在超过 MAX_TRAIN_ROWS 时才会被 Step 1 截断
                # 2. 上采样少数类，使其达到多数类的一定比例 (例如 50%)，以恢复信号

                # 目标: 让少数类达到多数类的 50% (或者你认为合适的平衡点，0.5 是比较稳健的)
                target_min_count = int(len(df_majority) * 0.5)

                # 如果原始少数类太少，就过采样；如果已经够多(只是比例低)，保持原样
                if len(df_minority) < target_min_count:
                    df_min_up = resample(df_minority,
                                         replace=True,
                                         n_samples=target_min_count,
                                         random_state=42)
                else:
                    df_min_up = df_minority

                # 合并
                df = pd.concat([df_majority, df_min_up])
                logging.info(f"[{model_name}] Hybrid Resampling Complete. New shape: {df.shape}")

    # --- 步骤 3: 精度压缩 (Downcasting) ---
    if ResourceConfig.ENABLE_DTYPE_DOWNCAST:
        float_cols = df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype("float32")

    # 最后打乱顺序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def train_and_generate(real_data, metadata, num_rows_to_generate, models_config):
    os.makedirs(PathConfig.SYNTH_DIR, exist_ok=True)
    os.makedirs(PathConfig.EMISSIONS_DIR, exist_ok=True)
    os.makedirs(PathConfig.MODELS_DIR, exist_ok=True)

    sustainability_report = {}

    for name, cfg in models_config.items():
        logging.info("=" * 80)
        logging.info(f"--- Processing Model: {name} ---")

        model_class = cfg["class"]
        model_params = cfg.get("params", {})

        # 准备数据
        data_for_model = _prepare_data_for_model(real_data, name)

        tracker = _create_emissions_tracker_for_model(name)

        start_time = time.time()
        if tracker:
            tracker.start()

        try:
            # 训练
            model = model_class(metadata, **model_params)
            model.fit(data_for_model)

            # 生成
            synth = model.sample(num_rows_to_generate)

            # 停止追踪 (关键修复：只调用一次 stop)
            emissions_kg = None
            energy_kwh = None

            if tracker:
                emissions_kg = tracker.stop()  # stop() 返回的是 emissions
                # 尝试获取更详细的数据
                energy_kwh = getattr(tracker, 'final_energy', None)

            elapsed = time.time() - start_time

            # 记录结果
            sustainability_report[name] = {
                "training_time_sec": elapsed,
                "co2_eq_kg": emissions_kg,
                "energy_kwh": energy_kwh,
                "sustainability_coverage": "full" if emissions_kg is not None else "unknown"
            }

            # 保存
            model.save(os.path.join(PathConfig.MODELS_DIR, f"{name.lower()}.pkl"))
            synth.to_csv(os.path.join(PathConfig.SYNTH_DIR, f"synth_{name.lower()}.csv"), index=False)

        except Exception as e:
            logging.error(f"[{name}] Failed: {e}", exc_info=True)
            if tracker:
                try:
                    tracker.stop()
                except:
                    pass

            sustainability_report[name] = {
                "training_time_sec": None,
                "co2_eq_kg": None,
                "sustainability_coverage": "error"
            }

    return sustainability_report