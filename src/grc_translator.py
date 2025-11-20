import os
import json
import logging
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

from src.config import (
    MODELS_CONFIG,
    RAGThresholdConfig,
    DatasetConfig,
    ResourceConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - INFO - %(message)s")

# 指标映射配置：将原始指标映射到 GRC 类别
# Metric Mapping: Maps raw metrics to GRC categories
METRIC_MAP = {
    "fidelity_jsd_avg": {
        "category": "Quality",
        "display_name": "Distribution (JSD Score)",
        "thresholds": RAGThresholdConfig.QUALITY_JSD,
    },
    "fidelity_nmi_avg": {
        "category": "Quality",
        "display_name": "Correlation (NMI Score)",
        "thresholds": RAGThresholdConfig.QUALITY_NMI,
    },
    "utility_tstr_f1": {
        "category": "Utility",
        "display_name": "ML (TSTR F1)",
        "thresholds": RAGThresholdConfig.UTILITY_TSTR_F1,
    },
    "privacy_mia_auc": {
        "category": "Risk",
        "display_name": "Privacy (MIA AUC)",
        "thresholds": RAGThresholdConfig.PRIVACY_MIA,
    },
    "avg_fairness": {
        "category": "Risk",
        "display_name": "Fairness (Avg Diff)",
        "thresholds": RAGThresholdConfig.FAIRNESS,
    },
    "co2_eq_kg": {
        "category": "Sustainability",
        "display_name": "CO2 Emissions (kg)",
        "thresholds": RAGThresholdConfig.SUSTAIN_CO2,
    },
    "training_time_sec": {
        "category": "Sustainability",
        "display_name": "Training Time (s)",
        "thresholds": RAGThresholdConfig.SUSTAIN_TIME,
    },
}

# 颜色方案 / Color Scheme
RAG_COLORS = {
    "Green": "#B7E1CD",
    "Amber": "#FFE9A3",
    "Red": "#F4B4AE",
    "N/A": "#E6E6E6",
}


def _get_rag_status(metric_key: str, value: float, thresholds: dict) -> str:
    """
    根据阈值确定 RAG 状态（支持“越高越好”和“越低越好”）。
    Determine RAG status based on thresholds (supports high-is-better and low-is-better).
    """
    if pd.isna(value):
        return "N/A"

    # 定义指标方向 / Define metric direction
    hi_better = {"fidelity_jsd_avg", "fidelity_nmi_avg", "utility_tstr_f1"}
    lo_better = {
        "privacy_mia_auc",
        "avg_fairness",
        "co2_eq_kg",
        "training_time_sec",
    }

    green = thresholds["green"]
    amber = thresholds["amber"]

    if metric_key in hi_better:
        if value >= green:
            return "Green"
        if value >= amber:
            return "Amber"
        return "Red"

    if metric_key in lo_better:
        if value <= green:
            return "Green"
        if value <= amber:
            return "Amber"
        return "Red"

    return "N/A"


def create_grc_scorecard(all_metrics, models_config):
    """
    将 metrics_report.json 转换为多级索引的记分卡 DataFrame。
    Convert metrics_report.json into a MultiIndex Scorecard DataFrame.
    """
    rows = []

    # 计算 CO2 全局最大值，用于“接近零”逻辑
    # Calculate global CO2 max for near-zero logic
    co2_vals = []
    for m in all_metrics.values():
        cov = m.get("sustainability_coverage", "full")
        v = m.get("co2_eq_kg")
        if cov == "full" and v is not None and not pd.isna(v):
            co2_vals.append(float(v))
    co2_max = float(np.nanmax(co2_vals)) if co2_vals else None
    co2_near_zero = (
            co2_max is not None and co2_max < RAGThresholdConfig.SUSTAIN_CO2_NEAR_ZERO
    )

    for model_name, metrics in all_metrics.items():
        # 聚合公平性指标 / Aggregate fairness
        fair_vals = [
            v for k, v in metrics.items() if k.startswith("fairness_") and pd.notna(v)
        ]
        metrics["avg_fairness"] = (
            float(np.nanmean(fair_vals)) if fair_vals else np.nan
        )

        coverage = metrics.get("sustainability_coverage", "full")

        for metric_key, cfg in METRIC_MAP.items():
            if metric_key not in metrics:
                continue

            raw_value = metrics.get(metric_key)
            value = raw_value
            rag = "N/A"

            # 处理可持续性覆盖率 / Handle sustainability coverage
            if metric_key == "co2_eq_kg" and coverage != "full":
                value = np.nan
                rag = "N/A"
            else:
                thresholds = cfg["thresholds"]
                if metric_key == "co2_eq_kg" and co2_near_zero and pd.notna(value):
                    rag = "Green"
                elif pd.notna(value):
                    rag = _get_rag_status(metric_key, float(value), thresholds)
                else:
                    rag = "N/A"

            # 添加覆盖率标记 / Add coverage markers
            metric_label = cfg["display_name"]
            if cfg["category"] == "Sustainability":
                if coverage == "partial":
                    metric_label += " [*]"
                elif coverage != "full":
                    metric_label += " [N/A]"

            rows.append(
                {
                    "Category": cfg["category"],
                    "Metric": metric_label,
                    "Model": model_name,
                    "Score": value,
                    "RAG": rag,
                }
            )

    df = pd.DataFrame(rows)

    # 透视表转换 / Pivot
    pivot = df.pivot_table(
        index=["Category", "Metric"],
        columns="Model",
        values=["Score", "RAG"],
        aggfunc="first",
    )

    pivot = pivot.swaplevel(0, 1, axis=1)
    model_order = list(models_config.keys())
    metric_order = ["Score", "RAG"]
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product([model_order, metric_order])
    )

    pivot = pivot.sort_index(level="Category", sort_remaining=False)

    return pivot


def _infer_sample_and_coverage_text():
    """
    推断样本信息文本。
    Infer sample info text.
    """
    effective_rows = getattr(DatasetConfig, "SAMPLE_SIZE", None)
    if effective_rows is None:
        effective_rows = getattr(ResourceConfig, "MAX_ROWS_TSTR", None)

    stratified = getattr(DatasetConfig, "STRATIFY_BY_TARGET", False)

    # 尝试推断原始行数 / Try to infer original rows
    original_rows = None
    try:
        raw_path = getattr(DatasetConfig, "RAW_PATH", None)
        if raw_path and os.path.exists(raw_path):
            with open(raw_path, "r", errors="ignore") as f:
                original_rows = sum(1 for _ in f) - 1
    except Exception as e:
        logging.warning(f"Failed to infer original dataset size: {e}")

    if effective_rows is not None:
        if original_rows is not None and original_rows > effective_rows:
            sample_info = (
                f"Effective sample size: {effective_rows:,} rows "
                f"(original dataset: {original_rows:,} rows)."
            )
        else:
            sample_info = f"Effective sample size: {effective_rows:,} rows."
    else:
        sample_info = "Effective sample size is configuration-dependent."

    if stratified:
        sampling_info = "Sampling method: stratified by target variable."
    else:
        sampling_info = "Sampling method: random sampling (no stratification)."

    return sample_info, sampling_info, effective_rows, stratified


def save_scorecard_as_image(scorecard_df, output_path: str | None = None):
    """
    生成高质量的 GRC 记分卡图像。
    Generate a high-quality GRC scorecard figure.
    """
    try:
        score_data = scorecard_df.xs("Score", level=1, axis=1)
        rag_data = scorecard_df.xs("RAG", level=1, axis=1)
    except Exception as e:
        logging.error(f"Scorecard missing Score/RAG: {e}")
        return

    n_rows, n_cols = score_data.shape
    if n_rows == 0 or n_cols == 0:
        logging.error("Empty scorecard – nothing to plot.")
        return

    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, "results", "grc_scorecard.png")

    sample_info_line, sampling_line, effective_rows, stratified = (
        _infer_sample_and_coverage_text()
    )

    # 检查是否需要显示注脚 / Check for footnotes
    metric_names = [
        idx[1] if isinstance(idx, tuple) else str(idx) for idx in scorecard_df.index
    ]
    has_partial = any("[*]" in m for m in metric_names)
    has_na = any("[N/A]" in m for m in metric_names)

    # 图表尺寸 / Figure size
    fig_width = max(10.0, 2.6 * n_cols)
    fig_height = max(9.0, 0.55 * n_rows + 7.2)
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 布局设置 / Layout setup
    outer_gs = fig.add_gridspec(
        4, 1, height_ratios=[1.3, 1.4, 3.5, 1.4]
    )
    ax_title = fig.add_subplot(outer_gs[0])
    ax_text = fig.add_subplot(outer_gs[1])
    ax_heat = fig.add_subplot(outer_gs[2])
    ax_leg = fig.add_subplot(outer_gs[3])

    ax_title.set_axis_off()
    ax_text.set_axis_off()
    ax_leg.set_axis_off()

    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.spines["left"].set_position(("data", 0))

    # ========= 1. 标题区域 / Header Area =========
    title = "GRC Quality, Risk, Sustainability and Utility Scorecard"
    subtitle = (
        "Comparison of synthetic data generators across governance-relevant dimensions: "
        "data quality, risk (privacy & fairness), sustainability, and utility."
    )

    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)

    title_y = 0.82
    subtitle_top_y = 0.58
    subtitle_line_h = 0.08
    sep_y = 0.35

    # [OPTIMIZATION] Bringing sample info closer together
    sample_y1 = 0.18  # was 0.22
    sample_y2 = 0.11  # was 0.08 (gap reduced from 0.14 to 0.07)

    ax_title.text(
        0.5,
        title_y,
        title,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        transform=ax_title.transAxes,
    )

    subtitle_lines = textwrap.wrap(subtitle, width=80)
    for i, line in enumerate(subtitle_lines):
        ax_title.text(
            0.5,
            subtitle_top_y - i * subtitle_line_h,
            line,
            ha="center",
            va="top",
            fontsize=11,
            transform=ax_title.transAxes,
        )

    line = plt.Line2D(
        [0.08, 0.92],
        [sep_y, sep_y],
        transform=ax_title.transAxes,
        color="#CCCCCC",
        linewidth=0.8,
    )
    ax_title.add_line(line)

    # [OPTIMIZATION] Smaller font (8.8) and closer positioning
    ax_title.text(
        0.5,
        sample_y1,
        sample_info_line,
        ha="center",
        va="top",
        fontsize=8.8,
        color='#333333',
        style="italic",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5,
        sample_y2,
        sampling_line,
        ha="center",
        va="top",
        fontsize=8.8,
        color='#333333',
        style="italic",
        transform=ax_title.transAxes,
    )

    # ========= 2. 文本说明框 / Gray Text Boxes =========
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    # [OPTIMIZATION] Extended box height to accommodate larger line spacing
    left_box_y0 = 0.12  # lowered from 0.20
    left_box_h = 0.86  # increased from 0.78
    right_box_y0 = 0.12
    right_box_h = 0.86

    left_box = FancyBboxPatch(
        (0.02, left_box_y0),
        0.45,
        left_box_h,
        boxstyle="round,pad=0.03",
        linewidth=0.6,
        edgecolor="#DDDDDD",
        facecolor="#F9F9F9",
        transform=ax_text.transAxes,
        zorder=0,
    )
    right_box = FancyBboxPatch(
        (0.53, right_box_y0),
        0.45,
        right_box_h,
        boxstyle="round,pad=0.03",
        linewidth=0.6,
        edgecolor="#DDDDDD",
        facecolor="#F9F9F9",
        transform=ax_text.transAxes,
        zorder=0,
    )
    ax_text.add_patch(left_box)
    ax_text.add_patch(right_box)

    box_top_y = left_box_y0 + left_box_h

    # Headers
    # Moved slightly higher to give body text more room
    title_y_in_box = box_top_y - 0.03
    ax_text.text(
        0.045,
        title_y_in_box,
        "Dimensions:",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        transform=ax_text.transAxes,
    )
    ax_text.text(
        0.555,
        title_y_in_box,
        "How to read the scorecard",
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        transform=ax_text.transAxes,
    )

    # Content
    left_bullets = [
        "• Quality – how closely synthetic data matches the patterns and relationships in the real data.",
        "• Risk – privacy and fairness risk, including potential bias between groups and risk of re-identification.",
        "• Sustainability – CO\u2082 emissions and computational cost of generating the data. Lower is better.",
        "• Utility – downstream ML performance when models are trained on synthetic data and evaluated on real data (TSTR).",
    ]

    right_bullets = [
        "• Each row is a metric under one of the four GRC dimensions (Quality, Risk, Sustainability, Utility).",
        "• Each column is a synthetic data generator (model).",
        "• Green / Amber / Red / Grey follow the legend below. Higher scores are better for quality and utility; lower scores are better for risk and sustainability.",
    ]

    def _draw_paragraph_fixed(ax, bullets, x_start, start_y, wrap_width, transform):
        import textwrap as _tw

        # [OPTIMIZATION] Significantly increased spacing
        line_height = 0.065  # Increased from 0.052
        para_gap = 0.042  # Increased from 0.035

        current_y = start_y
        for idx, bullet in enumerate(bullets):
            lines = _tw.wrap(bullet, width=wrap_width)
            for line in lines:
                ax.text(
                    x_start,
                    current_y,
                    line,
                    ha="left",
                    va="top",
                    fontsize=8.6,
                    fontweight='normal',
                    color='#222222',
                    transform=transform,
                )
                current_y -= line_height
            current_y -= para_gap

    # Starting text position
    text_start_y = box_top_y - 0.15

    _draw_paragraph_fixed(
        ax_text,
        left_bullets,
        x_start=0.045,
        start_y=text_start_y,
        wrap_width=55,
        transform=ax_text.transAxes,
    )
    _draw_paragraph_fixed(
        ax_text,
        right_bullets,
        x_start=0.555,
        start_y=text_start_y,
        wrap_width=47,
        transform=ax_text.transAxes,
    )

    # ========= 3. Score Matrix =========
    ax_heat.set_xlim(-0.15, n_cols)
    ax_heat.set_ylim(0, n_rows)
    ax_heat.invert_yaxis()

    for i in range(n_rows):
        for j in range(n_cols):
            val = score_data.iat[i, j]
            rag = rag_data.iat[i, j]
            color = RAG_COLORS.get(rag, "#FFFFFF")

            ax_heat.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            if pd.notna(val):
                if isinstance(val, (int, float, np.floating)):
                    text_val = f"{val:.3f}"
                else:
                    try:
                        text_val = f"{float(val):.3f}"
                    except:
                        text_val = str(val)

                ax_heat.text(
                    j + 0.5,
                    i + 0.5,
                    text_val,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                )

    ax_heat.set_xticks([j + 0.5 for j in range(n_cols)])
    ax_heat.set_xticklabels(score_data.columns.tolist(), fontsize=11)
    y_centers = [i + 0.5 for i in range(n_rows)]
    ax_heat.set_yticks(y_centers)
    ax_heat.set_yticklabels([])

    prev_cat = None
    for row_i, idx in enumerate(score_data.index):
        if isinstance(idx, tuple):
            category, metric = idx
        else:
            category, metric = "", str(idx)

        if prev_cat is not None and category != prev_cat:
            ax_heat.axhline(row_i, color="#DDDDDD", linewidth=0.8)
        prev_cat = category
        y = y_centers[row_i]

        ax_heat.text(
            -0.02,
            y - 0.18,
            category,
            ha="right",
            va="center",
            fontsize=8.3,
            fontweight="bold",
            transform=ax_heat.transData,
        )
        ax_heat.text(
            -0.02,
            y + 0.15,
            metric,
            ha="right",
            va="center",
            fontsize=8.1,
            transform=ax_heat.transData,
        )

    ax_heat.set_xlabel("Model", fontsize=11, labelpad=8)
    ax_heat.set_ylabel("GRC assessment dimension", fontsize=11)
    ax_heat.yaxis.set_label_coords(-0.16, 0.5)
    ax_heat.tick_params(axis="x", length=4, width=0.8)
    ax_heat.tick_params(axis="y", length=4, width=0.8)

    # ========= 4. Legend & Footnotes =========
    patches = [
        mpatches.Patch(color=RAG_COLORS["Green"], label="Green – Good / Low-Risk"),
        mpatches.Patch(color=RAG_COLORS["Amber"], label="Amber – Review Required"),
        mpatches.Patch(color=RAG_COLORS["Red"], label="Red – High-Risk"),
        mpatches.Patch(color=RAG_COLORS["N/A"], label="Grey – N/A"),
    ]

    # [OPTIMIZATION] Pushed Legend DOWN to 0.60 to clear the "Model" label
    ax_leg.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.60),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )

    # --- [FIX] 修复：重新添加注脚文本的生成逻辑 ---
    # --- [FIX] Restore footnote text generation logic ---
    footnote_texts = []

    if effective_rows is not None:
        if stratified:
            fn1 = (
                f"Evaluations in this thesis were performed on a fixed "
                f"{effective_rows:,}-row stratified sample for reproducibility. "
                "A resource-aware automatic mode is also available for "
                "practical deployment."
            )
        else:
            fn1 = (
                "Evaluations in this thesis were performed on a fixed "
                f"{effective_rows:,}-row sample for reproducibility. "
                "A resource-aware automatic mode is also available for "
                "practical deployment."
            )
        footnote_texts.append(fn1)

    if has_partial:
        footnote_texts.append(
            "[*] Sustainability rows are based on partial energy measurements "
            "(for example, only CPU/RAM energy was available; GPU energy "
            "could not be measured for at least one model)."
        )

    if has_na:
        footnote_texts.append(
            "[N/A] Sustainability metrics could not be computed for at least "
            "one model because no valid emissions data were available."
        )

    start_y = 0.35
    gap_y = 0.25

    for idx, text in enumerate(footnote_texts):
        wrapped = "\n".join(textwrap.wrap(text, width=115))
        y_pos = start_y - (idx * gap_y)
        ax_leg.text(
            0.5,
            y_pos,
            wrapped,
            ha="center",
            va="top",
            fontsize=8.3,
            transform=ax_leg.transAxes,
        )

    plt.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.97,
        bottom=0.07,
        hspace=0.05,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info(f"GRC Scorecard saved to {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    metrics_path = os.path.join(project_root, "results", "metrics_report.json")
    csv_path = os.path.join(project_root, "results", "grc_scorecard.csv")

    try:
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        model_keys = list(metrics_data.keys())
        mock_config = {k: {} for k in model_keys}

        scorecard = create_grc_scorecard(metrics_data, mock_config)
        scorecard.to_csv(csv_path, float_format="%.3f")
        save_scorecard_as_image(scorecard)

        print("Preview of GRC scorecard:")
        print(scorecard)

    except FileNotFoundError:
        logging.error(
            f"Metrics report not found at {metrics_path}. Please run main.py first."
        )
    except Exception as e:
        logging.error(f"Error generating scorecard: {e}")