import textwrap

import pandas as pd
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import matplotlib.patches as mpatches
from src.config import MODELS_CONFIG, RAGThresholdConfig
from textwrap import fill

# --- 配置日志 / Configure logging ---
# [!!] 修正: 将日志级别设置为 INFO，并将消息更改为英文
# [!!] Fix: Set log level to INFO and change messages to English
logging.basicConfig(level=logging.INFO, format='%(asctime)s - INFO - %(message)s')

# --- GRC 启发式阈值 / GRC Heuristic Thresholds ---
# 这些是您在论文中定义和论证的业务规则
# These are the business rules you define and justify in your thesis
#
# [!!] 修正: JSD 分数 (1-JSD) 越高越好 [1]
# [!!] Fix: JSD Score (1-JSD) is "higher is better" [1]
JSD_THRESHOLDS = {'green': 0.9, 'amber': 0.8}  # >0.9=G, 0.8-0.9=A, <0.8=R
# NMI: 越高越好 / Higher is better.
NMI_THRESHOLDS = {'green': 0.8, 'amber': 0.6}  # >0.8=G, 0.6-0.8=A, <0.6=R
# TSTR F1: 越高越好 / Higher is better.
TSTR_THRESHOLDS = {'green': 0.76, 'amber': 0.70}
# MIA AUC: 越低越好 / Lower is better. (0.5 是完美的 / 0.5 is perfect)
MIA_THRESHOLDS = {'green': 0.55, 'amber': 0.65}  # <0.55=G, 0.55-0.65=A, >0.65=R
# 公平性 (Avg Diff): 越低越好 / Lower is better. (0 是完美的 / 0 is perfect)
FAIR_THRESHOLDS = {'green': 0.1, 'amber': 0.2}  # <0.1=G, 0.1-0.2=A, >0.2=R

# [!!] 修正: 将所有 'display_name' 更改为英文以修复“乱码” [Image 11]
# [!!] Fix: Change all 'display_name' to English to fix "mojibake" [Image 11]
METRIC_MAP = {
    # Quality / Fidelity
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

    # Utility
    "utility_tstr_f1": {
        "category": "Utility",
        "display_name": "ML (TSTR F1)",
        "thresholds": RAGThresholdConfig.UTILITY_TSTR_F1,
    },

    # Risk
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

    # Sustainability
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

# [!!] 新增: 用于可视化的 RAG 颜色 / New: RAG colors for visualization [1]
RAG_COLORS = {
    "Green": "#B7E1CD",   # 柔和绿色
    "Amber": "#FFE9A3",   # 柔和琥珀色
    "Red":   "#F4B4AE",   # 柔和红色（偏珊瑚）
    "N/A":   "#E6E6E6",   # 中性灰
}
def _get_rag_status(metric_key: str, value: float, thresholds: dict) -> str:
    """
    通用 RAG 逻辑：
    - 对 JSD / NMI / TSTR：越高越好
    - 对 MIA / FAIR / CO2 / 时间：越低越好
    """
    if pd.isna(value):
        return "N/A"

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
    将原始 metrics_report.json 转成 GRC 记分卡 DataFrame：
    MultiIndex 行（Category, Metric），列为 (Model, Score/RAG)。
    RAG 阈值全部来自 config.RAGThresholdConfig。
    """
    scorecard_rows = []

    # --- 计算 CO2 的整体范围，用于 near-zero 保护 ---
    co2_vals = []
    for m in all_metrics.values():
        cov = m.get("sustainability_coverage", "full")
        v = m.get("co2_eq_kg")
        if cov == "full" and v is not None and not pd.isna(v):
            co2_vals.append(float(v))

    if co2_vals:
        co2_max = float(np.nanmax(co2_vals))
    else:
        co2_max = None

    co2_near_zero = (
        co2_max is not None
        and co2_max < RAGThresholdConfig.SUSTAIN_CO2_NEAR_ZERO
    )

    # --- 遍历每个模型 ---
    for model_name, metrics in all_metrics.items():
        # 聚合公平性为 avg_fairness
        fair_vals = [
            v
            for k, v in metrics.items()
            if k.startswith("fairness_") and pd.notna(v)
        ]
        if fair_vals:
            metrics["avg_fairness"] = float(np.nanmean(fair_vals))
        else:
            metrics["avg_fairness"] = np.nan

        coverage = metrics.get("sustainability_coverage", "full")

        # 针对每个我们关心的指标填一行
        for metric_key, cfg in METRIC_MAP.items():
            if metric_key not in metrics:
                continue

            raw_value = metrics.get(metric_key)
            value = raw_value
            rag = "N/A"

            # 可持续性：如果 coverage 不是 full，则标为 N/A（表示只测到部分/没测到）
            if metric_key == "co2_eq_kg" and coverage != "full":
                value = np.nan
                rag = "N/A"
            else:
                thresholds = cfg["thresholds"]
                if metric_key == "co2_eq_kg" and co2_near_zero and pd.notna(value):
                    # ⭐ 所有模型 CO2 都极小：不在它们之间搞红黄绿差异，统一 Green
                    rag = "Green"
                elif pd.notna(value):
                    rag = _get_rag_status(metric_key, float(value), thresholds)
                else:
                    rag = "N/A"

            # ------------ 这里新增：给可持续性行加 [*] / [N/A] 标记 ------------
            metric_label = cfg["display_name"]
            if cfg["category"] == "Sustainability":
                if coverage == "partial":
                    metric_label += " [*]"
                elif coverage != "full":
                    metric_label += " [N/A]"
            # -------------------------------------------------------------

            scorecard_rows.append(
                {
                    "Category": cfg["category"],
                    "Metric": metric_label,  # ← 用带标记的行名
                    "Model": model_name,
                    "Score": value,
                    "RAG": rag,
                }
            )

    df = pd.DataFrame(scorecard_rows)

    # 透视成 MultiIndex 记分卡
    pivot = df.pivot_table(
        index=["Category", "Metric"],
        columns="Model",
        values=["Score", "RAG"],
        aggfunc="first",
    )

    # 列层级换顺序：外层是 Model，内层是 Score/RAG
    pivot = pivot.swaplevel(0, 1, axis=1)

    model_order = list(models_config.keys())
    metric_order = ["Score", "RAG"]

    # 确保列顺序：按 config 里模型顺序 + ["Score","RAG"]
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product([model_order, metric_order])
    )

    # 按 Category 排行
    pivot = pivot.sort_index(level="Category", sort_remaining=False)

    return pivot

def save_scorecard_as_image(scorecard_df, output_path: str | None = None):
    """
    Generate a publication-quality GRC scorecard figure.
    """
    # ---- 0. 取 Score / RAG ----
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

    # ---- 1. 画布 & GridSpec ----
    fig_width = max(10.0, 2.6 * n_cols)
    fig_height = max(8.0, 0.55 * n_rows + 6.5)
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 调整各块高度比例，让整体更紧凑：标题略薄、灰框略薄、矩阵略厚、图例略薄
    outer_gs = fig.add_gridspec(4, 1, height_ratios=[0.65, 1.4, 3.5, 0.7])
    ax_title = fig.add_subplot(outer_gs[0])
    ax_text  = fig.add_subplot(outer_gs[1])
    ax_heat  = fig.add_subplot(outer_gs[2])
    ax_leg   = fig.add_subplot(outer_gs[3])

    ax_title.set_axis_off()
    ax_text.set_axis_off()
    ax_leg.set_axis_off()

    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    # 让 y 轴刻度线贴在矩阵左边界
    ax_heat.spines["left"].set_position(("data", 0))

    # ---- 2. 标题 & 副标题 ----
    title = "GRC Quality, Risk, Sustainability and Utility Scorecard"
    subtitle = (
        "Comparison of synthetic data generators across governance-relevant dimensions: "
        "data quality, risk (privacy & fairness), sustainability, and utility."
    )
    ax_title.text(
        0.5, 0.80, title,
        ha="center", va="top",
        fontsize=17, fontweight="bold",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, 0.25,
        "\n".join(textwrap.wrap(subtitle, width=90)),
        ha="center", va="top",
        fontsize=10.5,
        transform=ax_title.transAxes,
    )

    # ---- 3. 上方两个灰框 ----
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    left_box = FancyBboxPatch(
        (0.02, 0.04), 0.45, 0.92,
        boxstyle="round,pad=0.03",
        linewidth=0.5, edgecolor="#DDDDDD", facecolor="#F9F9F9",
        transform=ax_text.transAxes, zorder=0,
    )
    right_box = FancyBboxPatch(
        (0.53, 0.04), 0.45, 0.92,
        boxstyle="round,pad=0.03",
        linewidth=0.5, edgecolor="#DDDDDD", facecolor="#F9F9F9",
        transform=ax_text.transAxes, zorder=0,
    )
    ax_text.add_patch(left_box)
    ax_text.add_patch(right_box)

    # ------------ 左框：Dimensions ------------
    ax_text.text(
        0.045, 0.96, "Dimensions:",
        ha="left", va="top",
        fontsize=11.0, fontweight="bold",
        transform=ax_text.transAxes,
    )

    left_bullets = [
        "• Quality – how closely synthetic data matches the patterns and relationships in the real data.",
        "• Risk – privacy and fairness risk, including potential bias between groups and risk of re-identification.",
        "• Sustainability – CO\u2082 emissions and computational cost of generating the data. Lower is better.",
        "• Utility – downstream ML performance when models are trained on synthetic data and evaluated on real data (TSTR).",
    ]
    # 稍微往下挪一点起始位置，同时减小行间距 & 段间距，避免溢出
    left_y = 0.845
    left_line_h = 0.065
    left_block_extra = 0.045
    left_wrap_width = 50

    for bullet in left_bullets:
        wrapped = textwrap.wrap(bullet, width=left_wrap_width)
        if not wrapped:
            continue
        for j, line in enumerate(wrapped):
            ax_text.text(
                0.045,
                left_y - j * left_line_h,
                line,
                ha="left", va="top",
                fontsize=9.3,
                transform=ax_text.transAxes,
            )
        total_h = len(wrapped) * left_line_h
        left_y -= total_h + left_block_extra

    # ------------ 右框：How to read the scorecard ------------
    ax_text.text(
        0.555, 0.96, "How to read the scorecard",
        ha="left", va="top",
        fontsize=11.0, fontweight="bold",
        transform=ax_text.transAxes,
    )

    right_bullets = [
        "• Each row is a metric under one of the four GRC dimensions (Quality, Risk, Sustainability, Utility).",
        "• Each column is a synthetic data generator (model).",
        "• Green / Amber / Red / Grey follow the legend below. Higher scores are better for quality and utility; lower scores are better for risk and sustainability.",
    ]
    right_y = 0.845
    right_line_h = 0.065
    right_block_extra = 0.045
    right_wrap_width = 44

    for bullet in right_bullets:
        wrapped = textwrap.wrap(bullet, width=right_wrap_width)
        if not wrapped:
            continue
        for j, line in enumerate(wrapped):
            ax_text.text(
                0.555,
                right_y - j * right_line_h,
                line,
                ha="left", va="top",
                fontsize=9.3,
                transform=ax_text.transAxes,
            )
        total_h = len(wrapped) * right_line_h
        right_y -= total_h + right_block_extra

    # ---- 4. 记分卡矩阵 ----
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
                    (j, i), 1, 1,
                    facecolor=color,
                    edgecolor="white",  # 改成白色
                    linewidth=0.8,  # 稍微粗一点，让分隔清晰
                )
            )
            # 尝试把 val 当 float 画，如果实在转不了就直接写 str(val)
            if pd.notna(val):
                val_to_show = None
                if isinstance(val, (int, float, np.floating)):
                    val_to_show = f"{val:.3f}"
                else:
                    try:
                        val_to_show = f"{float(val):.3f}"
                    except (TypeError, ValueError):
                        val_to_show = str(val)

                ax_heat.text(
                    j + 0.5, i + 0.5,
                    val_to_show,
                    ha="center", va="center",
                    fontsize=8.5,
                )

    # 列名（模型）
    ax_heat.set_xticks([j + 0.5 for j in range(n_cols)])
    ax_heat.set_xticklabels(score_data.columns.tolist(), fontsize=11)

    # 行名：两行（维度 + 指标）
    y_centers = [i + 0.5 for i in range(n_rows)]
    ax_heat.set_yticks(y_centers)
    ax_heat.set_yticklabels([])

    # 在画行名之前，加一条「维度之间」的淡分割线
    prev_category = None
    for row_i, idx in enumerate(score_data.index):
        if isinstance(idx, tuple):
            category, metric = idx
        else:
            category, metric = "", str(idx)
        if prev_category is not None and category != prev_category:
            y = row_i  # 当前行上方的坐标
            ax_heat.axhline(
                y,
                color="#DDDDDD",
                linewidth=0.8,
            )
        prev_category = category

        y = y_centers[row_i]

        ax_heat.text(
            -0.02, y - 0.12,
            category,
            ha="right", va="center",
            fontsize=8.3,
            fontweight="bold",
            transform=ax_heat.transData,
        )
        ax_heat.text(
            -0.02, y + 0.08,
            metric,
            ha="right", va="center",
            fontsize=8.1,
            transform=ax_heat.transData,
        )

    # 轴标题：X 轴用 labelpad，Y 轴用 coords，避免和行名重叠
    ax_heat.set_xlabel("Model", fontsize=11, labelpad=10)
    ax_heat.set_ylabel("GRC assessment dimension", fontsize=11)
    ax_heat.yaxis.set_label_coords(-0.16, 0.5)

    # 刻度线
    ax_heat.tick_params(axis="x", length=4, width=0.8)
    ax_heat.tick_params(axis="y", length=4, width=0.8)

    # ---- 5. 底部图例 ----
    legend_patches = [
        mpatches.Patch(color=RAG_COLORS["Green"], label="Green – Good / Low-Risk"),
        mpatches.Patch(color=RAG_COLORS["Amber"], label="Amber – Review Required"),
        mpatches.Patch(color=RAG_COLORS["Red"],  label="Red – High-Risk"),
        mpatches.Patch(color=RAG_COLORS["N/A"],  label="Grey – N/A"),
    ]
    ax_leg.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.25),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    # ---- 5.1 可持续性脚注：放在图例下方 ----
    # 从 MultiIndex 中取出 Metric 文本，判断是否包含 [*] 或 [N/A]
    metric_names = [idx[1] if isinstance(idx, tuple) else str(idx)
                    for idx in scorecard_df.index]
    has_partial = any("[*]" in name for name in metric_names)
    has_na = any("[N/A]" in name for name in metric_names)

    footnote_lines = []
    if has_partial:
        footnote_lines.append(
            "[*] Sustainability rows are based on partial energy measurements "
            "(e.g., only CPU/RAM energy was available; GPU energy could not be "
            "measured for at least one model)."
        )
    if has_na:
        footnote_lines.append(
            "[N/A] Sustainability metrics could not be computed for at least "
            "one model because no valid energy/emissions data were available."
        )

    if footnote_lines:
        ax_leg.text(
            0.5,
            0.05,
            "\n".join(textwrap.wrap(" ".join(footnote_lines), width=110)),
            ha="center",
            va="bottom",
            fontsize=8.0,
            transform=ax_leg.transAxes,
        )

    # ---- 6. 整体布局 ----
    plt.subplots_adjust(
        left=0.18,   # 整体向左挪一点，缩小左侧留白
        right=0.98,
        top=0.97,
        bottom=0.07,
        hspace=0.18,  # 纵向间距更紧凑
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logging.info(f"GRC Scorecard saved to {output_path}")


if __name__ == "__main__":
    # 允许此脚本直接运行以进行测试 / Allow this script to be run directly for testing

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 'src' 的上一级 / Parent of 'src'

    METRICS_REPORT_PATH = os.path.join(PROJECT_ROOT, "results", "metrics_report.json")
    GRC_SCORECARD_PATH = os.path.join(PROJECT_ROOT, "results", "grc_scorecard.csv")
    GRC_IMAGE_PATH = os.path.join(PROJECT_ROOT, "results",
                                  "grc_scorecard.png")  # [!!] 修正: 使用构建的路径 / Fix: Use built path

    logging.info(f"Loading metrics from {METRICS_REPORT_PATH}...")
    try:
        with open(METRICS_REPORT_PATH, 'r') as f:
            metrics_data = json.load(f)

        # [!!] 修正: 从加载的数据中动态获取键
        # [!!] Fix: Dynamically get keys from loaded data
        model_keys = list(metrics_data.keys())
        mock_config = {key: {} for key in model_keys}
        logging.info(f"Found models: {model_keys}")

        scorecard = create_grc_scorecard(metrics_data, mock_config)
        scorecard.to_csv(GRC_SCORECARD_PATH, float_format="%.3f")
        logging.info(f"GRC Scorecard CSV saved to {GRC_SCORECARD_PATH}")

        # [!!] 修正: 调用新函数时不带参数 (路径是自动的)
        # [!!] Fix: Call new function without path argument (path is automatic)
        save_scorecard_as_image(scorecard)

        print("\n--- GRC Scorecard (Preview) ---")
        print(scorecard)
        print("---------------------------------")

    except FileNotFoundError:
        logging.error(f"Metrics report not found at {METRICS_REPORT_PATH}. Please run main.py first.")
    except Exception as e:
        logging.error(f"Error generating scorecard: {e}")
        import traceback

        traceback.print_exc()