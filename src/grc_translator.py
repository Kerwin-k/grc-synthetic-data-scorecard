import textwrap

import pandas as pd
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import matplotlib.patches as mpatches
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
    'fidelity_jsd_avg': {'thresholds': JSD_THRESHOLDS, 'display_name': 'Quality: Distribution (JSD Score)'},
    'fidelity_nmi_avg': {'thresholds': NMI_THRESHOLDS, 'display_name': 'Quality: Correlation (NMI Score)'},
    'utility_tstr_f1': {'thresholds': TSTR_THRESHOLDS, 'display_name': 'Utility: ML (TSTR F1)'},
    'privacy_mia_auc': {'thresholds': MIA_THRESHOLDS, 'display_name': 'Risk: Privacy (MIA AUC)'},
    'avg_fairness': {'thresholds': FAIR_THRESHOLDS, 'display_name': 'Risk: Fairness (Avg Diff)'},
    'co2_eq_kg': {'thresholds': None, 'display_name': 'Sustainability: CO2 Emissions (kg)'},
    'training_time_sec': {'thresholds': None, 'display_name': 'Sustainability: Training Time (s)'}
}

# [!!] 新增: 用于可视化的 RAG 颜色 / New: RAG colors for visualization [1]
RAG_COLORS = {
    "Green": "#B7E1CD",   # 柔和绿色
    "Amber": "#FFE9A3",   # 柔和琥珀色
    "Red":   "#F4B4AE",   # 柔和红色（偏珊瑚）
    "N/A":   "#E6E6E6",   # 中性灰
}
def _get_rag_status(metric_name, value, thresholds):
    """
    辅助函数，用于分配 RAG 状态。
    Helper function to assign RAG status.
    """
    if pd.isna(value):
        return "N/A"

    # '越高越好' 的指标 (NMI, TSTR, JSD Score)
    # 'Higher is better' metrics (NMI, TSTR, JSD Score)
    if metric_name in ['fidelity_nmi_avg', 'utility_tstr_f1', 'fidelity_jsd_avg']:
        if value >= thresholds['green']: return "Green"
        if value >= thresholds['amber']: return "Amber"
        return "Red"
    # '越低越好' 的指标 (MIA, Fairness)
    # 'Lower is better' metrics (MIA, Fairness)
    else:
        if value <= thresholds['green']: return "Green"
        if value <= thresholds['amber']: return "Amber"
        return "Red"


def create_grc_scorecard(all_metrics, models_config):
    """
    将原始指标字典转化为 GRC 就绪的、人类可读的 DataFrame。
    Transforms the raw metrics dict into a GRC-ready, human-readable DataFrame.
    """
    scorecard_data = []

    # ---- 先看 Sustainability coverage 情况 ----
    coverage_values = [
        m.get("sustainability_coverage")
        for m in all_metrics.values()
        if "sustainability_coverage" in m
    ]

    any_partial = any(c == "partial" for c in coverage_values)
    all_none = (not coverage_values) or all(
        c in (None, "none") for c in coverage_values
    )

    # 行名后缀：有 partial → [*]，全部 none → [N/A]
    sustainability_label_suffix = ""
    if any_partial and not all_none:
        sustainability_label_suffix = " [*]"
    elif all_none:
        sustainability_label_suffix = " [N/A]"

    # ---- 只用 coverage == full 的模型来算 co2/time 的基线 ----
    all_co2 = [
        m.get("co2_eq_kg", np.inf)
        for m in all_metrics.values()
        if (m.get("co2_eq_kg") is not None)
        and (m.get("sustainability_coverage") == "full")
    ]
    min_co2 = min(all_co2) if all_co2 else np.inf

    all_time = [
        m.get("training_time_sec", np.inf)
        for m in all_metrics.values()
        if (m.get("training_time_sec") is not None)
        and (m.get("sustainability_coverage") == "full")
    ]
    min_time = min(all_time) if all_time else np.inf

    # ---- 遍历模型，构造每一行 ----
    for model_name, metrics in all_metrics.items():

        # 聚合公平性指标 / Aggregate fairness metrics
        fair_metrics = [
            v for k, v in metrics.items()
            if k.startswith("fairness_") and pd.notna(v)
        ]
        avg_fairness = np.nanmean(fair_metrics) if fair_metrics else np.nan
        metrics["avg_fairness"] = avg_fairness  # 添加以便映射 / Add for mapping

        coverage = metrics.get("sustainability_coverage")

        # --- 定义记分卡行 / Define scorecard rows ---
        for key, config in METRIC_MAP.items():
            if key not in metrics:
                continue

            value = metrics.get(key)
            status = "N/A"

            # ⭐ 对可持续性两个指标：只要 coverage 不是 full，一律视为 N/A（灰色）
            if key in ("co2_eq_kg", "training_time_sec") and coverage != "full":
                value = np.nan
                status = "N/A"
            else:
                if config["thresholds"] is not None:
                    # 使用绝对阈值 / Use absolute thresholds
                    status = _get_rag_status(key, value, config["thresholds"])
                elif pd.notna(value):
                    # 使用相对 RAG 状态进行可持续性排名 / Use relative RAG for sustainability
                    if key == "co2_eq_kg":
                        if value <= (min_co2 * 1.1):
                            status = "Green"  # 10% 容差
                        elif value <= (min_co2 * 2.0):
                            status = "Amber"  # 2倍以内
                        else:
                            status = "Red"
                    elif key == "training_time_sec":
                        if value <= (min_time * 1.1):
                            status = "Green"
                        elif value <= (min_time * 2.0):
                            status = "Amber"
                        else:
                            status = "Red"

            # 拆分英文显示名称 / Split English display name
            category, metric_display = config["display_name"].split(": ")

            # 对 Sustainability 两个指标加上 [*] / [N/A] 后缀
            if key in ("co2_eq_kg", "training_time_sec") and sustainability_label_suffix:
                metric_display = metric_display + sustainability_label_suffix

            scorecard_data.append(
                {
                    "Model": model_name,
                    "Category": category,
                    "Metric": metric_display,
                    "Score": value,
                    "RAG": status,
                }
            )

    # --- 透视 DataFrame / Pivot the DataFrame ---
    df = pd.DataFrame(scorecard_data)

    scorecard_pivot = df.pivot_table(
        index=["Category", "Metric"],
        columns="Model",
        values=["Score", "RAG"],
        aggfunc="first",
    )

    # 重新排序列以实现逻辑呈现 / Reorder columns for logical presentation
    scorecard_pivot = scorecard_pivot.swaplevel(0, 1, axis=1)

    # 确保模型和指标的顺序 / Ensure model and metric order
    model_order = list(models_config.keys())
    metric_order = ["Score", "RAG"]

    scorecard_pivot = scorecard_pivot.reindex(
        columns=pd.MultiIndex.from_product([model_order, metric_order])
    )

    # 按类别排序索引 / Sort index by Category
    scorecard_pivot = scorecard_pivot.sort_index(level='Category', sort_remaining=False)

    return scorecard_pivot


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