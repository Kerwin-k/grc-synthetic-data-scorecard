import textwrap

import pandas as pd
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
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
RAG_COLORS = {'Green': '#90EE90', 'Amber': '#FFBF00', 'Red': '#F08080', 'N/A': '#D3D3D3'}


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

    # 计算可持续性基线 / Calculate sustainability baselines
    all_co2 = [m.get('co2_eq_kg', np.inf) for m in all_metrics.values() if m.get('co2_eq_kg') is not None]
    min_co2 = min(all_co2) if all_co2 else np.inf

    all_time = [m.get('training_time_sec', np.inf) for m in all_metrics.values() if
                m.get('training_time_sec') is not None]
    min_time = min(all_time) if all_time else np.inf

    for model_name, metrics in all_metrics.items():

        # 聚合公平性指标 / Aggregate fairness metrics
        fair_metrics = [v for k, v in metrics.items() if k.startswith('fairness_') and pd.notna(v)]
        avg_fairness = np.nanmean(fair_metrics) if fair_metrics else np.nan
        metrics['avg_fairness'] = avg_fairness  # 添加以便映射 / Add for mapping

        # --- 定义记分卡行 / Define scorecard rows ---
        for key, config in METRIC_MAP.items():
            if key not in metrics:
                continue

            value = metrics.get(key)
            status = "N/A"

            if config['thresholds'] is not None:
                # 使用绝对阈值 / Use absolute thresholds
                status = _get_rag_status(key, value, config['thresholds'])
            elif pd.notna(value):
                # 使用相对 RAG 状态进行可持续性排名 / Use relative RAG for sustainability
                if key == 'co2_eq_kg':
                    if value <= (min_co2 * 1.1):
                        status = "Green"  # 10% 容差
                    elif value <= (min_co2 * 2.0):
                        status = "Amber"  # 2倍以内
                    else:
                        status = "Red"
                elif key == 'training_time_sec':
                    if value <= (min_time * 1.1):
                        status = "Green"
                    elif value <= (min_time * 2.0):
                        status = "Amber"
                    else:
                        status = "Red"

            # [!!] 修正: 拆分英文显示名称 / Fix: Split English display name
            category, metric_display = config['display_name'].split(': ')

            scorecard_data.append({
                'Model': model_name,
                'Category': category,
                'Metric': metric_display,
                'Score': value,
                'RAG': status
            })

    # --- 透视 DataFrame / Pivot the DataFrame ---
    df = pd.DataFrame(scorecard_data)

    scorecard_pivot = df.pivot_table(
        index=['Category', 'Metric'],
        columns='Model',
        values=['Score', 'RAG'],aggfunc = 'first')

    # 重新排序列以实现逻辑呈现 / Reorder columns for logical presentation
    scorecard_pivot = scorecard_pivot.swaplevel(0, 1, axis=1)

    # 确保模型和指标的顺序 / Ensure model and metric order
    model_order = list(models_config.keys())
    metric_order = [ 'Score', 'RAG']

    # [!!] 修正: 移除 'level=0' 以修复 'TypeError:... ambiguous' [1]
    # [!!] Fix: Remove 'level=0' to fix 'TypeError:... ambiguous' [1]
    scorecard_pivot = scorecard_pivot.reindex(
        columns=pd.MultiIndex.from_product([model_order, metric_order])
    )

    # 按类别排序索引 / Sort index by Category
    scorecard_pivot = scorecard_pivot.sort_index(level='Category', sort_remaining=False)

    # 格式化分数使其更易读 / Format scores for readability
    def format_value(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.4f}"
        return x

    for model in model_order:
        if (model, 'Score') in scorecard_pivot.columns:
            scorecard_pivot = scorecard_pivot.apply(format_value)

    return scorecard_pivot


def save_scorecard_as_image(scorecard_df, output_path: str | None = None):
    """
    GRC 记分卡可视化：将 DataFrame 保存为带 RAG 颜色、标题和图例的 PNG 图像。
    GRC Scorecard visualisation: save the DataFrame as a PNG image with RAG colours, title and legend.
    """

    # 如果未指定输出路径，则自动定位到项目根目录下的 results/grc_scorecard.png
    # If no output path is provided, automatically use results/grc_scorecard.png under the project root.
    if output_path is None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
        output_path = os.path.join(PROJECT_ROOT, "results", "grc_scorecard.png")

    logging.info(f"Visualizing GRC Scorecard as image: {output_path}...")

    # 复制一份，以免修改到原始 DataFrame
    # Work on a copy to avoid mutating the original DataFrame.
    plot_df = scorecard_df.copy()

    # ---------- 提取 RAG 层用于着色 / Extract the RAG layer for colouring ----------
    try:
        rag_df = plot_df.xs("RAG", level=1, axis=1)
    except KeyError:
        logging.error("Could not find 'RAG' columns in scorecard. Skipping colouring.")
        rag_df = pd.DataFrame()

    # 只保留 Score 层用于展示数值
    # Keep only the 'Score' layer for displaying numeric values.
    try:
        plot_df = plot_df.xs("Score", level=1, axis=1)
    except KeyError:
        logging.error("Could not find 'Score' columns in scorecard. Cannot generate image.")
        return

    # 将层级索引中的 (Category, Metric) 还原为普通列，方便遍历
    # Reset multi-index (Category, Metric) to columns for easier iteration.
    plot_df.reset_index(inplace=True)

    # ---------- 数值格式化，便于论文展示 / Format numeric scores for display ----------
    # 只对模型得分列做格式化，不动 Category / Metric
    # Only format generator score columns; keep Category/Metric as they are.
    score_cols = [c for c in plot_df.columns if c not in ["Category", "Metric"]]

    def _format_score(x):
        # 针对浮点数保留三位小数，其它类型原样返回
        # Keep three decimals for floats; leave other types unchanged.
        if isinstance(x, (int, float)):
            return f"{x:.3f}"
        return x

    plot_df[score_cols] = plot_df[score_cols].applymap(_format_score)

    # ---------- 根据行列数动态调整画布大小 / Dynamically size the figure ----------
    n_rows, n_cols = plot_df.shape
    fig_width = max(13, 1.3 * (n_cols + 1))      # 稍微收窄一点，更像论文版式 / slightly narrower
    fig_height = max(8.0, 0.55 * (n_rows + 8))   # 保持横向论文比例 / keep a landscape, paper-like ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # 将表格整体下移，为标题、解释和图例保留空间
    # Move the table down to leave room for headings, interpretation text and legend.
    # [left, bottom, width, height] in figure coordinates
    ax.set_position([0.05, 0.18, 0.9, 0.60])

    # ---------- 创建表格 / Create the table ----------
    # 使用 cellText + colLabels，避免自动添加 0,1,2... 行号
    # Use cellText + colLabels to avoid automatic row indices (0,1,2,...).
    tab = ax.table(
        cellText=plot_df.values,
        colLabels=plot_df.columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.auto_set_column_width(col=list(range(n_cols)))
    tab.scale(1.0, 1.15)  # 稍微拉高行距 / Slightly stretch row height

    # ---------- 应用 RAG 颜色逻辑 / Apply RAG colour logic ----------
    for (row, col), cell in tab.get_celld().items():
        # 表头行和前两列（Category / Metric）统一灰底并加粗
        # Header row and the first two columns get a grey background and bold text.
        if row == 0 or col < 2:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(weight="bold")
            # 左侧两列靠左显示并允许换行
            # Left columns left-aligned with wrapping.
            if row > 0 and col < 2:
                cell.set_text_props(weight="bold", ha="left", wrap=True)
            continue

        # 其余单元格根据 RAG 值上色
        # Other cells use the RAG value for background colour.
        try:
            model_name = plot_df.columns[col]
            metric_cat = plot_df.iloc[row - 1][["Category", "Metric"]]
            rag_value = rag_df.loc[(metric_cat["Category"], metric_cat["Metric"]), model_name]
            color = RAG_COLORS.get(rag_value, "#FFFFFF")
            cell.set_facecolor(color)
        except (IndexError, KeyError) as e:
            logging.warning(f"Could not set colour for cell ({row}, {col}): {e}")
            cell.set_facecolor("#FFFFFF")

    # ---------- 顶部标题与说明（仅英文）/ Top title and explanations (English only) ----------
    # 根据画布宽度设定换行宽度，使长句在窄屏上也能美观换行
    # Wrap width depends on figure width so long sentences wrap nicely.
    wrap_width = max(70, int(fig_width * 4.0))

    # 主标题：简洁正式的标题
    # Main title: concise, formal title for the thesis.
    title_text = fig.text(
        0.5,
        0.96,
        "GRC Quality & Risk Scorecard",
        ha="center",
        va="top",
        fontsize=18,
        weight="bold",
    )

    # 副标题：一句话说明比较什么
    # Subtitle: one-line description of what is being compared.
    subtitle = textwrap.fill(
        "Comparison of synthetic tabular data generators across quality, risk, "
        "sustainability, and utility metrics.",
        width=wrap_width,
    )
    subtitle_text = fig.text(
        0.5,
        0.915,
        subtitle,
        ha="center",
        va="top",
        fontsize=11,
        linespacing=1.25,
    )
    subtitle_text.set_wrap(True)

    # 解释文本：左对齐的多行短句，更接近高水平论文风格
    # Interpretation block: left-aligned multi-line text, closer to a high-quality paper layout.
    interpretation_lines = [
        "Interpretation:",
        "• Rows represent evaluation metrics and columns represent data generators.",
        "• Numerical values are normalised scores: higher scores are preferred for "
        "quality and utility, whereas lower values are preferred for privacy, fairness, "
        "and sustainability (CO₂ emissions and training time).",
        "• Background colours summarise governance risk levels: green = low risk, "
        "amber = requires review, red = high risk, grey = metric not available.",
    ]
    interpretation_text = "\n".join(
        [textwrap.fill(line, width=wrap_width) for line in interpretation_lines]
    )

    # 让解释文本在表格上方、靠左排版，看起来像“图内说明”
    # Place the interpretation text above the table, left-aligned, similar to in-figure notes.
    guidance_text = fig.text(
        0.06,                # 左边距与表格齐平 / align with table left
        0.86,                # 稍高于表格顶部 / just above the table
        interpretation_text,
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.35,
    )
    guidance_text.set_wrap(True)

    # ---------- 图例：解释颜色含义 / Legend: explain colour meaning ----------
    green_patch = mpatches.Patch(
        color=RAG_COLORS["Green"],
        label="Good / Low-Risk / Best-in-Class",
    )
    amber_patch = mpatches.Patch(
        color=RAG_COLORS["Amber"],
        label="Warning / Requires Review",
    )
    red_patch = mpatches.Patch(
        color=RAG_COLORS["Red"],
        label="Bad / High-Risk / Worst-in-Class",
    )
    na_patch = mpatches.Patch(
        color=RAG_COLORS["N/A"],
        label="N/A (e.g., Metric Failed)",
    )

    legend = fig.legend(
        handles=[green_patch, amber_patch, red_patch, na_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )

    # ---------- 保存图片，确保标题/说明/图例都不会被裁剪 / Save image without cutting text ----------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(legend, title_text, subtitle_text, guidance_text),
    )
    logging.info(f"GRC Scorecard image saved to {output_path}")


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
        scorecard.to_csv(GRC_SCORECARD_PATH)
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