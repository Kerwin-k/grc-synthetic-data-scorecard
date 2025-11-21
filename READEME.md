# SynTab-GRC: A Multi-Dimensional Evaluation Framework for Synthetic Tabular Data

**SynTab-GRC** is a comprehensive evaluation system designed to bridge the "operational impasse" between data science innovation and governance oversight. 

This framework benchmarks synthetic data generators (Statistical, GAN-based, and VAE-based) across five responsible AI dimensions: **Fidelity, Utility, Privacy, Fairness, and Sustainability**. It features a novel **GRC Scorecard** mechanism that translates complex technical metrics into actionable, audit-ready risk signals (Red/Amber/Green).

> **Context:** This repository contains the implementation code for the Master's Thesis: *"Responsible AI in Practice: A Multi-Dimensional Evaluation Framework for Synthetic Tabular Data."*

---

## 🚀 Key Features

### 1. Multi-Dimensional Benchmarking
Beyond the traditional Fidelity-Utility-Privacy (FUP) trilemma, this framework integrates:
* **Algorithmic Fairness:** Measures bias propagation using *Statistical Parity Difference (SPD)*.
* **Computational Sustainability:** Tracks CO₂ emissions and energy consumption via *CodeCarbon*, implementing a tiered tracking logic (GPU priority with CPU fallback).

### 2. Resource-Aware & Robust Execution
* **Dynamic Stratified Sampling:** Automatically detects hardware constraints (RAM/GPU) and downsamples large datasets to a manageable size (e.g., 50k rows) while strictly preserving class distributions.
* **OOM Protection:** Prevents training crashes on standard hardware when using heavy deep learning models (CTGAN, TVAE).

### 3. Handling Mode Collapse (Imbalanced Data Support)
* **Hybrid Resampling Strategy:** Implements a conditional upsampling mechanism in the training pipeline. This mitigates "mode collapse" in highly imbalanced datasets (e.g., Financial Fraud), ensuring that generative models capture minority class signals to preserve downstream Utility (F1-Score) and Fairness metrics.

### 4. Governance-Oriented Visualization
* **Automated Scorecard:** Converts raw floating-point metrics into categorical risk levels based on configurable, context-aware thresholds (defined in `config.py`).

---

## 📂 Repository Structure

```bash
SynTab-GRC/
├── data/
│   ├── raw/                 # Place original CSV files here (e.g., application_train.csv)
│   └── processed/           # Cleaned and sampled data (auto-generated)
├── metadata/                # SDV metadata JSON files (auto-generated)
├── models/                  # Saved generator models (.pkl)
├── results/
│   ├── emissions/           # Carbon footprint logs (CodeCarbon)
│   ├── synthetic_data/      # Generated synthetic datasets
│   ├── metrics_report.json  # Raw calculated metrics
│   └── grc_scorecard.png    # Final visualized scorecard
├── src/
│   ├── config.py            # Central Control Panel (Thresholds, Paths, Models)
│   ├── data_loader.py       # Resource-aware ingestion & sampling
│   ├── model_trainer.py     # Training loop with resampling & carbon tracking
│   ├── evaluation_engine.py # 5-dimensional metric computation
│   ├── grc_translator.py    # RAG logic application & visualization
│   └── utils.py             # Helper functions
├── main.py                  # Pipeline orchestrator (Entry Point)
├── requirements.txt         # Dependencies
└── README.md
```
## 🛠️ Installation & Prerequisites

### Requirements
* Python >= 3.8
* CUDA-enabled GPU (Recommended for CTGAN/TVAE training, but works on CPU)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/SynTab-GRC.git](https://github.com/YourUsername/SynTab-GRC.git)
    cd SynTab-GRC
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies include: `sdv`, `sdmetrics`, `codecarbon`, `fairlearn`, `scikit-learn`, `pandas`, `pynvml`.*

---

## ⚙️ Configuration

This project uses a **Configuration-Driven Architecture**. You do not need to modify the core logic code to switch datasets or adjust risk policies.

All settings are managed in `src/config.py`:

1.  **Dataset Selection:**
    Modify `DatasetConfig` to point to your target CSV.
    ```python
    class DatasetConfig:
        RAW_DATA_FILE = "application_train.csv"  # Target file
        TARGET_COLUMN = "TARGET"                 # Target variable for prediction
        POSITIVE_LABEL = 1                       # Minority class label
        SENSITIVE_FEATURES = ["CODE_GENDER"]     # For fairness assessment
        SAMPLE_SIZE = 50000                      # Effective training size
    ```

2.  **Risk Thresholds (RAG Logic):**
    Adjust `RAGThresholdConfig` based on the domain context (e.g., Finance vs. Healthcare).
    ```python
    class RAGThresholdConfig:
        # Adjust thresholds based on task difficulty (e.g., Imbalanced vs Balanced)
        UTILITY_TSTR_F1 = {"green": 0.40, "amber": 0.20} 
        PRIVACY_MIA = {"green": 0.55, "amber": 0.65}
    ```

---

## 🚀 Usage

### Running the Full Pipeline
To execute the complete workflow (Data Ingestion → Training → Evaluation → Scorecard Generation):

```bash
python main.py
```
### Workflow Details
1.  **Ingestion:** The system loads the raw data, checks system resources, and applies stratified sampling if necessary (`data_loader.py`).
2.  **Training:** Three models (GaussianCopula, CTGAN, TVAE) are trained. If class imbalance is detected, the system applies **stratified upsampling** to prevent mode collapse (`model_trainer.py`).
3.  **Evaluation:** Synthetic data is evaluated against real data using JSD, NMI, TSTR (F1), MIA (AUC), and SPD (`evaluation_engine.py`).
4.  **Reporting:** Results are aggregated, mapped to RAG colors, and exported as a PNG image (`grc_translator.py`).

---

## 📊 Methodology & Metrics

### 1. Generative Models Benchmarked
* **Gaussian Copula:** Statistical baseline (Multivariate covariance).
* **CTGAN:** Conditional Generative Adversarial Network (Deep Learning).
* **TVAE:** Tabular Variational Autoencoder (Deep Learning).

### 2. Evaluation Dimensions
| Dimension | Metric | Description |
| :--- | :--- | :--- |
| **Quality** | JSD & NMI | Measures marginal distribution (Shape) and correlation (Structure) retention. |
| **Utility** | TSTR F1 | *Train-Synthetic-Test-Real*. Evaluates if synthetic data retains predictive signals. |
| **Privacy** | Adversarial AUC | Measures how easily a discriminator can distinguish synthetic from real records. |
| **Fairness** | SPD | *Statistical Parity Difference*. Checks if algorithmic bias is amplified. |
| **Sustainability** | CO₂eq (kg) | Environmental cost of training, tracked via CodeCarbon. |

### 3. Note on Resampling
For highly imbalanced datasets (e.g., Home Credit Default Risk), this framework employs a **Fidelity-Utility Trade-off Mechanism**. It minimally upsamples the minority class during training to ensure the model learns risk signals (Utility), acknowledging a controlled trade-off in statistical distribution fidelity.

---

## 📄 Outputs

After execution, check the `results/` folder:
* **`grc_scorecard.png`**: The visual artifact summarizing model performance for stakeholders.
* **`metrics_report.json`**: Full technical logs containing raw float values.
* **`emissions/`**: CSV logs detailed energy consumption (RAM, CPU, GPU).

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🖊️ Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{SynTabGRC2025,
  author = {Liu Kun},
  title = {Responsible AI in Practice: A Multi-Dimensional Evaluation Framework for Synthetic Tabular Data},
  school = {UCSI University},
  year = {2025}
}