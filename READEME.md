# SynTab-GRC: A Multi-Dimensional Governance Scorecard for Synthetic Tabular Data

The project implements an end-to-end evaluation pipeline for synthetic **tabular** data generators (e.g. GaussianCopula, CTGAN, TVAE).  
It translates technical metrics into a **GRC-oriented scorecard** that can be read by non-technical governance, risk and compliance (GRC) stakeholders.

---

## 1. Overview

The toolkit provides:

1. **Data ingestion & preparation**  
   - Load a configured raw CSV dataset.  
   - Apply lightweight cleaning and column dropping.  
   - Automatically infer SDV `SingleTableMetadata`.

2. **Synthetic data generation with sustainability tracking**  
   - Train multiple SDV single-table synthesizers (GaussianCopula, CTGAN, TVAE).  
   - Generate synthetic datasets with 1:1 row counts.  
   - Track **CO₂ emissions and energy consumption** using `codecarbon`.

3. **Multi-dimensional quantitative evaluation**  
   - **Fidelity / Quality**: distributional similarity and correlation structure (JSD-based scores, NMI) using `sdmetrics`.  
   - **Utility**: Train-on-Synthetic, Test-on-Real (TSTR) downstream ML performance (F1, AUC) using `scikit-learn`.  
   - **Privacy risk**: membership-inference-style risk proxy (MIA AUC) from `sdmetrics`.
   - **Fairness**: demographic parity and equalized odds differences across sensitive attributes using `fairlearn`.
   - **Sustainability**: CO₂ emissions (kg CO₂eq) and training time (seconds) from `codecarbon`.

4. **GRC translation & reporting**
   - Map raw metrics to **RAG (Red / Amber / Green)** ratings using configurable thresholds.
   - Aggregate results into a **GRC scorecard DataFrame**.
   - Export as:
     - `results/metrics_report.json` – raw metrics for each model.
     - `results/grc_scorecard.csv` – tabular scorecard.
     - `results/grc_scorecard.png` – publication-quality scorecard figure.

The entire experiment is orchestrated through a single entry point (`main.py`) and a single configuration file (`src/config.py`).

---

## 2. Repository layout

Expected project structure (after unpacking this project):

```text
thesis_project/
├── data/
│   ├── raw/
│   │   └── adult.csv              # Example dataset (not distributed in this repo)
│   └── processed/                 # Auto-generated cleaned dataset(s)
├── metadata/                      # Auto-generated SDV metadata JSONs
├── models/                        # Saved synthesizer objects (optional)
├── results/
│   ├── synthetic_data/            # Generated synthetic tables
│   ├── emissions/                 # CodeCarbon emission CSV files
│   ├── metrics_report.json        # Raw metric dictionary (per model)
│   ├── grc_scorecard.csv          # Final GRC scorecard (tabular)
│   └── grc_scorecard.png          # Final GRC scorecard (figure)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Single control panel for the framework
│   ├── data_loader.py             # Load & clean raw data, infer metadata
│   ├── model_trainer.py           # Train SDV models + CodeCarbon tracking
│   ├── evaluation_engine.py       # Fidelity / Utility / Privacy / Fairness engine
│   └── grc_translator.py          # GRC mapping + scorecard generation & plotting
├── main.py                        # Orchestrates the full 5-step pipeline
├── requirements.txt
└── README.md
```
> **Note:** The raw dataset (e.g. `adult.csv` in `data/raw/`) is **not** distributed with this repository due to licensing and size.
> You must place your own dataset there or update the configuration to point to a different file.

---

## 3. Installation

### 3.1. Python version

The project was developed and tested with **Python 3.10–3.11**.

### 3.2. Create and activate a virtual environment

**Windows (PowerShell):**

```bash
cd thesis_project

python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux:**

```bash
cd thesis_project

python -m venv .venv
source .venv/bin/activate
```

### 3.3. Install dependencies
```bash
pip install -r requirements.txt
```
## 4. Quickstart: run the full pipeline
### 1. Place a raw CSV dataset
Copy your tabular dataset into data/raw/, for example:

````
data/raw/adult.csv
````

### 2. Update configuration in src/config.py
In DatasetConfig, set at least:
````python
class DatasetConfig:
    RAW_DATA_FILE = "adult.csv"      # file name in data/raw/
    TARGET_COLUMN = "income"         # label column used for TSTR & fairness
    POSITIVE_LABEL = ">50K"          # the "positive" class for F1
    SENSITIVE_FEATURES = ["sex", "race"]  # protected attributes for fairness
    COLS_TO_DROP = ["fnlwgt", "education"]  # IDs / weights / redundant cols
````
If you use a different dataset, adjust these fields accordingly.

### 3. Run the pipeline
From the project root (with the virtual environment activated):
````bash
python main.py
````
   The script will execute the following steps:

   1. Ingest & preprocess the raw dataset (`data_loader.py`).  
   2. Train synthetic data generators & track sustainability (`model_trainer.py`).  
   3. Evaluate quality, utility, privacy, and fairness (`evaluation_engine.py`).  
   4. Translate metrics into a GRC scorecard (`grc_translator.py`).  
   5. Export the scorecard as CSV and PNG.

4. **Inspect outputs**

   - Check the logs in the console.  
   - Key artefacts are saved under `results/`:
     - `metrics_report.json`  
     - `grc_scorecard.csv`  
     - `grc_scorecard.png`

---

## 5. Configuration details (`src/config.py`)

`src/config.py` is the **single control panel** for the framework.

### 5.1. Paths – `PathConfig`

```python
class PathConfig:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    SYNTH_DIR = os.path.join(RESULTS_DIR, "synthetic_data")
    EMISSIONS_DIR = os.path.join(RESULTS_DIR, "emissions")

    METRICS_REPORT_PATH = os.path.join(RESULTS_DIR, "metrics_report.json")
    GRC_SCORECARD_CSV_PATH = os.path.join(RESULTS_DIR, "grc_scorecard.csv")
    GRC_SCORECARD_IMG_PATH = os.path.join(RESULTS_DIR, "grc_scorecard.png")
```
Usually you can keep these defaults.

### 5.2. Dataset – `DatasetConfig`

Key fields to adapt when switching datasets:

```python
class DatasetConfig:
    RAW_DATA_FILE = "adult.csv"
    TARGET_COLUMN = "income"
    POSITIVE_LABEL = ">50K"
    SENSITIVE_FEATURES = ["sex", "race"]
    COLS_TO_DROP = ["fnlwgt", "education"]

    RAW_PATH = os.path.join(PathConfig.RAW_DIR, RAW_DATA_FILE)
    CLEAN_FILE = RAW_DATA_FILE.replace(".csv", "_clean.csv")
    PROCESSED_PATH = os.path.join(PathConfig.PROCESSED_DIR, CLEAN_FILE)
    META_FILE = RAW_DATA_FILE.replace(".csv", "_metadata.json")
    METADATA_PATH = os.path.join(PathConfig.METADATA_DIR, META_FILE)
```

When switching to a new dataset:

1. Place `my_dataset.csv` in `data/raw/`.  
2. Set `RAW_DATA_FILE = "my_dataset.csv"`.  
3. Update `TARGET_COLUMN`, `POSITIVE_LABEL`, `SENSITIVE_FEATURES`, and `COLS_TO_DROP`.

### 5.3. Models – `MODELS_CONFIG`

```python
MODELS_CONFIG = {
    "GaussianCopula": {
        "class": GaussianCopulaSynthesizer,
        "params": {}
    },
    "CTGAN": {
        "class": CTGANSynthesizer,
        "params": {"epochs": 5}
    },
    "TVAE": {
        "class": TVAESynthesizer,
        "params": {"epochs": 5}
    }
}
```

You can increase `epochs` for more realistic experiments or remove/add models as needed.

---

## 6. Outputs

After a successful run, the main artefacts are:

- **`results/metrics_report.json`** – nested dictionary with all quantitative metrics per model.  
- **`results/grc_scorecard.csv`** – tabular GRC scorecard.  
- **`results/grc_scorecard.png`** – publication-quality figure for the thesis and GRC reports.  
- **`results/emissions/*.csv`** – detailed emission logs from `codecarbon`.

---

## 7. Reproducing the thesis experiments

To reproduce the experiments reported in the thesis:

1. Obtain the original datasets (e.g. UCI Adult).  
2. Place them in `data/raw/` with the expected file names.  
3. Ensure `DatasetConfig` matches the column names and labels.  
4. Run `python main.py`.  
5. Archive the resulting `results/` directory.

---

## 8. License & citation

The code is provided for academic and research purposes.

If you use this repository, please cite the MSc thesis:

> *Liu Kun*, **"Responsible AI in Practice: A Multi-Dimensional Evaluation Framework for Synthetic Tabular Data"**
