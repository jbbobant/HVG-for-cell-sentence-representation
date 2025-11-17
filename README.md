# CellSentence: Evaluating HVG Selection for Single-Cell Analysis

This repository contains a production-ready, configurable Python codebase for the research project:
**"Cell Sentences and Language Models for Single-Cell Analysis: Evaluating the Role of HVG Selection."**

---

## 1. Introduction & Research Motivation

This study builds on the framework of **AI-powered virtual cells (AIVC)**, exploring new ways to represent cellular heterogeneity. Traditional single-cell analysis relies on numerical gene expression values. Instead, we ask:

> **Can a cell's gene expression profile be treated as a language?**

Using the **Cell2Sentence (C2S)** framework, we convert ranked gene expression profiles into textual "sentences," enabling the use of Natural Language Processing (NLP) models. We leverage the **C2S-Pythia-410M Large Language Model** to perform downstream tasks.

### **Core Research Question**
A standard preprocessing step in scRNA-seq analysis is the selection of **Highly Variable Genes (HVGs)** (often top 2000). This project investigates:

> **What is the impact of using HVG-based sentences versus full-genome sentences on downstream analysis tasks?**

### **Key Findings**
Experiments benchmarked using **scEval** highlight:

- **Gene Expression Reconstruction:**  
  Cell2Sentence is nearly reversible. A simple linear model reconstructs normalized expression from ranks alone with:
  - **R² = 0.80–0.85**
  - **Pearson = 0.92**

- **Cell-Type Annotation:**  
  A simple MLP classifier achieves **state-of-the-art performance (Macro-F1 0.85–0.89)** using C2S embeddings.  
  Crucially, **HVG-based sentences perform as well as full-genome ones**, offering major efficiency gains.

This repository provides all tools necessary to reproduce and extend these findings.

---

## 2. Key Features

- **Modular & Scalable:** Fully refactored from notebooks into an object‑oriented Python codebase.
- **Config-Driven:** All hyperparameters and paths are controlled via YAML files.
- **Reproducible:** Global seeding via `src/utils/common.py`.

### **Two Core Tasks**
- **Cell-Type Annotation:** MLP classifier (PyTorch), evaluated with Accuracy and Macro-F1.
- **Gene Expression Reconstruction:** Linear regressor, evaluated with R², Pearson, and Spearman correlations.

- **Tested:** Includes pytest unit tests for data integrity.

---

## 3. Project Structure

```
cell_sentence_project/
│
├── configs/                # The control center. All parameters are in .yaml files.
│   ├── base_config.yaml    # Default settings for all tasks
│   └── experiments/        # (Optional) Configs for specific runs (e.g., pancreas.yaml)
│
├── data/
│   ├── raw/                # -> Place your raw .h5ad files here
│   └── processed/          # -> Processed .h5ad or .arrow files are saved here
│
├── src/                    # The core engine of the project.
│   ├── __init__.py
│   ├── utils/              # Logging, seeding, visualization, and config helpers.
│   │   ├── common.py       # seed_everything(), get_device(), load_config()
│   │   ├── logger.py       # setup_logger()
│   │   └── visualization.py # plot_reconstruction(), plot_confusion_matrix()
│   │
│   ├── data/               # Scanpy preprocessing & PyTorch data loading.
│   │   ├── preprocessing.py # Preprocessor class (QC, norm, HVG)
│   │   └── dataset.py      # CellDataset & ReconstructionDataset classes
│   │
│   ├── models/             # Neural Network architectures.
│   │   ├── classifier.py   # CellClassifier (MLP)
│   │   └── reconstructor.py # GeneReconstructor (Linear)
│   │
│   └── engine/             # Training and evaluation loops.
│       ├── trainer.py      # Trainer for classification (CrossEntropyLoss)
│       └── trainer_regression.py # Trainer for reconstruction (MSELoss)
│
├── scripts/                # Main entry points to run tasks.
│   ├── train_annotation.py    # -> Run this for cell-type annotation
│   └── run_reconstruction.py  # -> Run this for gene reconstruction
│
├── tests/                  # Pytest unit tests.
│   ├── conftest.py         # Mock data fixtures
│   └── test_data_pipeline.py # Tests for data loaders and splits
│
└── requirements.txt        # All dependencies
```

---

## 4. Data & Modeling Pipeline

This repository provides an automated pipeline from raw `.h5ad` to final benchmark results.

### **Step 1 — Configuration**
Edit `configs/base_config.yaml`:
- `paths.input_data`: path to raw data
- `preprocessing.use_hvg`: toggle HVG vs full genome
- `model`: architectures & hyperparameters
- `training`: batch size, epochs, early stopping

### **Step 2 — Data Ingestion**
Handled automatically by the **Preprocessor** class.

### **Step 3 — Quality Control**
QC follows the standard Scanpy procedure:
- Filter cells & genes
- Compute mitochondrial percentage
- Apply thresholds from config

### **Step 4 — Normalization**
`sc.pp.normalize_total` → `sc.pp.log1p(base=10)`  
Base-10 log is required for compatibility with Cell2Sentence.

### **Step 5 — HVG Selection**
If enabled, selects top `n_hvg` genes (Seurat v3 flavor).

### **Step 6 — Data Splitting**
`create_dataloaders` performs:
- Extraction of C2S embeddings
- Label encoding
- Stratified train/val/test splits

### **Step 7 — Model Training**
Two training engines:
- Classification: CrossEntropyLoss + Accuracy + Macro-F1
- Reconstruction: MSELoss + R² + Pearson + Spearman

Includes progress logging and early stopping.

### **Step 8 — Evaluation & Visualization**
Visual outputs include:
- Reconstruction scatter plots
- Confusion matrices

---

## 5. Installation

Clone the repository:
```bash
git clone https://github.com/your-username/cell_sentence_project.git
cd cell_sentence_project
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Place your raw `.h5ad` file into `data/raw/`.

---

## 6. Usage / Quickstart

Run all tasks from `scripts/`.

### **Cell-Type Annotation**
```bash
python scripts/train_annotation.py --config configs/base_config.yaml
```

### **Gene Expression Reconstruction**
```bash
python scripts/run_reconstruction.py --config configs/base_config.yaml
```

### **Override Data Path**
```bash
python scripts/train_annotation.py \
    --config configs/base_config.yaml \
    --data_path data/raw/pskin_hvg.h5ad
```

---

## 7. Testing

Install testing tools:
```bash
pip install pytest pytest-cov
```

Run all tests:
```bash
pytest
```

Check coverage:
```bash
pytest --cov=src
```

---

## Acknowledgements
My acknowledgements go to 
- Professor **Seiya Imoto**
- Professor **Yaozhong Zhang**
- Dr. **Yusri Dwi Heryanto**

for their insightful supervision through this project.
