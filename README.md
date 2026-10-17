# 🧬 MolPredict: Molecular Property Prediction Pipeline

A production-grade, modular Python codebase and deep learning framework to predict molecular properties (such as LogP, solubility, binding affinity) from SMILES strings using RDKit and PyTorch.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

---

## 🌟 Architecture & Highlights

- **Data Preprocessing & Validation**:
  - Validates and canonicalizes SMILES via RDKit.
  - Cleans unparseable records and duplicates.
  - Stratified continuous splitting (Train / Validation / Test) and target standardization.
- **Feature Engineering**:
  - Circular Morgan Fingerprints (ECFP4 / configurable radius and bit vector length).
  - 2D Physicochemical Descriptors (MW, LogP, TPSA, HBD, HBA, Rotatable Bonds).
  - Automatic feature scaling to avoid leakage.
- **Deep Learning Architecture**:
  - Fully configurable PyTorch Multi-Layer Perceptron (MLP).
  - BatchNorm, Dropout, Cosine Annealing Learning Rate Schedule, Gradient Clipping.
- **Comprehensive Evaluation**:
  - RMSE, MAE, Coefficient of Determination ($R^2$), Pearson ($r$), and Spearman ($\rho$).

---

## 📁 Repository Structure

```
molpredict/
├── config/
│   └── config.yaml               # Hyperparameters & data paths
├── data/
│   └── sample_data.csv           # Benchmark SMILES & LogP dataset
├── molpredict/
│   ├── data/
│   │   ├── preprocessing.py      # Cleaning, validation, stratified splits
│   │   └── dataset.py            # PyTorch MolecularDataset
│   ├── features/
│   │   └── engineering.py        # Morgan fingerprints & descriptors
│   ├── models/
│   │   └── mlp.py                # Deep MLP architecture
│   ├── training/
│   │   └── trainer.py            # Train loop, validation & early stopping
│   └── evaluation/
│       └── metrics.py            # RMSE, MAE, R², Pearson correlation
├── scripts/
│   └── train.py                  # End-to-end training entry point
├── tests/
│   └── test_pipeline.py          # Pytest unit tests
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/prajwalkkhadka-cpu/molpredict.git
cd molpredict
pip install -r requirements.txt
pip install -e .
```

### 2. Run Training Pipeline

```bash
python scripts/train.py --config config/config.yaml
```

### 3. Run Unit Tests

```bash
pytest tests/ -v
```

---

## 👤 Author

**Prajwal Khadka** ([@prajwalkkhadka-cpu](https://github.com/prajwalkkhadka-cpu))

## 📄 License
This project is licensed under the MIT License.
