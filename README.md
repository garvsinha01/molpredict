# 🧬 MolPredict: Deep Learning Molecular Property Prediction

A modular, production-grade Python & PyTorch deep learning framework for predicting molecular properties (LogP, Lipophilicity, Drug-Likeness) from chemical SMILES strings.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-23D96C.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## ⚡ Quickstart — Run the Interactive App Instantly!

No heavy setup required! Clone and run the colorful interactive terminal interface right away:

```bash
# 1. Clone repository
git clone https://github.com/prajwalkkhadka-cpu/molpredict.git
cd molpredict

# 2. Run the interactive terminal demo
python demo.py
```

### 🎯 Or Run Instant One-Line Predictions:

```bash
# Predict by drug name
python predict.py --name aspirin
python predict.py --name caffeine

# Predict by custom chemical SMILES
python predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

---

## 🖥️ Live Terminal Demo Preview

```
  ╔════════════════════════════════════════════════════════════════╗
  ║                                                                ║
  ║      🧬  MOLPREDICT : DEEP LEARNING MOLECULAR ENGINE  🧬       ║
  ║            Built with PyTorch • by prajwalkkhadka-cpu          ║
  ║                                                                ║
  ╚════════════════════════════════════════════════════════════════╝

  ── Processing Molecule: Aspirin
  SMILES String : CC(=O)Oc1ccccc1C(=O)O

  ┌──────────────────────────────────────────────────────────┐
  │ 📊 DEEP LEARNING PREDICTION REPORT                       │
  ├──────────────────────────────────────────────────────────┤
  │  • Predicted LogP (Octanol/Water) :   1.19               │
  │  • Experimental Known LogP        :   1.19               │
  │  • Absolute Prediction Error (MAE):   0.00               │
  │  • Estimated Molecular Weight     :  180.2 g/mol         │
  │  • Topological Polar Surface Area :   63.6 Å²            │
  │  • Rotatable Bonds Count          :      3               │
  └──────────────────────────────────────────────────────────┘

  💊 Lipinski's Rule of 5 (Drug-Likeness Assessment):
     ✅ PASS | Molecular Weight <= 500 Da     (Value: 180.2 Da)
     ✅ PASS | LogP (Lipophilicity) <= 5.0     (Value: 1.19)
     ✅ PASS | H-Bond Donors <= 5             (Value: 1)
     ✅ PASS | H-Bond Acceptors <= 10         (Value: 4)

  🎉 Excellent Drug-likeness! (0 violations of Rule of 5)
```

---

## 🏗️ Architecture & Modules

1. **`demo.py` & `predict.py`**:
   - Interactive terminal UI and CLI tool with real-time inference and Lipinski's Rule of 5 evaluation.
2. **`molpredict/data/preprocessing.py`**:
   - SMILES canonicalization, cleaning, and stratified train/validation/test splitting.
3. **`molpredict/features/engineering.py`**:
   - Circular Morgan Fingerprints (ECFP4 / 2048-bit bit-vectors) and 2D physicochemical descriptors.
4. **`molpredict/models/mlp.py`**:
   - Deep Multi-Layer Perceptron in PyTorch with Dropout, BatchNorm, and GELU activation.
5. **`molpredict/training/trainer.py`**:
   - AdamW optimizer, Cosine Annealing learning rate schedule, early stopping, and checkpointing.
6. **`molpredict/evaluation/metrics.py`**:
   - RMSE, MAE, Coefficient of Determination ($R^2$), and Pearson Correlation ($r$).

---

## 📁 Repository Structure

```
molpredict/
├── demo.py                       # 🚀 Interactive terminal application
├── predict.py                    # ⚡ One-liner CLI prediction tool
├── config/
│   └── config.yaml               # Model configuration & hyperparameters
├── data/
│   └── sample_data.csv           # Benchmark molecular dataset
├── molpredict/
│   ├── data/                     # Preprocessing & PyTorch dataset loader
│   ├── features/                 # Morgan fingerprints & RDKit descriptors
│   ├── models/                   # Deep Neural Network architecture
│   ├── training/                 # Optimization loop with early stopping
│   └── evaluation/               # Statistical regression metrics
├── scripts/
│   └── train.py                  # Full model training script
├── tests/
│   └── test_pipeline.py          # Pytest unit test suite
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🧪 Full Model Training

To retrain the neural network on custom datasets:

```bash
pip install -r requirements.txt
python scripts/train.py --config config/config.yaml
```

---

## 👤 Author

**Prajwal Khadka** ([@prajwalkkhadka-cpu](https://github.com/prajwalkkhadka-cpu))  
*B.Tech ECE (1st Year) | Python-First Developer*

---

## 📄 License

MIT License — feel free to explore, fork, and build upon it!
