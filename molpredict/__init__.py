"""
MolPredict — Molecular Property Prediction with Deep Learning
=============================================================
A modular, production-ready ML pipeline for predicting molecular properties
from SMILES strings using Morgan fingerprints and deep neural networks.

Author : Prajwal Khadka (@prajwalkkhadka-cpu)
License: MIT
"""

__version__ = "1.0.0"
__author__  = "Prajwal Khadka"

from molpredict.data.preprocessing import DataPreprocessor
from molpredict.features.engineering import FeatureEngineer
from molpredict.models.mlp import MolecularMLP
from molpredict.models.gnn import MolecularGNN
from molpredict.training.trainer import Trainer
from molpredict.evaluation.metrics import MolecularMetrics

__all__ = [
    "DataPreprocessor",
    "FeatureEngineer",
    "MolecularMLP",
    "MolecularGNN",
    "Trainer",
    "MolecularMetrics",
]
