"""
Unit Tests for MolPredict Pipeline
==================================
"""

import numpy as np
import pytest
import torch
from molpredict.features.engineering import FeatureEngineer
from molpredict.models.mlp import MolecularMLP
from molpredict.evaluation.metrics import MolecularMetrics


def test_morgan_fingerprint_generation():
    engineer = FeatureEngineer(radius=2, n_bits=1024, use_descriptors=False)
    smiles = "CCO"  # Ethanol
    fp = engineer.smiles_to_morgan_fingerprint(smiles)
    assert fp is not None
    assert len(fp) == 1024
    assert fp.sum() > 0


def test_descriptor_computation():
    engineer = FeatureEngineer(use_descriptors=True)
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    desc = engineer.compute_descriptors(smiles)
    assert desc is not None
    assert len(desc) == len(engineer.descriptor_names)


def test_mlp_forward_pass():
    model = MolecularMLP(input_dim=128, hidden_dims=[64, 32], dropout=0.1)
    dummy_input = torch.randn(8, 128)
    output = model(dummy_input)
    assert output.shape == torch.Size([8])


def test_metrics_computation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    res = MolecularMetrics.compute_all(y_true, y_pred)
    assert "rmse" in res
    assert "mae" in res
    assert "r2" in res
    assert res["r2"] > 0.95
