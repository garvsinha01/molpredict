"""
PyTorch Dataset for Molecular Data
====================================
Wraps pre-computed feature matrices and target arrays into a
``torch.utils.data.Dataset`` for use with DataLoader.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MolecularDataset(Dataset):
    """
    PyTorch Dataset for molecular property prediction.

    Stores feature vectors (fingerprints + descriptors) and target
    values as tensors, ready for batched training.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Pre-computed feature matrix (one row per molecule).
    targets : np.ndarray, shape (N,)
        Target property values (normalized or raw).
    smiles : list[str], optional
        Original SMILES strings; stored for traceability.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        smiles: list[str] | None = None,
    ) -> None:
        if len(features) != len(targets):
            raise ValueError(
                f"Features ({len(features)}) and targets ({len(targets)}) "
                "must have the same length."
            )

        # Convert to float32 tensors for PyTorch compatibility
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(targets,  dtype=torch.float32)
        self.smiles   = smiles or [""] * len(features)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict:
        """
        Return a single sample as a dictionary.

        Returns
        -------
        dict with keys:
          - "features" : float32 tensor of shape (D,)
          - "target"   : float32 scalar tensor
          - "smiles"   : SMILES string
        """
        return {
            "features": self.features[idx],
            "target":   self.targets[idx],
            "smiles":   self.smiles[idx],
        }

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vector."""
        return self.features.shape[1]


class GraphMolecularDataset(Dataset):
    """
    PyTorch Dataset for GNN-based molecular property prediction.

    Each sample is a molecular graph represented as:
      - node features (atom features)
      - edge index (adjacency in COO format)
      - edge features (bond features)

    Parameters
    ----------
    graphs : list[dict]
        Each element is a dict with keys:
          "node_features", "edge_index", "edge_features", "target", "smiles"
    """

    def __init__(self, graphs: list[dict]) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict:
        g = self.graphs[idx]
        return {
            "node_features": torch.tensor(g["node_features"], dtype=torch.float32),
            "edge_index":    torch.tensor(g["edge_index"],    dtype=torch.long),
            "edge_features": torch.tensor(g["edge_features"], dtype=torch.float32),
            "target":        torch.tensor(g["target"],        dtype=torch.float32),
            "smiles":        g["smiles"],
        }
