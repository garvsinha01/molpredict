"""
Feature Engineering Module
==========================
Extracts molecular representations from SMILES strings using RDKit:
  1. Morgan Fingerprints (ECFP / circular fingerprints)
  2. Physical-chemical molecular descriptors (MW, LogP, TPSA, HBD, HBA, etc.)
  3. Feature normalization and combination
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Computes Morgan fingerprints and 2D molecular descriptors.

    Parameters
    ----------
    radius : int
        Morgan fingerprint radius (2 = ECFP4, 3 = ECFP6). Default is 2.
    n_bits : int
        Bit vector size for Morgan fingerprints. Default is 2048.
    use_descriptors : bool
        Whether to calculate 2D physicochemical descriptors. Default is True.
    normalize_descriptors : bool
        Whether to scale descriptors using StandardScaler. Default is True.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_descriptors: bool = True,
        normalize_descriptors: bool = True,
    ) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self.use_descriptors = use_descriptors
        self.normalize_descriptors = normalize_descriptors
        self.descriptor_scaler = StandardScaler()
        self._is_scaler_fitted = False

        # Selected stable 2D descriptors
        self.descriptor_names = [
            "MolWt",
            "MolLogP",
            "NumHDonors",
            "NumHAcceptors",
            "TPSA",
            "NumRotatableBonds",
            "NumAromaticRings",
            "FractionCSP3",
            "RingCount",
        ]

    def smiles_to_morgan_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """Convert a SMILES string to Morgan fingerprint bit vector."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.radius, nBits=self.n_bits)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def compute_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate specified 2D chemical descriptors for a molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        values = []
        for name in self.descriptor_names:
            func = getattr(Descriptors, name, None)
            if func is not None:
                try:
                    val = float(func(mol))
                except Exception:
                    val = 0.0
            else:
                val = 0.0
            values.append(val)
        return np.array(values, dtype=np.float32)

    def extract_features(
        self,
        smiles_list: List[str],
        fit_scaler: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract fingerprint and descriptor features for a list of SMILES.

        Parameters
        ----------
        smiles_list : list of str
            List of valid canonical SMILES.
        fit_scaler : bool
            Fit the descriptor scaler (set True for train split only).

        Returns
        -------
        Tuple[np.ndarray, list of int]
            Combined feature matrix and indices of successfully parsed molecules.
        """
        fps = []
        descs = []
        valid_indices = []

        for idx, smi in enumerate(smiles_list):
            fp = self.smiles_to_morgan_fingerprint(smi)
            if fp is None:
                continue

            fps.append(fp)
            valid_indices.append(idx)

            if self.use_descriptors:
                desc = self.compute_descriptors(smi)
                descs.append(desc if desc is not None else np.zeros(len(self.descriptor_names), dtype=np.float32))

        X_fp = np.array(fps, dtype=np.float32)

        if not self.use_descriptors:
            return X_fp, valid_indices

        X_desc = np.array(descs, dtype=np.float32)

        if self.normalize_descriptors:
            if fit_scaler:
                X_desc = self.descriptor_scaler.fit_transform(X_desc)
                self._is_scaler_fitted = True
            elif self._is_scaler_fitted:
                X_desc = self.descriptor_scaler.transform(X_desc)

        X_combined = np.hstack([X_fp, X_desc])
        return X_combined, valid_indices
