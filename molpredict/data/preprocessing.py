"""
Data Preprocessing Module
=========================
Handles loading, validation, cleaning, and splitting of molecular datasets.

Key responsibilities:
  - Load CSV data with SMILES strings and target properties
  - Validate SMILES strings using RDKit
  - Remove duplicate and invalid molecules
  - Stratified train/validation/test splitting
  - Target property normalization (StandardScaler)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    End-to-end data preprocessing pipeline for molecular datasets.

    Parameters
    ----------
    smiles_col : str
        Column name containing SMILES strings.
    target_col : str
        Column name containing the target property to predict.
    test_size : float
        Fraction of data reserved for the test set (default 0.15).
    val_size : float
        Fraction of training data reserved for validation (default 0.15).
    random_seed : int
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        smiles_col: str = "smiles",
        target_col: str = "logP",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        self.smiles_col  = smiles_col
        self.target_col  = target_col
        self.test_size   = test_size
        self.val_size    = val_size
        self.random_seed = random_seed

        # Fitted during preprocessing; used later for inverse-transform
        self.target_scaler = StandardScaler()
        self._is_fitted    = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> pd.DataFrame:
        """
        Load a CSV dataset and return a raw DataFrame.

        Parameters
        ----------
        path : str or Path
            File path to the CSV dataset.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame with at minimum `smiles_col` and `target_col`.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If required columns are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)
        logger.info("Loaded %d rows from %s", len(df), path)

        # Validate required columns
        for col in [self.smiles_col, self.target_col]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Available: {list(df.columns)}"
                )

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by:
          1. Removing rows with missing SMILES or target values.
          2. Validating SMILES strings via RDKit.
          3. Canonicalizing valid SMILES.
          4. Removing duplicate molecules (by canonical SMILES).
          5. Removing rows with non-numeric target values.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame from :meth:`load`.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with a new ``canonical_smiles`` column.
        """
        n_raw = len(df)
        df = df.copy()

        # ── Step 1: Drop missing values ───────────────────────────────────────
        df.dropna(subset=[self.smiles_col, self.target_col], inplace=True)

        # ── Step 2: Coerce target to numeric ──────────────────────────────────
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")
        df.dropna(subset=[self.target_col], inplace=True)

        # ── Step 3: Validate & canonicalize SMILES ────────────────────────────
        canonical = []
        valid_idx = []
        for idx, smi in df[self.smiles_col].items():
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                canonical.append(Chem.MolToSmiles(mol))   # canonical form
                valid_idx.append(idx)
            else:
                logger.debug("Invalid SMILES skipped: %s", smi)

        df = df.loc[valid_idx].copy()
        df["canonical_smiles"] = canonical

        # ── Step 4: Remove duplicates ─────────────────────────────────────────
        n_before_dedup = len(df)
        df.drop_duplicates(subset=["canonical_smiles"], keep="first", inplace=True)

        logger.info(
            "Cleaning: %d → %d rows  "
            "(dropped %d invalid SMILES, %d duplicates)",
            n_raw,
            len(df),
            n_raw - n_before_dedup,
            n_before_dedup - len(df),
        )
        return df.reset_index(drop=True)

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split cleaned data into train / validation / test sets.

        The split preserves the approximate distribution of the target
        property by binning values into deciles for stratification.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame from :meth:`clean`.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, val_df, test_df)
        """
        # Create decile bins for stratified splitting
        y = df[self.target_col].values
        bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=bins,
            random_state=self.random_seed,
        )

        # Recompute bins for the remaining train+val data
        y_tv  = train_val[self.target_col].values
        bins_tv = pd.qcut(y_tv, q=10, labels=False, duplicates="drop")

        train, val = train_test_split(
            train_val,
            test_size=self.val_size,
            stratify=bins_tv,
            random_state=self.random_seed,
        )

        logger.info(
            "Split → train: %d | val: %d | test: %d",
            len(train),
            len(val),
            len(test),
        )
        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    def normalize_targets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit a StandardScaler on training targets and transform all splits.

        The scaler is stored as ``self.target_scaler`` for inverse-transform
        during evaluation.

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            Split DataFrames from :meth:`split`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (y_train, y_val, y_test) — normalized target arrays.
        """
        y_train = train_df[self.target_col].values.reshape(-1, 1)
        y_val   = val_df[self.target_col].values.reshape(-1, 1)
        y_test  = test_df[self.target_col].values.reshape(-1, 1)

        # Fit only on training data to avoid data leakage
        y_train_norm = self.target_scaler.fit_transform(y_train).ravel()
        y_val_norm   = self.target_scaler.transform(y_val).ravel()
        y_test_norm  = self.target_scaler.transform(y_test).ravel()

        self._is_fitted = True
        logger.info(
            "Target normalization: mean=%.4f, std=%.4f",
            self.target_scaler.mean_[0],
            self.target_scaler.scale_[0],
        )
        return y_train_norm, y_val_norm, y_test_norm

    def inverse_transform_targets(self, y_norm: np.ndarray) -> np.ndarray:
        """
        Inverse-transform normalized predictions back to original scale.

        Parameters
        ----------
        y_norm : np.ndarray
            Normalized prediction array.

        Returns
        -------
        np.ndarray
            Predictions in original (un-normalized) scale.
        """
        if not self._is_fitted:
            raise RuntimeError("Call normalize_targets() before inverse_transform.")
        return self.target_scaler.inverse_transform(
            y_norm.reshape(-1, 1)
        ).ravel()

    def run(
        self, path: str | Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full preprocessing pipeline: load → clean → split.

        Parameters
        ----------
        path : str or Path
            Path to the input CSV file.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, val_df, test_df)
        """
        df = self.load(path)
        df = self.clean(df)
        return self.split(df)
