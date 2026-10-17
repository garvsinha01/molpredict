"""
Evaluation Metrics Module
=========================
Standard benchmarking metrics for regression on chemical properties:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Coefficient of Determination (R²)
  - Pearson Correlation Coefficient (r)
  - Spearman Rank Correlation (ρ)
"""

from typing import Dict
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MolecularMetrics:
    """Computes statistical and regression metrics for model predictions."""

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate full set of regression metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target property values.
        y_pred : np.ndarray
            Model predicted values.

        Returns
        -------
        dict
            Dictionary containing RMSE, MAE, R2, Pearson_r, and Spearman_rho.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # Check for non-constant arrays before calculating correlation
        if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
            pearson_r, _ = stats.pearsonr(y_true, y_pred)
            spearman_rho, _ = stats.spearmanr(y_true, y_pred)
        else:
            pearson_r, spearman_rho = 0.0, 0.0

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "pearson_r": round(float(pearson_r), 4),
            "spearman_rho": round(float(spearman_rho), 4),
        }
