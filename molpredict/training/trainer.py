"""
Training and Optimization Loop
==============================
Contains the Trainer class handling:
  - Mini-batch gradient descent
  - Learning rate scheduling (Cosine / Step)
  - Validation evaluation
  - Early stopping
  - Checkpoint saving
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from molpredict.evaluation.metrics import MolecularMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages the training, validation, and checkpointing lifecycle.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        checkpoint_dir: str = "checkpoints/",
        patience: int = 10,
        device: str = "auto",
        target_scaler=None,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.target_scaler = target_scaler

        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        self.best_val_loss = float("inf")
        self.best_metrics: Dict[str, float] = {}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            x = batch["features"].to(self.device)
            y = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x)
            loss = self.criterion(preds, y)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in loader:
            x = batch["features"].to(self.device)
            y = batch["target"].to(self.device)

            preds = self.model(x)
            loss = self.criterion(preds, y)
            total_loss += loss.item() * len(y)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        y_p = np.array(all_preds)
        y_t = np.array(all_targets)

        # Invert scaling if scaler provided
        if self.target_scaler is not None:
            y_p = self.target_scaler.inverse_transform(y_p.reshape(-1, 1)).ravel()
            y_t = self.target_scaler.inverse_transform(y_t.reshape(-1, 1)).ravel()

        metrics = MolecularMetrics.compute_all(y_t, y_p)
        return avg_loss, metrics

    def fit(self) -> Dict[str, float]:
        """Runs full training loop with early stopping."""
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:03d}/{self.epochs:03d} | Train Loss: {train_loss:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} | Val R2: {val_metrics['r2']:.4f}"
            )

            # Checkpoint best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_metrics = val_metrics
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    self.checkpoint_dir / "best_model.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        return self.best_metrics
