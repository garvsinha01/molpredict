"""
Training Entry Point
====================
CLI script to run data preprocessing, feature engineering, model training,
and evaluation.
"""

import argparse
import logging
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from molpredict.data.preprocessing import DataPreprocessor
from molpredict.data.dataset import MolecularDataset
from molpredict.features.engineering import FeatureEngineer
from molpredict.models.mlp import MolecularMLP
from molpredict.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train MolPredict deep learning model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info("Initializing DataPreprocessor...")
    preprocessor = DataPreprocessor(
        smiles_col=cfg["data"]["smiles_col"],
        target_col=cfg["data"]["target_col"],
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_seed=cfg["data"]["random_seed"],
    )

    train_df, val_df, test_df = preprocessor.run(cfg["data"]["path"])
    y_train_norm, y_val_norm, y_test_norm = preprocessor.normalize_targets(train_df, val_df, test_df)

    logger.info("Extracting Molecular Features...")
    engineer = FeatureEngineer(
        radius=cfg["features"]["fingerprint"]["radius"],
        n_bits=cfg["features"]["fingerprint"]["n_bits"],
        use_descriptors=cfg["features"]["descriptors"]["enabled"],
        normalize_descriptors=cfg["features"]["descriptors"]["normalize"],
    )

    X_train, valid_tr = engineer.extract_features(train_df["canonical_smiles"].tolist(), fit_scaler=True)
    X_val, valid_val = engineer.extract_features(val_df["canonical_smiles"].tolist(), fit_scaler=False)
    X_test, valid_te = engineer.extract_features(test_df["canonical_smiles"].tolist(), fit_scaler=False)

    train_dataset = MolecularDataset(X_train, y_train_norm[valid_tr], train_df["canonical_smiles"].iloc[valid_tr].tolist())
    val_dataset = MolecularDataset(X_val, y_val_norm[valid_val], val_df["canonical_smiles"].iloc[valid_val].tolist())
    test_dataset = MolecularDataset(X_test, y_test_norm[valid_te], test_df["canonical_smiles"].iloc[valid_te].tolist())

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_dataset.feature_dim
    logger.info(f"Feature Dimension: {input_dim}")

    model = MolecularMLP(
        input_dim=input_dim,
        hidden_dims=cfg["model"]["mlp"]["hidden_dims"],
        dropout=cfg["model"]["mlp"]["dropout"],
        use_batch_norm=cfg["model"]["mlp"]["batch_norm"],
        activation=cfg["model"]["mlp"]["activation"],
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        checkpoint_dir=cfg["checkpoint"]["dir"],
        patience=cfg["training"]["early_stopping"]["patience"],
        target_scaler=preprocessor.target_scaler,
    )

    logger.info("Starting training loop...")
    best_val_metrics = trainer.fit()
    logger.info(f"Training completed! Best Validation Metrics: {best_val_metrics}")

    # Evaluate on test set
    _, test_metrics = trainer.evaluate(test_loader)
    logger.info(f"📊 Final Test Set Metrics: {test_metrics}")


if __name__ == "__main__":
    main()
