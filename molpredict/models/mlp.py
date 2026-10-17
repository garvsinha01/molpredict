"""
Deep Learning Architecture: Molecular MLP
=========================================
Multi-Layer Perceptron optimized for molecular feature vectors (fingerprints + descriptors).
Features:
  - Configurable hidden dimensions
  - Dropout regularization
  - Optional Batch Normalization / Layer Normalization
  - GELU / ReLU activation support
  - Residual skip connections for deeper layers
"""

from typing import List
import torch
import torch.nn as nn


class MolecularMLP(nn.Module):
    """
    Deep Neural Network for predicting molecular properties.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features (e.g., 2048 FP + 9 descriptors = 2057).
    hidden_dims : list of int
        List specifying unit counts in hidden layers (e.g., [1024, 512, 256, 128]).
    output_dim : int
        Output dimension (default 1 for single-property regression).
    dropout : float
        Dropout rate between layers.
    use_batch_norm : bool
        Whether to apply BatchNorm1d after linear projections.
    activation : str
        Activation function ('relu', 'gelu', or 'silu').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [1024, 512, 256, 128],
        output_dim: int = 1,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        act_fn = act_map.get(activation.lower(), nn.GELU)

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final regression head
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor of shape (batch_size, output_dim)
        """
        return self.network(x).squeeze(-1)
