"""This file contains PyTorch neural network modules."""

from torch import nn, Tensor


class LitNetwork(nn.Module):
    """PyTorch neural network module."""

    def __init__(self):
        """Initialize a network."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the architecture."""
        pass
