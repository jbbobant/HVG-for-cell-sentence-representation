import torch.nn as nn
import torch

class GeneReconstructor(nn.Module):
    """
    A Linear Reconstruction model.
    Maps Cell Embeddings -> Gene Expression Profile.
    
    As per Report Section II.A.2: "linear function of the log-transformed rank"
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Dimension of the cell embedding
            output_dim: Number of genes to reconstruct (e.g., 2000)
        """
        super().__init__()
        # Simple Linear mapping (Weights = beta coefficients)
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)