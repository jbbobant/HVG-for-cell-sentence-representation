import torch
import torch.nn as nn
from typing import Dict, Any

class CellClassifier(nn.Module):
    """
    A compact MLP classifier for cell-type annotation.
    Architecture based on CellFM design:
    Linear -> BN -> GELU -> Dropout -> Linear -> BN -> GELU -> Dropout -> Linear
    """
    
    def __init__(self, input_dim: int, num_classes: int, config: Dict[str, Any]):
        """
        Args:
            input_dim: Dimension of input features (e.g., 2000 for HVG genes)
            num_classes: Number of target cell types
            config: Model configuration dictionary
        """
        super().__init__()
        
        h1 = config['model']['hidden_dim_1']  # e.g., 512
        h2 = config['model']['hidden_dim_2']  # e.g., 256
        dropout_rate = config['model']['dropout']
        
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Layer 2
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Output Layer
            nn.Linear(h2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)