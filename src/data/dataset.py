import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import anndata
import logging
from typing import Tuple, List, Dict, Optional
from scipy.sparse import issparse

class ReconstructionDataset(Dataset):
    """
    Dataset for Regression: Input (Embeddings) -> Output (Gene Expression)
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Args:
            X: Embeddings (Cells x Hidden_Dim)
            Y: Normalized Expression Matrix (Cells x Genes)
        """
        self.x = torch.tensor(X, dtype=torch.float32)
        # Ensure Y is dense for PyTorch training
        if issparse(Y):
            Y = Y.toarray()
        self.y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_reconstruction_loaders(
    adata: anndata.AnnData,
    config: dict,
    embedding_key: str = "c2s_cell_embeddings"
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates loaders for the Reconstruction task.
    Target (Y) is the actual gene expression data (adata.X).
    """
    
    # 1. Subsampling (As per report )
    n_subsample = config['reconstruction'].get('subsample_size', None)
    if n_subsample and n_subsample < adata.n_obs:
        logger.info(f"Subsampling dataset to {n_subsample} cells for reconstruction...")
        # Use sc.pp.subsample to maintain distribution or random indices
        sc.pp.subsample(adata, n_obs=n_subsample)
    
    # 2. Extract Features (X)
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding key '{embedding_key}' not found.")
    X = adata.obsm[embedding_key].astype(np.float32)
    
    # 3. Extract Targets (Y) - The Gene Expression Profile
    # We assume adata.X is already normalized/log1p transformed [cite: 72]
    Y = adata.X
    
    # 4. Split (Train/Test)
    # For reconstruction, we usually just need a test set to validate correlation
    seed = config['project']['seed']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )
    
    train_ds = ReconstructionDataset(X_train, Y_train)
    test_ds = ReconstructionDataset(X_test, Y_test)
    
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader