import sys
import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import Preprocessor
from src.data.dataset import create_dataloaders, create_reconstruction_loaders
from tests.conftest import N_CELLS, N_GENES, N_HVG, EMB_DIM, BATCH_SIZE

# We don't test the Preprocessor logic (like filtering) here
# as it's mainly calling Scanpy, which is already tested.
# We focus on our *glue code*: the DataLoader creation.

def test_create_annotation_dataloaders(mock_processed_adata, mock_config):
    """
    Tests the main annotation data loader.
    Checks: correct splits, batch shapes, and dtypes.
    """
    adata = mock_processed_adata
    config = mock_config

    train_loader, val_loader, test_loader, encoder = create_dataloaders(
        adata=adata,
        config=config,
        embedding_key="c2s_cell_embeddings",
        label_key="cell_type"
    )
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert isinstance(encoder, LabelEncoder)
    
    # Check classes
    assert list(encoder.classes_) == ['type_A', 'type_B']
    
    # Check splits (80% train_full, 10% of that is val)
    # 100 * 0.8 = 80 (train_full)
    # 100 * 0.2 = 20 (test)
    # 80 * 0.1 = 8 (val)
    # 80 * 0.9 = 72 (train)
    assert len(train_loader.dataset) == 72
    assert len(val_loader.dataset) == 8
    assert len(test_loader.dataset) == 20
    
    # Check one batch
    x, y = next(iter(train_loader))
    
    assert x.shape == (BATCH_SIZE, EMB_DIM)
    assert y.shape == (BATCH_SIZE,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long  # Crucial for CrossEntropyLoss

def test_create_reconstruction_loaders(mock_processed_adata, mock_config):
    """
    Tests the reconstruction data loader.
    Checks: batch shapes and dtypes (X and Y should be float).
    """
    adata = mock_processed_adata
    config = mock_config

    train_loader, test_loader = create_reconstruction_loaders(
        adata=adata,
        config=config,
        embedding_key="c2s_cell_embeddings"
    )
    
    # Check splits (subsample=80, test_size=0.2)
    # 80 * 0.8 = 64 (train)
    # 80 * 0.2 = 16 (test)
    assert len(train_loader.dataset) == 64
    assert len(test_loader.dataset) == 16
    
    # Check one batch
    x, y = next(iter(train_loader))
    
    assert x.shape == (BATCH_SIZE, EMB_DIM)
    assert y.shape == (BATCH_SIZE, N_HVG) # Target is gene expression
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32 # Crucial for MSELoss