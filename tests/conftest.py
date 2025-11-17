import pytest
import numpy as np
import pandas as pd
import anndata
from scipy.sparse import csr_matrix

# Define constants for our mock data
N_CELLS = 100
N_GENES = 500
N_HVG = 100
EMB_DIM = 50
BATCH_SIZE = 16

@pytest.fixture
def mock_config() -> dict:
    """Provides a minimal config for testing."""
    return {
        'project': {'seed': 1234, 'device': 'cpu'},
        'preprocessing': {
            'min_genes_per_cell': 1,
            'min_cells_per_gene': 1,
            'max_n_genes_by_counts': 10000,
            'max_pct_mt': 50.0,
            'n_hvg': N_HVG,
            'use_hvg': True
        },
        'reconstruction': {
            'subsample_size': 80,
            'model': {'learning_rate': 0.01}
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'train_split': 0.8,
            'val_split': 0.1
        },
        'model': {
            'hidden_dim_1': 32,
            'hidden_dim_2': 16,
            'dropout': 0.1,
            'learning_rate': 0.001
        }
    }

@pytest.fixture
def mock_processed_adata() -> anndata.AnnData:
    """
    Creates a small, processed AnnData object simulating the output
    of the Preprocessor, ready for the DataLoader functions.
    """
    # 1. Create expression data (X)
    # Use sparse matrix as it's common in scRNA-seq
    X_sparse = csr_matrix(np.random.rand(N_CELLS, N_HVG).astype(np.float32))
    
    # 2. Create observation metadata (obs)
    obs_df = pd.DataFrame(
        {
            'cell_type': ['type_A' if i % 2 == 0 else 'type_B' for i in range(N_CELLS)]
        },
        index=[f'cell_{i}' for i in range(N_CELLS)]
    )
    
    # 3. Create variable metadata (var)
    var_df = pd.DataFrame(
        index=[f'gene_{i}' for i in range(N_HVG)]
    )
    
    # 4. Create embedding data (obsm)
    obsm_dict = {
        'c2s_cell_embeddings': np.random.rand(N_CELLS, EMB_DIM).astype(np.float32)
    }
    
    # 5. Assemble AnnData
    adata = anndata.AnnData(X=X_sparse, obs=obs_df, var=var_df, obsm=obsm_dict)
    
    return adata