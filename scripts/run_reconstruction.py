import sys
import os
import argparse
import logging
import torch

# Ensure src is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.common import load_config, setup_logger, seed_everything, get_device
from src.data import Preprocessor
from src.data.dataset import create_reconstruction_loaders
from src.models.reconstructor import GeneReconstructor
from src.engine.trainer_regression import RegressionTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Expression Reconstruction")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger = setup_logger("CellSentence.Reconstruction", config['paths']['output_dir'])
    
    seed_everything(config['project']['seed'])
    device = get_device(config['project']['device'])

    # 1. Data
    preprocessor = Preprocessor(config['preprocessing'])
    preprocessor.load_data(config['paths']['input_data'])
    adata = preprocessor.adata
    
    train_loader, test_loader = create_reconstruction_loaders(
        adata=adata,
        config=config
    )
    
    # 2. Model
    # Input: Embedding dim, Output: Number of Genes (from adata.shape)
    input_dim = adata.obsm["c2s_cell_embeddings"].shape[1]
    output_dim = adata.shape[1] # The number of genes (Y)
    
    logger.info(f"Reconstructing {output_dim} genes from {input_dim} dimensions.")
    
    model = GeneReconstructor(input_dim, output_dim)
    
    # 3. Train
    trainer = RegressionTrainer(model, config, device)
    trainer.fit(train_loader, test_loader)

if __name__ == "__main__":
    main()